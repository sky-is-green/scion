"""Solver QAT on the 9B Clef f16 body (rental, student-only).

Trains the ternary masters of the reverse-loaded Clef body against a
precomputed f16 teacher hidden-state cache (built locally by `clef_cache.py
--generic-only`).  The recipe is the derisk-validated one (`dense/qat_derisk.py`
on Qwen3.5-0.8B): block-Hadamard fold, Lloyd g128 STE, hidden-state MSE +
cosine, Adafactor, grad checkpointing.

Memory plan (48 GB card): bf16 masters (13.8 GB) + bf16 grads (13.8 GB) +
frozen bf16 weights (~4 GB) + checkpointed activations (batch 2 x 512) —
fits with ~8 GB headroom; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

Usage (on the pod):
    PYTHONPATH=/path/gguf-py python dense/qat_9b.py \
        --model ./clef-flash-ternary --cache ./teacher-cache \
        --out ./qat-run --steps 800 --batch 2 --seq 512
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WORKSPACE = Path("/home/penis/Desktop/work")
sys.path.insert(0, str(WORKSPACE / "bonsai2-ternary-forensics"))

from clef_dense_load import load_text_model_streamed, patch_recurrent_gdn  # noqa: E402
from qat_derisk import make_rot, ROTATED, UNROTATED  # noqa: E402
from quant import _lloyd_scale  # noqa: E402
from bonsai_forensics import rotation as bf_rotation  # noqa: E402
from transformers.optimization import Adafactor  # noqa: E402


def ternary_lloyd_ste(w: torch.Tensor, group: int = 128) -> torch.Tensor:
    """Lloyd ternary with STE; works for bf16 masters (math in f32)."""
    with torch.no_grad():
        wf = w.float()
        g = wf.reshape(wf.shape[0], wf.shape[-1] // group, group)
        mean = g.abs().mean(dim=-1)
        scale = _lloyd_scale(g, mean).unsqueeze(-1)
        codes = torch.clamp(torch.round(g / scale.clamp_min(1e-12)), -1, 1)
        q = (codes * scale).reshape(wf.shape).to(w.dtype)
    return w + (q - w).detach()


def _fwht_last(x: torch.Tensor) -> torch.Tensor:
    """Fast Walsh-Hadamard transform over the last (power-of-two) axis."""
    n = x.shape[-1]
    h = 1
    while h < n:
        x = x.reshape(*x.shape[:-1], n // (2 * h), 2, h)
        a = x[..., 0, :].clone()
        b = x[..., 1, :].clone()
        x = torch.stack([a + b, a - b], dim=-2).reshape(*x.shape[:-3], n)
        h *= 2
    return x


def apply_rotation_fast(x: torch.Tensor, signs: torch.Tensor,
                        g: int) -> torch.Tensor:
    """``R x = H (S ⊙ x)/sqrt(g)`` per block, matching the forensics basis."""
    d = x.shape[-1]
    y = x.reshape(*x.shape[:-1], d // g, g) * signs
    y = _fwht_last(y) / math.sqrt(g)
    return y.reshape(*x.shape[:-1], d)


class TernaryLinear9B(torch.nn.Module):
    """Linear with a ternary STE master, folded once, fast block-Hadamard input.

    ``fold`` is the dense ``Rᵀ`` used only at init; the forward applies the
    O(d log d) butterfly.  The Lloyd scale is refreshed every ``refresh``
    forwards (it moves slowly under lr 5e-5), keeping the dominant cost to one
    bf16 codes pass.
    """

    def __init__(self, base: torch.nn.Linear, fold: torch.Tensor | None,
                 signs: torch.Tensor | None, g: int, group: int = 128,
                 refresh: int = 50):
        super().__init__()
        w = base.weight.detach().to(torch.float32)
        if fold is not None:
            w = w @ fold
        self.weight = torch.nn.Parameter(w)
        self.group = group
        self.g = g
        self.refresh = max(1, int(refresh))
        self._calls = 0
        self._scale = None
        self.register_buffer("signs", signs, persistent=False)
        if base.bias is not None:
            self.bias = torch.nn.Parameter(base.bias.detach().to(torch.float32),
                                           requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def _refresh_scale(self) -> None:
        with torch.no_grad():
            wf = self.weight.float()
            g = wf.reshape(wf.shape[0], -1, self.group)
            mean = g.abs().mean(dim=-1)
            self._scale = _lloyd_scale(g, mean).unsqueeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._scale is None or (self._calls % self.refresh) == 0:
            self._refresh_scale()
        self._calls += 1
        w = self.weight
        scale = self._scale.to(w.dtype)
        with torch.no_grad():
            g = w.reshape(w.shape[0], -1, self.group)
            codes = torch.clamp(torch.round(g / scale.clamp_min(1e-12)), -1, 1)
            q = (codes * scale).reshape(w.shape)
        wq = (w + (q - w).detach()).to(x.dtype)
        if self.signs is not None:
            x = apply_rotation_fast(x, self.signs.to(x.dtype), self.g)
        return F.linear(x, wq, self.bias)


def wrap_model(model, seed: int, group: int, dtype: torch.dtype,
               skip: tuple[str, ...] = ()) -> dict:
    for p in model.parameters():
        p.requires_grad_(False)
    cache: dict[int, torch.Tensor] = {}
    n_rot = n_unrot = 0
    params = 0
    device = next(model.parameters()).device
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, torch.nn.Linear):
            continue
        if any(name.endswith(s) for s in skip):
            continue
        rot = any(name.endswith(t) for t in ROTATED)
        unrot = any(name.endswith(t) for t in UNROTATED)
        if not (rot or unrot):
            continue
        a = fold = signs = None
        g = 1024
        if rot:
            width = mod.in_features
            if width not in cache:
                rots = bf_rotation.rotations_for(width, seed)
                g = len(rots[0])
                signs = torch.from_numpy(
                    np.stack(rots).astype(np.float32)).to(device)
                fold = make_rot(width, seed, str(device))
                cache[width] = (fold, signs, g)
            fold, signs, g = cache[width]
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        wrapped = TernaryLinear9B(mod, fold, signs, g, group)
        wrapped.weight.data = wrapped.weight.data.to(dtype)
        setattr(parent, child, wrapped)
        params += wrapped.weight.numel()
        n_rot += int(rot)
        n_unrot += int(unrot)
    return {"rotated": n_rot, "unrotated": n_unrot, "master_params": params,
            "rotation_widths": sorted(cache)}


class HiddenCache:
    """Memory-mapped teacher cache from `clef_cache.py --generic-only`."""

    def __init__(self, path: Path):
        idx = json.loads((path / "index.json").read_text())
        self.dir = path
        self.entries = [e for e in idx["entries"] if e.get("kind") == "text"]
        if not self.entries:
            raise SystemExit(f"{path}: no text entries in index.json")

    def __len__(self) -> int:
        return len(self.entries)

    def get(self, i: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        e = self.entries[i]
        z = np.load(self.dir / e["file"])
        ids = torch.from_numpy(z["input_ids"].astype(np.int64)).unsqueeze(0)
        hid = torch.from_numpy(z["hidden"]).unsqueeze(0)
        return ids.to(device), hid.to(device)


@torch.no_grad()
def eval_cache(model, cache: HiddenCache, idxs: list[int], device: str) -> dict:
    model.eval()
    mse = cos = n = 0.0
    for i in idxs:
        ids, th = cache.get(i, device)
        sh = model(input_ids=ids, use_cache=False).last_hidden_state
        mse += float(F.mse_loss(sh.float(), th.float()))
        cos += float(F.cosine_similarity(sh.float(), th.float(), dim=-1).mean())
        n += 1
    model.train()
    return {"mse": mse / n, "cos": cos / n}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True,
                    help="dir containing clef-flash-f16.gguf")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--holdout", type=int, default=32,
                    help="last N cache windows reserved for eval")
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--skip-targets", default="")
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    ap.add_argument("--kernel", default="chunked", choices=("chunked", "recurrent"),
                    help="GDN prefill kernel; the cache was captured recurrent and "
                         "chunked matches it at cos 0.9999 on the pod")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-steps-wall", type=int, default=6 * 3600)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = args.device
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cache = HiddenCache(Path(args.cache))
    holdout = list(range(len(cache) - args.holdout, len(cache)))
    train_idx = list(range(0, len(cache) - args.holdout))
    print(f"cache: {len(cache)} windows ({len(train_idx)} train / {args.holdout} holdout)",
          flush=True)

    print("loading f16 body (bf16, recurrent GDN) ...", flush=True)
    model, _, loaded = load_text_model_streamed(
        Path(args.model) / "clef-flash-f16.gguf", device=device, dtype=dtype)
    if args.kernel == "recurrent":
        patch_recurrent_gdn(model)
    model.config.use_cache = False
    print(f"model loaded: {loaded} params (kernel {args.kernel})", flush=True)

    coverage = wrap_model(model, args.seed, args.group, dtype,
                          skip=tuple(s for s in args.skip_targets.split(",") if s))
    print(f"wrapped: {coverage}", flush=True)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.train()

    base = eval_cache(model, cache, holdout, device)
    print(f"post-hoc ternary (holdout): mse {base['mse']:.4f} cos {base['cos']:.4f}",
          flush=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adafactor(
        trainable, lr=args.lr, eps=(1e-30, 0.001), clip_threshold=1.0,
        decay_rate=-0.8, beta1=None, weight_decay=0.0, scale_parameter=False,
        relative_step=False, warmup_init=False)

    def lr_lambda(step: int) -> float:
        if step < args.warmup:
            return (step + 1) / max(1, args.warmup)
        p = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    rng = np.random.default_rng(args.seed)
    history = [{"step": 0, **base}]
    t0 = time.time()
    for step in range(1, args.steps + 1):
        if time.time() - t0 > args.max_steps_wall:
            print("wall-clock guard hit; stopping", flush=True)
            break
        sel = rng.choice(len(train_idx), size=args.batch, replace=False)
        blobs = [cache.get(train_idx[int(i)], device) for i in sel]
        ids = torch.cat([b[0] for b in blobs], dim=0)
        th = torch.cat([b[1] for b in blobs], dim=0)
        sh = model(input_ids=ids, use_cache=False).last_hidden_state
        loss_h = F.mse_loss(sh.float(), th.float()) + (
            1.0 - F.cosine_similarity(sh.float(), th.float(), dim=-1)).mean()
        loss_h.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        if step % 10 == 0 or step == 1:
            print(f"[{time.strftime('%H:%M:%S')}] step {step}/{args.steps} "
                  f"loss {float(loss_h):.4f} lr {scheduler.get_last_lr()[0]:.2e}",
                  flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            ev = eval_cache(model, cache, holdout, device)
            ev["step"] = step
            history.append(ev)
            print(f"[{time.strftime('%H:%M:%S')}] eval step {step}: "
                  f"mse {ev['mse']:.4f} cos {ev['cos']:.4f}", flush=True)
            model.train()
        if step % args.save_every == 0 or step == args.steps:
            ckpt = out / f"masters-step{step}.pt"
            torch.save({n: m.weight.detach().cpu()
                        for n, m in model.named_modules()
                        if isinstance(m, TernaryLinear9B)}, ckpt)
            print(f"  saved {ckpt}", flush=True)

    (out / "qat.json").write_text(json.dumps({
        "coverage": coverage, "args": vars(args), "history": history,
        "elapsed_s": time.time() - t0}, indent=1))
    print(f"done in {time.time()-t0:.0f}s; wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
