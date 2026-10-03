"""End-to-end residual corrections for the deployed Clef-Flash PQ2_0 body.

The dense analogue of Scion's MoE correction trainer, but for the hybrid Qwen3.5
dense stack: the frozen deployed body runs **with the quantiser in the loop**,
and rank-512 branches on the attention output and the MLP output are trained by
hidden-state KD to the f16 teacher (primary) and decision KD through the frozen
joint head (secondary).

Dense-specific decisions, grounded in the Bonsai-2 27B forensics
(`bonsai2-ternary-forensics/docs/FAILURES.md` F4/F6/F7 and the recipe ledger):
  * **end-to-end only** -- per-layer/block-wise KD compounds and is a dead-end;
  * the deployed CPU runtime uses the **recurrent** GDN; patch it in, or the
    corrections will not transfer;
  * managed LR decay; no exotic quantiser tricks;
  * clean holdout (the 30 test records are never trained on).

Training forward: the reverse-loaded deployed PQ2_0 weights (exact), frozen.
Decision KD is computed on CPU with a manual gradient bridge so the joint head
and lm_head never occupy VRAM.

Usage:
    python dense/clef_corrections.py --cache <dir> --out <dir> --smoke
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
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from clef_dense_load import (load_text_model_streamed, patch_recurrent_gdn,  # noqa: E402
                             patch_truncated_gdn)
from quant import ternary_lloyd, ternary_absmean  # noqa: E402
import clef_head as H  # noqa: E402

DEFAULT_MODEL = "/home/penis/Desktop/work/models/clef-flash-ternary"
HIVE = Path("/home/penis/Desktop/work/hivebench/experiments/cascade")
EVAL_JSON = {
    "train": HIVE / "results/clef-flash-judge-eval-train.json",
    "test": HIVE / "results/clef-flash-judge-eval-test.json",
}


# -------------------------------------------------------------------- branches

class CorrectionBranch(nn.Module):
    """Low-rank residual branch, zero-initialised on the output side.

    ``quant="g128"`` ternarises both factors with the deployed Lloyd rule under
    STE (Scion D5: post-hoc ternarisation is 14-25x worse); ``"rank"`` folds a
    per-rank scale into the up factor; ``"fp32"`` is the reference.
    """

    def __init__(self, hidden: int, rank: int, quant: str = "g128",
                 out_dim: int | None = None):
        super().__init__()
        self.down = nn.Linear(hidden, rank, bias=False)
        self.up = nn.Linear(rank, out_dim or hidden, bias=False)
        self.quant = quant
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def _weights(self):
        wd, wu = self.down.weight, self.up.weight
        if self.quant == "fp32":
            return wd, wu
        with torch.no_grad():
            if self.quant == "g128":
                wdq, wuq = ternary_lloyd(wd, 128), ternary_lloyd(wu, 128)
            else:  # rank component scales folded into the up factor
                s = wd.abs().mean(dim=1).clamp_min(1e-8)
                qd = torch.clamp(torch.round(wd / s[:, None]), -1, 1)
                t = wu.abs().mean(dim=0).clamp_min(1e-8)
                qu = torch.clamp(torch.round(wu / t[None, :]), -1, 1)
                wdq, wuq = qd, qu * (s * t)[None, :]
        return wdq + (wd - wd.detach()), wuq + (wu - wu.detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wd, wu = self._weights()
        xf = x.float()
        return F.linear(F.linear(xf, wd), wu).to(x.dtype)


class BranchWrap(nn.Module):
    """``sub(x) + branch(x)``; the branch reads ``x`` (the submodule's input)."""

    def __init__(self, sub: nn.Module, hidden: int, rank: int, quant: str,
                 out_dim: int | None = None):
        super().__init__()
        self.sub = sub
        self.branch = CorrectionBranch(hidden, rank, quant, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sub(x) + self.branch(x)


def attach_branches(model, rank: int, quant: str, target: str = "both") -> None:
    """Wrap the attention output (GDN/full) and/or the MLP output of each block."""
    hidden = model.config.hidden_size
    for layer in model.layers:
        dev = next(layer.mlp.parameters()).device
        if target in ("attn_out", "both"):
            if getattr(layer, "layer_type", "") == "linear_attention":
                p = layer.linear_attn.out_proj
                layer.linear_attn.out_proj = BranchWrap(
                    p, p.in_features, rank, quant, out_dim=p.out_features).to(dev)
            else:
                p = layer.self_attn.o_proj
                layer.self_attn.o_proj = BranchWrap(
                    p, p.in_features, rank, quant, out_dim=p.out_features).to(dev)
        if target in ("mlp_out", "both"):
            layer.mlp = BranchWrap(layer.mlp, hidden, rank, quant).to(dev)


def freeze_body(model) -> int:
    for p in model.parameters():
        p.requires_grad_(False)
    for name, p in model.named_parameters():
        if ".branch." in name:
            p.requires_grad_(True)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------- cache

class TeacherCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.entries = json.loads((self.root / "index.json").read_text())["entries"]

    def __len__(self) -> int:
        return len(self.entries)

    def load(self, i: int, device) -> dict:
        e = self.entries[i]
        z = np.load(self.root / e["file"])
        out = {"input_ids": torch.tensor(z["input_ids"], dtype=torch.long, device=device),
               "hidden": torch.tensor(z["hidden"], dtype=torch.float32, device=device),
               "entry": e}
        if "option_logits" in z:
            out["option_logits"] = torch.tensor(z["option_logits"], dtype=torch.float32)
        return out

    def decision_indices(self, split: str | None = None) -> list[int]:
        return [i for i, e in enumerate(self.entries)
                if e.get("has_decision") and (split is None or e["split"] == split)]


# ----------------------------------------------------------------------- loss

def hidden_kd(student: torch.Tensor, teacher: torch.Tensor, mse_weight: float = 0.1):
    s, t = student.float(), teacher.float()
    cos = (1.0 - F.cosine_similarity(s, t, dim=-1)).mean()
    if mse_weight <= 0:
        return cos
    # scale-aware MSE: residual energy relative to the teacher's, so the term
    # cannot dominate a low-cosine start and destabilise the recurrent body.
    mse = F.mse_loss(s, t) / t.pow(2).mean().clamp_min(1e-6)
    return cos + mse_weight * mse


@torch.no_grad()
def _head_logits(head, hidden_cpu_f32, ids_cpu, enc, lm_head):
    return head(hidden_cpu_f32.unsqueeze(0).to(torch.bfloat16), ids_cpu.unsqueeze(0),
                torch.ones(1, ids_cpu.numel(), dtype=torch.long), [enc], lm_head)[0][0]


def decision_grad(hidden: torch.Tensor, entry: dict, tok, head, lm_head,
                  teacher_logits: torch.Tensor, temp: float):
    """KL(student || teacher) on the noul logits and its grad w.r.t. ``hidden``.

    The head runs on CPU over a detached copy; the gradient is handed back so the
    body's graph gets it without the head occupying VRAM.  Returns
    ``(loss_value, grad_on_hidden_device)``.
    """
    enc = H.encode(tok, entry["prompt"], entry["candidate"])
    ids_cpu = torch.tensor(enc.input_ids, dtype=torch.long)
    h = hidden.detach().float().cpu().requires_grad_(True)
    logits = head(h.unsqueeze(0), ids_cpu.unsqueeze(0),
                  torch.ones(1, ids_cpu.numel(), dtype=torch.long), [enc], lm_head)[0][0]
    logp = F.log_softmax(logits.float() / temp, dim=-1)
    tlogp = F.log_softmax(teacher_logits.float() / temp, dim=-1)
    kl = (tlogp.exp() * (tlogp - logp)).sum() * (temp ** 2)
    g = torch.autograd.grad(kl, h)[0]
    return float(kl.detach()), g.to(hidden.device)


# -------------------------------------------------------------------- training

def _clip_norm(x: torch.Tensor, max_norm: float) -> torch.Tensor:
    """Bound a gradient tensor's L2 norm (bf16 body backward overflows above it)."""
    n = x.norm()
    return x * (max_norm / n.clamp_min(1e-9)) if float(n) > max_norm else x


def train(args) -> None:
    device = args.device
    # preflight: never touch the GPU unless there is room for the body + margin.
    if device.startswith("cuda") and torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        free_gb, total_gb = free / 1e9, total / 1e9
        need = args.min_free_gb
        print(f"preflight: GPU free {free_gb:.1f}/{total_gb:.1f} GB, need >= {need:.1f}",
              flush=True)
        if free_gb < need:
            raise SystemExit(f"preflight: only {free_gb:.1f} GB free VRAM, need {need:.1f}")
    dt = torch.float32 if args.dtype == "float32" else torch.bfloat16
    print(f"loading deployed PQ2_0 student ({args.dtype}) ...", flush=True)
    model, _, loaded = load_text_model_streamed(
        Path(args.model) / "clef-flash-PQ2_0.gguf", device=device, dtype=dt,
        layer_limit=(args.prefix_layers or None))
    n_gdn = 0
    if args.gdn == "recurrent":
        n_gdn = patch_recurrent_gdn(model)
    elif args.gdn == "truncated":
        n_gdn = patch_truncated_gdn(model, args.gdn_chunk)
    model.config.use_cache = False
    attach_branches(model, args.rank, args.branch_quant, args.target)
    n_train = freeze_body(model)
    print(f"student {loaded} params, {n_gdn} GDN patched; "
          f"trainable {n_train/1e6:.1f}M branches (rank {args.rank}, {args.branch_quant})", flush=True)
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    print(f"grad checkpointing: {getattr(model, 'is_gradient_checkpointing', None)}", flush=True)

    # run the frozen head/lm_head in fp32 on CPU: bf16 overflows in its backward
    head, _ = H.load_joint_head(args.model, device="cpu", dtype=torch.float32)
    lm_head = H.load_lm_head(args.model, device="cpu", dtype=torch.float32)
    tok = H.load_tokenizer(args.model)
    cache = TeacherCache(args.cache)
    print(f"cache {len(cache)} sequences "
          f"(decision train {len(cache.decision_indices('train'))}, "
          f"test {len(cache.decision_indices('test'))})", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    from transformers.optimization import Adafactor
    opt = Adafactor(params, lr=args.lr, eps=(1e-30, 0.001), clip_threshold=1.0,
                    decay_rate=-0.8, beta1=None, weight_decay=0.0,
                    scale_parameter=False, relative_step=False, warmup_init=False)
    print(f"peak VRAM after load+attach: {torch.cuda.max_memory_allocated()/1e9:.2f} GB",
          flush=True)

    rng = np.random.default_rng(args.seed)
    model.train()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    step = 0
    t0 = time.time()
    order = list(range(len(cache)))
    rng.shuffle(order)
    for epoch in range(args.epochs):
        for i in order:
            if args.max_tokens and cache.entries[i]["n_tokens"] > args.max_tokens:
                continue
            rec = cache.load(i, device)
            if not torch.isfinite(rec["hidden"]).all():
                continue  # a non-finite teacher target would poison the run
            ids = rec["input_ids"].unsqueeze(0)
            hidden = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                           use_cache=False).last_hidden_state[0]
            loss_h = hidden_kd(hidden, rec["hidden"], args.mse_weight)
            g = torch.autograd.grad(loss_h, hidden, retain_graph=True)[0]
            total = float(loss_h.detach())
            dec = 0.0
            if args.decision_weight > 0 and "option_logits" in rec:
                kl, g_dec = decision_grad(hidden, rec["entry"], tok, head, lm_head,
                                          rec["option_logits"], args.temp)
                # bound each source before it enters the bf16 body backward: the
                # head's gradient is ~10x the hidden KD's and overflows bf16.
                g_dec = _clip_norm(g_dec, args.upstream_clip)
                g = g + args.decision_weight * g_dec
                total += args.decision_weight * kl
                dec = kl
            g = _clip_norm(g, args.upstream_clip)
            opt.zero_grad(set_to_none=True)
            hidden.backward(g)
            gn = torch.nn.utils.clip_grad_norm_(params, args.clip)
            opt.step()
            step += 1
            if not torch.isfinite(torch.tensor(total)) or not torch.isfinite(gn):
                print(f"step {step}: non-finite (total {total}, grad_norm {gn}); aborting",
                      flush=True)
                torch.save({k: v for k, v in model.state_dict().items() if ".branch." in k},
                           out / f"branches-diverged-step{step}.pt")
                return
            if args.lr_half_every and step >= args.lr_decay_start and step % args.lr_half_every == 0:
                for pg in opt.param_groups:
                    pg["lr"] *= 0.5
            if step % args.log_every == 0 or step == 1:
                mem = torch.cuda.max_memory_allocated() / 1e9
                print(f"step {step} hidden {float(loss_h.detach()):.5f} dec {dec:.5f} "
                      f"total {total:.5f} gnorm {float(gn):.3e} "
                      f"lr {opt.param_groups[0]['lr']:.2e} "
                      f"peakVRAM {mem:.2f}GB ({time.time()-t0:.0f}s)", flush=True)
            if step >= args.steps:
                break
        if step >= args.steps:
            break

    ckpt = out / f"branches-r{args.rank}-{args.branch_quant}-step{step}.pt"
    sd = {k: v for k, v in model.state_dict().items() if ".branch." in k}
    torch.save(sd, ckpt)
    print(f"saved {ckpt} ({sum(v.numel() for v in sd.values())/1e6:.1f}M params)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16",
                    help="body dtype; float32 is the stable rental config")
    ap.add_argument("--rank", type=int, default=512)
    ap.add_argument("--branch-quant", choices=["g128", "rank", "fp32"], default="g128")
    ap.add_argument("--target", choices=["attn_out", "mlp_out", "both"], default="both")
    ap.add_argument("--gdn", choices=["recurrent", "truncated", "chunked"], default="truncated",
                    help="prefill form; 'truncated' = recurrent forward (matches the "
                         "CPU runtime) with the state detached every --gdn-chunk tokens")
    ap.add_argument("--gdn-chunk", type=int, default=64, help="truncated-BPTT horizon")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-half-every", type=int, default=0)
    ap.add_argument("--lr-decay-start", type=int, default=0)
    ap.add_argument("--temp", type=float, default=2.0)
    ap.add_argument("--decision-weight", type=float, default=0.5)
    ap.add_argument("--mse-weight", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--upstream-clip", type=float, default=1.0,
                    help="L2 bound on each gradient source before the body backward")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=0, help="skip cache entries longer than this")
    ap.add_argument("--prefix-layers", type=int, default=0, help="load only N layers (code smoke)")
    ap.add_argument("--min-free-gb", type=float, default=16.0, help="VRAM preflight floor")
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--grad-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--smoke", action="store_true", help="tiny run: 10 steps")
    args = ap.parse_args()
    if args.smoke:
        args.steps = 10
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
