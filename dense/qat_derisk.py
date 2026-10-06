"""QAT derisk on Qwen3.5-0.8B (same hybrid arch as Clef).

Validates the rotation-in-the-loop ternary QAT + teacher-KD recipe planned for
the 9B Clef run, at a scale that fits the local box.

  * student = Qwen3.5-0.8B text model (24 layers, full_attention_interval=4,
    hidden 1024) with the attention/MLP linears replaced by block-Hadamard
    folded ternary masters: GDN ``in_proj_qkv``/``in_proj_z`` and all
    ``q/k/v/o`` + MLP projections rotated, ``out_proj`` unrotated-ternary;
    ``in_proj_a/b`` and every norm stay f32-frozen;
  * ternary = deployed Lloyd group rule (g128) with straight-through gradients
    on the folded master weights;
  * teacher = the same f16 weights (frozen); KD = hidden-state MSE + cosine
    plus temperature KL on the tied-embedding logits;
  * Adafactor per the Bonsai forensics recipe, grad checkpointing, cosine LR.

Measures: f16 PPL -> post-hoc ternary PPL -> QAT PPL curve (wikitext-2 test,
c512 windows) on the non-display GPU.

Usage:
    HIP_VISIBLE_DEVICES=1 .venv-rocm/bin/python dense/qat_derisk.py \
        --model-dir <hf-snapshot-dir> \
        --out models/qwen35-0.8b-qat-derisk/run1 --steps 300
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from clef_paths import WORKSPACE  # noqa: E402

_BONSAI = WORKSPACE / "bonsai2-ternary-forensics"
if _BONSAI.is_dir():
    sys.path.insert(0, str(_BONSAI))

from quant import _lloyd_scale  # noqa: E402
from bonsai_forensics import rotation as bf_rotation  # noqa: E402

from transformers import AutoConfig, AutoTokenizer  # noqa: E402
from transformers.models.qwen3_5 import Qwen3_5TextModel  # noqa: E402
from transformers.optimization import Adafactor  # noqa: E402

ROTATED = (
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
UNROTATED = ("linear_attn.out_proj",)


def ternary_lloyd_ste(w: torch.Tensor, group: int = 128) -> torch.Tensor:
    """Lloyd-rule ternary with straight-through gradients (scale detached)."""
    out_f, in_f = w.shape
    g = w.reshape(out_f, in_f // group, group)
    mean = g.abs().mean(dim=-1)
    scale = _lloyd_scale(g.detach(), mean.detach()).unsqueeze(-1)
    codes = torch.clamp(torch.round(g / scale.clamp_min(1e-12)), -1, 1)
    q = (codes * scale).reshape(out_f, in_f)
    return w + (q - w).detach()


class TernaryLinear(torch.nn.Module):
    """Linear with a ternary STE weight; optional block-Hadamard input fold.

    ``rot`` is ``A`` with ``x' = x @ A`` and ``W' = W @ A`` (row convention),
    built from the forensics' ``materialize_rotation`` (``R x`` convention), so
    ``A = Rᵀ``.
    """

    def __init__(self, base: torch.nn.Linear, rot: torch.Tensor | None,
                 group: int = 128):
        super().__init__()
        w = base.weight.detach().to(torch.float32)
        if rot is not None:
            w = w @ rot
        self.weight = torch.nn.Parameter(w)
        self.group = group
        self.register_buffer("rot", rot, persistent=False)
        if base.bias is not None:
            self.bias = torch.nn.Parameter(base.bias.detach().to(torch.float32),
                                           requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = ternary_lloyd_ste(self.weight, self.group)
        if self.rot is not None:
            x = x.to(self.rot.dtype) @ self.rot
        return F.linear(x, w, self.bias)


def load_text_state(snap: Path) -> dict[str, torch.Tensor]:
    """Checkpoint -> Qwen3_5TextModel state dict (f32), prefix stripped."""
    from safetensors import safe_open
    files = glob.glob(str(snap / "*.safetensors"))
    if not files:
        raise SystemExit(f"no safetensors in {snap}")
    out: dict[str, torch.Tensor] = {}
    with safe_open(files[0], framework="pt") as f:
        for key in f.keys():
            if key.startswith("model.language_model."):
                out[key[len("model.language_model."):]] = \
                    f.get_tensor(key).to(torch.float32)
    return out


def load_text_model(snap: Path, device: str,
                    dtype: torch.dtype = torch.float32) -> Qwen3_5TextModel:
    cfg = AutoConfig.from_pretrained(str(snap))
    model = Qwen3_5TextModel(cfg.text_config)
    state = load_text_state(snap)
    model.load_state_dict(state, strict=True)
    del state
    model.config.use_cache = False
    return model.to(dtype).to(device)


def make_rot(width: int, seed: int, device: str) -> torch.Tensor:
    r = bf_rotation.materialize_rotation(width, seed)
    a = torch.from_numpy(r).to(torch.float32).t().contiguous()
    return a.to(device)


def wrap_model(model: Qwen3_5TextModel, seed: int, group: int,
               use_rotation: bool = True, skip: tuple[str, ...] = ()) -> dict:
    """Replace target linears with TernaryLinear; freeze everything else."""
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
        a = None
        if rot and use_rotation:
            width = mod.in_features
            if width not in cache:
                cache[width] = make_rot(width, seed, str(device))
            a = cache[width]
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        wrapped = TernaryLinear(mod, a, group)
        setattr(parent, child, wrapped)
        params += wrapped.weight.numel()
        n_rot += int(rot)
        n_unrot += int(unrot)
    return {"rotated": n_rot, "unrotated": n_unrot, "master_params": params,
            "rotation_widths": sorted(cache)}


def load_windows(tok, n: int, seq: int, split: str) -> list[list[int]]:
    from datasets import load_dataset
    ds = None
    for name, config in (("Salesforce/wikitext", "wikitext-2-raw-v1"),
                         ("Salesforce/wikitext", "wikitext-103-raw-v1")):
        try:
            ds = load_dataset(name, config, split=split)
            print(f"  corpus {name}/{config}:{split}", flush=True)
            break
        except Exception as e:
            print(f"  dataset {name}/{config}:{split} unavailable: "
                  f"{type(e).__name__}", flush=True)
    if ds is None:
        raise SystemExit("no wikitext dataset available locally")
    buf: list[int] = []
    windows: list[list[int]] = []
    for row in ds:
        text = row.get("text", "")
        if not text.strip():
            continue
        buf.extend(tok(text, add_special_tokens=False).input_ids)
        while len(buf) >= seq and len(windows) < n:
            windows.append(buf[:seq])
            buf = buf[seq:]
        if len(windows) >= n:
            break
    return windows


def sample_batches(windows: list[list[int]], batch: int, steps: int, seed: int):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(windows))
    pos = 0
    for _ in range(steps):
        if pos + batch > len(order):
            order = rng.permutation(len(windows))
            pos = 0
        sel = order[pos:pos + batch]
        pos += batch
        yield [windows[int(i)] for i in sel]


@torch.no_grad()
def generate(model: Qwen3_5TextModel, tok, prompt: str, n: int = 120,
             temp: float = 0.7, seed: int = 0) -> str:
    """Plain completion from the (wrapped ternary) student, tied head."""
    model.eval()
    g = torch.Generator(device=model.device).manual_seed(seed)
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    for _ in range(n):
        h = model(input_ids=ids).last_hidden_state[:, -1, :]
        logits = h @ model.embed_tokens.weight.to(h.dtype).t()
        if temp <= 0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            probs = (logits.float() / temp).softmax(-1)
            nxt = torch.multinomial(probs[0], 1, generator=g).unsqueeze(0)
        ids = torch.cat([ids, nxt], dim=1)
    return tok.decode(ids[0], skip_special_tokens=True)


def logits_kd(sh: torch.Tensor, th: torch.Tensor, head_w: torch.Tensor,
              temp: float = 1.0, chunk: int = 128) -> torch.Tensor:
    """KL(student || teacher) over token chunks; head projection checkpointed.

    Keeps peak memory to ~one chunk of vocab logits (full-vocab KD at
    4x512 tokens costs ~2 GB each side on a 248k vocab).
    """
    total = sh.new_zeros(())
    n = sh.shape[1]
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        seg = sh[:, s:e, :]
        slogits = checkpoint(lambda z: z @ head_w.to(z.dtype).t(), seg,
                             use_reentrant=False)
        with torch.no_grad():
            tlogits = th[:, s:e, :] @ head_w.to(th.dtype).t()
        total = total + F.kl_div(
            F.log_softmax(slogits / temp, dim=-1),
            F.log_softmax(tlogits / temp, dim=-1),
            log_target=True, reduction="sum") * temp ** 2
    return total / (sh.shape[0] * n)


@torch.no_grad()
def perplexity(model: Qwen3_5TextModel, windows: list[list[int]],
               device: str, batch: int = 2) -> float:
    was_training = model.training
    model.eval()
    nll, ntok = 0.0, 0
    for i in range(0, len(windows), batch):
        ids = torch.tensor(windows[i:i + batch], device=device)
        h = model(ids).last_hidden_state
        logits = h @ model.embed_tokens.weight.to(h.dtype).t()
        shift_logits = logits[:, :-1].reshape(-1, logits.shape[-1]).float()
        shift_ids = ids[:, 1:].reshape(-1)
        nll += float(F.cross_entropy(shift_logits, shift_ids, reduction="sum"))
        ntok += int(shift_ids.numel())
    if was_training:
        model.train()
    return math.exp(nll / ntok)


def quantized_state(model: Qwen3_5TextModel) -> dict[str, torch.Tensor]:
    """Export each wrapped master as its deployed ternary (f32) values."""
    out: dict[str, torch.Tensor] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, TernaryLinear):
            with torch.no_grad():
                w = ternary_lloyd_ste(mod.weight, mod.group)
            out[name + ".weight"] = w.detach().cpu()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--train-windows", type=int, default=2048)
    ap.add_argument("--corpus-npy", default="",
                    help="pre-tokenized int32 [n, seq] windows (mixed corpus); "
                         "overrides --train-windows construction")
    ap.add_argument("--sample-every", type=int, default=0,
                    help="generate the probe suite every N steps (0 = final only)")
    ap.add_argument("--eval-windows", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--kd-logits", type=float, default=0.0,
                    help="weight for logits KL; 0 = hidden-state KD only "
                         "(vocab logits are ~2 GB at batch 4 x 512)")
    ap.add_argument("--kd-temp", type=float, default=1.0)
    ap.add_argument("--gen-samples", action="store_true",
                    help="generate prose/code/math samples at the end")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-rotation", action="store_true",
                    help="debug/control: ternarize targets without the fold")
    ap.add_argument("--skip-targets", default="",
                    help="comma-separated linear suffixes to leave f16 "
                         "(e.g. mlp.down_proj for the mixed base)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.batch, args.eval_every, args.train_windows = 2, 1, 1, 8

    snap = Path(args.model_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = args.device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tok = AutoTokenizer.from_pretrained(str(snap))
    print(f"[{time.strftime('%H:%M:%S')}] loading teacher", flush=True)
    teacher = load_text_model(snap, device, dtype=torch.bfloat16)
    teacher.eval()
    print(f"[{time.strftime('%H:%M:%S')}] loading student", flush=True)
    student = load_text_model(snap, device, dtype=torch.float32)

    eval_windows = load_windows(tok, args.eval_windows, args.seq, "test")
    if args.corpus_npy:
        train_windows = [list(map(int, w)) for w in np.load(args.corpus_npy)]
        print(f"corpus {args.corpus_npy}: {len(train_windows)} windows x "
              f"{args.seq}", flush=True)
    else:
        train_windows = load_windows(tok, args.train_windows, args.seq, "train")
    print(f"windows: train {len(train_windows)} eval {len(eval_windows)} x {args.seq}",
          flush=True)

    ppl_f16 = perplexity(teacher, eval_windows, device)
    print(f"f16 PPL: {ppl_f16:.4f}", flush=True)

    coverage = wrap_model(student, args.seed, args.group,
                          use_rotation=not args.no_rotation,
                          skip=tuple(s for s in args.skip_targets.split(",") if s))
    print(f"wrapped: {coverage}", flush=True)
    if hasattr(student, "gradient_checkpointing_enable"):
        student.gradient_checkpointing_enable()
    student.train()

    ppl_rtn = perplexity(student, eval_windows, device)
    print(f"post-hoc ternary PPL ({coverage['master_params']/1e6:.0f}M masters): "
          f"{ppl_rtn:.4f}", flush=True)
    if torch.cuda.is_available():
        print(f"VRAM after eval: "
              f"{torch.cuda.memory_allocated()/1e9:.1f} GB allocated, "
              f"{torch.cuda.max_memory_allocated()/1e9:.1f} GB peak",
              flush=True)
        torch.cuda.empty_cache()

    trainable = [p for p in student.parameters() if p.requires_grad]
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
    history = [{"step": 0, "ppl": ppl_rtn, "loss": None}]

    PROBE_PROMPTS = [
        ("The history of the Roman Empire begins with", 0.7, 120),
        ("def fibonacci(n):", 0.7, 120),
        ("Question: A train travels 240 km in 3 hours. What is its "
         "average speed? Answer:", 0.0, 100),
    ]
    samples: list[dict] = []

    def take_samples(step: int) -> None:
        for prompt, temp, n in PROBE_PROMPTS:
            text = generate(student, tok, prompt, n=n, temp=temp, seed=args.seed)
            samples.append({"step": step, "prompt": prompt,
                            "temperature": temp, "text": text})
            print(f"--- sample step {step} temp {temp}: {prompt}\n"
                  f"{text[:400]}\n", flush=True)

    if args.sample_every:
        print("post-hoc ternary samples (step 0)", flush=True)
        take_samples(0)
        student.train()

    t0 = time.time()
    for step, batch in enumerate(sample_batches(
            train_windows, args.batch, args.steps, args.seed), start=1):
        ids = torch.tensor(batch, device=device)
        with torch.no_grad():
            th = teacher(ids).last_hidden_state
        sh = student(ids).last_hidden_state
        loss_h = F.mse_loss(sh, th) + (1.0 - F.cosine_similarity(
            sh, th, dim=-1)).mean()
        loss = loss_h
        loss_kd = torch.zeros((), device=device)
        if args.kd_logits > 0.0:
            loss_kd = logits_kd(sh, th, student.embed_tokens.weight,
                                temp=args.kd_temp)
            loss = loss + args.kd_logits * loss_kd
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        if step % 5 == 0 or step == 1:
            print(f"[{time.strftime('%H:%M:%S')}] step {step}/{args.steps} "
                  f"loss {float(loss):.4f} (h {float(loss_h):.4f} "
                  f"kd {float(loss_kd):.4f}) lr {scheduler.get_last_lr()[0]:.2e}",
                  flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            ppl = perplexity(student, eval_windows, device)
            history.append({"step": step, "ppl": ppl, "loss": float(loss)})
            print(f"[{time.strftime('%H:%M:%S')}] eval step {step}: PPL {ppl:.4f}",
                  flush=True)
            student.train()
        if args.sample_every and step % args.sample_every == 0:
            take_samples(step)
            student.train()

    if args.gen_samples and (not samples or samples[-1]["step"] != args.steps):
        take_samples(args.steps)
        student.train()

    result = {
        "model_dir": str(snap),
        "coverage": coverage,
        "args": vars(args),
        "ppl_f16": ppl_f16,
        "ppl_posthoc": ppl_rtn,
        "history": history,
        "samples": samples,
        "elapsed_s": time.time() - t0,
    }
    (out / "derisk.json").write_text(json.dumps(result, indent=1))
    torch.save({k: v for k, v in student.state_dict().items()},
               out / "student-masters.pt")
    torch.save(quantized_state(student), out / "student-ternary.pt")
    best = min(h["ppl"] for h in history)
    peak = (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0
    print(f"done: f16 {ppl_f16:.3f} -> ternary {ppl_rtn:.3f} -> best {best:.3f} "
          f"({result['elapsed_s']:.0f}s, peak VRAM {peak:.1f} GB); wrote {out}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
