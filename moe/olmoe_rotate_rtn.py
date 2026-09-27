"""Rotation-aware RTN baseline for OLMoE experts (no training).

Applies the Bonsai-style blockwise Walsh-Hadamard rotation to the input axis of
each expert bank, absmean-RTN at g128, then unrotates, so the model runs in its
original basis with a better ternary approximation of W.

Measures: weight rel-err / zero-share (raw vs rotated), held-out PPL and
per-layer router agreement vs the FP teacher.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from moe_proxy import ternary_absmean  # noqa: E402
from olmoe_proxy import MODEL, load_model, windows, gate_hook  # noqa: E402

sys.path.insert(0, str(HERE.parent))          # public repo root
from scion_moe import rotation as R  # noqa: E402

ART = Path(os.environ.get("MOE_ARTIFACTS", HERE / "artifacts"))
OUT = ART / "olmoe"


def fwht(x: torch.Tensor) -> torch.Tensor:
    """Normalized Walsh-Hadamard transform along the last axis (power of 2)."""
    n = x.shape[-1]
    lead = x.shape[:-1]
    x = x.reshape(-1, n).clone()
    h = 1
    while h < n:
        x = x.reshape(-1, n // (2 * h), 2, h)
        a = x[:, :, 0, :]
        b = x[:, :, 1, :]
        x = torch.cat([a + b, a - b], dim=2).reshape(-1, n)
        h *= 2
    return (x / math.sqrt(n)).reshape(*lead, n)


def absorb_torch(w: torch.Tensor, signs: list, block: int) -> torch.Tensor:
    lead = w.shape[:-1]
    x = w.reshape(*lead, w.shape[-1] // block, block)
    s = torch.stack([torch.as_tensor(v, device=w.device, dtype=w.dtype) for v in signs])
    return fwht(x * s).reshape(*lead, w.shape[-1])


def unabsorb_torch(w: torch.Tensor, signs: list, block: int) -> torch.Tensor:
    lead = w.shape[:-1]
    x = w.reshape(*lead, w.shape[-1] // block, block)
    x = fwht(x)
    s = torch.stack([torch.as_tensor(v, device=w.device, dtype=w.dtype) for v in signs])
    return (x * s).reshape(*lead, w.shape[-1])


@torch.no_grad()
def rotate_quantize(w: torch.Tensor, seed: int, group: int = 128, chunk: int = 8):
    width = w.shape[-1]
    block = R.block_size(width)
    signs = R.rotations_for(width, seed)
    out = torch.empty_like(w, dtype=torch.float32)
    err2 = ref2 = 0.0
    zero = tot = 0
    for s0 in range(0, w.shape[0], chunk):
        part = w[s0:s0 + chunk].float()
        wr = absorb_torch(part, signs, block)
        q = ternary_absmean(wr, group)
        we = unabsorb_torch(q, signs, block)
        out[s0:s0 + chunk] = we
        err2 += float(((we - part) ** 2).sum())
        ref2 += float((part ** 2).sum())
        zero += int((q == 0).sum())
        tot += q.numel()
        del part, wr, q, we
        torch.cuda.empty_cache()
    rel = math.sqrt(err2 / (ref2 + 1e-12))
    return out.to(w.dtype), rel, zero / max(tot, 1)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--eval-windows", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    data = windows(tok, args.eval_windows, args.seq, 999, "wikitext")

    # teacher reference (PPL + router indices)
    teacher, _ = load_model(args.device)
    ref, t_total, t_ntok = {}, 0.0, 0
    for i in range(len(data)):
        ids = data[i:i + 1].to(args.device)
        store = {}
        hs = [layer.mlp.gate.register_forward_hook(gate_hook(store, j))
              for j, layer in enumerate(teacher.model.layers)]
        logits = teacher(ids).logits
        for h in hs:
            h.remove()
        ref[i] = {j: store[j][2].cpu() for j in store}
        t_total += F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                                   ids[:, 1:].reshape(-1), reduction="sum").item()
        t_ntok += ids[:, 1:].numel()
    teacher_ppl = math.exp(t_total / t_ntok)
    print(f"teacher ppl {teacher_ppl:.4f}", flush=True)
    del teacher
    torch.cuda.empty_cache()

    # student: FP weights, experts replaced by rotate+RTN+unrotate
    student, _ = load_model(args.device)
    stats = []
    for i, layer in enumerate(student.model.layers):
        exp = layer.mlp.experts
        for name in ("gate_up_proj", "down_proj"):
            w = getattr(exp, name)
            w_raw = w.data.clone()
            with torch.no_grad():
                wq_raw = ternary_absmean(w_raw, args.group)
                rel_raw = float((wq_raw - w_raw).norm() / w_raw.norm())
            we, rel_rot, zero = rotate_quantize(w_raw, args.seed + i, args.group)
            w.data = we
            stats.append({"layer": i, "param": name,
                          "rel_raw": round(rel_raw, 5),
                          "rel_rotated": round(rel_rot, 5),
                          "zero_share": round(zero, 4)})
            print(f"  L{i} {name}: rel_raw {rel_raw:.4f} -> rel_rot {rel_rot:.4f} "
                  f"zero {zero:.3f}", flush=True)

    total, ntok, agree = 0.0, 0, {i: [] for i in range(len(student.model.layers))}
    for i in range(len(data)):
        ids = data[i:i + 1].to(args.device)
        store = {}
        hs = [layer.mlp.gate.register_forward_hook(gate_hook(store, j))
              for j, layer in enumerate(student.model.layers)]
        logits = student(ids).logits
        for h in hs:
            h.remove()
        total += F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                                 ids[:, 1:].reshape(-1), reduction="sum").item()
        ntok += ids[:, 1:].numel()
        for j in store:
            a = ref[i][j].to(store[j][2].device)
            b = store[j][2]
            agree[j].append(float((a.unsqueeze(-1) == b.unsqueeze(-2)).any(-1).float().mean()))
    ppl = math.exp(total / ntok)
    agg = sum(sum(v) / len(v) for v in agree.values()) / len(agree)
    out = {"teacher_ppl": round(teacher_ppl, 4),
           "rtn_rotated_ppl": round(ppl, 4),
           "ratio": round(ppl / teacher_ppl, 2),
           "router_agree_mean": round(agg, 4),
           "weights": stats}
    print(json.dumps({k: v for k, v in out.items() if k != "weights"}, indent=2))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rotate-rtn.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT/'rotate-rtn.json'}")


if __name__ == "__main__":
    main()
