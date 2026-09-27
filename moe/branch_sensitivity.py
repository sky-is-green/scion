"""Per-layer sidecar sensitivity: ternarise one correction branch at a time.

Loads an fp32-branch checkpoint, evaluates the 8-window held-out PPL, then
converts a single layer's correction branch to the deployed g128 ternary format
(post-hoc, no retraining) and re-evaluates.  The per-layer deltas show which
layers carry the sidecar cost and therefore which ones a mixed-precision
sidecar should keep at fp16.

Usage:
  MOE_ARTIFACTS=... HIP_VISIBLE_DEVICES=1 python branch_sensitivity.py \
      --rank 512 --load $MOE_ARTIFACTS/olmoe/olmoe-corr-r512-d4096-step4096.pt \
      --out $MOE_ARTIFACTS/olmoe/branch-sensitivity-r512-d4096.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from olmoe_corrections import build, load_branch_state  # noqa: E402
from olmoe_proxy import MODEL, windows  # noqa: E402


@torch.no_grad()
def eval_ppl(model, data, device):
    total, ntok = 0.0, 0
    for i in range(len(data)):
        ids = data[i:i + 1].to(device)
        logits = model(ids).logits[:, :-1]
        total += F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                                 ids[:, 1:].reshape(-1), reduction="sum").item()
        ntok += ids[:, 1:].numel()
    return math.exp(total / ntok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=512)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--load", required=True)
    ap.add_argument("--layer", type=int, default=-1,
                    help="single layer to probe (default: every layer)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    data = windows(tok, args.windows, args.seq, 999, "wikitext")

    ns = SimpleNamespace(device_map=args.device, group=args.group, rank=args.rank,
                         branch_quant="fp32", branch_quant_fp16_layers="",
                         quant="absmean")
    model, _ = build(ns)
    missing, unexpected = load_branch_state(model, args.load)
    print(f"loaded {args.load}: missing={len(missing)} unexpected={len(unexpected)}",
          flush=True)
    model.eval()
    branches = [layer.mlp.branch for layer in model.model.layers]
    base = eval_ppl(model, data, args.device)
    print(f"baseline fp32: ppl {base:.4f}", flush=True)

    rows = []
    todo = range(len(branches)) if args.layer < 0 else [args.layer]
    for i in todo:
        branches[i].quant = "g128"
        ppl = eval_ppl(model, data, args.device)
        branches[i].quant = "fp32"
        rows.append({"layer": i, "ppl_g128": round(ppl, 4),
                     "delta": round(ppl - base, 4), "ratio": round(ppl / base, 4)})
        print(f"layer {i:2d}: ppl {ppl:9.2f}  x{ppl/base:.3f}  (+{ppl-base:.2f})",
              flush=True)

    out = {"checkpoint": args.load, "rank": args.rank, "windows": args.windows,
           "baseline_fp32_ppl": round(base, 4), "per_layer_g128": rows}
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"wrote {args.out}")
    else:
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
