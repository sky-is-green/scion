"""Dump top-k next-token logprobs from the Python fp8 prefix loader (PLE rows).

Used as the reference side of the PLE export proof: the same prompt through
the Python model (official shards, bf16 body, PLE rows gathered per window)
and through the exported GGUF in the fork must agree on the top tokens.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen4exp_proxy as q4  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    ids = tok(args.prompt, return_tensors="pt").input_ids
    model, missing, unexpected, _ = q4.load_fp8_prefix(
        args.layers, args.device, model_dir=args.model_dir, ple="rows",
        ple_ids=ids)
    if missing or unexpected:
        raise SystemExit(f"load mismatch: missing {len(missing)} "
                         f"unexpected {len(unexpected)}")
    with torch.no_grad():
        logits = q4.model_logits(model, ids)[0, -1].float()
        lp = torch.log_softmax(logits, -1)
        vals, idx = lp.topk(args.top_k)
    res = {"prompt": args.prompt, "layers": args.layers,
           "tokens": ids[0].tolist(),
           "top": [{"token": int(i), "text": tok.decode([int(i)]),
                    "logprob": float(v)} for v, i in zip(vals, idx)]}
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
