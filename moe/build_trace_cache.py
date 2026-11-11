#!/usr/bin/env python3
"""Build a correction-training cache from a llama-perplexity logits dump.

Pipeline (teacher = Scion-FlashNext-176B, student = Scion-35B-A3B):

  1. capture_rollouts.py against the teacher server writes trace JSONs;
  2. concatenate the trace texts into one file and run
       llama-perplexity -m teacher.gguf -f traces.txt -c 512 --chunks N \
         --save-all-logits logits.bin ...
     The logits file stores, per 512-token chunk, the 511 next-token rows as
     16-bit affine log-probabilities (llama.cpp's `_logits_` format);
  3. this script decodes those rows to top-k log-probs + support mass and
     writes the `cache.pt` the qwen35 trainer consumes.  Each record also
     carries the exact token ids, so the trainer does not need a parallel
     corpus windowing pass.

Usage:
  python moe/build_trace_cache.py --logits logits.bin --out cache.pt \
      --top-k 512 --seq 512

The file format (read from `tools/perplexity/perplexity.cpp`):
  "_logits_" (8) | u32 n_ctx | i32 n_vocab | i32 n_chunk
  | i32 tokens[n_chunk * n_ctx]
  | per chunk: (n_ctx - 1) rows of nv u16, nv = 2*((n_vocab+1)//2)+4
  row = f32 scale (2 u16) | f32 min_log_prob (2 u16) | n_vocab u16 codes
  log_prob[i] = scale * code[i] + min_log_prob
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=512)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--limit-chunks", type=int, default=0)
    args = ap.parse_args()

    with open(args.logits, "rb") as f:
        magic = f.read(8)
        if magic != b"_logits_":
            raise SystemExit(f"{args.logits}: not a _logits_ file ({magic!r})")
        n_ctx = struct.unpack("<I", f.read(4))[0]
        n_vocab = struct.unpack("<i", f.read(4))[0]
        n_chunk = struct.unpack("<i", f.read(4))[0]
        if n_ctx != args.seq:
            print(f"warning: logits n_ctx {n_ctx} != --seq {args.seq}")
        tokens = np.frombuffer(f.read(n_ctx * n_chunk * 4),
                               dtype="<i4").reshape(n_chunk, n_ctx).copy()
        nv = 2 * ((n_vocab + 1) // 2) + 4
        n_rows = n_ctx - 1
        recs = []
        use = n_chunk if not args.limit_chunks else min(n_chunk, args.limit_chunks)
        for c in range(use):
            raw = f.read(n_rows * nv * 2)
            if len(raw) != n_rows * nv * 2:
                raise SystemExit(f"short logits at chunk {c}")
            rows = np.frombuffer(raw, dtype="<u2").reshape(n_rows, nv)
            idx_out = np.empty((1, n_rows, args.top_k), dtype=np.int32)
            val_out = np.empty((1, n_rows, args.top_k), dtype=np.float16)
            w_out = np.empty((1, n_rows), dtype=np.float16)
            for p in range(n_rows):
                head = rows[p, :4].tobytes()
                scale, min_log_prob = struct.unpack("<ff", head)
                codes = rows[p, 4:4 + n_vocab].astype(np.float32)
                logp = scale * codes + min_log_prob
                if args.top_k < n_vocab:
                    part = np.argpartition(-logp, args.top_k - 1)[:args.top_k]
                else:
                    part = np.arange(n_vocab)
                order = part[np.argsort(-logp[part])]
                v = logp[order]
                idx_out[0, p] = order.astype(np.int32)
                val_out[0, p] = v.astype(np.float16)
                w_out[0, p] = np.exp(v.astype(np.float32)).sum()
            recs.append({
                "ids": torch.from_numpy(tokens[c].astype(np.int32)),
                "idx": torch.from_numpy(idx_out),
                "val": torch.from_numpy(val_out),
                "w": torch.from_numpy(w_out),
            })
            if c % 50 == 0:
                print(f"decoded chunk {c}/{use} (teacher mass {float(w_out.mean()):.4f})",
                      flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(recs, out)
    print(f"wrote {out} ({len(recs)} records, {out.stat().st_size/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
