#!/usr/bin/env python3
"""Build a correction-training cache from a llama-perplexity logits dump.

Pipeline (teacher = Scion-FlashNext-176B, student = Scion-35B-A3B):

  1. capture_rollouts.py against the teacher server writes trace JSONs;
  2. concatenate the trace texts into one file and run
        llama-perplexity -m teacher.gguf -f traces.txt -c 512 --chunks N \
          --save-all-logits logits.bin ...
     The logits file stores, per 512-token chunk, the scored next-token rows
     as 16-bit affine log-probabilities (llama.cpp's `_logits_` format);
  3. this script decodes those rows to top-k log-probs + support mass and
     writes the `cache.pt` the qwen35 trainer consumes.  Each record also
     carries the exact token ids, so the trainer does not need a parallel
     corpus windowing pass.

Two dump layouts exist and are auto-detected from the file size:

  - full:   n_ctx-1 scored rows per chunk (positions 0..n_ctx-2 predict
            tokens 1..n_ctx-1).  `ids` is the whole chunk; row i of
            idx/val/w predicts ids[i+1] -- the trainer's `logits[:, :-1]`
            rows line up directly.
  - half:   the prism forks score only the second half of each chunk
            (`first = n_ctx/2`): n_ctx-1-first rows, positions
            first..n_ctx-2 predicting tokens first+1..n_ctx-1, conditioned
            on the full chunk.  `ids` is then the scored subwindow
            tokens[first:], so trainer row i again predicts ids[i+1]
            (both sides see the same targets; the teacher saw the first
            half as extra context -- the conditioning asymmetry is inherent
            to the dump format, not a bug here).

File format (read from `tools/perplexity/perplexity.cpp`):
  "_logits_" (8) | u32 n_ctx | i32 n_vocab | i32 n_chunk
  | i32 tokens[n_chunk * n_ctx]
  | per chunk: (rows) rows of nv u16, nv = 2*((n_vocab+1)//2)+4
  row = f32 scale (2 u16) | f32 min_log_prob (2 u16) | n_vocab u16 codes
  log_prob[i] = scale * code[i] + min_log_prob

Usage:
  python moe/build_trace_cache.py --logits logits.bin --out cache.pt \
      --top-k 512 --seq 512
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import torch


def parse_dump(path: str) -> dict:
    """Read a `_logits_` dump header and detect the scored-row layout.

    Returns ``{"tokens": [n_chunk, n_ctx] int32, "n_vocab", "n_chunk",
    "layout": "full"|"half", "rows": scored rows per chunk, "first": first
    scored input position}``.  Raises SystemExit on a bad magic or a size
    that matches neither layout.
    """
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != b"_logits_":
            raise SystemExit(f"{path}: not a _logits_ file ({magic!r})")
        n_ctx = struct.unpack("<I", f.read(4))[0]
        n_vocab = struct.unpack("<i", f.read(4))[0]
        n_chunk = struct.unpack("<i", f.read(4))[0]
        tokens = np.frombuffer(f.read(n_ctx * n_chunk * 4),
                               dtype="<i4").reshape(n_chunk, n_ctx).copy()
        nv = 2 * ((n_vocab + 1) // 2) + 4
        row_b = nv * 2
        data_start = f.tell()
        f.seek(0, 2)
        payload = f.tell() - data_start

        # the scored-row layout: full (upstream) or half (the prism forks
        # score only the second half of each chunk); detect from the size
        rows_full = n_ctx - 1
        rows_half = n_ctx - 1 - n_ctx // 2
        layout, n_rows, first = None, 0, 0
        for cand, frst in ((rows_full, 0), (rows_half, n_ctx // 2)):
            if cand and payload == n_chunk * cand * row_b:
                layout, n_rows, first = ("full" if frst == 0 else "half"), cand, frst
                break
        if layout is None:
            exp = " or ".join(str(n_chunk * c * row_b) for c in (rows_full, rows_half))
            raise SystemExit(
                f"{path}: payload {payload} B matches neither the full "
                f"({rows_full} rows/chunk) nor the half ({rows_half} rows/chunk) "
                f"layout for {n_chunk} chunks (expected {exp} B)")
    return {"tokens": tokens, "n_vocab": n_vocab, "n_chunk": n_chunk,
            "layout": layout, "rows": n_rows, "first": first}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=512)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--limit-chunks", type=int, default=0)
    args = ap.parse_args()

    d = parse_dump(args.logits)
    tokens, n_vocab = d["tokens"], d["n_vocab"]
    n_chunk, n_rows, first = d["n_chunk"], d["rows"], d["first"]
    layout = d["layout"]
    nv = 2 * ((n_vocab + 1) // 2) + 4

    recs = []
    use = n_chunk if not args.limit_chunks else min(n_chunk, args.limit_chunks)
    with open(args.logits, "rb") as f:
        f.seek(8 + 12 + n_chunk * tokens.shape[1] * 4)   # skip to the rows
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
            # ids aligned to the scored rows: the full layout keeps the whole
            # chunk (row i predicts ids[i+1]); the half layout keeps the
            # scored subwindow so the same alignment holds
            recs.append({
                "ids": torch.from_numpy(tokens[c, first:].astype(np.int32)),
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
    print(f"wrote {out} ({len(recs)} records, {out.stat().st_size/1e9:.2f} GB) "
          f"[{layout}: {n_rows} rows/chunk, n_vocab {n_vocab}]")


if __name__ == "__main__":
    main()
