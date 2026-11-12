#!/usr/bin/env python3
"""Build a correction-training cache from a llama-perplexity logits dump.

Pipeline (teacher = Scion-FlashNext-176B, student = Scion-35B-A3B):

  1. capture_rollouts.py against the teacher server writes trace JSONs;
  2. concatenate the trace texts into one file and run
        llama-perplexity -m teacher.gguf -f traces.txt -c 512 --chunks N \
          --save-all-logits logits.bin ...
     The logits file stores, per 512-token chunk, the scored next-token rows
     as 16-bit affine log-probabilities (llama.cpp's `_logits_` format);
  3. this script decodes those rows to top-k log-probs + support mass, and
     (with --tail-logits) samples teacher-tail tokens for the D_KL2 term,
     then writes the `cache.pt` the qwen35 trainer consumes.  Each record
     also carries the exact token ids, so the trainer does not need a
     parallel corpus windowing pass.

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

A dump truncated by a crashed teacher pass is accepted as partial data
(floor(payload / rows-per-chunk) chunks) with a loud warning: an hour of
teacher GPU time should not be lost to the last chunk.

File format (read from `tools/perplexity/perplexity.cpp`):
  "_logits_" (8) | u32 n_ctx | i32 n_vocab | i32 n_chunk
  | i32 tokens[n_chunk * n_ctx]
  | per chunk: (rows) rows of nv u16, nv = 2*((n_vocab+1)//2)+4
  row = f32 scale (2 u16) | f32 min_log_prob (2 u16) | n_vocab u16 codes
  log_prob[i] = scale * code[i] + min_log_prob

Usage:
  python moe/build_trace_cache.py --logits logits.bin --out cache.pt \
      --top-k 512 --seq 512 [--tail-logits 64 --seed 0]
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
    scored input position}``.  A truncated dump is accepted as partial data
    (with a warning) when at least one whole chunk of rows is present;
    anything else raises SystemExit.
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
        # score only the second half of each chunk); detect from the size.
        # Exact-and-complete wins; otherwise an exact partial fit (truncated
        # dump) with the most complete chunks wins; ties prefer half, the
        # production layout.
        rows_full = n_ctx - 1
        rows_half = n_ctx - 1 - n_ctx // 2
        layout, n_rows, first, have = None, 0, 0, 0
        best = None
        for cand, frst in ((rows_full, 0), (rows_half, n_ctx // 2)):
            if not cand or payload < row_b or payload % row_b:
                continue
            total_rows = payload // row_b
            have_c, rem = divmod(total_rows, cand)
            if rem or not (1 <= have_c <= n_chunk):
                continue
            complete = have_c == n_chunk
            is_half = frst != 0
            score = (1 if complete else 0, have_c, 1 if is_half else 0)
            if best is None or score > best[0]:
                best = (score, cand, frst, have_c, complete)
        if best is None:
            exp = " or ".join(str(n_chunk * c * row_b) for c in (rows_full, rows_half))
            raise SystemExit(
                f"{path}: payload {payload} B matches neither the full "
                f"({rows_full} rows/chunk) nor the half ({rows_half} rows/chunk) "
                f"layout for {n_chunk} chunks (expected {exp} B)")
        _, n_rows, first, have, complete = best
        layout = "half" if first else "full"
        if not complete:
            print(f"warning: {path} declares {n_chunk} chunks but holds {have} "
                  f"({layout} layout) - truncated dump, using the complete "
                  f"chunks", flush=True)
            n_chunk = have
    return {"tokens": tokens[:n_chunk], "n_vocab": n_vocab, "n_chunk": n_chunk,
            "layout": layout, "rows": n_rows, "first": first}


def decode_chunk(rows: np.ndarray, n_vocab: int, top_k: int
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode one chunk's quantized rows to (idx, val, w), vectorized.

    ``rows`` is ``[n_rows, nv] u16``; each row is f32 scale | f32 min |
    n_vocab u16 codes of ``logp = scale*code + min``.  Returns the top-k
    token ids, their log-probs (fp16) and the support mass (fp16).

    The within-row order is this vectorized sort's tie order (quantized
    log-probs tie heavily, so it differs from the pre-vectorisation scalar
    path's order).  That is order-invariant downstream: the trainer gathers
    the student logits at ``idx`` and compares them to ``val``
    position-aligned, so only the (id, logp) pairs matter, and the support
    mass ``w`` is bit-identical to the scalar path (pinned by test).
    """
    n_rows = rows.shape[0]
    heads = rows[:, :4].copy().view("<f4").reshape(n_rows, 2)
    scale, min_lp = heads[:, 0], heads[:, 1]
    codes = rows[:, 4:4 + n_vocab].astype(np.float32)
    logp = scale[:, None] * codes + min_lp[:, None]
    if top_k < n_vocab:
        part = np.argpartition(logp, -top_k, axis=-1)[:, -top_k:]
    else:
        part = np.tile(np.arange(n_vocab), (n_rows, 1))
    vals = np.take_along_axis(logp, part, axis=-1)
    order = np.argsort(-vals, axis=-1)
    idx = np.take_along_axis(part, order, axis=-1).astype(np.int32)
    val = np.take_along_axis(vals, order, axis=-1).astype(np.float16)
    w = np.exp(val.astype(np.float32)).sum(-1, keepdims=True).astype(np.float16)
    return idx[None], val[None], w[None]      # [1, n_rows, k] cache format


def sample_tail(logp: np.ndarray, top_idx: np.ndarray, m: int,
                generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample teacher-tail tokens + their conditional log-probs per row.

    Mirrors ``kd_loss.sample_tail_tokens`` in log-prob space (the dump rows
    are log-probabilities): mask the top-k support, renormalise the tail,
    draw ``m`` tokens per row and record ``log p_t(v | ~S)``.  Runs on the
    CPU in fp32; the generator is seeded per chunk by the caller so a cache
    rebuilt with the same flags is reproducible.
    """
    lp = torch.as_tensor(logp)                        # [n_rows, V] fp32
    supp = torch.as_tensor(top_idx).long()            # [n_rows, k]
    mask = torch.zeros_like(lp, dtype=torch.bool).scatter_(-1, supp, True)
    masked = lp.masked_fill(mask, float("-inf"))
    log_tt = torch.logsumexp(masked, dim=-1, keepdim=True)      # log(1 - w)
    probs = (masked - log_tt).exp()
    idx = torch.multinomial(probs, m, replacement=True, generator=generator)
    tail_lp = lp.gather(-1, idx) - log_tt
    return idx[None], tail_lp[None]                   # [1, n_rows, m]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=512)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--limit-chunks", type=int, default=0)
    ap.add_argument("--tail-logits", type=int, default=0,
                    help="sampled teacher-tail tokens per position for the "
                         "D_KL2 term (0 = off; 64 is the validated size).  The "
                         "dump holds the full distribution, so the samples come "
                         "from the decoded log-probs - the full FlashNext "
                         "recipe runs off a real dump")
    ap.add_argument("--seed", type=int, default=0,
                    help="per-chunk seed for the tail sampling (seed * 1e6+3 + "
                         "chunk index); same flags -> same cache")
    args = ap.parse_args()

    d = parse_dump(args.logits)
    tokens, n_vocab = d["tokens"], d["n_vocab"]
    n_chunk, n_rows, first = d["n_chunk"], d["rows"], d["first"]
    layout = d["layout"]
    nv = 2 * ((n_vocab + 1) // 2) + 4

    recs = []
    use = n_chunk if not args.limit_chunks else min(n_chunk, args.limit_chunks)
    with open(args.logits, "rb") as f:
        f.seek(8 + 12 + tokens.shape[0] * tokens.shape[1] * 4)  # skip to the rows
        for c in range(use):
            raw = f.read(n_rows * nv * 2)
            if len(raw) != n_rows * nv * 2:
                raise SystemExit(f"short logits at chunk {c}")
            rows = np.frombuffer(raw, dtype="<u2").reshape(n_rows, nv)
            idx_out, val_out, w_out = decode_chunk(rows, n_vocab, args.top_k)
            rec = {"ids": torch.from_numpy(tokens[c, first:].astype(np.int32)),
                   "idx": torch.from_numpy(idx_out),
                   "val": torch.from_numpy(val_out),
                   "w": torch.from_numpy(w_out)}
            if args.tail_logits:
                if args.top_k >= n_vocab:
                    raise SystemExit("--tail-logits needs a top-k support that "
                                     "leaves a non-empty tail")
                logp = rows[:, :4].copy().view("<f4").reshape(n_rows, 2)
                scale, min_lp = logp[:, 0], logp[:, 1]
                codes = rows[:, 4:4 + n_vocab].astype(np.float32)
                full_logp = scale[:, None] * codes + min_lp[:, None]
                gen = torch.Generator().manual_seed(
                    int(args.seed) * 1_000_003 + c)
                tidx, tlp = sample_tail(full_logp, idx_out[0],
                                        args.tail_logits, gen)
                rec["tidx"] = tidx.to(torch.int32)
                rec["tlp"] = tlp.to(torch.float16)
            recs.append(rec)
            if c % 50 == 0:
                print(f"decoded chunk {c}/{use} (teacher mass {float(w_out.mean()):.4f})",
                      flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(recs, out)
    tail = f", tail {args.tail_logits}/position" if args.tail_logits else ""
    print(f"wrote {out} ({len(recs)} records, {out.stat().st_size/1e9:.2f} GB) "
          f"[{layout}: {n_rows} rows/chunk, n_vocab {n_vocab}{tail}]")


if __name__ == "__main__":
    main()
