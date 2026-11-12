"""Tests for build_trace_cache.py's `_logits_` dump layouts.

The prism forks' ``llama-perplexity --save-all-logits`` scores only the second
half of each 512-token chunk (``first = n_ctx/2`` -> 255 rows), while upstream
writes all n_ctx-1 rows.  The parser must detect the layout from the byte
count and produce records whose ``ids`` align with the scored rows: trainer
row i of idx/val/w predicts ``ids[i+1]`` in both layouts.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import build_trace_cache as btc  # noqa: E402


def write_dump(path: Path, n_ctx: int, n_vocab: int, n_chunk: int,
               rows: int, first: int) -> None:
    """A minimal `_logits_` dump: one row of zeros per scored position."""
    nv = 2 * ((n_vocab + 1) // 2) + 4
    row = np.zeros(nv, dtype="<u2")
    with open(path, "wb") as f:
        f.write(b"_logits_")
        f.write(struct.pack("<Iii", n_ctx, n_vocab, n_chunk))
        tokens = np.arange(n_chunk * n_ctx, dtype="<i4")
        f.write(tokens.tobytes())
        for _ in range(n_chunk):
            f.write(row.tobytes() * rows)


def test_parse_full_layout(tmp_path):
    n_ctx, n_vocab, n_chunk = 8, 64, 3
    p = tmp_path / "full.bin"
    write_dump(p, n_ctx, n_vocab, n_chunk, rows=n_ctx - 1, first=0)
    d = btc.parse_dump(str(p))
    assert d["layout"] == "full"
    assert d["rows"] == n_ctx - 1
    assert d["first"] == 0
    assert d["n_vocab"] == n_vocab and d["n_chunk"] == n_chunk


def test_parse_half_layout_is_the_prism_fork_behaviour(tmp_path):
    n_ctx, n_vocab, n_chunk = 512, 248320, 4
    p = tmp_path / "half.bin"
    write_dump(p, n_ctx, n_vocab, n_chunk, rows=255, first=256)
    d = btc.parse_dump(str(p))
    assert d["layout"] == "half"
    assert d["rows"] == 255
    assert d["first"] == n_ctx // 2


def test_parse_rejects_unmatchable_size(tmp_path):
    """A payload that fits neither layout (even partially) must be rejected."""
    p = tmp_path / "bad.bin"
    write_dump(p, 8, 64, 1, rows=4, first=0)   # 4 rows: neither 7 (full) nor 3 (half)
    with pytest.raises(SystemExit):
        btc.parse_dump(str(p))


def test_parse_accepts_truncated_dump_as_partial(tmp_path):
    """A crashed teacher pass truncates the dump after N complete chunks; the
    parser must keep the complete chunks with a warning instead of losing
    the pass."""
    p = tmp_path / "trunc.bin"
    write_dump(p, 16, 64, 4, rows=7, first=0)  # declares 4 full chunks...
    import os
    nv = 2 * ((64 + 1) // 2) + 4
    header = 8 + 12 + 4 * 16 * 4
    with open(p, "r+b") as f:                  # ...then keep only 2 chunks
        f.truncate(header + 2 * (16 - 1) * nv * 2)
    d = btc.parse_dump(str(p))
    assert d["n_chunk"] == 2 and d["layout"] == "full"


def test_records_align_ids_with_scored_rows(tmp_path):
    """Both layouts must yield records where idx row i predicts ids[i+1].

    The dump rows are built so that the teacher's top-1 IS the true next
    token (a synthetic well-calibrated teacher); the built records must then
    show a perfect top-1 hit rate against ids[1:].
    """
    n_ctx, n_vocab, n_chunk = 16, 64, 2
    nv = 2 * ((n_vocab + 1) // 2) + 4
    with open(tmp_path / "in.bin", "wb") as f:
        f.write(b"_logits_")
        f.write(struct.pack("<Iii", n_ctx, n_vocab, n_chunk))
        tokens = np.tile(np.arange(n_ctx, dtype="<i4"), n_chunk)
        f.write(tokens.tobytes())
        for c in range(n_chunk):
            for i in range(n_ctx - 1):
                tgt = int(tokens[c * n_ctx + i + 1])          # row i predicts i+1
                row = np.zeros(nv, dtype="<u2")
                row[4:4 + n_vocab] = 0                        # all-zero codes
                # scale=1.0, min_log_prob=-63.0: logp = code + min; make the
                # target the argmax with log-prob 0 and everything else -63
                row[:2] = np.frombuffer(struct.pack("<f", 1.0), dtype="<u2")
                row[2:4] = np.frombuffer(struct.pack("<f", -63.0), dtype="<u2")
                row[4 + tgt] = 63                             # -63 + 63 = 0.0
                f.write(row.tobytes())
    out = tmp_path / "cache.pt"
    sys.argv = ["build_trace_cache.py", "--logits", str(tmp_path / "in.bin"),
                "--out", str(out), "--top-k", "8", "--seq", str(n_ctx)]
    btc.main()
    recs = torch.load(out)
    for r in recs:
        top1 = r["idx"][0, :, 0].long()
        assert torch.equal(top1, r["ids"].long()[1:])
        assert abs(float(r["val"][0, :, 0].max()) - 0.0) < 1e-3


def _write_dump_with_dist(tmp_path, name, n_ctx=16, n_vocab=64, n_chunk=1):
    """A dump whose rows hold a known log-prob distribution per position."""
    nv = 2 * ((n_vocab + 1) // 2) + 4
    rows = np.zeros((n_chunk, n_ctx - 1 - n_ctx // 2, nv), dtype="<u2")
    tokens = np.tile(np.arange(n_ctx, dtype="<i4"), n_chunk)
    logp = np.full((n_chunk, rows.shape[1], n_vocab), -6.0, dtype=np.float32)
    for c in range(n_chunk):
        for i in range(rows.shape[1]):
            tgt = int(tokens[c * n_ctx + i + 1])
            logp[c, i] = -6.0
            logp[c, i, tgt] = 0.0                      # peaked at the target
            heads = np.array([1.0, -63.0], dtype="<f4").view("<u2")  # scale=1, min=-63
            row = np.zeros(nv, dtype="<u2")
            row[:2] = heads[0:2]
            row[2:4] = heads[2:4]
            codes = np.round((logp[c, i] + 63.0)).astype(np.int32)
            row[4:4 + n_vocab] = np.clip(codes, 0, 65535).astype("<u2")
            rows[c, i] = row
    p = tmp_path / name
    with open(p, "wb") as f:
        f.write(b"_logits_")
        f.write(struct.pack("<Iii", n_ctx, n_vocab, n_chunk))
        f.write(tokens.tobytes())
        f.write(rows.tobytes())
    return p, tokens


def test_tail_samples_are_off_support_with_exact_conditionals(tmp_path):
    """--tail-logits: samples never land on the top-k support and tlp is
    exactly log p(v | ~S) = logp[tidx] - logsumexp(tail logp)."""
    p, tokens = _write_dump_with_dist(tmp_path, "tail.bin")
    out = tmp_path / "cache-tail.pt"
    sys.argv = ["build_trace_cache.py", "--logits", str(p), "--out", str(out),
                "--top-k", "8", "--seq", "16", "--tail-logits", "4", "--seed", "0"]
    btc.main()
    recs = torch.load(out)
    for r in recs:
        assert r["tidx"].shape == (1, r["idx"].shape[1], 4)
        on = (r["idx"][0].long().unsqueeze(1)
              == r["tidx"][0].long().unsqueeze(-1)).any(-1)
        assert not bool(on.any())
        # rebuild the full logp row and verify the conditional exactly
        nv = 2 * ((64 + 1) // 2) + 4
        rows = np.frombuffer((p.read_bytes()[8 + 12 + tokens.size * 4:]),
                             dtype="<u2").reshape(-1, nv)
        heads = rows[:, :4].copy().view("<f4").reshape(rows.shape[0], 2)
        full = torch.from_numpy(
            heads[:, 0:1] * rows[:, 4:4 + 64].astype(np.float32) + heads[:, 1:2])
        masked = full.clone()
        masked.scatter_(-1, r["idx"][0].long(), float("-inf"))
        log_tt = torch.logsumexp(masked, dim=-1)
        sel = full.gather(-1, r["tidx"][0].long())
        err = (sel - log_tt.unsqueeze(-1) - r["tlp"][0].float()).abs()
        assert float(err.max()) < 0.01      # tlp is stored fp16 (eps ~1e-2 at |v|<16)


def test_tail_sampling_reproducible_per_chunk(tmp_path):
    p, _ = _write_dump_with_dist(tmp_path, "tail2.bin", n_chunk=2)
    outs = []
    for i in range(2):
        out = tmp_path / f"cache-rep{i}.pt"
        sys.argv = ["build_trace_cache.py", "--logits", str(p), "--out", str(out),
                    "--top-k", "8", "--seq", "16", "--tail-logits", "4",
                    "--seed", "3"]
        btc.main()
        outs.append(torch.load(out))
    for r1, r2 in zip(*outs):
        assert torch.equal(r1["tidx"], r2["tidx"])
        assert torch.equal(r1["tlp"], r2["tlp"])
