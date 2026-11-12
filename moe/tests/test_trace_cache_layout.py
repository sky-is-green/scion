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


def test_parse_rejects_ambiguous_size(tmp_path):
    p = tmp_path / "bad.bin"
    write_dump(p, 8, 64, 3, rows=2, first=0)
    with pytest.raises(SystemExit):
        btc.parse_dump(str(p))


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
    btc.main.__wrapped__ if hasattr(btc.main, "__wrapped__") else None
    sys.argv = ["build_trace_cache.py", "--logits", str(tmp_path / "in.bin"),
                "--out", str(out), "--top-k", "8", "--seq", str(n_ctx)]
    btc.main()
    recs = torch.load(out)
    for r in recs:
        top1 = r["idx"][0, :, 0].long()
        assert torch.equal(top1, r["ids"].long()[1:])
        assert abs(float(r["val"][0, :, 0].max()) - 0.0) < 1e-3
