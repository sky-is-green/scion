"""Tests for the PLE A/B extensions (mixed/off/SVD arms).

CPU-only: allocation math, SVD bytes/reconstruction, mixed-row quantize.
The GPU stage (`ple-ab` on the 2-layer mirror) is the manual run in the
runlog, not this file.
"""

from __future__ import annotations

import pytest
import torch

from qwen4exp_proxy import (alloc_mixed_bits, parse_mixed_spec,
                            quantize_mixed_rows, quantize_rows, svd_compress)


def test_parse_mixed_spec():
    assert parse_mixed_spec("8:0.1,4:0.6,2:0.3") == [(8, 0.1), (4, 0.6), (2, 0.3)]


def test_alloc_mixed_bits_by_frequency():
    freq = torch.tensor([10, 1, 5, 0, 7, 3, 9, 2, 6, 4])
    bits = alloc_mixed_bits(freq, [(8, 0.2), (4, 0.5), (2, 0.3)])
    assert bits.tolist() == [8, 2, 4, 2, 4, 4, 8, 2, 4, 4]
    # fractions must sum to 1.
    with pytest.raises(ValueError):
        alloc_mixed_bits(freq, [(8, 0.5), (4, 0.4)])
    # uniform frequency degrades to order (stable sort).
    u = alloc_mixed_bits(torch.ones(10), [(8, 0.2), (2, 0.8)])
    assert u.tolist() == [8, 8] + [2] * 8


def test_svd_compress_bytes_and_error():
    torch.manual_seed(0)
    rows = torch.randn(200, 160)
    rec, bpr = svd_compress(rows, 32)
    assert rec.shape == rows.shape
    # rank-32 of flat-spectrum Gaussian keeps little energy (err large but
    # bounded); the full-rank check below proves exactness instead.
    assert 0 < float((rec - rows).abs().mean()) < 0.7
    n, d = rows.shape
    assert bpr == pytest.approx(((n * 32 + 32 + 32 * d) * 2.0) / n)
    # reconstruction error shrinks with rank (the property that matters).
    rec_big, _ = svd_compress(rows, 120)
    assert float((rec_big - rows).abs().mean()) < \
        float((rec - rows).abs().mean())
    with pytest.raises(ValueError):
        svd_compress(rows, 0)
    with pytest.raises(ValueError):
        svd_compress(rows, 160)


def test_quantize_mixed_rows_matches_uniform():
    torch.manual_seed(1)
    rows = torch.randn(40, 160)
    bits = torch.tensor([8] * 10 + [4] * 20 + [2] * 10)
    deq, bpr = quantize_mixed_rows(rows, bits, 32)
    assert deq.shape == rows.shape
    for b, sel in ((8, slice(0, 10)), (4, slice(10, 30)), (2, slice(30, 40))):
        ref, rbpr = quantize_rows(rows[sel], b, 32)
        assert torch.equal(deq[sel], ref)
    _, b8 = quantize_rows(rows[:1], 8, 32)
    _, b4 = quantize_rows(rows[:1], 4, 32)
    _, b2 = quantize_rows(rows[:1], 2, 32)
    assert bpr == pytest.approx((10 * b8 + 20 * b4 + 10 * b2) / 40)
    with pytest.raises(ValueError):
        quantize_mixed_rows(rows, torch.full((40,), 3))
