"""Top-k mass coverage: the number that decides which tail term to build.

Computed from the parked full-vocab log-probs (which ARE the true normalised
distribution), so it is not the tautological "softmax over the cached entries
sums to 1" that a first attempt at this measured.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kld_eval import _coverage, topk_coverage


def _logprobs(rows: list[list[float]]) -> torch.Tensor:
    return torch.log_softmax(torch.tensor(rows), dim=-1)


def test_full_capture_when_the_tail_is_negligible():
    """A peaked teacher: 4 entries hold essentially all the mass."""
    V = 5000
    logits = torch.full((1, V), -60.0)
    logits[0, :4] = torch.tensor([10.0, 8.0, 6.0, 4.0])
    lp = torch.log_softmax(logits, dim=-1)
    cov = topk_coverage(logits, logits.topk(4, dim=-1).values)
    assert cov["mean_topk_mass"] == pytest.approx(1.0, abs=1e-6)
    assert cov["mean_residual_mass"] < 1e-6


def test_half_capture_on_a_flat_teacher():
    """A uniform teacher over 8: capturing 4 entries holds exactly half."""
    logits = torch.zeros(1, 8)
    lp = torch.log_softmax(logits, dim=-1)
    cov = topk_coverage(logits, logits.topk(4, dim=-1).values)
    assert cov["mean_topk_mass"] == pytest.approx(0.5, abs=1e-6)
    assert cov["mean_residual_mass"] == pytest.approx(0.5, abs=1e-6)


def test_coverage_is_monotone_in_k():
    torch.manual_seed(0)
    logits = torch.randn(1, 200) * 2
    prev = -1.0
    for k in (1, 2, 4, 8, 16, 32, 64, 200):
        got = topk_coverage(logits, logits.topk(k, dim=-1).values)["mean_topk_mass"]
        assert got >= prev - 1e-6, f"k={k} captured less than a smaller k"
        prev = got
    assert prev == pytest.approx(1.0, abs=1e-5)


def test_coverage_is_never_above_one():
    torch.manual_seed(1)
    for _ in range(25):
        logits = torch.randn(1, 300) * 5
        for k in (1, 8, 64):
            cov = topk_coverage(logits, logits.topk(k, dim=-1).values)
            assert cov["mean_topk_mass"] <= 1.0 + 1e-6
            assert cov["min_topk_mass"] <= 1.0 + 1e-6
            assert cov["mean_residual_mass"] >= -1e-6


def test_coverage_reports_per_token_min_and_p01():
    """A single flattened token must not collapse the shape contract."""
    V = 400
    logits = torch.full((1, V), -80.0)
    logits[0, 0] = 30.0                     # one very peaked token
    logits[0, 1:, ] = 0.0
    cov = topk_coverage(logits, logits.topk(1, dim=-1).values)
    assert cov["n_tokens"] == 1
    assert cov["top_k"] == 1
    assert 0.0 <= cov["min_topk_mass"] <= 1.0
    assert 0.0 <= cov["p01_topk_mass"] <= cov["mean_topk_mass"] + 1e-6


def test_coverage_helper_matches_the_direct_function():
    """``_coverage`` over parked windows must agree with the per-window version."""
    torch.manual_seed(2)
    V = 600
    parked, tops = [], []
    for _ in range(3):
        logits = torch.randn(1, V) * 2
        parked.append(torch.log_softmax(logits, dim=-1))
        tops.append(logits.topk(16, dim=-1).values)
    batched = _coverage(parked, tops, 16)
    assert batched["n_tokens"] == 3
    assert batched["top_k"] == 16
    # same quantity, computed from log-probs: the top-16 of a log-prob row
    # exponentiated IS the mass the cache would carry
    direct = torch.cat([torch.topk(p, 16, dim=-1).values.exp().sum(-1) for p in parked])
    assert batched["mean_topk_mass"] == pytest.approx(float(direct.mean()), abs=1e-6)


def test_empty_input_is_reported_as_empty_not_crashed():
    """The reduce helpers need a guard; an empty batch must still return a dict."""
    cov = topk_coverage(torch.zeros(0, 50), torch.zeros(0, 8))
    assert cov["n_tokens"] == 0
    assert cov["top_k"] == 8


def test_the_tautology_is_avoided():
    """Renormalising the cached entries must NOT be what we report.

    This is the bug the first implementation had: softmax over just the cached
    entries returns 1.0 for any input, so it would have reported ~100% coverage
    no matter how flat the teacher was.
    """
    logits = torch.zeros(1, 100)                  # perfectly flat
    cached_only = logits.topk(10, dim=-1).values
    assert cached_only.softmax(-1).sum() == pytest.approx(1.0)   # the trap
    cov = topk_coverage(logits, cached_only)
    assert cov["mean_topk_mass"] == pytest.approx(0.10, abs=1e-6)
