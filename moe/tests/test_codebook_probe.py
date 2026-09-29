"""CPU tests for the codebook probe's pure math (no safetensors needed)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

MOE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MOE))

import codebook_probe as cb  # noqa: E402
from moe_proxy import ternary_absmean  # noqa: E402


def test_hadamard_is_orthonormal_and_self_inverse():
    torch.manual_seed(0)
    x = torch.randn(3, 64)
    y = cb.hadamard(x)
    assert torch.allclose(cb.hadamard(y), x, atol=1e-5)      # symmetric & orthonormal
    assert torch.allclose(y.pow(2).sum(-1), x.pow(2).sum(-1), atol=1e-5)


def test_hadamard_rejects_non_power_of_two():
    with pytest.raises(ValueError):
        cb.hadamard(torch.randn(4, 96))


def test_normalized_mse_is_zero_for_an_exact_match_and_scale_free():
    torch.manual_seed(1)
    w = torch.randn(8, 128)
    assert cb.normalized_mse(w, w) == pytest.approx(0.0, abs=1e-12)
    assert cb.normalized_mse(w, w * 1.0) == pytest.approx(0.0, abs=1e-12)


def test_fixed_gauss_codebook_is_near_optimal_on_gaussian_data():
    torch.manual_seed(2)
    w = torch.randn(32, 256)          # standard normal
    nmse = cb.normalized_mse(w, cb.fixed_gauss_ternary(w, 128))
    # the 3-level N(0,1) Lloyd-Max quantizer is ~0.19 normalized MSE
    assert 0.15 < nmse < 0.25


def test_compander_with_p_one_is_plain_absmean():
    torch.manual_seed(3)
    w = torch.randn(8, 128)
    assert torch.allclose(cb.companded_absmean(w, 128, p=1.0),
                          ternary_absmean(w, 128), atol=1e-6)


def test_per_group_stats_shape():
    torch.manual_seed(4)
    w = torch.randn(4, 256)
    stats = cb.per_group_nmse(w, w, 128)
    assert stats["p50"] == 0.0 and stats["p90"] == 0.0
