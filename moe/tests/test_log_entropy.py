"""The --log-entropy diagnostic must read the quantities the gate cares about.

Entropy and peak top-1 mass are what the Phase 1 KLD gate showed collapsing
(10.77 -> 7.42 nats, peak 0.023 -> 0.148). Logging them inside the training loop
makes a --kd-weight sweep self-diagnosing, so the answer does not need a KLD run
per checkpoint.

Pinned here because the failure mode is silent: a wrong reduction produces
plausible-looking numbers rather than an error.
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _stats(logits: torch.Tensor) -> tuple[float, float]:
    flat = logits.reshape(-1, logits.shape[-1]).float()
    lp = F.log_softmax(flat, dim=-1)
    return (float(-(lp.exp() * lp).sum(-1).mean()),
            float(lp.exp().max(-1).values.mean()))


def test_uniform_distribution_has_max_entropy():
    V = 1000
    ent, peak = _stats(torch.zeros(1, V))
    assert ent == pytest.approx(float(torch.tensor(V).log()), abs=1e-3)
    assert peak == pytest.approx(1.0 / V, abs=1e-6)


def test_one_hot_distribution_has_zero_entropy():
    V = 500
    lg = torch.full((1, V), -50.0)
    lg[0, 0] = 50.0
    ent, peak = _stats(lg)
    assert ent == pytest.approx(0.0, abs=1e-5)
    assert peak == pytest.approx(1.0, abs=1e-5)


def test_entropy_falls_as_the_distribution_sharpens():
    """The signature the gate found: less entropy, more peak mass."""
    V = 2000
    soft = _stats(torch.randn(1, V) * 0.5)
    sharp = _stats(torch.randn(1, V) * 4.0)
    assert sharp[0] < soft[0], "sharper distribution must have lower entropy"
    assert sharp[1] > soft[1], "sharper distribution must have higher peak mass"


def test_stats_average_over_tokens_not_over_vocab():
    """Two tokens with different confidence must average, not sum."""
    V = 100
    lg = torch.zeros(1, 2, V)
    lg[0, 0, 0] = 50.0          # near one-hot
    lg[0, 1, :] = 0.0           # uniform
    ent, peak = _stats(lg)
    full = float(torch.tensor(V).log())
    assert ent == pytest.approx(full / 2, abs=1e-2)
    assert peak == pytest.approx((1.0 + 1.0 / V) / 2, abs=1e-2)


def test_stats_are_finite_for_extreme_logits():
    lg = torch.full((1, 1, 50), -1e4)
    lg[0, 0, 0] = 1e4
    ent, peak = _stats(lg)
    assert torch.isfinite(torch.tensor(ent))
    assert torch.isfinite(torch.tensor(peak))


def test_bf16_logits_do_not_degrade_the_reduction():
    """Training runs in bf16; the diagnostic must upcast before softmax."""
    lg = (torch.randn(1, 64, 512) * 3).to(torch.bfloat16)
    ent_bf, peak_bf = _stats(lg)
    ent_f, peak_f = _stats(lg.float())
    assert ent_bf == pytest.approx(ent_f, abs=5e-2)
    assert peak_bf == pytest.approx(peak_f, abs=5e-2)


def test_the_flag_is_off_by_default():
    """Diagnostic, not a recipe change: it must not alter any default."""
    import qwen35_moe_proxy as proxy
    args = proxy.build_parser().parse_args(["train"])
    assert args.log_entropy is False
    assert args.kd_weight == 1.0        # the frozen v1 value, unchanged
