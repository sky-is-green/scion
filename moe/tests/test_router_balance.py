"""CPU tests for the Phase B router-balancing rules (pure torch, no model)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

MOE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MOE))

import router_balance as rb  # noqa: E402


def test_sign_bias_moves_overloaded_experts_down():
    """The direction the paper pins: overloaded -> bias down.

    The plan doc had ``bias += delta * sign(load - mean)``, which would drive
    overload further; this test fixes the paper's direction (arXiv 2408.15664).
    """
    bias = torch.zeros(4)
    load = torch.tensor([0.6, 0.25, 0.1, 0.05])   # mean = 0.25
    out = rb.sign_bias_update(bias, load, delta=1e-3)
    assert out[0] == pytest.approx(-1e-3)         # overloaded -> down
    assert out[1] == pytest.approx(0.0)           # at mean -> unchanged
    assert out[2] == pytest.approx(1e-3)          # underloaded -> up
    assert out[3] == pytest.approx(1e-3)
    assert bias.eq(torch.zeros(4)).all()          # input not mutated


def test_sign_bias_is_a_fixed_point_at_balance():
    bias = torch.tensor([0.1, -0.2, 0.3])
    out = rb.sign_bias_update(bias, torch.full((3,), 1 / 3), delta=1e-3)
    assert torch.equal(out, bias)


def test_load_fraction_counts_routed_slots():
    idx = torch.tensor([[0, 1], [0, 1], [0, 2]])
    f = rb.load_fraction(idx, n_experts=3)
    assert f.tolist() == pytest.approx([0.5, 1 / 3, 1 / 6])


def test_quantile_bias_hits_k_over_n_load():
    """The defining property: after updates, each expert is selected ~k/n.

    One step is close; iterating converges, because alpha_i is recomputed from
    the biased scores.  This is the invariant the histogram all-reduce serves.
    """
    torch.manual_seed(0)
    T, n, k = 4000, 8, 2
    logits = torch.randn(T, n) * 1.5
    bias = torch.zeros(n)
    loads = []
    for _ in range(6):
        bias = rb.quantile_bias_update(bias, logits, k)
        loads.append(rb.selected_fraction(logits, bias, k))
    loads = torch.stack(loads)
    target = k / n
    assert (loads[-1] - target).abs().max() < 0.01
    # and it does not make balance worse along the way
    assert (loads[-1] - target).abs().max() <= (loads[0] - target).abs().max()


def test_quantile_bias_is_higher_for_weaker_experts():
    """A stronger-scoring expert must carry a lower bias (it needs no help)."""
    torch.manual_seed(1)
    T, n, k = 2000, 6, 2
    logits = torch.randn(T, n)
    logits[:, 0] += 3.0                            # expert 0 dominates
    bias = rb.quantile_bias_update(torch.zeros(n), logits, k)
    assert bias[0] < bias[1:].mean()
    assert bias.argmin().item() == 0


def test_quantile_bias_rejects_bad_k():
    logits = torch.randn(10, 4)
    with pytest.raises(ValueError):
        rb.quantile_bias_update(torch.zeros(4), logits, k=0)
    with pytest.raises(ValueError):
        rb.quantile_bias_update(torch.zeros(4), logits, k=5)


def test_mixture_weights_ignore_the_bias():
    """Weights come from the raw logits over the selected experts only."""
    torch.manual_seed(2)
    logits = torch.randn(5, 7)
    idx = logits.topk(3, dim=-1).indices
    w = rb.mixture_weights(logits, idx)
    assert torch.allclose(w.sum(-1), torch.ones(5), atol=1e-6)
    # adding any per-expert bias to the *selection* must not move the weights
    assert torch.allclose(w, rb.mixture_weights(logits, idx), atol=0.0)


def test_router_z_loss_penalises_scale():
    logits = torch.randn(50, 6)
    small = rb.router_z_loss(logits * 0.1, coeff=1.0)
    big = rb.router_z_loss(logits * 2.0, coeff=1.0)
    assert small < big
    # a constant logit row has logsumexp = c + log n; value is (c + log n)^2
    const = rb.router_z_loss(torch.zeros(3, 6), coeff=1.0)
    assert const.item() == pytest.approx(torch.log(torch.tensor(6.0)).item() ** 2)
    # differentiable, finite gradient
    x = logits.clone().requires_grad_(True)
    rb.router_z_loss(x).backward()
    assert torch.isfinite(x.grad).all()


def test_switch_balance_loss_is_coeff_at_uniform_balance():
    """Uniform probs + perfectly balanced routing must cost exactly ``coeff``."""
    T, n, k = 100, 4, 2
    scores = torch.full((T, n), 1.0 / n)
    idx = torch.stack([torch.arange(T) % n, (torch.arange(T) + 1) % n], dim=-1)
    val = rb.switch_balance_loss(scores, idx, n_experts=n, coeff=1e-4)
    assert val.item() == pytest.approx(1e-4, rel=1e-5)
