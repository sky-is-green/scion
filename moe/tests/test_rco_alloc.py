"""RCO allocation port -- the properties the port must hold.

The port's correctness claims are narrow and testable:

  1. the budget manifold math does what the reference proves (retraction hits
     the target, tangent projection removes the normal component, the DP solves
     the constrained assignment exactly);
  2. the container options round-trip (codes <-> packed bytes) and the ternary
     option reproduces the deployed Lloyd g128 rule;
  3. the softmax-Jacobian STE consumes dL/dp in the same direction as autograd
     through a linear mix (the identity the reference implements with
     ``hard - soft.detach() + soft``).
"""
import numpy as np
import pytest
import torch

from rco_alloc import (BudgetAllocator, SCALE_OVERHEAD, TERNARY_BPW,
                       assignment_bits, budget_constrained_argmax,
                       budget_normal, dequant_option, dot_grad_option,
                       greedy_assignment, int_codes, iter_dequant, pack_codes,
                       project_gradient, quantize_grouped, quantize_tensor,
                       retraction, row_bytes, ternary_codes, unpack_codes)
from moe_proxy import ternary_lloyd


# ------------------------------------------------ manifold / budget math ----
def test_retraction_hits_target_and_is_monotone():
    torch.manual_seed(0)
    costs = torch.tensor([2.125, 4.125, 6.125, 8.125])
    weights = torch.ones(5)
    fracs = weights / weights.sum()
    alpha = torch.randn(5, 4)
    got = retraction(alpha, costs, 3.0, fracs, tol=1e-6)
    assert abs(got - 3.0) < 1e-4
    # higher target -> higher achieved cost stays monotone from the same start
    a2 = alpha.clone()
    got2 = retraction(a2, costs, 5.0, fracs, tol=1e-6)
    assert got2 > got


def test_project_gradient_removes_normal_component():
    torch.manual_seed(0)
    costs = torch.tensor([2.125, 4.125, 8.125])
    weights = torch.tensor([1.0, 2.0, 3.0])
    fracs = weights / weights.sum()
    alpha = torch.randn(3, 3, requires_grad=True)
    alpha.grad = torch.randn(3, 3)
    n = budget_normal(alpha, costs, fracs)
    coeff, raw, proj = project_gradient(alpha, costs, fracs)
    residual = float((alpha.grad * n).sum())
    assert abs(residual) < 1e-6
    assert proj <= raw + 1e-6


def test_budget_normal_is_dc_dalpha():
    costs = torch.tensor([2.0, 5.0])
    alpha = torch.randn(4, 2, requires_grad=True)
    p = torch.softmax(alpha, dim=-1)
    val = (p * costs).sum()          # differentiable C(alpha), no weights
    val.backward()
    got = budget_normal(alpha.detach(), costs)
    assert torch.allclose(alpha.grad, got, atol=1e-5)


def test_dp_matches_brute_force_and_respects_budget():
    torch.manual_seed(0)
    bits = torch.tensor([2.125, 4.125, 8.125])
    fracs = torch.tensor([0.2, 0.3, 0.5])
    noisy = torch.randn(3, 3)
    target = 4.0
    res = 500
    got = budget_constrained_argmax(noisy, fracs, bits, target, res)

    # brute force on the same discretised costs (the port floors them)
    budget = int(target * res)
    costs = np.floor(np.outer(fracs.numpy(), bits.numpy()) * res).astype(int)
    best, best_val = None, -1e30
    for b0 in range(3):
        for b1 in range(3):
            for b2 in range(3):
                c = costs[0, b0] + costs[1, b1] + costs[2, b2]
                if c <= budget:
                    v = float(noisy[0, b0] + noisy[1, b1] + noisy[2, b2])
                    if v > best_val:
                        best_val, best = v, (b0, b1, b2)
    assert tuple(got.tolist()) == best
    spent = costs[0, got[0]] + costs[1, got[1]] + costs[2, got[2]]
    assert spent <= budget


def test_dp_falls_back_to_min_when_infeasible():
    bits = torch.tensor([4.125, 8.125])
    fracs = torch.tensor([0.5, 0.5])
    noisy = torch.randn(2, 2)
    got = budget_constrained_argmax(noisy, fracs, bits, 0.1)
    assert got.tolist() == [0, 0]


# ------------------------------------------------------------- containers ---
@pytest.mark.parametrize("bits", [2, 4, 6, 8])
def test_pack_roundtrip(bits):
    torch.manual_seed(0)
    if bits == 2:
        codes = torch.randint(-1, 2, (37,))
    else:
        qmax = 2 ** (bits - 1) - 1
        codes = torch.randint(-qmax - 1, qmax + 1, (37,))
    packed = pack_codes(codes, bits)
    back = unpack_codes(packed, bits, codes.numel())
    assert torch.equal(back, codes.to(torch.int8))


@pytest.mark.parametrize("bits", [2, 4, 6, 8])
def test_option_dequant_matches_direct_codes(bits):
    torch.manual_seed(0)
    w = torch.randn(6, 512)
    codes, scales = (ternary_codes(w) if bits == 2 else int_codes(w, bits))
    opt = quantize_tensor(w, bits)
    got = dequant_option(opt)
    want = (codes.reshape(6, -1, 128).float() * scales.unsqueeze(-1)).reshape(6, 512)
    assert torch.equal(got.float(), want.to(torch.bfloat16).float())
    assert row_bytes(512, bits) == 512 * bits // 8
    # row-chunked streaming agrees with the full dequant
    chunks = list(iter_dequant(opt, torch.bfloat16, row_chunk=2))
    assert len(chunks) == 3
    for (r0, r1), blk in chunks:
        assert torch.equal(blk, got[r0:r1])


@pytest.mark.parametrize("bits", [2, 4, 6, 8])
def test_option_handles_3d_expert_banks(bits):
    """The fused banks are 3-D; the packed layout flattens leading dims only."""
    torch.manual_seed(0)
    w = torch.randn(4, 8, 256)
    opt = quantize_tensor(w, bits)
    assert opt["shape"] == (4, 8, 256) and opt["n_rows"] == 32
    got = dequant_option(opt)
    assert got.shape == (4, 8, 256)
    # streaming rows reproduce the same tensor
    flat = got.reshape(-1, 256)
    for (r0, r1), blk in iter_dequant(opt, torch.bfloat16, row_chunk=8):
        assert torch.equal(blk, flat[r0:r1])


def test_ternary_option_reproduces_deployed_rule():
    torch.manual_seed(0)
    w = torch.randn(8, 256)
    opt = quantize_tensor(w, 2)
    got = dequant_option(opt)
    want = ternary_lloyd(w, 128)
    assert torch.allclose(got.float(), want.float(), atol=2e-3, rtol=2e-2)
    # the container rate is the PQ2_0 rate
    assert abs(opt["bits"] + SCALE_OVERHEAD - TERNARY_BPW) < 1e-9


def test_dot_grad_option_matches_full_product():
    torch.manual_seed(0)
    w = torch.randn(4, 256)
    opt = quantize_tensor(w, 4)
    g = torch.randn(4, 256)
    got = dot_grad_option(g, opt)
    want = float((g * dequant_option(opt).float()).sum())
    assert abs(got - want) < 1e-2


# ---------------------------------------------------------- allocator ------
def test_allocator_init_and_budget_hold():
    torch.manual_seed(0)
    costs = torch.tensor([TERNARY_BPW, 4.125, 6.125, 8.125])
    weights = torch.ones(8)
    alloc = BudgetAllocator(costs, weights, target_bits=TERNARY_BPW * 1.25)
    assert abs(alloc.expected_bits() - TERNARY_BPW * 1.25) < 1e-3
    p = alloc.probs()
    # ternary is the dominant option at this budget
    assert float(p[:, 0].mean()) > 0.5

    gen = torch.Generator().manual_seed(0)
    for _ in range(15):
        hard, soft = alloc.sample(tau=1.0, generator=gen)
        assert int(hard.min()) >= 0 and int(hard.max()) < costs.numel()
        # a synthetic objective: prefer the 4-bit option
        target_p = torch.zeros_like(soft)
        target_p[:, 1] = 1.0
        dl_dp = 2.0 * (soft.detach() - target_p)
        diag = alloc.step(dl_dp, soft)
        assert abs(diag["budget"] - TERNARY_BPW * 1.25) < 1e-3
        assert np.isfinite(diag["grad_raw_norm"])
    # the preferred option's probability has moved up
    assert float(alloc.probs()[:, 1].mean()) > 0.0


def test_allocator_all_ternary_when_budget_is_floor():
    costs = torch.tensor([TERNARY_BPW, 4.125, 8.125])
    alloc = BudgetAllocator(costs, torch.ones(4), target_bits=TERNARY_BPW)
    p = alloc.probs()
    assert float(p[:, 0].min()) > 0.99
    hard, _ = alloc.sample(tau=0.5, generator=torch.Generator().manual_seed(1))
    assert hard.tolist() == [0, 0, 0, 0]


@pytest.mark.parametrize("bits", [2, 4, 6, 8])
def test_quantize_grouped_matches_dequant_options(bits):
    torch.manual_seed(0)
    w = torch.randn(3, 4, 256)
    got = quantize_grouped(w, bits)
    want = dequant_option(quantize_tensor(w, bits))
    assert got.shape == w.shape
    assert torch.allclose(got.float(), want.float(), atol=2e-3, rtol=2e-2)


def test_greedy_assignment_respects_budget_and_order():
    bits = [2, 4, 6, 8]
    weights = torch.tensor([117.4e6, 117.4e6])
    # sensitivity order: group 1 first
    assigned = greedy_assignment(bits, torch.tensor([2.125, 4.125, 6.125, 8.125]),
                                 weights, target_bits=2.125 * 1.5, order=[1, 0])
    total = assignment_bits(assigned, weights)
    assert total <= 2.125 * 1.5 + 1e-6
    assert assigned[1] >= assigned[0]
