"""The exact coarsened D_KL1: the support-conditional weighted by w_t.

The top-k KD term renormalises both sides over the cached support, so the
chain rule for the full-vocab divergence needs each token's support-conditional
contribution weighted by the teacher's mass on that support:

    KL(P_t || P_s) = marginal
                   + w_t * KL(P_t(.|S) || P_s(.|S))      <- the new weighting
                   + (1 - w_t) * KL(P_t(.|~S) || P_s(.|~S))

The recipe currently leaves the middle term unweighted (kd weight 2.0 on a
piece the chain rule weights w_t ~ 0.17, i.e. ~6x over-weighted).  These tests
pin:

  1. ``kd_filtered(..., token_weight=w)`` is exactly the weighted mean of the
     per-token KL, composes with the loss filter, and reduces to the plain mean
     at ``w = 1``;
  2. the identity above holds to 1e-5 on a small full-vocab case with the
     complement enumerated exactly, so the weighted term really is the piece
     the marginal and tail terms are missing (and the *unweighted* support term
     is not).
"""
import torch

from kd_loss import (kd_filtered, kl_per_token, residual_mass_kl,
                     support_mass)


def test_token_weight_is_the_weighted_mean():
    torch.manual_seed(0)
    s = torch.randn(5, 11)
    t = torch.randn(5, 11)
    w = torch.rand(5)
    want = (w * kl_per_token(s, t, temp=2.0)).mean()
    got = kd_filtered(s, t, temp=2.0, token_weight=w)
    assert torch.allclose(got, want, atol=1e-6)


def test_token_weight_ones_matches_unweighted():
    torch.manual_seed(0)
    s = torch.randn(4, 9)
    t = torch.randn(4, 9)
    assert torch.allclose(kd_filtered(s, t, token_weight=torch.ones(4)),
                          kd_filtered(s, t), atol=1e-6)


def test_token_weight_composes_with_filtering():
    """The weight multiplies *before* the filter: dropping the largest
    weighted losses is what the combined flags do."""
    torch.manual_seed(0)
    s = torch.randn(10, 7)
    t = torch.randn(10, 7)
    w = torch.rand(10)
    kl = kl_per_token(s, t) * w
    keep = torch.sort(kl.reshape(-1)).values[:-1]  # filter 0.1 of 10 -> drop 1
    want = keep.mean()
    got = kd_filtered(s, t, filter_frac=0.1, token_weight=w)
    assert torch.allclose(got, want, atol=1e-6)


def test_exact_chain_rule_identity_with_wt_support():
    """marginal + w_t*support + (1-w_t)*tail == full-vocab KL, exactly.

    Enumerate the complement (V=64, k=8) so the tail piece is exact rather than
    sampled, and use temp 1 so the KD temperature scaling is out of the way.
    """
    torch.manual_seed(0)
    v, k, n = 64, 8, 3
    t_logits = torch.randn(n, v) * 2.0
    s_logits = torch.randn(n, v)
    idx = t_logits.topk(k, dim=-1).indices

    p = torch.softmax(t_logits, dim=-1)
    q = torch.softmax(s_logits, dim=-1)
    full = (p * (p.log() - q.log())).sum(-1)

    w_t = support_mass(t_logits, idx)
    w_s = support_mass(s_logits, idx)
    marginal = residual_mass_kl(w_s, w_t)

    support_cond = kl_per_token(s_logits.gather(-1, idx),
                                t_logits.gather(-1, idx), temp=1.0)
    weighted_support = w_t * support_cond

    mask = torch.zeros_like(t_logits, dtype=torch.bool).scatter_(-1, idx, True)
    all_idx = torch.arange(v).expand(n, v)
    comp = torch.stack([all_idx[i][~mask[i]] for i in range(n)])
    tail_cond = kl_per_token(s_logits.gather(-1, comp),
                             t_logits.gather(-1, comp), temp=1.0)
    tail = (1.0 - w_t) * tail_cond

    total = marginal + weighted_support + tail
    assert torch.allclose(total, full, atol=1e-5)

    # the unweighted support term is a different (wrong) objective: with w_t ~
    # 0.1-0.4 here the difference is far above the 1e-5 identity tolerance
    unweighted = marginal + support_cond + tail
    assert not torch.allclose(unweighted, full, atol=1e-3)
