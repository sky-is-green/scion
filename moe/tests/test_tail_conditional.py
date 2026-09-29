"""TAD's D_KL2: the sampled tail-conditional piece of the full-vocab KL.

The chain-rule decomposition measured that 71-82% of the gate's KLD lives in
the tail-conditional piece, which a top-k cache cannot hold.  The build
samples ~64 tokens per position from the teacher's own tail conditional
(Sparse Logit Sampling) and the loss fits the piece on those samples.

These tests pin the properties the build depends on:

  1. ``sample_tail_tokens`` samples the *tail* (never the support) and returns
     exact conditional log-probs;
  2. ``tail_conditional_piece`` matches the gate instrument's measured piece
     exactly when the samples enumerate the tail, is zero at a match, and
     constrains the tail *shape* only (the mass is the marginal term's job);
  3. the wiring: flags off by default, the cache fields required (refusal
     instead of a plausible fallback), and the loss-side term sharing one
     full-vocab pass with the marginal term.
"""
from pathlib import Path

import pytest
import torch

from kd_loss import (residual_mass_kl, sample_tail_tokens, support_mass,
                     support_mass_lse, tail_conditional_piece)


def _cond_lp(logits: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """Conditional log p(v | ~S) for every token, with -inf on the support."""
    mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, support, True)
    masked = logits.masked_fill(mask, float("-inf"))
    log_tt = torch.logsumexp(masked, dim=-1, keepdim=True)
    return masked - log_tt


# ---------------------------------------------------- sample_tail_tokens ----

def test_sample_tail_tokens_never_draws_the_support():
    torch.manual_seed(0)
    logits = torch.randn(6, 40)
    idx = logits.topk(8, dim=-1).indices
    tidx, tlp = sample_tail_tokens(logits, idx, 16,
                                   generator=torch.Generator().manual_seed(1))
    assert tidx.shape == (6, 16) and tlp.shape == (6, 16)
    assert tidx.dtype == torch.int64 and tlp.dtype == torch.float32
    in_support = (tidx.unsqueeze(-1) == idx.unsqueeze(1)).any(-1)
    assert not in_support.any(), "a support token was sampled"


def test_sample_tail_tokens_returns_exact_conditional_logprobs():
    """``lp`` must be log p(v | ~S), not the raw logit and not log p(v)."""
    torch.manual_seed(2)
    logits = torch.randn(4, 30)
    idx = logits.topk(5, dim=-1).indices
    tidx, tlp = sample_tail_tokens(logits, idx, 12,
                                   generator=torch.Generator().manual_seed(3))
    assert torch.allclose(tlp, _cond_lp(logits, idx).gather(-1, tidx), atol=1e-5)
    # and the conditional draws are proper samples -- the empirical mass over
    # many draws concentrates on the teacher's highest-probability tail tokens
    gen = torch.Generator().manual_seed(4)
    many, _ = sample_tail_tokens(logits[:1], idx[:1], 4000, generator=gen)
    got = torch.bincount(many.reshape(-1), minlength=30).float() / 4000
    want = _cond_lp(logits[:1], idx[:1]).squeeze(0).exp()
    assert (got - want).abs().max() < 0.03


def test_sample_tail_tokens_is_reproducible_with_a_generator():
    torch.manual_seed(5)
    logits = torch.randn(3, 20)
    idx = logits.topk(4, dim=-1).indices
    a = sample_tail_tokens(logits, idx, 8,
                           generator=torch.Generator().manual_seed(7))
    b = sample_tail_tokens(logits, idx, 8,
                           generator=torch.Generator().manual_seed(7))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_sample_tail_tokens_rejects_bad_shapes_and_m():
    logits = torch.randn(2, 9)
    idx = logits.topk(3, dim=-1).indices
    with pytest.raises(ValueError):
        sample_tail_tokens(logits, idx, 0)
    with pytest.raises(ValueError):
        sample_tail_tokens(logits, torch.arange(9).expand(2, 9), 2)
    with pytest.raises(ValueError):
        sample_tail_tokens(logits, torch.zeros(3, 3, dtype=torch.long), 2)


# ------------------------------------------------ tail_conditional_piece ----

def test_piece_matches_the_eval_instrument_when_the_tail_is_uniform():
    """Loss side vs gate side, exactly.

    The lane sweep validated ``kld_eval.tail_sample_estimate`` against the
    exact decomposition; this pins that the *loss* function is the same
    quantity.  A uniform tail makes the unweighted sample mean the p-weighted
    expectation, so enumerating the complement must reproduce
    ``kld_decompose``'s "tail" piece exactly.
    """
    from kld_eval import kld_decompose, log_probs
    torch.manual_seed(10)
    t = torch.randn(5, 24)
    t[:, :6] += 5.0
    t[:, 6:] = -3.0                      # uniform tail over the complement
    s = torch.randn(5, 24) * 2.0
    k = 6
    exact = kld_decompose(log_probs(t), log_probs(s), k)["tail"]
    supp = t.topk(k, dim=-1).indices
    tidx, tlp = sample_tail_tokens(t, supp, 24 - k, replacement=False,
                                   generator=torch.Generator().manual_seed(11))
    got = tail_conditional_piece(s, tidx, tlp, supp, support_mass(t, supp))
    assert torch.allclose(got, exact, atol=1e-5)


def test_piece_is_exact_for_a_degenerate_tail():
    """One tail token with all the tail mass: the sample must hit it."""
    from kld_eval import kld_decompose, log_probs
    t = torch.full((3, 10), -30.0)
    t[:, 0], t[:, 1], t[:, 7] = 0.0, -1.0, -2.0     # support {0,1}; tail mass on 7
    s = torch.full((3, 10), -30.0)
    s[:, 0], s[:, 1], s[:, 8], s[:, 7] = -0.5, -1.5, -3.0, -4.0
    supp = t.topk(2, dim=-1).indices
    tidx, tlp = sample_tail_tokens(t, supp, 4)
    assert (tidx == 7).all(), "the only tail token must be drawn"
    got = tail_conditional_piece(s, tidx, tlp, supp, support_mass(t, supp))
    exact = kld_decompose(log_probs(t), log_probs(s), k=2)["tail"]
    assert torch.allclose(got, exact, atol=1e-5)


def test_piece_is_zero_at_a_match():
    torch.manual_seed(12)
    t = torch.randn(4, 32) * 2.0
    supp = t.topk(6, dim=-1).indices
    tidx, tlp = sample_tail_tokens(t, supp, 20,
                                   generator=torch.Generator().manual_seed(13))
    got = tail_conditional_piece(t, tidx, tlp, supp, support_mass(t, supp))
    assert got.abs().max() < 1e-5


def test_piece_is_positive_when_the_student_reshapes_its_tail():
    """The piece sees the tail's *shape*, and a distorted tail must cost.

    Head sharpening alone (the arms' over-concentration on the support) is
    invisible to this term by construction -- it is a KL between *conditional*
    tail distributions.  What it exists to fix is the measured tail mismatch:
    arms carry 80-82% of their KLD in the conditional shape, so a student whose
    tail is peakier than the teacher's must have a positive piece.
    """
    torch.manual_seed(14)
    t = torch.randn(2, 48) * 2.0
    supp = t.topk(8, dim=-1).indices
    tidx, tlp = sample_tail_tokens(t, supp, 24,
                                   generator=torch.Generator().manual_seed(15))
    tail_cols = torch.ones(48, dtype=torch.bool)
    tail_cols[supp.reshape(-1).unique()] = False
    s = t.clone()
    s[:, tail_cols] *= 1.5                # peakier tail than the teacher's
    got = tail_conditional_piece(s, tidx, tlp, supp, support_mass(t, supp))
    assert got.mean().item() > 0, "a reshaped tail must be penalised"


def test_piece_constrains_the_tail_shape_not_its_mass():
    """A uniform shift of the tail logits is the same tail conditional.

    The piece is a KL between *conditional* distributions, so rescaling the
    whole tail leaves it unchanged -- mass matching is the marginal term's job
    (``residual_mass_kl``).  If this drifted, the two tail terms would be
    fighting over the same parameter, which is exactly the double-count the
    chain rule avoids.
    """
    torch.manual_seed(16)
    t = torch.randn(1, 40) * 2.0
    supp = t.topk(6, dim=-1).indices
    tidx, tlp = sample_tail_tokens(t, supp, 20,
                                   generator=torch.Generator().manual_seed(17))
    s = torch.randn(1, 40) * 2.0
    tail_cols = torch.ones(40, dtype=torch.bool)
    tail_cols[supp.reshape(-1)] = False
    s_shift = s.clone()
    s_shift[:, tail_cols] += 2.5
    a = tail_conditional_piece(s, tidx, tlp, supp, support_mass(t, supp))
    b = tail_conditional_piece(s_shift, tidx, tlp, supp, support_mass(t, supp))
    assert torch.allclose(a, b, atol=1e-4)


def test_one_gradient_step_fits_the_teacher_tail():
    """It is a fit, not a ratchet: a small descent step on the piece reduces it."""
    torch.manual_seed(18)
    t = torch.randn(2, 40) * 2.0
    supp = t.topk(6, dim=-1).indices
    tidx, tlp = sample_tail_tokens(t, supp, 24,
                                   generator=torch.Generator().manual_seed(19))
    w_t = support_mass(t, supp)
    s = (t + torch.randn(2, 40)).requires_grad_(True)
    before = tail_conditional_piece(s, tidx, tlp, supp, w_t).mean()
    g = torch.autograd.grad(before, s)[0]
    after = tail_conditional_piece((s - 0.1 * g).detach(), tidx, tlp, supp, w_t)
    assert after.mean() < before.item()


def test_piece_accepts_a_precomputed_student_mass_and_lse():
    """The trainer shares one full-vocab pass between both tail terms."""
    torch.manual_seed(20)
    t = torch.randn(3, 30) * 2.0
    s = torch.randn(3, 30) * 2.0
    supp = t.topk(5, dim=-1).indices
    tidx, tlp = sample_tail_tokens(t, supp, 15,
                                   generator=torch.Generator().manual_seed(21))
    w_s, lse_s = support_mass_lse(s, supp)
    a = tail_conditional_piece(s, tidx, tlp, supp, support_mass(t, supp))
    b = tail_conditional_piece(s, tidx, tlp, supp, support_mass(t, supp),
                               student_mass=w_s, student_lse=lse_s)
    assert torch.allclose(a, b, atol=1e-6)
    assert torch.allclose(w_s, support_mass(s, supp), atol=1e-6)


def test_support_mass_lse_returns_the_full_vocab_normaliser():
    torch.manual_seed(22)
    logits = torch.randn(3, 50)
    idx = logits.topk(5, dim=-1).indices
    w, lse = support_mass_lse(logits, idx)
    assert torch.allclose(w, support_mass(logits, idx), atol=1e-6)
    assert torch.allclose(lse, torch.logsumexp(logits, dim=-1), atol=1e-5)


# ---------------------------------------------------------------- wiring ---

def test_flags_are_off_by_default():
    """Candidates, not shipped defaults: v1 must be numerically unchanged."""
    import qwen35_moe_proxy as proxy
    train = proxy.build_parser().parse_args(["train"])
    assert train.kd_tailcond_weight == 0.0
    assert train.kd_tail_weight == 0.0
    assert train.kd_weight == 1.0
    cache = proxy.build_parser().parse_args(["cache"])
    assert cache.tail_logits == 0
    assert cache.top_logits == 50


def test_cache_stage_records_the_sampled_tail():
    """The fields have to be written, or the term can never run."""
    import qwen35_moe_proxy as proxy
    src = Path(proxy.__file__).read_text()
    assert "sample_tail_tokens(logits[:, :-1], t_top.indices," in src
    assert '"tidx"' in src and '"tlp"' in src
    # seeded per window so a rebuild samples the same tokens
    assert "manual_seed(" in src[src.index("gen = torch.Generator"):][:120]


def test_train_refuses_a_cache_without_tail_samples():
    """The failure mode worth being loud about: skip the term and the run looks
    healthy while measuring nothing."""
    import qwen35_moe_proxy as proxy
    src = Path(proxy.__file__).read_text()
    assert '"tidx" not in rec' in src
    idx = src.index('"tidx" not in rec')
    assert "SystemExit" in src[idx:idx + 700], "the refusal must be explicit"
    assert "cache-tail" in src[idx:idx + 700], "and say how to fix it"


def test_train_shares_one_full_vocab_pass_between_the_tail_terms():
    """Two full-width logsumexps per step would double the measured 2-3x cost."""
    import qwen35_moe_proxy as proxy
    src = Path(proxy.__file__).read_text()
    assert "w_s, lse_s = support_mass_lse(" in src
    assert "student_mass=w_s, student_lse=lse_s" in src
    # and the marginal term is fed the same w_s the shared pass produced
    assert "residual_mass_kl(w_s, w_t)" in src


def test_piece_is_stable_on_bf16_logits():
    """The trainer's student logits arrive in bf16; the term upcasts internally.

    A missing upcast (gathering in bf16, or an fp16 normaliser) would show up
    here as a large deviation or a non-finite value before it ever reaches the
    GPU run.
    """
    torch.manual_seed(25)
    t = torch.randn(3, 40) * 2.0
    s = torch.randn(3, 40) * 2.0
    supp = t.topk(5, dim=-1).indices
    tidx, tlp = sample_tail_tokens(t, supp, 15,
                                   generator=torch.Generator().manual_seed(26))
    a = tail_conditional_piece(s, tidx, tlp, supp, support_mass(t, supp))
    b = tail_conditional_piece(s.bfloat16(), tidx, tlp, supp,
                               support_mass(t, supp))
    assert torch.isfinite(b).all()
    assert (a - b).abs().max() < 5e-2


def test_cache_and_train_shapes_agree_end_to_end():
    """Both stages write/read on ``logits[:, :-1]`` -- pin the exact path.

    A one-position drift (caching the full T, or a shifted gather) would line
    up, stay finite, and quietly supervise the wrong token.  This mirrors the
    cache stage's writes and the trainer's reshape/gather chain with a fake
    record, including the int32/fp16 storage round trip.
    """
    torch.manual_seed(23)
    T, V, k, m = 9, 64, 8, 12
    teacher = torch.randn(1, T, V) * 2.0
    student = torch.randn(1, T, V) * 2.0
    t_top = teacher[:, :-1].topk(k, dim=-1)
    tidx, tlp = sample_tail_tokens(teacher[:, :-1], t_top.indices, m,
                                   generator=torch.Generator().manual_seed(24))
    rec = {"idx": t_top.indices.to(torch.int32),
           "w": support_mass(teacher[:, :-1], t_top.indices).to(torch.float16),
           "tidx": tidx.to(torch.int32), "tlp": tlp.to(torch.float16)}

    # --- stage_train, verbatim ---
    logits = student
    ti = rec["idx"]
    w_t = rec["w"].float().reshape(-1)
    w_s, lse_s = support_mass_lse(logits[:, :-1].reshape(-1, V),
                                  ti.reshape(-1, k))
    piece = tail_conditional_piece(
        logits[:, :-1].reshape(-1, V),
        rec["tidx"].reshape(-1, m), rec["tlp"].float().reshape(-1, m),
        ti.reshape(-1, k), w_t,
        student_mass=w_s, student_lse=lse_s)
    assert piece.shape == (T - 1,)
    assert torch.isfinite(piece).all()
    # the teacher's own samples, scored against the teacher, are ~zero
    # (up to the fp16 storage rounding of the cached log-probs)
    as_teacher = tail_conditional_piece(
        teacher[:, :-1].reshape(-1, V),
        rec["tidx"].reshape(-1, m), rec["tlp"].float().reshape(-1, m),
        ti.reshape(-1, k), w_t)
    assert as_teacher.abs().mean() < 5e-3
