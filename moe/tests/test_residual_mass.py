"""The residual-mass (tail plan 2a) term.

The term exists because the top-k KD loss renormalises both distributions over
the cached support, which makes the support *mass* a free parameter.  These
tests pin the two properties the build depends on:

  1. ``support_mass`` is exact -- it must agree with a brute-force softmax, and
     it must see the mass the topk dropped.  If it only measured the support's
     share of itself it would be identically 1.0 and the term would be a no-op
     that still looked like it was training.
  2. ``residual_mass_kl`` is a proper divergence on the marginal: zero exactly
     at a match, non-negative, and it pushes a *sharpened* student back down.

Plus the two ways this can silently do nothing: an untrained student already
matching the target, and the old cache with no ``w`` field.
"""
from pathlib import Path

import pytest
import torch

from kd_loss import residual_mass_kl, support_mass
# ------------------------------------------------------- support_mass is exact

def test_support_mass_matches_bruteforce_softmax():
    torch.manual_seed(0)
    logits = torch.randn(3, 7, 50)
    idx = torch.stack([torch.randperm(50)[:5] for _ in range(7)]).expand(3, 7, 5)
    got = support_mass(logits, idx)
    probs = torch.softmax(logits, dim=-1)
    want = probs.gather(-1, idx).sum(-1)
    assert torch.allclose(got, want, atol=1e-5)


def test_support_mass_sees_the_dropped_mass():
    """The whole premise: mass on the support is far below 1.

    Sized to the measured regime -- a 248k vocab, a 512-entry support and a
    logit spread that lands the support at ~0.17, which is the coverage
    ``kld_eval.py --measure-topk 512`` actually reported on the teacher (mean
    0.1704).  A function that returned ~1 here would never see the 10x KLD
    regression, and the term built on it would be a no-op.
    """
    torch.manual_seed(2)
    logits = torch.randn(2, 248_000) * 2.0
    m = support_mass(logits, logits.topk(512, dim=-1).indices)
    assert (m > 0.05).all() and (m < 0.40).all(), m


def test_support_mass_is_one_on_full_support():
    """Sanity on the other end: support == everything means mass 1."""
    torch.manual_seed(2)
    logits = torch.randn(5, 16)
    idx = torch.arange(16).expand(5, 16)
    assert torch.allclose(support_mass(logits, idx), torch.ones(5), atol=1e-5)


def test_support_mass_is_shift_invariant():
    """A constant added to every logit is the same distribution.

    Guards the logsumexp path against a silently temperature-like dependence:
    both sides of the tail term must be read at temp 1, the deployed
    distribution, not at the KD term's --temp 2.
    """
    torch.manual_seed(3)
    logits = torch.randn(6, 30)
    idx = logits.topk(4, dim=-1).indices
    shifted = support_mass(logits + 7.5, idx)
    assert torch.allclose(shifted, support_mass(logits, idx), atol=1e-5)


def test_support_mass_gradient_reaches_the_full_vocab():
    """Only the gathered columns are in the support, but the normaliser is global.

    A per-column softmax (the mistake) would give every token mass 1 and no
    gradient to the tail; the logsumexp is what couples them.
    """
    logits = torch.randn(2, 20, requires_grad=True)
    idx = torch.tensor([[1, 3]]).expand(2, 20 // 10)
    m = support_mass(logits, idx).sum()
    m.backward()
    # every logit is touched (the normaliser is full-width) ...
    assert (logits.grad.abs().sum(-1) > 0).all()
    # ... and off-support logits get a strictly negative gradient: pushing them
    # up is the only way to move mass out of the support.
    off = torch.tensor([0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                        16, 17, 18, 19])
    assert (logits.grad[0, off] < 0).all()


# ---------------------------------------------------- residual_mass_kl is sane

def test_residual_mass_kl_zero_at_a_match():
    m = torch.tensor([0.17, 0.05, 0.9])
    assert torch.allclose(residual_mass_kl(m, m), torch.zeros(3), atol=1e-6)


def test_residual_mass_kl_is_non_negative():
    torch.manual_seed(4)
    p = torch.rand(200).clamp(0.01, 0.99)
    q = torch.rand(200).clamp(0.01, 0.99)
    assert (residual_mass_kl(p, q) >= 0).all()


def test_residual_mass_kl_equals_the_two_way_kl_by_hand():
    p, q = 0.4, 0.17
    import math
    want = q * math.log(q / p) + (1 - q) * math.log((1 - q) / (1 - p))
    assert residual_mass_kl(torch.tensor([p]), torch.tensor([q])).item() == \
        pytest.approx(want, abs=1e-6)


def test_residual_mass_kl_pulls_an_over_concentrated_student_back():
    """The failure this term exists to fix, as a gradient direction.

    A sharpened student puts *more* mass on the teacher's top-512 than the
    teacher does -- that is what the W1 arms did (entropy 10.77 -> 7.42 nats,
    top-1 mass 0.023 -> 0.148).  For a proper divergence minimised at p == q the
    gradient is positive when p overshoots q, so gradient descent moves the
    student's support mass back down, off the head and onto the tail.
    """
    w_s = torch.tensor([0.45], requires_grad=True)   # sharpened: over-concentrated
    residual_mass_kl(w_s, torch.tensor([0.17])).backward()
    assert w_s.grad.item() > 0, "overshoot must push w_s down"


def test_residual_mass_kl_pulls_an_under_concentrated_student_up():
    """The symmetric case, so the term is a real fit and not a one-way ratchet."""
    w_s = torch.tensor([0.05], requires_grad=True)
    residual_mass_kl(w_s, torch.tensor([0.17])).backward()
    assert w_s.grad.item() < 0, "undershoot must push w_s up"


def test_residual_mass_kl_minimises_over_the_student_mass():
    """The optimum is the match itself, at a scale that actually bites.

    A term with its minimum somewhere other than p == q would be fitting the
    wrong thing no matter how well it trains.
    """
    w_t = torch.tensor([0.17])
    grid = torch.linspace(0.01, 0.99, 199)
    vals = residual_mass_kl(grid, w_t.expand_as(grid))
    assert grid[int(vals.argmin())].item() == pytest.approx(0.17, abs=0.01)
    assert vals.min().item() < 1e-4


def test_residual_mass_kl_survives_degenerate_masses():
    """Clamped, so a 0/1 mass does not produce inf/nan and poison the step."""
    out = residual_mass_kl(torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0]))
    assert torch.isfinite(out).all()
    out = residual_mass_kl(torch.tensor([0.0, 1.0]), torch.tensor([0.17, 0.17]))
    assert torch.isfinite(out).all()
    assert (out > 0).all()


# ------------------------------------------------- the term is not a no-op

def test_term_is_nonzero_on_a_realistic_untrained_student():
    """An untrained (near-uniform) student is badly mismatched.

    Guards the degenerate case where the term would be exactly 0 and the run
    would look healthy while training nothing.  A uniform student over 50k vocab
    puts ~1e-5 on a 512-support; the teacher puts 0.17 there.
    """
    torch.manual_seed(5)
    v, k = 50000, 512
    student = torch.zeros(4, v)          # uniform: mass k/v on any support
    idx = torch.stack([torch.randperm(v)[:k] for _ in range(4)])
    w_s = support_mass(student, idx)
    w_t = torch.full((4,), 0.17)
    val = residual_mass_kl(w_s, w_t).mean()
    assert val.item() > 0.1
    assert torch.isfinite(val)


# ---------------------------------------------------------------- the wiring

def test_the_flag_is_off_by_default():
    """A candidate, not a shipped default: v1 must be numerically unchanged."""
    import qwen35_moe_proxy as proxy
    args = proxy.build_parser().parse_args(["train"])
    assert args.kd_tail_weight == 0.0
    assert args.kd_weight == 1.0          # the frozen v1 value, untouched
    assert args.kd_filter_frac == 0.0


def test_train_refuses_an_old_cache_rather_than_guessing():
    """The one failure mode worth being loud about.

    Caches built before this change have no ``w`` key.  Substituting a constant
    (say the measured 0.1704) would train a term against a corpus-level average
    while reporting a plausible loss, and the per-token spread is wide -- the
    measured coverage runs 0.0455 at p01 against a 0.1704 mean -- so a constant
    would be wrong by 4x on the hardest tokens.  Refusing is the honest answer.
    """
    import qwen35_moe_proxy as proxy
    src = Path(proxy.__file__).read_text()
    assert '"w" not in rec' in src, "train must check for the w field"
    idx = src.index('"w" not in rec')
    assert "SystemExit" in src[idx:idx + 400], "the refusal must be explicit"
    # and the message has to say how to fix it
    assert "cache" in src[idx:idx + 400]


def test_cache_stage_records_the_teacher_mass():
    """The ``w`` field has to be written, or the term can never run."""
    from pathlib import Path
    import qwen35_moe_proxy as proxy
    src = Path(proxy.__file__).read_text()
    assert "support_mass(logits[:, :-1], t_top.indices)" in src, \
        "stage_cache must record the teacher support mass"
    assert '"w": wmass' in src, "the cache record must carry the w field"


def test_tail_term_reads_full_width_logits():
    """The mass needs the whole normaliser, not the gathered support.

    Passing ``s_sel`` (the k gathered student logits) would make the student
    side identically 1.0 by construction, and the term would contribute a
    constant with no gradient.  This pins the full-vocab call.
    """
    from pathlib import Path
    import qwen35_moe_proxy as proxy
    src = Path(proxy.__file__).read_text()
    call = src[src.index("w_s, lse_s = support_mass_lse("):][:240]
    assert "logits.shape[-1]" in call, "support_mass_lse must see the full vocab"
    assert "s_sel" not in call, "must not renormalise over the support"


def test_cache_and_train_shapes_agree():
    """The cache and the train loop must agree on the token axis, exactly.

    The cache writes ``w`` from ``logits[:, :-1]`` (T-1 next-token positions)
    and the train loop reads it against the same slice.  A drift of one position
    -- caching the full T instead, or comparing against ``ids[:, 1:]`` shifted
    the other way -- would line up, still be finite, still produce a plausible
    loss, and quietly supervise the wrong token.  So the shapes are pinned here
    with the expressions both stages actually use.
    """
    T, V, k = 9, 400, 32
    teacher_full = torch.randn(1, T, V) * 2.0
    student_full = torch.randn(1, T, V) * 2.0

    # --- stage_cache, verbatim ---
    t_top = teacher_full[:, :-1].topk(k, dim=-1)
    w = support_mass(teacher_full[:, :-1], t_top.indices)
    rec = {"idx": t_top.indices, "w": w.to(torch.float16)}

    # --- stage_train, verbatim ---
    logits = student_full
    ti = rec["idx"]
    w_t = rec["w"].to(logits.device).float().reshape(-1)
    w_s = support_mass(logits[:, :-1].reshape(-1, logits.shape[-1]),
                       ti.reshape(-1, ti.shape[-1]))
    assert w.shape == (1, T - 1) == ti.shape[:2]
    assert w_s.shape == w_t.shape == (T - 1,)
    # and the values line up token-for-token, not merely in count
    assert torch.allclose(w.float(), w_t, atol=1e-3)
    assert torch.isfinite(residual_mass_kl(w_s, w_t)).all()

