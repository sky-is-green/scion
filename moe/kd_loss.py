"""KD loss variants for correction training.

SignRoundV2 (arXiv 2512.04746) section 3.4 excludes the top fraction of
losses when fitting the quantizer (``k`` = 0.1% of elements) to keep outlier
samples from dominating; the same trick applies to the output-KD term here.
Pure tensor functions so the trainer change is small and CPU-testable.

Reference: SignRoundV2, Eq. 12 (mean over the losses with the top-k values
removed).

The last functions are the two tail builds on top of the residual-mass (tail
plan 2a) term: ``support_mass``/``residual_mass_kl`` (the marginal piece) and
``sample_tail_tokens``/``tail_conditional_piece`` (the tail-conditional piece,
TAD's D_KL2, estimated from sampled teacher-tail tokens).  See their
docstrings for why each is shaped this way.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def kl_per_token(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                 temp: float = 2.0) -> torch.Tensor:
    """Per-token KL(teacher || student) over the selected vocab, scaled by T^2."""
    logp = F.log_softmax(student_logits.float() / temp, dim=-1)
    tlogp = F.log_softmax(teacher_logits.float() / temp, dim=-1)
    return (tlogp.exp() * (tlogp - logp)).sum(-1) * (temp ** 2)


def filtered_mean(losses: torch.Tensor, filter_frac: float = 0.0) -> torch.Tensor:
    """Mean of ``losses`` after dropping the largest ``filter_frac`` fraction."""
    flat = losses.reshape(-1).float()
    if filter_frac <= 0.0 or flat.numel() == 0:
        return flat.mean()
    # never drop everything: at least one loss survives even at filter_frac=1
    n_drop = min(int(flat.numel() * filter_frac), flat.numel() - 1)
    if n_drop <= 0:
        return flat.mean()
    return torch.sort(flat).values[:-n_drop].mean()


def kd_filtered(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                temp: float = 2.0, filter_frac: float = 0.0) -> torch.Tensor:
    """Output-KD term with optional loss filtering (`filter_frac` = 0 keeps all)."""
    return filtered_mean(kl_per_token(student_logits, teacher_logits, temp), filter_frac)


# --------------------------------------------------------- residual mass ----

def support_mass_lse(logits: torch.Tensor, idx: torch.Tensor
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """``support_mass`` plus the full-vocab logsumexp it already computes.

    The D_KL2 tail term needs both: the mass ``w_s`` for its weight, and the
    normaliser ``lse`` to build the complement term ``log(1 - w_s) + lse``.
    Returning them together keeps the two tail terms on one full-width pass.
    """
    lse = torch.logsumexp(logits.float(), dim=-1)              # (...)
    sel = logits.gather(-1, idx.long()).float()                 # (..., k)
    return (sel - lse.unsqueeze(-1)).exp().sum(-1), lse


def support_mass(logits: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Probability mass a distribution puts on the support ``idx``.

    ``logits`` is ``(..., V)`` over the *full* vocabulary and ``idx`` is
    ``(..., k)``; the result is ``(...)``.  The normalisation is a
    ``logsumexp`` against the full axis, and the sum is a gather, so the
    *exponentiation* is width ``k`` rather than ``V`` -- at a 248k vocab that is
    the difference between ~500 MB of temporaries and ~4 MB.  The upcast to
    fp32 is still full width while it lasts; that is the same temporary the LM
    cross-entropy already allocates one line earlier, and it is freed between
    the two terms.

    Works for either side: the teacher's own top-k support is its mass ``w_t``,
    the student's on the same indices is ``w_s``.
    """
    return support_mass_lse(logits, idx)[0]


def residual_mass_kl(student_mass: torch.Tensor, teacher_mass: torch.Tensor,
                     eps: float = 1e-6) -> torch.Tensor:
    """Marginal KL of the two-way support/complement split, per token.

    This is the term the top-k KD loss throws away.  Coarsening the vocabulary
    to the two outcomes "in the support / in the tail" turns the full-vocab
    divergence into three pieces by the chain rule for relative entropy::

        KL(P_t || P_s) = KL(marginal_t || marginal_s)      <-- this function
                       + E_t[ KL(P_t(.|S) || P_s(.|S)) ]   <-- what kd_filtered fits
                       + KL(P_t(.|~S) || P_s(.|~S))       <-- unconstrained

    ``kd_filtered`` only sees the middle piece, because it renormalises both
    sides over the support.  So the support mass itself is free to drift, and
    that is exactly the failure the W1 gate measured: the trained arms are ~10x
    worse than the uncorrected body on full-vocab KLD while PPL improves, and
    they *sharpen* (entropy 10.77 -> 7.42 nats, top-1 mass 0.023 -> 0.148).
    Sharpening inflates ``w_s`` above ``w_t``, so this term pushes back on the
    same axis the collapse happens on, and it costs one scalar per token.

    Matching the mass is equivalent to matching the residual (tail) mass, since
    the two sum to 1 -- but the support form is the one the cache can supply.
    """
    p = student_mass.float().clamp(eps, 1.0 - eps)
    q = teacher_mass.float().clamp(eps, 1.0 - eps)
    return q * (q.log() - p.log()) + (1.0 - q) * ((1.0 - q).log() - (1.0 - p).log())


# ------------------------------------------------------ tail conditional ----

def sample_tail_tokens(logits: torch.Tensor, support_idx: torch.Tensor, m: int,
                       generator: torch.Generator | None = None,
                       replacement: bool = True, chunk: int = 64,
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``m`` tokens per row from the teacher's tail conditional.

    ``logits`` is ``(..., V)`` raw full-width logits and ``support_idx`` is
    ``(..., k)``, so the samples never land on the support.  Returns
    ``(idx, lp)`` with shape ``(..., m)``: the sampled token ids and the
    teacher's *conditional* log-probabilities ``log p_t(v | ~S)`` -- exactly
    the pair a top-k cache cannot supply and the D_KL2 term needs.

    Sampling from the teacher's own tail conditional (rather than uniformly
    and reweighting) is what makes the Sparse Logit Sampling estimate unbiased
    with no importance weights: the proposal equals the target, so each sample
    contributes its own log-ratio (see ``tail_conditional_piece``).  The
    weights are computed in fp32 because fp16 underflows the tail of a 248k
    softmax; rows are processed in chunks because those weights are V wide --
    at a 248k vocab and chunk 64 that is ~65 MB per temporary instead of
    ~1.5 GB for a whole window at once.
    """
    if m <= 0:
        raise ValueError("m must be positive")
    v = logits.shape[-1]
    if support_idx.shape[-1] >= v:
        raise ValueError("the support must leave a non-empty tail")
    flat = logits.reshape(-1, v)
    supp = support_idx.reshape(-1, support_idx.shape[-1]).long()
    if flat.shape[0] != supp.shape[0]:
        raise ValueError("logits and support_idx disagree on the token axis")
    idx_parts, lp_parts = [], []
    for i in range(0, flat.shape[0], chunk):
        rows = flat[i:i + chunk].float()
        mask = torch.zeros_like(rows, dtype=torch.bool).scatter_(
            -1, supp[i:i + chunk], True)
        masked = rows.masked_fill(mask, float("-inf"))
        log_tt = torch.logsumexp(masked, dim=-1, keepdim=True)
        probs = (masked - log_tt).exp()          # support entries are exp(-inf)=0
        idx = torch.multinomial(probs, m, replacement=replacement,
                                generator=generator)
        lp_parts.append(rows.gather(-1, idx) - log_tt)
        idx_parts.append(idx)
    shape = (*logits.shape[:-1], m)
    return (torch.cat(idx_parts).reshape(shape),
            torch.cat(lp_parts).reshape(shape))


def tail_conditional_piece(student_logits: torch.Tensor,
                           tail_idx: torch.Tensor, tail_logp: torch.Tensor,
                           support_idx: torch.Tensor, teacher_mass: torch.Tensor,
                           student_mass: torch.Tensor | None = None,
                           student_lse: torch.Tensor | None = None,
                           eps: float = 1e-6) -> torch.Tensor:
    """Per-token sampled estimate of the tail-conditional chain-rule piece.

    The full-vocab KL splits exactly into three pieces; the top-k KD term fits
    the support-conditional one, ``residual_mass_kl`` the marginal, and this is
    the third::

        (1 - w_t) * KL( p_t(.|~S) || p_s(.|~S) )        <-- this function

    A top-k cache cannot hold the ~V-k probabilities that expectation needs, so
    it is estimated from ``m`` tokens drawn from the *teacher's* tail
    conditional (``sample_tail_tokens``; Sparse Logit Sampling, ACL 2025 --
    same estimator as ``kld_eval.tail_sample_estimate``, which the lane sweep
    validated against the exact piece at ratio 1.001).  The teacher side is
    constant for the student, so the gradient is unbiased too, and the student
    side is exact: its complement normaliser is ``lse + log1p(-w_s)`` from the
    full-vocab logsumexp, never a probability-space ``1 - w``.

    The return value is the *weighted* piece -- the same "tail" piece
    ``kld_eval.kld_decompose`` measures -- so a unit loss weight adds exactly
    the term the decomposition says is missing, and TAD's ``alpha_K * D_KL2``
    is this function's per-token value before the token mean.
    """
    w_t = teacher_mass.float()
    tail_idx = tail_idx.long()
    support_idx = support_idx.long()
    if student_mass is None or student_lse is None:
        student_mass, student_lse = support_mass_lse(student_logits, support_idx)
    w_s = student_mass.float().clamp(max=1.0 - eps)
    log_ts = student_lse.float() + torch.log1p(-w_s)
    cond_s = student_logits.gather(-1, tail_idx).float() - log_ts.unsqueeze(-1)
    return (1.0 - w_t) * (tail_logp.float() - cond_s).mean(-1)
