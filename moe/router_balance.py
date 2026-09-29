"""Expert-balancing rules for the Phase B router A/B/C.

The router path stores ``(logits, scores, indices)`` per gate (see
``olmoe_proxy.gate_hook``).  Every balancer here steers *which* experts are
selected; the mixture weights stay the raw softmax over the selected experts, so
a bias cannot change how the chosen experts are mixed:

1. ``sign_bias_update`` — DeepSeek auxiliary-loss-free rule (Wang et al. 2024,
   arXiv 2408.15664).  After each step, overloaded experts get their bias
   *decreased* and underloaded ones increased::

       b_e <- b_e - delta * sign(load_e - mean(load))

   (The plan doc wrote ``+=``; the paper's update is ``p_k <- p_k - u`` for
   overloaded experts, so the plan's sign is a typo — pinned by a test.)  The
   bias is a per-expert scalar, free at inference, and balancing adds no
   gradient to the router.
2. ``quantile_bias_update`` — Kimi K3 Quantile Balancing.  Set each expert's
   bias from the ``(1 - k/n)``-quantile of its per-token margins
   ``s_i,e + b_e - alpha_i`` (``alpha_i`` = the token's top-k cutoff), so the
   expert's expected load is exactly ``k/n``.  The quantile is a per-expert
   order statistic, which is what makes it all-reduce-able as a histogram
   across ranks (one reduce, a few hundred bins) instead of gathering margins.
3. ``router_z_loss`` — OLMoE router z-loss on ``logsumexp(logits)^2``; the
   cheap control arm.
4. ``switch_balance_loss`` — the classic Switch aux loss, kept as the tiny
   sequence-level safety net next to a bias rule (and the current v1 term).

Pure torch, CPU-testable, no model imports.  Every flag that would use these
defaults off: the frozen v1 recipe stays comparable, and the balancers are
candidate arms until the E1 harness says otherwise.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _flat(x: torch.Tensor) -> torch.Tensor:
    """Flatten every leading axis into tokens; keep the expert axis last."""
    return x.reshape(-1, x.shape[-1])


def load_fraction(indices: torch.Tensor, n_experts: int | None = None) -> torch.Tensor:
    """Per-expert fraction of the routed (token, slot) assignments.

    ``indices`` is (..., k) of expert ids; the result is (n_experts,).
    """
    idx = _flat(indices).long()
    n = int(n_experts) if n_experts is not None else int(idx.max()) + 1
    return torch.bincount(idx.reshape(-1), minlength=n).float() / idx.numel()


def sign_bias_update(bias: torch.Tensor, load: torch.Tensor,
                     delta: float = 1e-3) -> torch.Tensor:
    """DeepSeek ALF-LB step: overloaded experts move down, underloaded up.

    ``bias`` and ``load`` are both (n_experts,); returns the new bias (does not
    mutate).  At exactly mean load the bias is unchanged.
    """
    return bias - delta * torch.sign(load - load.mean())


def selection_scores(logits: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Biased scores ``logits + bias`` (the quantity top-k selection uses)."""
    return _flat(logits).float() + bias


def selection_cutoff(logits: torch.Tensor, bias: torch.Tensor,
                     k: int) -> torch.Tensor:
    """The k-th largest biased score per token, ``alpha_i`` — (T, 1)."""
    return selection_scores(logits, bias).topk(k, dim=-1).values[:, -1:]


def margins(logits: torch.Tensor, bias: torch.Tensor, k: int) -> torch.Tensor:
    """Per-token margins ``s_i,e + b_e - alpha_i``; expert e is selected iff >= 0."""
    return selection_scores(logits, bias) - selection_cutoff(logits, bias, k)


def quantile_bias_step(bias: torch.Tensor, marg: torch.Tensor, k: int,
                       damp: float = 1.0) -> torch.Tensor:
    """Bias update from precomputed margins (the forward already has them).

    Setting the ``(1 - k/n)``-quantile of expert e's margins to zero makes the
    measure of tokens with ``margin >= 0`` (i.e. its load) exactly ``k/n``.
    ``damp < 1`` trades convergence speed for stability, as with any quantile
    controller.
    """
    n = marg.shape[-1]
    if not 0 < k <= n:
        raise ValueError(f"k={k} must be in (0, n_experts={n}]")
    q = torch.quantile(marg.float(), 1.0 - k / n, dim=0)
    return bias - damp * q


def quantile_bias_update(bias: torch.Tensor, logits: torch.Tensor, k: int,
                         damp: float = 1.0) -> torch.Tensor:
    """K3 Quantile Balancing: ``b_e <- b_e - damp * Quantile_{1-k/n}(margins_e)``.

    One call is a one-step update — ``alpha_i`` moves slightly with the biases,
    so a handful of iterations tightens it (see the tests).  This computes the
    margins from the logits; ``quantile_bias_step`` takes them directly when the
    forward pass already produced them (``router_bias``).
    """
    n = _flat(logits).shape[-1]
    if not 0 < k <= n:
        raise ValueError(f"k={k} must be in (0, n_experts={n}]")
    return quantile_bias_step(bias, margins(logits, bias, k), k, damp)


def selected_fraction(logits: torch.Tensor, bias: torch.Tensor,
                      k: int) -> torch.Tensor:
    """Per-expert fraction of tokens that select it under ``logits + bias``.

    Ties count as selected; with continuous scores that measure is exactly the
    load the quantile rule targets.
    """
    return (margins(logits, bias, k) >= 0).float().mean(0)


def mixture_weights(logits: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Raw softmax over the selected experts — the bias never enters weights.

    ``logits`` is (..., E) and ``indices`` (..., k); returns (..., k).
    """
    sel = _flat(logits).float().gather(-1, _flat(indices).long())
    return F.softmax(sel, dim=-1)


def router_z_loss(logits: torch.Tensor, coeff: float = 1e-3) -> torch.Tensor:
    """OLMoE z-loss: ``coeff * mean(logsumexp(logits)^2)``.

    Penalises router logit scale (not just spread), which is what the OLMoE
    report used to stabilise routing.  Differentiable, no reduction surprises.
    """
    z = torch.logsumexp(_flat(logits).float(), dim=-1)
    return coeff * (z ** 2).mean()


def switch_balance_loss(scores: torch.Tensor, indices: torch.Tensor,
                        n_experts: int | None = None,
                        coeff: float = 1e-4) -> torch.Tensor:
    """Classic Switch aux loss ``coeff * n * sum_e f_e * P_e``.

    ``scores`` is the router softmax (..., E), ``indices`` the selected ids
    (..., k).  ``f_e`` is the fraction of routed slots to expert e, ``P_e`` the
    mean router probability for e.  This is the tiny safety net next to a bias
    rule — the plan keeps it at ~1e-4, and it is the term the current v1 uses.
    """
    p = _flat(scores).float()
    idx = _flat(indices).long()
    n = int(n_experts) if n_experts is not None else p.shape[-1]
    f = torch.bincount(idx.reshape(-1), minlength=n).float() / idx.numel()
    return coeff * n * (f * p.mean(0)).sum()
