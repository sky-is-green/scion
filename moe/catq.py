"""CAT-Q-style ternary reconstruction (weight-space port).

Source: CAT-Q, arXiv 2606.26650 (ICML 2026 oral, Intel Labs China); reference
code github.com/IntelChina-AI/BitTern (Apache-2.0).  This is an independent,
simplified port of the two components:

- learnable modulation (LM, paper Eq. 3): per-group mean/scale/threshold
  factors that redistribute the FP weights before ternarization;
- softened ternarization (ST, paper Eq. 5-6): a tanh transition scheduled
  from identity to hard ternarization, so the scale/threshold parameters get
  gradients before the codes harden.

The paper optimises sliding *layer windows* against their FP outputs; that
needs activations and a model (GPU stage).  This module keeps the quantizer
core in weight space so it is CPU-testable and can be dropped into
``quantize_bank_inplace(kind="catq")`` when a card frees up.
"""
from __future__ import annotations

import torch

DEFAULT_GROUP = 128
DEFAULT_STEPS = 200
DEFAULT_LR = 0.05
DEFAULT_GAMMA = 0.8      # fraction of steps in the soft phase (paper: 0.8)
DEFAULT_S0 = 30.0        # initial sharpness (paper: 30)
DEFAULT_THRESHOLD = 0.5  # Delta_0 on transformed weights (paper: 0.5)
DEFAULT_POLISH = 0.1     # lr factor for the hard phase (STE polish)


def _grouped(w: torch.Tensor, group: int) -> torch.Tensor:
    if w.shape[-1] % group:
        raise ValueError(f"last dim {w.shape[-1]} not divisible by group {group}")
    return w.reshape(-1, group).float()


def soft_ternarize(w_hat: torch.Tensor, s, threshold) -> torch.Tensor:
    """ST transition function f (paper Eq. 5); ``s``/``threshold`` broadcast."""
    if not torch.is_tensor(s):
        s = torch.as_tensor(s, dtype=w_hat.dtype, device=w_hat.device)
    return (torch.tanh(s * (w_hat - threshold))
            + torch.tanh(s * (w_hat + threshold))) / (2 * torch.tanh(s))


def hard_ternarize(w_hat: torch.Tensor, threshold) -> torch.Tensor:
    """Hard ternarization Q (paper Eq. 2) around ``+/- threshold``."""
    return torch.where(w_hat > threshold, torch.ones_like(w_hat),
                       torch.where(w_hat < -threshold, -torch.ones_like(w_hat),
                                   torch.zeros_like(w_hat)))


def stats_per_group(w: torch.Tensor, group: int = DEFAULT_GROUP):
    """Per-group mean and absolute-mean (the LM initializers mu_0, alpha_0)."""
    v = _grouped(w, group)
    mu0 = v.mean(dim=1)
    alpha0 = (v - mu0[:, None]).abs().mean(dim=1)
    return mu0, alpha0


def transformed_weights(w: torch.Tensor, group: int, delta_mu, delta_alpha):
    """LM transform (paper Eq. 3); returns grouped ``w_hat`` and ``alpha``."""
    v = _grouped(w, group)
    mu0, alpha0 = stats_per_group(w, group)
    mu = mu0 + delta_mu * alpha0
    alpha = (delta_alpha * alpha0).clamp_min(1e-12)
    return (v - mu[:, None]) / alpha[:, None], alpha


def catq_reconstruct(w: torch.Tensor, group: int = DEFAULT_GROUP,
                     steps: int = DEFAULT_STEPS, lr: float = DEFAULT_LR,
                     gamma: float = DEFAULT_GAMMA, s0: float = DEFAULT_S0,
                     threshold: float = DEFAULT_THRESHOLD,
                     polish: float = DEFAULT_POLISH):
    """Learn per-group LM factors under the ST schedule.

    Returns ``(codes, scales)``: codes have ``w``'s shape and values in
    {-1, 0, +1}; scales are positive, one per group, with shape
    ``w.shape[:-1] + (w.shape[-1] // group,)``.  Deployment reconstruction is
    ``alpha * codes`` (paper: W ~= alpha*T; mu is only used to redistribute
    the weights for ternarization).

    ``polish`` scales the learning rate for the hard phase (t > gamma) where
    the codes are straight-through; the paper's LM constraints are enforced
    every step (delta_mu in (-1, 1), scales/threshold positive).
    """
    v = _grouped(w, group).detach()
    n_groups = v.shape[0]
    delta_mu = torch.zeros(n_groups, device=v.device, requires_grad=True)
    delta_alpha = torch.ones(n_groups, device=v.device, requires_grad=True)
    delta_delta = torch.ones(n_groups, device=v.device, requires_grad=True)
    opt = torch.optim.Adam([delta_mu, delta_alpha, delta_delta], lr=lr)

    with torch.enable_grad():   # works under a caller's torch.no_grad() too
        for step in range(1, steps + 1):
            t = step / steps
            for g in opt.param_groups:
                g["lr"] = lr if t <= gamma else lr * polish
            w_hat, alpha = transformed_weights(v, group, delta_mu, delta_alpha)
            thr = delta_delta * threshold
            if t <= gamma:
                tern = soft_ternarize(w_hat, (t / gamma) * s0, thr[:, None])
            else:
                hard = hard_ternarize(w_hat, thr[:, None])
                tern = hard.detach() + w_hat - w_hat.detach()   # STE carry
            loss = ((v - alpha[:, None] * tern) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                # paper constraints: delta_mu in (-1, 1); scales/threshold positive
                delta_mu.clamp_(-0.999, 0.999)
                delta_alpha.clamp_(1e-6, None)
                delta_delta.clamp_(1e-6, None)

    with torch.no_grad():
        w_hat, alpha = transformed_weights(v, group, delta_mu, delta_alpha)
        codes = hard_ternarize(w_hat, (delta_delta * threshold)[:, None])
    return codes.reshape(w.shape), alpha.reshape(w.shape[:-1] + (w.shape[-1] // group,))


def catq_dequantize(codes: torch.Tensor, scales: torch.Tensor, group: int) -> torch.Tensor:
    """``alpha * codes`` back to the weight shape."""
    return (codes * scales.repeat_interleave(group, dim=-1)).reshape(codes.shape)


def ternary_catq(w: torch.Tensor, group: int = DEFAULT_GROUP, **kw) -> torch.Tensor:
    """Dequantised CAT-Q weights; drop-in shape for the bank quantizer."""
    codes, scales = catq_reconstruct(w, group=group, **kw)
    return catq_dequantize(codes, scales, group).to(w.dtype)
