"""Ternary branch quantizer for the dense route.

A vendored copy of the deployable Lloyd-Max group scale from
``moe/moe_proxy.py`` (same rule the fork's ``Q1_0_g128`` / ``PQ2_0`` quantizer
uses), kept local so ``dense/`` does not import the MoE harness.  The *body* is
loaded from the deployed GGUF (exact values, no quantizer needed); this is only
for the correction-branch weights, which train with STE in the deployed format
(Scion D5: post-hoc ternarisation is 14-25x worse).
"""
from __future__ import annotations

import torch


def _lloyd_scale(g: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    """Lloyd-Max / TWN fixed-point group scale, matching the Q1_0_g128 rule."""
    best_a = mean.clone()
    best_obj = torch.full_like(mean, -1.0)
    for init in (0.5, 0.7, 0.9, 1.1):
        a = init * mean
        s1 = torch.zeros_like(mean)
        sw = torch.zeros_like(mean)
        for _ in range(8):
            mask = g.abs() > 0.5 * a.unsqueeze(-1)
            s1 = (g.abs() * mask).sum(-1)
            sw = mask.sum(-1).float()
            a = torch.where(sw > 0, s1 / sw, torch.zeros_like(a))
        obj = torch.where(sw > 0, s1 * s1 / sw.clamp_min(1e-9), torch.zeros_like(s1))
        take = obj > best_obj
        best_obj = torch.where(take, obj, best_obj)
        best_a = torch.where(take, a, best_a)
    return torch.where(best_a > 0, best_a, mean)


def ternary_lloyd(w: torch.Tensor, group: int = 128) -> torch.Tensor:
    """Ternary with Lloyd-refined per-group scales (the deployable rule).

    Same storage as absmean (2-bit codes + fp16 scale per group); the fp16
    round-trip of the scale is reproduced so training sees the deployed scale.
    """
    if group <= 0 or w.shape[-1] % group != 0:
        group = w.shape[-1]
    g = w.float().reshape(*w.shape[:-1], w.shape[-1] // group, group)
    mean = g.abs().mean(-1)
    a = _lloyd_scale(g, mean).half().float()
    q = torch.clamp(torch.round(g / a.unsqueeze(-1).clamp_min(1e-12)), -1, 1)
    return (q * a.unsqueeze(-1)).reshape(w.shape).to(w.dtype)


def ternary_lloyd_scales(w: torch.Tensor, group: int = 128) -> torch.Tensor:
    """Per-group Lloyd scales ``(..., n_groups)``, fp16-rounded as deployed."""
    if group <= 0 or w.shape[-1] % group != 0:
        group = w.shape[-1]
    g = w.float().reshape(*w.shape[:-1], w.shape[-1] // group, group)
    mean = g.abs().mean(-1)
    return _lloyd_scale(g, mean).half().float()


def ternary_absmean(w: torch.Tensor, group: int = 128) -> torch.Tensor:
    """Straight-through ternary with per-group absmean scales."""
    if group <= 0 or w.shape[-1] % group != 0:
        alpha = w.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
        return torch.clamp(torch.round(w / alpha), -1, 1) * alpha
    g = w.reshape(*w.shape[:-1], w.shape[-1] // group, group)
    alpha = g.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
    q = torch.clamp(torch.round(g / alpha), -1, 1) * alpha
    return q.reshape(w.shape)
