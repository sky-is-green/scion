"""Bias-based router balancing for the qwen35 prefix (Phase B, P0).

The frozen v1 prefix recipe has **no** balancing term: the routers are trained by
the LM/KD gradient alone.  This module adds the three Phase B balancers as a
patch of ``Qwen3_5MoeTopKRouter``:

- ``bias``     : DeepSeek ALF-LB (arXiv 2408.15664) — select by ``logits + bias``,
                 weights stay the raw softmax over the selected experts, and
                 after each optimizer step ``bias -= delta * sign(load - mean)``.
- ``quantile`` : Kimi K3 Quantile Balancing — bias from the ``(1 - k/n)``-quantile
                 of the per-token margins, so each expert's expected load is
                 ``k/n``.
- ``zloss``    : OLMoE router z-loss — a differentiable term on
                 ``logsumexp(logits)^2``.

Everything is off unless ``--balance`` says otherwise, so the frozen v1 recipe
and the in-flight runs are untouched.  The bias is a registered buffer on each
gate, so ``save()`` (which keeps ``.gate.`` keys) persists it and the eval
instrument must be run with the same ``--balance`` setting to load it.

Why a patch: the stock gate computes ``softmax`` first and top-k on the
probabilities.  A bias has to be added to the *scores* before selection, and the
mixture weights must stay the unbiased softmax over the selected experts — that
is the ALF-LB rule.  ``router_balance.py`` holds the pure math; this file is
only plumbing and state.
"""
from __future__ import annotations

import types

import torch
import torch.nn.functional as F

from router_balance import margins, quantile_bias_step, sign_bias_update

KINDS = ("none", "bias", "quantile", "zloss")


def _router_class():
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeTopKRouter)
    return Qwen3_5MoeTopKRouter


def _patched(model):
    return [m for m in model.modules()
            if getattr(m, "_balance_kind", None) is not None]


def _new_stats(n: int) -> dict:
    return {"counts": torch.zeros(n, dtype=torch.long), "margins": [],
            "z_terms": []}


def _balanced_forward(self, hidden_states):
    """Stock forward + bias on the selection scores; weights stay unbiased."""
    hidden = hidden_states.reshape(-1, self.hidden_dim)
    logits = F.linear(hidden, self.weight)                     # raw scores
    bias = getattr(self, "balance_bias", None)
    choice = logits if bias is None else logits + bias         # selection scores
    probs = F.softmax(choice, dtype=torch.float, dim=-1)
    _, idx = torch.topk(probs, self.top_k, dim=-1)
    # mixture weights: raw softmax over the selected experts (bias never enters)
    weights = F.softmax(logits.gather(-1, idx).float(), dim=-1).to(probs.dtype)

    kind = getattr(self, "_balance_kind", None)
    if kind is not None and self.training:
        st = self._balance_stats
        st["counts"] = st["counts"] + torch.bincount(
            idx.reshape(-1).cpu(), minlength=self.num_experts)
        if kind == "quantile":
            st["margins"].append(margins(logits, bias, self.top_k).detach())
        elif kind == "zloss":
            z = torch.logsumexp(logits.float(), dim=-1)
            st["z_terms"].append((z ** 2).mean())
    return probs, weights, idx


def patch_gate(gate, kind: str) -> None:
    """Patch one gate: forward, bias buffer, stats, kind (idempotent)."""
    gate.forward = types.MethodType(_balanced_forward, gate)
    if not hasattr(gate, "balance_bias"):
        gate.register_buffer("balance_bias", torch.zeros(gate.num_experts))
    gate._balance_kind = kind
    gate._balance_stats = _new_stats(gate.num_experts)


def patch_router_balance(model, kind: str) -> int:
    """Patch every gate in ``model`` for ``kind``; returns the gate count.

    Patches the class (so later instances behave) *and* rebinds the loaded
    instances, because ``device_map`` binds the pre-patch forward on each
    module — the same reason ``patch_experts`` rebinds.
    """
    if kind not in KINDS or kind == "none":
        raise ValueError(f"kind must be one of {KINDS[1:]}, got {kind!r}")
    cls = _router_class()
    if not getattr(cls, "_balance_patched", False):
        cls.forward = _balanced_forward
        cls._balance_patched = True
    n = 0
    for m in model.modules():
        if isinstance(m, cls):
            patch_gate(m, kind)
            n += 1
    return n


def balance_diagnostics(model) -> dict:
    """Load-entropy diagnostic over the gates (no reset)."""
    ents = []
    for m in _patched(model):
        c = m._balance_stats["counts"].float()
        if c.sum() == 0:
            continue
        p = c / c.sum()
        ents.append(float(-(p * (p + 1e-12).log()).sum()))
    return {"n_gates": len(_patched(model)),
            "mean_load_entropy": (sum(ents) / len(ents)) if ents else float("nan")}


def balance_reset(model) -> None:
    for m in _patched(model):
        m._balance_stats = _new_stats(m.num_experts)


def balance_update(model, kind: str, delta: float = 1e-3) -> dict:
    """Apply one bias update from the stats the last forward accumulated.

    Call *after* ``opt.step()``.  Returns the load diagnostics before the reset
    (the log wants them), then clears the stats for the next step.
    """
    if kind not in ("bias", "quantile"):
        raise ValueError(f"no update for kind {kind!r}")
    for m in _patched(model):
        st = m._balance_stats
        with torch.no_grad():
            if kind == "bias":
                total = int(st["counts"].sum())
                if total == 0:
                    continue
                load = st["counts"].float() / total
                m.balance_bias.copy_(sign_bias_update(m.balance_bias, load, delta))
            else:
                if not st["margins"]:
                    continue
                marg = torch.cat(st["margins"], dim=0)
                m.balance_bias.copy_(
                    quantile_bias_step(m.balance_bias, marg, m.top_k))
    diag = balance_diagnostics(model)
    balance_reset(model)
    return diag


def balance_z_loss(model, coeff: float = 1e-3):
    """Differentiable z-loss over the collected gate terms; resets the stats.

    Returns ``(loss, diagnostics)``.  The terms were accumulated in the forward
    with their graph intact, so the loss must be added to the training loss and
    backpropagated in the same step.
    """
    terms = [t for m in _patched(model) for t in m._balance_stats["z_terms"]]
    if terms:
        loss = coeff * torch.stack(terms).mean()
    else:
        loss = torch.zeros((), requires_grad=False)
    diag = balance_diagnostics(model)
    balance_reset(model)
    return loss, diag
