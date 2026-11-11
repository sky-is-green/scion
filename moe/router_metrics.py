"""Routing diagnostics shared by the MoE correction trainers.

Pure torch and no model/transformers import, so every metric can be unit-tested
on CPU.  The point of the module is to make the router-freeze A/B read the same
axes MiMo-V2.6 reads in its RL report: top-k agreement alone hides load
collapse, and PPL alone hides a router that has co-adapted to different experts.
"""

from __future__ import annotations

import numbers

import torch


def set_trainable(model, freeze_router: bool = False) -> int:
    """Freeze the body, train the branches, and optionally the routers.

    The router modules are the ``...mlp.gate.weight`` parameters; the branch's
    own per-channel gates are ``...branch.read_gate`` / ``...branch.write_gate``
    and do not contain the ``.gate.`` substring.  Returns the router parameter
    count that stayed trainable, so a run log states which arm it is.
    """
    n_router = 0
    for name, p in model.named_parameters():
        is_branch = ".branch." in name
        is_router = ".gate." in name
        train = is_branch or (is_router and not freeze_router)
        p.requires_grad_(train)
        if is_router and train:
            n_router += p.numel()
    return n_router


def split_params(model):
    """Return ``(branch_params, router_params)`` in named order."""
    branch, router = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (router if ".gate." in name else branch).append(p)
    return branch, router


@torch.no_grad()
def load_stats(indices: torch.Tensor, n_experts: int) -> dict:
    """Expert-load statistics for one layer over the routed slots.

    ``indices`` is ``(tokens, top_k)``.  ``cv``, ``peak`` and ``cold`` follow
    MiMo-V2.6 §5.4: coefficient of variation, max/mean, and the fraction of
    experts below 0.1x the mean load.  ``dead`` counts experts with zero load.
    """
    idx = indices.reshape(-1).long()
    counts = torch.bincount(idx, minlength=int(n_experts)).float()
    total = counts.sum().clamp_min(1.0)
    p = counts / total
    mean = p.mean().clamp_min(1e-12)
    return {
        "cv": float(p.std(unbiased=False) / mean),
        "peak": float(p.max() / mean),
        "cold": float((p < 0.1 * mean).float().mean()),
        "entropy": float(-(p * (p + 1e-12).log()).sum()),
        "dead": int((counts == 0).sum()),
    }


@torch.no_grad()
def top1_agreement(student_idx: torch.Tensor, teacher_idx: torch.Tensor) -> float:
    """Fraction of tokens whose first (highest-scoring) expert matches."""
    return float((student_idx[:, 0] == teacher_idx[:, 0]).float().mean())


@torch.no_grad()
def topk_agreement(student_idx: torch.Tensor, teacher_idx: torch.Tensor) -> float:
    """Mean token-level overlap of the two top-k sets, divided by k."""
    hit = (student_idx.unsqueeze(-1) == teacher_idx.unsqueeze(-2)).any(-1).float()
    return float(hit.mean())


@torch.no_grad()
def bucket_agreement(student_idx: torch.Tensor, teacher_idx: torch.Tensor,
                     bucket: int = 64) -> list[float]:
    """Top-k agreement per contiguous position bucket.

    Long-horizon degradation is a compounding story, so the position profile
    is more informative than the sequence mean: a router that is fine at token
    0 and wrong by token 4k looks acceptable in the aggregate.
    """
    hit = (student_idx.unsqueeze(-1) == teacher_idx.unsqueeze(-2)).any(-1).float()
    per_token = hit.mean(-1)
    return [float(per_token[s:s + bucket].mean())
            for s in range(0, per_token.numel(), bucket)]


@torch.no_grad()
def topk_union_kl(student_idx: torch.Tensor, student_p: torch.Tensor,
                  teacher_idx: torch.Tensor, teacher_p: torch.Tensor,
                  n_experts: int) -> float:
    """KL(teacher || student) restricted to the union of the two top-k sets.

    The full teacher router logits are not in the cache, so the comparison is
    over the support both sides actually report.  This is a routing-decision
    metric, not a replacement for full-vocabulary KLD.
    """
    t = student_idx.shape[0]
    dev = student_idx.device
    s = torch.zeros(t, int(n_experts), device=dev, dtype=torch.float32)
    q = torch.zeros_like(s)
    s.scatter_(1, student_idx.long(), student_p.float())
    q.scatter_(1, teacher_idx.long(), teacher_p.float())
    s = s / s.sum(-1, keepdim=True).clamp_min(1e-9)
    q = q / q.sum(-1, keepdim=True).clamp_min(1e-9)
    return float((q * (q.clamp_min(1e-9).log() - s.clamp_min(1e-9).log())).sum(-1).mean())


@torch.no_grad()
def router_delta_stats(model, base: dict[str, torch.Tensor]) -> dict[str, float]:
    """Relative Frobenius change of each trained router from its base weights.

    A zero delta is a frozen router; a large delta is a router that moved far
    enough to re-define expert specialization.  The metric is per layer because
    the early layers usually move least.
    """
    out = {}
    for name, p in model.named_parameters():
        if ".gate." not in name or name not in base:
            continue
        ref = base[name].detach().float()
        cur = p.detach().float()
        out[name] = float((cur - ref).norm() / ref.norm().clamp_min(1e-12))
    return out


def summarize(layer_stats: list[dict]) -> dict:
    """Average a list of per-layer metric dicts, ignoring missing/non-finite."""
    keys = set().union(*(d.keys() for d in layer_stats)) if layer_stats else set()
    out = {}
    for k in sorted(keys):
        vals = [float(d[k]) for d in layer_stats
                if k in d and isinstance(d[k], numbers.Real)]
        vals = [v for v in vals if v == v]          # drop NaN
        if vals:
            out[k] = sum(vals) / len(vals)
    return out
