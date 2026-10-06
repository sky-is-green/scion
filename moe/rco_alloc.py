"""RCO allocation port -- budget-manifold bit allocation for the scion prefix.

Ported from IST-DASLab/RCO (arXiv 2605.00649, Apache-2.0):

  - ``src/manifold.py``: ``budget_normal``, ``project_gradient``,
    ``retraction``, ``vector_transport`` (tangent projection, monotone
    retraction, momentum transport on the softmax budget manifold).
  - ``src/search/quant.py``: ``budget_constrained_argmax`` -- the exact
    budget-constrained DP forward (Gumbel-perturbed logits -> one discrete
    assignment inside the byte budget).

Local adaptation of the RCO allocation:

  - groups = quantizable 2-D weight tensors (per-layer); options = container
    levels {2 = ternary (the deployable Lloyd g128 rule), 4, 6, 8} bits/param
    with fp16 scales per 128-weight group;
  - objective = our LM + KD loss on the cached teacher logits, not the
    reference repo's calibration KL;
  - the discrete forward applies the *hard* DP assignment; the
    straight-through gradient ``dL/dp_k = <dL/dW, W_k>`` is accumulated by
    per-parameter grad hooks and handed to the softmax Jacobian.  This is the
    same STE as the reference ``WeightInterpolation``
    (``hard - soft.detach() + soft``) without materialising an interpolated
    model.
"""

from __future__ import annotations

import math
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

from moe_proxy import ternary_lloyd

# Bit cost: b-bit codes + one fp16 scale per 128-weight group.
SCALE_OVERHEAD = 16.0 / 128.0
TERNARY_BPW = 2.0 + SCALE_OVERHEAD        # 2.125 -- the PQ2_0 container rate
SUPPORTED_BITS = (2, 4, 6, 8)


# --------------------------------------------------------------- manifold ---
def budget_normal(alpha: torch.Tensor, costs: torch.Tensor,
                  weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Normal vector to the budget surface at alpha (reference Prop. 2).

    ``dC/dalpha_ik = w_i * p_ik * (c_k - E_i[c])``.
    """
    probs = torch.softmax(alpha, dim=-1)
    e = (probs * costs).sum(dim=-1, keepdim=True)
    n = probs * (costs - e)
    if weights is not None:
        n = n * weights.unsqueeze(-1)
    return n


def budget_value(alpha: torch.Tensor, costs: torch.Tensor,
                 weights: Optional[torch.Tensor] = None) -> float:
    """Expected cost C(alpha) = sum_i w_i sum_k p_ik c_k."""
    with torch.no_grad():
        p = torch.softmax(alpha, dim=-1)
        row = (p * costs).sum(dim=-1)
        return float((row * weights).sum()) if weights is not None else float(row.sum())


def project_gradient(alpha: torch.Tensor, costs: torch.Tensor,
                     weights: Optional[torch.Tensor] = None
                     ) -> Tuple[float, float, float]:
    """Tangent projection of ``alpha.grad`` (in place).  Returns diagnostics."""
    with torch.no_grad():
        n = budget_normal(alpha, costs, weights)
        nf = n.flatten()
        g = alpha.grad.flatten()
        raw_norm = float(g.norm())
        coeff = (g @ nf) / (nf @ nf + 1e-12)
        alpha.grad.sub_(coeff * n)
        proj_norm = float(alpha.grad.norm())
    return float(coeff), raw_norm, proj_norm


def retraction(alpha: torch.Tensor, costs: torch.Tensor, target: float,
               weights: Optional[torch.Tensor] = None,
               max_iter: int = 60, tol: float = 1e-4) -> float:
    """Monotone bisection retraction back onto the budget surface (in place).

    Adds ``shift * costs`` to alpha; the map shift -> C is monotone, so the
    bracket can be expanded and bisected to machine tolerance (reference
    Prop. 3).  Returns the achieved cost.
    """
    def c_of(shift: float) -> float:
        with torch.no_grad():
            p = torch.softmax(alpha + shift * costs, dim=-1)
            row = (p * costs).sum(dim=-1)
            return float((row * weights).sum()) if weights is not None else float(row.sum())

    cur = c_of(0.0)
    if abs(cur - target) < tol:
        return cur

    lo, hi = -1.0, 1.0
    for _ in range(40):
        if c_of(lo) <= target <= c_of(hi):
            break
        if c_of(hi) < target:
            hi *= 2.0
        if c_of(lo) > target:
            lo *= 2.0

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        e = c_of(mid)
        if abs(e - target) < tol:
            lo = hi = mid
            break
        if e > target:
            hi = mid
        else:
            lo = mid

    with torch.no_grad():
        alpha.add_(0.5 * (lo + hi) * costs)
    return c_of(0.0)


def vector_transport(optimizer, alpha: torch.Tensor, costs: torch.Tensor,
                     weights: Optional[torch.Tensor] = None) -> None:
    """Re-project Adam's first moment onto the new tangent plane (in place)."""
    with torch.no_grad():
        n = budget_normal(alpha, costs, weights)
        nf = n.flatten()
        nsq = float(nf @ nf) + 1e-12
        for group in optimizer.param_groups:
            for p in group["params"]:
                state = optimizer.state.get(p)
                if state and "exp_avg" in state:
                    m = state["exp_avg"]
                    coeff = (m.flatten() @ nf) / nsq
                    m.sub_(coeff * n)


# ------------------------------------------------------------------- DP -----
def budget_constrained_argmax(noisy_logits: torch.Tensor, group_fracs: torch.Tensor,
                              actual_bits: torch.Tensor, target_bits: float,
                              resolution: int = 500) -> torch.Tensor:
    """Exact multiple-choice knapsack over groups x container options.

    Adapted from ``src/search/quant.py``.  ``group_fracs`` sums to 1;
    ``target_bits`` is the weighted-average bits budget.  Returns the option
    index per group (long tensor on ``noisy_logits``'s device).
    """
    n_groups, n_bw = noisy_logits.shape
    budget = max(int(target_bits * resolution), 0)

    logits_np = noisy_logits.detach().cpu().numpy()
    fracs = group_fracs.detach().cpu().numpy()
    bits_np = actual_bits.detach().cpu().numpy()
    # Floor discretisation (not the reference's round): sum_i floor(x_i) <=
    # floor(sum_i x_i), so the all-min assignment is feasible whenever the
    # target is at or above the container floor -- the round form can make the
    # budget infeasible by a few units at the floor.
    costs = np.floor(np.outer(fracs, bits_np) * resolution).astype(int)
    costs = np.clip(costs, 0, budget) if budget > 0 else np.zeros_like(costs)

    neg_inf = -1e30
    dp = np.full(budget + 1, neg_inf)
    dp[0] = 0.0
    backtrack = np.zeros((n_groups, budget + 1), dtype=np.int64)

    for g in range(n_groups):
        new_dp = np.full(budget + 1, neg_inf)
        for b in range(n_bw):
            c = int(costs[g, b])
            if c > budget:
                continue
            cand = dp[: budget + 1 - c] + logits_np[g, b] if c else dp + logits_np[g, b]
            seg = new_dp[c:]
            better = cand > seg
            seg[better] = cand[better]
            bt = backtrack[g, c:]
            bt[better] = b
        dp = new_dp

    best_j = int(np.argmax(dp))
    if not np.isfinite(dp[best_j]) or dp[best_j] <= neg_inf:
        min_idx = int(np.argmin(bits_np))
        return torch.full((n_groups,), min_idx, dtype=torch.long,
                          device=noisy_logits.device)

    assignment = np.zeros(n_groups, dtype=np.int64)
    j = best_j
    for g in range(n_groups - 1, -1, -1):
        b = int(backtrack[g, j])
        assignment[g] = b
        j -= int(costs[g, b])
    return torch.from_numpy(assignment).to(device=noisy_logits.device)


# ----------------------------------------------------------- quantisation ---
def _flatten_rows(w: torch.Tensor) -> torch.Tensor:
    """Row view: (prod(leading dims), last) -- the scale-group axis is last."""
    return w.reshape(-1, w.shape[-1])


def ternary_codes(w: torch.Tensor, group: int = 128
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """The deployable Lloyd g128 ternary rule -> (codes -1/0/1, fp16 scales).

    Works for any leading dims (the fused expert banks are 3-D); codes are
    returned flat as [rows, k], scales as [rows, k // group].
    """
    f = _flatten_rows(w)
    n, k = f.shape
    q = ternary_lloyd(f, group)
    g = q.float().reshape(n, -1, group)
    a = g.abs().amax(-1)
    codes = torch.where(a.unsqueeze(-1) > 0,
                        torch.round(g / a.unsqueeze(-1).clamp_min(1e-12)),
                        torch.zeros_like(g))
    return codes.to(torch.int8).reshape(n, -1), a.to(torch.float16)


def int_codes(w: torch.Tensor, bits: int, group: int = 128
              ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric g128 integer codes (fp16 scale round-trip, calibration-free)."""
    f = _flatten_rows(w)
    n, k = f.shape
    g = f.float().reshape(n, -1, group)
    qmax = float(2 ** (bits - 1) - 1)
    a = (g.abs().amax(-1).clamp_min(1e-12) / qmax).half().float()
    codes = torch.clamp(torch.round(g / a.unsqueeze(-1)), -qmax, qmax)
    return codes.to(torch.int8).reshape(n, -1), a.to(torch.float16)


def pack_codes(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack flat int8 codes into uint8 (2/4/6/8-bit).  Row alignment is the
    caller's contract (see ``row_bytes``)."""
    c = codes.reshape(-1)
    if bits == 8:
        return c.to(torch.uint8)
    if bits == 2:
        v = (c.to(torch.int16) + 1).to(torch.uint8)
        per = 4
    elif bits == 4:
        v = (c.to(torch.int16) + 8).to(torch.uint8)
        per = 2
    elif bits == 6:
        v = (c.to(torch.int16) + 32).to(torch.uint8)
        per = 4
    else:
        raise ValueError(f"unsupported bits {bits}")

    pad = (-v.numel()) % per
    if pad:
        v = torch.cat([v, torch.zeros(pad, dtype=torch.uint8, device=v.device)])
    if bits == 6:
        v32 = v.reshape(-1, 4).to(torch.int32)
        b0 = v32[:, 0] | (v32[:, 1] << 6)
        b1 = (v32[:, 1] >> 2) | (v32[:, 2] << 4)
        b2 = (v32[:, 2] >> 4) | (v32[:, 3] << 2)
        return (torch.stack([b0, b1, b2], dim=1) & 0xFF).reshape(-1).to(torch.uint8)
    v32 = v.reshape(-1, per).to(torch.int32)
    out = torch.zeros(v32.shape[0], dtype=torch.int32, device=v.device)
    for j in range(per):
        out |= v32[:, j] << (bits * j)
    return (out & 0xFF).to(torch.uint8)


def unpack_codes(packed: torch.Tensor, bits: int, numel: int) -> torch.Tensor:
    """Inverse of ``pack_codes``; returns int8 codes of length ``numel``."""
    if bits == 8:
        return packed.reshape(-1)[:numel].to(torch.int8)
    if bits == 6:
        b = packed.reshape(-1, 3).to(torch.int32)
        out = torch.empty(b.shape[0], 4, dtype=torch.int32, device=packed.device)
        out[:, 0] = b[:, 0] & 0x3F
        out[:, 1] = ((b[:, 0] >> 6) | (b[:, 1] << 2)) & 0x3F
        out[:, 2] = ((b[:, 1] >> 4) | (b[:, 2] << 4)) & 0x3F
        out[:, 3] = (b[:, 2] >> 2) & 0x3F
        return (out.reshape(-1)[:numel] - 32).to(torch.int8)
    per = 4 if bits == 2 else 2
    mask = (1 << bits) - 1
    b = packed.reshape(-1).to(torch.int32)
    cols = torch.stack([(b >> (bits * j)) & mask for j in range(per)], dim=1)
    vals = cols.reshape(-1)[:numel]
    off = 1 if bits == 2 else 8
    return (vals - off).to(torch.int8)


def row_bytes(k: int, bits: int) -> int:
    """Bytes per row for the packed layout.  Requires byte alignment."""
    if (k * bits) % 8:
        raise ValueError(f"row width {k} is not byte-aligned at {bits}-bit")
    return k * bits // 8


def quantize_tensor(w: torch.Tensor, bits: int, group: int = 128,
                    row_chunk: Optional[int] = None) -> Dict:
    """Pre-quantise one weight tensor (any leading dims) to a container option.

    Row-chunked: the float32 transients of a 537M-param bank are ~2 GB each,
    which does not fit next to the resident prefix and the packed options.
    """
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"unsupported bits {bits}")
    if w.ndim < 2:
        raise ValueError("quantize_tensor expects at least 2-D weights")
    k = w.shape[-1]
    if k % group:
        raise ValueError(f"width {k} not divisible by group {group}")
    if (k * bits) % 8:
        raise ValueError("width not byte-aligned for the packed layout")
    rows, ng = w.numel() // k, k // group
    if row_chunk is None:
        row_chunk = max(1, (4 << 20) // max(k, 1))
    codes = torch.empty(rows, k, dtype=torch.int8, device=w.device)
    scales = torch.empty(rows, ng, dtype=torch.float16, device=w.device)
    flat = w.reshape(-1, k)
    for r0 in range(0, rows, row_chunk):
        r1 = min(rows, r0 + row_chunk)
        if bits == 2:
            c, a = ternary_codes(flat[r0:r1], group)
        else:
            c, a = int_codes(flat[r0:r1], bits, group)
        codes[r0:r1] = c
        scales[r0:r1] = a
    return {"packed": pack_codes(codes, bits), "scales": scales, "bits": bits,
            "shape": tuple(w.shape), "group": group,
            "n_rows": int(codes.shape[0]), "n_params": int(w.numel())}


def iter_dequant(opt: Dict, dtype: torch.dtype, row_chunk: Optional[int] = None
                 ) -> Iterator[Tuple[Tuple[int, int], torch.Tensor]]:
    """Row-chunked dequantisation: yields ((r0, r1), [r1-r0, k] block).

    ``r0``/``r1`` index the flattened leading dims (``opt['shifted rows']``).
    """
    shape = opt["shape"]
    k, group, bits = shape[-1], opt["group"], opt["bits"]
    n, ng = opt["n_rows"], k // group
    if row_chunk is None:
        row_chunk = max(1, (4 << 20) // max(k, 1))
    bpr = row_bytes(k, bits)
    flat = opt["packed"].reshape(-1)
    for r0 in range(0, n, row_chunk):
        r1 = min(n, r0 + row_chunk)
        chunk = flat[r0 * bpr: r1 * bpr]
        codes = unpack_codes(chunk, bits, (r1 - r0) * k).reshape(-1, ng, group).float()
        sc = opt["scales"][r0:r1].float()
        yield (r0, r1), (codes * sc.unsqueeze(-1)).reshape(r1 - r0, k).to(dtype)


def dequant_option(opt: Dict, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Full dequantisation (prototype convenience; prefer ``iter_dequant``)."""
    out = torch.empty(opt["n_rows"], opt["shape"][-1], dtype=dtype,
                      device=opt["packed"].device)
    for (r0, r1), block in iter_dequant(opt, dtype):
        out[r0:r1] = block
    return out.reshape(opt["shape"])


def dot_grad_option(grad: torch.Tensor, opt: Dict) -> float:
    """<grad, W_option> over a weight tensor, row-chunked to bound memory."""
    g = _flatten_rows(grad)
    total = 0.0
    for (r0, r1), block in iter_dequant(opt, torch.bfloat16):
        total += float((g[r0:r1].float() * block.float()).sum())
    return total


def quantize_grouped(w: torch.Tensor, bits: int, group: int = 128,
                     row_chunk: Optional[int] = None) -> torch.Tensor:
    """Dequantised container values at ``bits`` (2 = deployable Lloyd ternary).

    The trainer/export path uses this so an RCO assignment trains against the
    same weights the gate measures; row-chunked for the 3-D expert banks.
    """
    if bits == 2:
        return ternary_lloyd(w, group)
    f = _flatten_rows(w)
    n, k = f.shape
    out = torch.empty_like(f)
    if row_chunk is None:
        row_chunk = max(1, (4 << 20) // max(k, 1))
    for r0 in range(0, n, row_chunk):
        r1 = min(n, r0 + row_chunk)
        c, a = int_codes(f[r0:r1], bits, group)
        block = (c.reshape(r1 - r0, -1, group).float()
                 * a.unsqueeze(-1).float()).reshape(r1 - r0, k)
        out[r0:r1] = block.to(out.dtype)
    return out.reshape(w.shape)


# ------------------------------------------------------------- allocator ---
class BudgetAllocator:
    """Projected Gumbel-STE loop over G groups x K container options.

    ``costs``  : [K] bits per parameter per option.
    ``weights``: [G] parameter count per group.
    ``target_bits``: weighted-average bits budget (e.g. 2.656 = ternary +25%).
    """

    def __init__(self, costs: torch.Tensor, weights: torch.Tensor,
                 target_bits: float, lr: float = 0.05, device="cpu"):
        self.costs = costs.to(device).float()
        self.weights = weights.to(device).float()
        self.fracs = self.weights / self.weights.sum()
        self.target = float(target_bits)
        self.g, self.k = self.weights.numel(), self.costs.numel()
        self.alpha = torch.zeros(self.g, self.k, device=device, requires_grad=True)
        self.opt = torch.optim.Adam([self.alpha], lr=lr)
        self.init_to_target()

    # -- init ---------------------------------------------------------------
    def init_to_target(self, lo: float = 0.0, hi: float = 60.0) -> None:
        """Set alpha = -beta*costs so E[bits] = target (reference init)."""
        bits = self.costs.detach()

        def expected(beta: float) -> float:
            p = torch.softmax(-beta * bits, dim=0)
            return float((p * bits).sum())

        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if expected(mid) > self.target:
                lo = mid
            else:
                hi = mid
        with torch.no_grad():
            self.alpha.copy_((-0.5 * (lo + hi) * bits).unsqueeze(0).expand_as(self.alpha))
        self.retract()

    def retract(self) -> float:
        return retraction(self.alpha, self.costs, self.target, self.fracs, tol=1e-4)

    def expected_bits(self) -> float:
        return budget_value(self.alpha, self.costs, self.fracs)

    def probs(self) -> torch.Tensor:
        with torch.no_grad():
            return torch.softmax(self.alpha, dim=-1)

    # -- step ---------------------------------------------------------------
    def sample(self, tau: float, generator: Optional[torch.Generator] = None):
        """Gumbel-perturbed sample: (hard assignment, soft probs with grad)."""
        u = torch.rand(self.alpha.shape, generator=generator,
                       device=self.alpha.device).clamp_(1e-20, 1 - 1e-20)
        gumbel = -torch.log(-torch.log(u))
        noisy = (self.alpha + gumbel) / tau
        hard = budget_constrained_argmax(noisy.detach(), self.fracs,
                                         self.costs, self.target)
        soft = torch.softmax(noisy, dim=-1)          # keeps the alpha graph
        return hard, soft

    def step(self, dl_dp: torch.Tensor, soft: torch.Tensor) -> Dict[str, float]:
        """One Riemannian Adam step given dL/dp and the soft probs."""
        self.alpha.grad = None
        (dl_dp.detach() * soft).sum().backward()
        diag: Dict[str, float] = {}
        if self.alpha.grad is not None:
            with torch.no_grad():
                diag["grad_finite"] = bool(torch.isfinite(self.alpha.grad).all())
                diag["grad_absmax"] = float(self.alpha.grad.abs().max())
            coeff, raw, proj = project_gradient(self.alpha, self.costs, self.fracs)
            diag.update(proj_coeff=coeff, grad_raw_norm=raw, grad_proj_norm=proj)
        self.opt.step()
        diag["budget_before_retract"] = self.expected_bits()
        diag["budget"] = self.retract()
        vector_transport(self.opt, self.alpha, self.costs, self.fracs)
        with torch.no_grad():
            p = torch.softmax(self.alpha, dim=-1)
            diag["alpha_entropy"] = float(-(p * (p + 1e-10).log()).sum(-1).mean())
            diag["decided"] = int((p.max(-1).values > 0.9).sum())
        return diag


# ------------------------------------------------------------------ misc ---
def greedy_assignment(options: Sequence[int], costs_bpw: torch.Tensor,
                      weights: torch.Tensor, target_bits: float,
                      order: Sequence[int]) -> Dict[int, int]:
    """Hand-rule baseline: upgrade groups in ``order`` while the budget lasts.

    ``order`` is a permutation of group indices (e.g. descending ternary error
    for the sensitivity rule, or identity for the uniform rule).
    """
    total_w = float(weights.sum())
    base = sum(float(weights[g]) * float(costs_bpw[0])
               for g in range(len(weights)))
    budget = float(target_bits) * total_w - base
    assigned = {int(g): int(options[0]) for g in range(len(weights))}
    label_to_cost = {int(b): float(costs_bpw[i]) for i, b in enumerate(options)}
    for g in order:
        g = int(g)
        for b in options:
            if b <= assigned[g]:
                continue
            extra = (label_to_cost[int(b)] - label_to_cost[assigned[g]]) * float(weights[g])
            if extra <= budget:
                budget -= extra
                assigned[g] = int(b)
    return assigned


def assignment_bits(assigned: Dict[int, int], weights: torch.Tensor) -> float:
    total = sum(float(weights[g]) * b for g, b in assigned.items())
    return total / float(weights.sum())


__all__ = [
    "SCALE_OVERHEAD", "TERNARY_BPW", "SUPPORTED_BITS",
    "budget_normal", "budget_value", "project_gradient", "retraction",
    "vector_transport", "budget_constrained_argmax",
    "ternary_codes", "int_codes", "pack_codes", "unpack_codes", "row_bytes",
    "quantize_tensor", "iter_dequant", "dequant_option", "dot_grad_option",
    "quantize_grouped", "BudgetAllocator", "greedy_assignment",
    "assignment_bits",
]
