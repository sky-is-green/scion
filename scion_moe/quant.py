"""T3 — ternary codec `w ≈ s_g · t`, `t ∈ {−1, 0, +1}` (spec §2).

Scale search is exactly the frozen recipe: absmean init, half-away-from-zero
rounding, up to 4 least-squares refinements, best-of(init, refined) per group.
Fully deterministic (no RNG). This module is also the base quantizer GPTQ (T4)
and the GGUF packer (T6) call into.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SPEC_SHA256 = "0d2c008b4aee726351f9b90e44ec003c18b579d8690db24c77a089d9e1fc652b"

DEFAULT_GROUP_SIZE = 256
REFINE_ITERS = 4


def round_half_away(x: np.ndarray) -> np.ndarray:
    """Round half away from zero — mirrors `lroundf` in llama.cpp."""
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def clip_codes(x: np.ndarray) -> np.ndarray:
    """Nearest ternary code: `clip(round(x), -1, +1)` as int8."""
    return np.clip(round_half_away(x), -1, 1).astype(np.int8)


@dataclass(frozen=True)
class TernaryQuant:
    """Codes + per-group scales for a last-axis-grouped tensor."""

    codes: np.ndarray
    scales: np.ndarray
    group_size: int
    orig_numel: int

    def dequantize(self) -> np.ndarray:
        if self.scales.shape[:-1] != self.codes.shape[:-1]:
            raise ValueError("codes/scales leading shapes disagree")
        expanded = np.repeat(self.scales, self.group_size, axis=-1)
        return (self.codes.astype(np.float64) * expanded)[..., : self.orig_numel]

    @property
    def num_groups(self) -> int:
        return int(self.scales.shape[-1])


def _quantize_groups(groups: np.ndarray, refine_iters: int) -> tuple[np.ndarray, np.ndarray]:
    """Core group kernel. `groups`: (..., group_size); returns codes, scales."""
    groups = np.asarray(groups, dtype=np.float64)
    if refine_iters < 0:
        raise ValueError("refine_iters must be >= 0")

    absmean = np.abs(groups).mean(axis=-1, keepdims=True)
    s0 = np.where(absmean > 0.0, absmean, 1.0)
    t0 = clip_codes(groups / s0)

    def residual(s: np.ndarray, t: np.ndarray) -> np.ndarray:
        return ((groups - s * t) ** 2).sum(axis=-1)

    best_s = np.squeeze(s0, axis=-1)
    best_t = t0
    best_r = residual(s0, t0)

    s = s0
    t = t0
    for _ in range(refine_iters):
        tt = (t.astype(np.float64) ** 2).sum(axis=-1, keepdims=True)
        tg = (t.astype(np.float64) * groups).sum(axis=-1, keepdims=True)
        s_ls = np.where((tt > 0.0) & (tg > 0.0), tg / np.where(tt > 0.0, tt, 1.0), s)
        s = s_ls
        t = clip_codes(groups / s)

    if refine_iters > 0:
        tt = (t.astype(np.float64) ** 2).sum(axis=-1, keepdims=True)
        tg = (t.astype(np.float64) * groups).sum(axis=-1, keepdims=True)
        s = np.where((tt > 0.0) & (tg > 0.0), tg / np.where(tt > 0.0, tt, 1.0), s)

    r = residual(s, t)
    take_refined = r <= best_r
    scales = np.where(take_refined, np.squeeze(s, axis=-1), best_s)
    codes = np.where(take_refined[..., None], t, best_t).astype(np.int8)

    all_zero = (codes == 0).all(axis=-1)
    return codes, np.where(all_zero, 0.0, scales)


def quantize(w: np.ndarray, group_size: int = DEFAULT_GROUP_SIZE, refine_iters: int = REFINE_ITERS) -> TernaryQuant:
    """Quantize `w` with groups along the last axis (spec §2.2)."""
    w = np.asarray(w, dtype=np.float64)
    if w.ndim == 0:
        raise ValueError("quantize expects at least a 1-D tensor")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    orig = w.shape[-1]
    n_groups = -(-orig // group_size) if orig else 0
    padded_in = n_groups * group_size
    padded = np.zeros(w.shape[:-1] + (padded_in,), dtype=np.float64)
    if orig:
        padded[..., :orig] = w
    groups = padded.reshape(w.shape[:-1] + (n_groups, group_size))
    codes, scales = _quantize_groups(groups, refine_iters)
    return TernaryQuant(codes.reshape(padded.shape), scales, group_size, orig)


def quantize_group(w: np.ndarray, refine_iters: int = REFINE_ITERS) -> tuple[np.ndarray, float]:
    """Single-group convenience used by GPTQ's per-column quantizer."""
    w = np.asarray(w, dtype=np.float64).reshape(1, -1)
    codes, scales = _quantize_groups(w, refine_iters)
    return codes[0], float(scales[0])


def quantize_rtn_absmax(w: np.ndarray, group_size: int = DEFAULT_GROUP_SIZE) -> TernaryQuant:
    """Control baseline: RTN with per-group absmax scale (llama.cpp ref style)."""
    w = np.asarray(w, dtype=np.float64)
    orig = w.shape[-1]
    n_groups = -(-orig // group_size) if orig else 0
    padded = np.zeros(w.shape[:-1] + (n_groups * group_size,), dtype=np.float64)
    if orig:
        padded[..., :orig] = w
    groups = padded.reshape(w.shape[:-1] + (n_groups, group_size))
    amax = np.abs(groups).max(axis=-1, keepdims=True)
    s = np.where(amax > 0.0, amax, 1.0)
    codes = clip_codes(groups / s)
    scales = np.where((codes == 0).all(axis=-1), 0.0, np.squeeze(s, axis=-1))
    return TernaryQuant(codes.reshape(padded.shape), scales, group_size, orig)


def quantize_rtn_absmean(w: np.ndarray, group_size: int = DEFAULT_GROUP_SIZE) -> TernaryQuant:
    """Control baseline: RTN with absmean scale, no LS refinement."""
    w = np.asarray(w, dtype=np.float64)
    orig = w.shape[-1]
    n_groups = -(-orig // group_size) if orig else 0
    padded = np.zeros(w.shape[:-1] + (n_groups * group_size,), dtype=np.float64)
    if orig:
        padded[..., :orig] = w
    groups = padded.reshape(w.shape[:-1] + (n_groups, group_size))
    absmean = np.abs(groups).mean(axis=-1, keepdims=True)
    s = np.where(absmean > 0.0, absmean, 1.0)
    codes = clip_codes(groups / s)
    scales = np.where((codes == 0).all(axis=-1), 0.0, np.squeeze(s, axis=-1))
    return TernaryQuant(codes.reshape(padded.shape), scales, group_size, orig)


# Bonsai-2-class (PQ2_0) canonical quantizer.  Gate 1 forensics
# (`research/gate1-forensics.md`, `bonsai_forensics/gate1_forensics.py`) show
# that plain absmean RTN of the rotated base reproduces Prism's released PQ2_0
# trits at 0.896-0.948 agreement, while the LS-refined spec quantizer drops to
# 0.854-0.874: the refinement moves 6-7% of trits *away* from Prism.  The
# PQ2_0-class path is therefore deliberately *unrefined*.
#
# This does not change the frozen TQ2_0 contract (spec §2.2: g256, refine x4);
# it fixes the Bonsai-2-aligned PQ2_0 path (g128, refine 0) so the pilot's
# deployed projection and the packed artifact use the same quantizer.
PQ2_0_GROUP = 128


def quantize_pq2_0(w: np.ndarray, group_size: int = PQ2_0_GROUP) -> TernaryQuant:
    """Bonsai-2-class quantizer: plain absmean RTN at g128 (no LS refinement)."""
    return quantize_rtn_absmean(w, group_size)
