"""T2 — Walsh–Hadamard rotation `R = (1/√n)·H_n·diag(S)` and its absorption.

Contract: `bonsai_forensics/spec.md` §1 (spec tbr-1.1, pinned by
`tests/ternary/test_rotation.py::test_spec_hash_is_pinned`). ADR-3: `R` is
folded into adjacent weights, no mainline kernel pays a runtime Hadamard.

Conventions (torch-style linear `y = x Wᵀ`, hidden axis last):

- `apply_rotation(x)` computes `R x` for every block along the last axis;
- `apply_rotation(x, transpose=True)` computes `Rᵀ x`;
- `absorb_input(W)`  → `W Rᵀ`  (rotated activations feed an unrotated output);
- `absorb_output(W)` → `R W`   (unrotated activations feed a rotated output).

Float64 reference implementation; production casts per the spec error budget.
The default sign source is the SHA-256 counter PRF in spec §1.4 (no RNG).

T26 adds Prism's **explicit** sign vectors as an alternative source: the
released 27B GGUF carries `prism.hadamard.*` metadata (block 1024, dims
5120/6144/17408) with sign files on disk. `load_sign_manifest` /
`load_sign_file` parse them and `resolve_rotations` selects explicit signs per
width, falling back to the PRF only when asked. The frozen spec hash is
unchanged — the manifest is a run-level basis override, not a new contract.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

SPEC_SHA256 = "0d2c008b4aee726351f9b90e44ec003c18b579d8690db24c77a089d9e1fc652b"

TBR_N = 1024
DEFAULT_DOMAIN = "hidden"


def block_size(d: int) -> int:
    """`g(d) = min(1024, 2^v2(d))` — no zero-padding (spec §1.2)."""
    if d <= 0:
        raise ValueError(f"dimension must be positive, got {d}")
    return min(d & -d, TBR_N)


def hadamard(g: int) -> np.ndarray:
    """Normalized Sylvester Hadamard `H_g[i, j] = (-1)^popcount(i & j)/√g`."""
    if g < 1 or g & (g - 1):
        raise ValueError(f"Hadamard size must be a power of two, got {g}")
    h = np.ones((1, 1), dtype=np.float64)
    while h.shape[0] < g:
        h = np.block([[h, h], [h, -h]])
    return h / math.sqrt(g)


def sign_vector(seed: int, domain: str = DEFAULT_DOMAIN, block: int = 0, g: int = TBR_N) -> np.ndarray:
    """`S ∈ {±1}^g` from the spec §1.4 SHA-256 counter PRF."""
    if g <= 0:
        raise ValueError(f"sign vector size must be positive, got {g}")
    out = np.empty(g, dtype=np.float64)
    tag = f"{int(seed)}|{domain}|{int(block)}|".encode()
    for i in range(g):
        digest = hashlib.sha256(tag + str(i // 8).encode()).digest()
        out[i] = 1.0 if (digest[0] >> (7 - i % 8)) & 1 else -1.0
    return out


def rotations_for(d: int, seed: int, domain: str = DEFAULT_DOMAIN,
                  block: int | None = None) -> list[np.ndarray]:
    """Per-block sign vectors for a dimension `d` (spec §1.2).

    `block` overrides the spec's `min(1024, 2^v2(d))` rule — used to isolate the
    block-size confound on a model whose width permits more than one divisor.
    Must divide `d`."""
    g = block_size(d) if block is None else int(block)
    if g <= 0 or d % g:
        raise ValueError(f"rotation block {g} does not divide width {d}")
    return [sign_vector(seed, domain, k, g) for k in range(d // g)]


def load_sign_file(path: str | Path) -> list[np.ndarray]:
    """Load one Prism `hadamard-signs-<d>.npy` vector as per-block ±1 signs (T26).

    Accepts a flat `(d,)` or `(blocks, g)` array; blocks follow the spec §1.2
    rule, so the file's width is the full rotated dimension."""
    arr = np.asarray(np.load(Path(path)), dtype=np.float64)
    if arr.ndim == 1:
        flat = arr
    elif arr.ndim == 2:
        flat = arr.reshape(-1)
    else:
        raise ValueError(f"sign vector must be 1-D or 2-D, got shape {arr.shape}")
    if flat.size == 0:
        raise ValueError("sign vector is empty")
    if not np.all(np.isin(flat, (-1.0, 1.0))):
        raise ValueError("sign vectors must be ±1")
    g = block_size(int(flat.size))
    if flat.size % g:
        raise ValueError(f"sign vector width {flat.size} is not a multiple of block {g}")
    return [flat[k * g : (k + 1) * g].copy() for k in range(flat.size // g)]


def load_sign_manifest(path: str | Path) -> dict[str, list[np.ndarray]]:
    """Load Prism's `hadamard-manifest.json` → `{width: per-block signs}` (T26).

    Validates the disclosed transform (normalized Sylvester Walsh–Hadamard,
    block 1024) and cross-checks `sign_files` against `sign_widths`; a mismatch
    in either direction is an error, never a silent partial basis."""
    manifest_path = Path(path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    block = int(data.get("block_size", TBR_N))
    if block != TBR_N:
        raise ValueError(f"manifest block_size {block} != spec n {TBR_N}")
    transform = data.get("transform")
    if transform not in (None, "normalized-sylvester-walsh-hadamard"):
        raise ValueError(f"unsupported manifest transform {transform!r}")
    declared = {str(int(w)) for w in data.get("sign_widths", [])}
    files = data.get("sign_files")
    if not files:
        raise ValueError("manifest has no sign_files")
    out: dict[str, list[np.ndarray]] = {}
    for name in files:
        signs = load_sign_file(manifest_path.parent / name)
        width = str(len(signs) * len(signs[0]))
        if int(width) % block:
            raise ValueError(f"sign file {name!r} width {width} is not a multiple of block {block}")
        if declared and width not in declared:
            raise ValueError(f"sign file {name!r} width {width} not declared in sign_widths")
        out[width] = signs
    missing = sorted(declared - set(out), key=int)
    if missing:
        raise ValueError(f"sign_widths declared without sign files: {missing}")
    return out


def resolve_rotations(
    d: int,
    sign_sets: Mapping[str, list[np.ndarray]] | None = None,
    seed: int | None = None,
    domain: str = DEFAULT_DOMAIN,
    *,
    strict: bool = False,
) -> list[np.ndarray]:
    """Per-block signs for width `d`: explicit `sign_sets[str(d)]` first, else
    the spec §1.4 PRF (`seed`/`domain`).

    `strict=True` refuses the PRF fallback when a sign manifest is supplied but
    lacks this width — mixing bases mid-pipeline is a correctness bug, not a
    default."""
    key = str(int(d))
    if sign_sets is not None and key in sign_sets:
        return sign_sets[key]
    if strict:
        raise ValueError(f"signs manifest has no sign set for width {d}")
    if seed is None:
        raise ValueError(f"no explicit signs for width {d} and no seed given")
    return rotations_for(d, seed, domain)


def rotation_matrix(signs: np.ndarray) -> np.ndarray:
    """Dense `R = (1/√g)·H_g·diag(S)` for one block (tests / small g)."""
    signs = np.asarray(signs, dtype=np.float64)
    return hadamard(len(signs)) * signs[None, :]


def sign_digest(d: int, seed: int, domain: str = DEFAULT_DOMAIN,
                block: int | None = None) -> str:
    """Return a stable digest of the PRF signs for one rotated width."""
    rotations = rotations_for(d, seed, domain, block=block)
    digest = hashlib.sha256()
    digest.update(f"width={int(d)}|seed={int(seed)}|domain={domain}|blocks={len(rotations)}|".encode())
    for signs in rotations:
        digest.update(np.asarray(signs, dtype=np.int8).tobytes())
    return digest.hexdigest()


def basis_digest(widths: Iterable[int], seed: int, domain: str = DEFAULT_DOMAIN,
                 block: int | None = None) -> str:
    """Digest a complete set of width-specific PRF sign vectors."""
    digest = hashlib.sha256()
    for width in sorted(set(int(x) for x in widths)):
        digest.update(str(width).encode())
        digest.update(b"|")
        digest.update(sign_digest(width, seed, domain, block).encode())
        digest.update(b"|")
    return digest.hexdigest()


def explicit_sign_digest(sign_sets: Mapping[str, list[np.ndarray]]) -> str:
    """Digest explicit per-width sign manifests independently of filenames."""
    digest = hashlib.sha256()
    for width in sorted((int(x) for x in sign_sets)):
        rotations = sign_sets[str(width)] if str(width) in sign_sets else sign_sets[width]
        digest.update(str(width).encode())
        digest.update(b"|")
        for signs in rotations:
            digest.update(np.asarray(signs, dtype=np.int8).tobytes())
        digest.update(b"|")
    return digest.hexdigest()


def materialize_rotation(d: int, seed: int, domain: str = DEFAULT_DOMAIN) -> np.ndarray:
    """Full block-diagonal `R_d` — test/analysis helper, O(d²) memory."""
    rots = rotations_for(d, seed, domain)
    g = len(rots[0])
    out = np.zeros((d, d), dtype=np.float64)
    for k, signs in enumerate(rots):
        out[k * g : (k + 1) * g, k * g : (k + 1) * g] = rotation_matrix(signs)
    return out


def _fwht(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Fast Walsh–Hadamard transform (unnormalized), O(n log n)."""
    x = np.moveaxis(np.asarray(x, dtype=np.float64), axis, -1).copy()
    n = x.shape[-1]
    if n == 0 or n & (n - 1):
        raise ValueError(f"FWHT length must be a power of two, got {n}")
    h = 1
    while h < n:
        x = x.reshape(*x.shape[:-1], n // (2 * h), 2, h)
        a = x[..., 0, :].copy()
        b = x[..., 1, :].copy()
        x[..., 0, :] = a + b
        x[..., 1, :] = a - b
        x = x.reshape(*x.shape[:-3], n)
        h *= 2
    return np.moveaxis(x, -1, axis)


def apply_rotation(x: np.ndarray, rotations: list[np.ndarray], transpose: bool = False) -> np.ndarray:
    """Apply `R` (`transpose=False`) or `Rᵀ` (`transpose=True`) blockwise.

    `R x = H (S ⊙ x)/√g` and `Rᵀ x = S ⊙ (H x/√g)`; both are O(g log g).
    """
    x = np.asarray(x)
    if x.shape[-1] == 0:
        return x.astype(np.float64, copy=True)
    g = len(rotations[0])
    if x.shape[-1] != g * len(rotations):
        raise ValueError(f"last axis {x.shape[-1]} does not match {len(rotations)} blocks of {g}")
    out = np.empty_like(x, dtype=np.float64)
    for k, signs in enumerate(rotations):
        blk = x[..., k * g : (k + 1) * g]
        if transpose:
            out[..., k * g : (k + 1) * g] = _fwht(blk) / math.sqrt(g) * signs
        else:
            out[..., k * g : (k + 1) * g] = _fwht(blk * signs) / math.sqrt(g)
    return out


def absorb_input(w: np.ndarray, rotations: list[np.ndarray]) -> np.ndarray:
    """`W' = W Rᵀ` — the consuming linear of a rotated activation (spec §1.3)."""
    w = np.asarray(w, dtype=np.float64)
    g = len(rotations[0])
    if w.shape[-1] != g * len(rotations):
        raise ValueError(f"input axis {w.shape[-1]} does not match {len(rotations)} blocks of {g}")
    out = np.empty_like(w)
    for k, signs in enumerate(rotations):
        out[..., k * g : (k + 1) * g] = _fwht(w[..., k * g : (k + 1) * g] * signs) / math.sqrt(g)
    return out


def absorb_output(w: np.ndarray, rotations: list[np.ndarray]) -> np.ndarray:
    """`W' = R W` — the producing linear of a rotated activation (spec §1.3)."""
    w = np.asarray(w, dtype=np.float64)
    g = len(rotations[0])
    if w.shape[0] != g * len(rotations):
        raise ValueError(f"output axis {w.shape[0]} does not match {len(rotations)} blocks of {g}")
    out = np.empty_like(w)
    for k, signs in enumerate(rotations):
        block = w[k * g : (k + 1) * g, :]
        out[k * g : (k + 1) * g, :] = _fwht(block * signs[:, None], axis=0) / math.sqrt(g)
    return out


def unabsorb_input(w: np.ndarray, rotations: list[np.ndarray]) -> np.ndarray:
    """Inverse of `absorb_input`: `W = (W Rᵀ) R` — recover the primal weight.

    Used by the PQ2_0 export bridge: an rmd-trained master is stored in the
    absorbed basis (`W Rᵀ`), so it must be un-rotated before the canonical
    exporter re-folds the norm and re-absorbs it.
    """
    w = np.asarray(w, dtype=np.float64)
    g = len(rotations[0])
    if w.shape[-1] != g * len(rotations):
        raise ValueError(f"input axis {w.shape[-1]} does not match {len(rotations)} blocks of {g}")
    return apply_rotation(w, rotations, transpose=True)


def unabsorb_output(w: np.ndarray, rotations: list[np.ndarray]) -> np.ndarray:
    """Inverse of `absorb_output`: `W = Rᵀ (R W)`."""
    w = np.asarray(w, dtype=np.float64)
    g = len(rotations[0])
    if w.shape[0] != g * len(rotations):
        raise ValueError(f"output axis {w.shape[0]} does not match {len(rotations)} blocks of {g}")
    out = np.empty_like(w)
    for k, signs in enumerate(rotations):
        block = w[k * g : (k + 1) * g, :]
        out[k * g : (k + 1) * g, :] = _fwht(block, axis=0) / math.sqrt(g) * signs[:, None]
    return out


def rms_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """γ-free RMSNorm; commutes exactly with `R` (spec §1.3)."""
    x = np.asarray(x, dtype=np.float64)
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)


def fold_norm_scale(w: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Fold a hidden-axis RMSNorm `γ` into its consuming linear."""
    w = np.asarray(w, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    if gamma.shape[-1] != w.shape[-1]:
        raise ValueError(f"gamma {gamma.shape} does not match linear input {w.shape[-1]}")
    return w * gamma


def unfold_norm_scale(hessian: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Remove `γ` from a Hessian captured on the original norm output (T22).

    `H = E[xᵀx]` with `x = γ ⊙ z` gives `H = D H_z D` (`D = diag(γ)`), so the
    Hessian of the γ-stripped stream is `H_z = D⁻¹ H D⁻¹ = H / outer(γ, γ)`.
    Apply before `rotate_hessian` (spec rule `unfold_then_rotate`).
    """
    hessian = np.asarray(hessian, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError(f"hessian must be square, got {hessian.shape}")
    if gamma.shape[-1] != hessian.shape[-1]:
        raise ValueError(f"gamma {gamma.shape} does not match hessian {hessian.shape}")
    return hessian / np.outer(gamma, gamma)
