"""Q4_0_G128 (GGUF type 147) numpy codec.

128 weights per block: one fp16 scale plus 64 bytes of nibbles.  Codes are
symmetric in [-7, 7] stored as code+8 in [1, 15]; the scale is ``amax/7``
computed in fp32 and then rounded to fp16.  Nibbles use the standard Q4_0
order: low = element ``j``, high = element ``j+64``.  This is exactly the
dequantisation used by ``rco_alloc.quantize_grouped(bits=4)`` and by the
``Q4_0_G128`` CPU/CUDA kernels.

The block payload is 66 bytes::

    <fp16 scale> <64 bytes of nibbles>

Rows are grouped along the last (contiguous) axis, matching the expert-bank
layout used by the 35B GGUF.
"""

from __future__ import annotations

import numpy as np

GROUP = 128
BLOCK_BYTES = 2 + GROUP // 2


def encode_np(w: np.ndarray, group: int = GROUP) -> np.ndarray:
    """Quantise a float array into Q4_0_G128 blocks.

    ``w`` is ``[..., cols]`` with ``cols % group == 0``.  Returns a uint8 array
    ``[..., cols // group, 66]``; the last axis is the block payload.
    """
    a = np.asarray(w)
    if group != GROUP:
        raise ValueError(f"Q4_0_G128 is group {GROUP}, got {group}")
    if a.shape[-1] % group:
        raise ValueError(f"last dim {a.shape[-1]} is not a multiple of {group}")
    ng = a.shape[-1] // group
    g = a.reshape(*a.shape[:-1], ng, group).astype(np.float32)

    amax = np.max(np.abs(g), axis=-1)
    scale = (amax / 7.0).astype(np.float16)
    s = scale.astype(np.float32)
    inv = np.where(s > 0, 1.0 / np.maximum(s, 1e-30), 0.0)
    codes = np.rint(g * inv[..., None])
    codes = np.clip(codes, -7, 7).astype(np.int16) + 8

    # standard Q4_0 halves: low = j, high = j+64
    lo = codes[..., :group // 2]
    hi = codes[..., group // 2:]
    packed = (lo | (hi << 4)).astype(np.uint8)

    out = np.empty((*a.shape[:-1], ng, BLOCK_BYTES), dtype=np.uint8)
    out[..., :2] = scale.view(np.uint8).reshape(*scale.shape, 2)
    out[..., 2:] = packed
    return out


def decode_np(blocks: np.ndarray, group: int = GROUP) -> np.ndarray:
    """Inverse of :func:`encode_np`; returns float32 ``[..., cols]``."""
    b = np.asarray(blocks)
    if b.shape[-1] != BLOCK_BYTES:
        raise ValueError(f"expected {BLOCK_BYTES}-byte blocks, got {b.shape[-1]}")
    scale = b[..., :2].copy().view("<f2")[..., 0].astype(np.float32)
    packed = b[..., 2:]
    lo = (packed & 0x0F).astype(np.float32) - 8.0
    hi = (packed >> 4).astype(np.float32) - 8.0
    vals = np.concatenate([lo, hi], axis=-1)
    return (vals * scale[..., None]).reshape(*b.shape[:-2], b.shape[-2] * group)


def encode_row_bytes(w: np.ndarray) -> bytes:
    """Encode one row (or a flat group-aligned vector) to contiguous bytes."""
    a = np.asarray(w, dtype=np.float32).reshape(-1)
    return encode_np(a).tobytes()


def quantize_grouped_torch(w, group: int = GROUP, chunk: int = 32):
    """Torch-side encoder used by the exporter/trainer equivalence test.

    Returns ``(packed uint8 [..., ng, 66], dequantised float32)``.
    """
    import torch

    if w.shape[-1] % group:
        raise ValueError(f"last dim {w.shape[-1]} is not a multiple of {group}")
    ng = w.shape[-1] // group
    g = w.reshape(*w.shape[:-1], ng, group).float()
    amax = g.abs().amax(-1)
    scale = (amax / 7.0).half()
    s = scale.float()
    inv = torch.where(s > 0, 1.0 / s.clamp_min(1e-30), torch.zeros_like(s))
    codes = torch.clamp(torch.round(g * inv.unsqueeze(-1)), -7, 7).to(torch.int16) + 8
    lo = codes[..., :group // 2]
    hi = codes[..., group // 2:]
    packed = (lo | (hi << 4)).to(torch.uint8)

    out = torch.empty((*w.shape[:-1], ng, BLOCK_BYTES), dtype=torch.uint8,
                      device=w.device)
    out[..., :2] = scale.view(torch.uint8).reshape(*scale.shape, 2)
    out[..., 2:] = packed
    deq = (codes.to(torch.float32) - 8.0) * s.unsqueeze(-1)
    return out, deq.reshape(*w.shape)
