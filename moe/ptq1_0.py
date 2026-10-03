"""PTQ1_0 base-3 repacking of ternary expert banks (Tier 1, free, local).

The training container (:class:`CompactBank`) holds 2-bit codes + one fp16
scale per 128-group (PQ2_0-equivalent, 2.125 bpw).  The release container packs
the SAME trits base-3 in the fork's ``PTQ1_0`` block layout (``block_ptq1_0``
in ``ggml-common.h``)::

    qs[24]  5 trits/byte, staged 16-byte (values 0..79) then 8-byte (80..119)
    qh[2]   4 trits/byte (values 120..127)
    d       the SAME fp16 group scale (2 bytes)

28 bytes per 128 params = **1.75 bpw**.  Packing is lossless for already-
ternary banks (Tier 1's ~5.7 GB expert saving at quality cost ~0).

The byte codec replicates ``quantize_row_ptq1_0_ref`` EXACTLY (base-3 field
order, the ``ceil(Q*256/243)`` byte scaling, the qh 4-trit left shift), and
:func:`unpack_trits_ptq1_0` transcribes ``dequantize_row_ptq1_0``'s integer
trit extraction 1:1 — including the load-bearing ``uint8_t`` truncation of
``byte * 3**n`` (without it the high trits decode wrong).  The exhaustive
tests prove the round-trip is the identity on every representable byte, so
:func:`decode_ptq1_0` is bit-equal to the 2-bit :func:`decode_ternary` and the
C++ runtime reads exactly what training shipped.
"""

from __future__ import annotations

import torch

PTQ1_0_QK = 128
PTQ1_0_QS = 24
PTQ1_0_QH = 2
PTQ1_0_SCALE_BYTES = 2
PTQ1_0_BLOCK_BYTES = PTQ1_0_QS + PTQ1_0_QH + PTQ1_0_SCALE_BYTES  # 28
PTQ1_0_BPW = PTQ1_0_BLOCK_BYTES * 8 / PTQ1_0_QK                 # 1.75

# base-3 place values, most-significant trit first (matches the C++ q loop:
# q = q*3 + xi, so trit n=0 carries 3**4).
_POW3W = torch.tensor([81, 27, 9, 3, 1], dtype=torch.int64)
_QH_W = torch.tensor([27, 9, 3, 1], dtype=torch.int64)


def _check_trits(t: torch.Tensor) -> torch.Tensor:
    if t.shape[-1] != PTQ1_0_QK:
        raise ValueError(
            f"trit groups must be {PTQ1_0_QK} wide, got {tuple(t.shape)}")
    if bool(((t != -1) & (t != 0) & (t != 1)).any()):
        raise ValueError("trits must be in {-1, 0, +1}")
    return (t.to(torch.int64) + 1)  # xi in {0, 1, 2}


def _scale_byte(q: torch.Tensor) -> torch.Tensor:
    """The C++ ``ceil(Q*256/243)`` byte scaling (243 == 3**5)."""
    return ((q * 256 + 242) // 243).to(torch.uint8)


def pack_trits_ptq1_0(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack ``[..., 128]`` int8 trits into ``(qs [..., 24], qh [..., 2])``.

    Stage order mirrors ``quantize_row_ptq1_0_ref`` with
    ``ptq1_0_stages = {32, 16, 8}``: the 32-wide stage covers nothing at
    qs-size 24, the 16-wide stage takes values 0..79, the 8-wide stage values
    80..119, and ``qh`` takes 120..127.  Byte ``j`` of a stage with stride
    ``c`` holds trits ``j + n*c`` (``n = 0..4``), most-significant first.
    """
    xi = _check_trits(t)
    pre = xi.shape[:-1]
    v = xi.reshape(-1, PTQ1_0_QK)
    # stage c=16: values 0..79 as [16, 5] with [j, n] = value j + 16*n.
    s16 = v[:, 0:80].reshape(-1, 5, 16).transpose(1, 2).reshape(-1, 16, 5)
    q16 = (s16 * _POW3W.to(v.device)).sum(-1)
    # stage c=8: values 80..119 as [8, 5] with [j, n] = 80 + j + 8*n.
    s8 = v[:, 80:120].reshape(-1, 5, 8).transpose(1, 2).reshape(-1, 8, 5)
    q8 = (s8 * _POW3W.to(v.device)).sum(-1)
    qs = _scale_byte(torch.cat([q16, q8], dim=-1))
    # qh: values 120..127 as [2, 4] with [h, m] = 120 + h + 2*m; the C++
    # shifts the 4-trit field left one trit (q *= 3) before byte scaling.
    h2 = v[:, 120:128].reshape(-1, 4, 2).transpose(1, 2).reshape(-1, 2, 4)
    qh = _scale_byte((h2 * _QH_W.to(v.device)).sum(-1) * 3)
    return (qs.reshape(*pre, PTQ1_0_QS), qh.reshape(*pre, PTQ1_0_QH))


def unpack_trits_ptq1_0(qs: torch.Tensor, qh: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`pack_trits_ptq1_0`; int8 trits in ``{-1, 0, +1}``.

    A 1:1 transcription of ``dequantize_row_ptq1_0``'s integer math:
    ``xi = ((byte * 3**n) mod 256 * 3) >> 8``.  The ``mod 256`` is the C++
    ``uint8_t`` truncation — dropping it corrupts trits n >= 2.
    """
    if qs.shape[-1] != PTQ1_0_QS or qh.shape[-1] != PTQ1_0_QH:
        raise ValueError(
            f"expected qs [..., {PTQ1_0_QS}] + qh [..., {PTQ1_0_QH}], got "
            f"{tuple(qs.shape)} + {tuple(qh.shape)}")
    if qs.shape[:-1] != qh.shape[:-1]:
        raise ValueError("qs and qh batch shapes differ")
    pre = qs.shape[:-1]
    b = qs.reshape(-1, PTQ1_0_QS).to(torch.int64)
    h = qh.reshape(-1, PTQ1_0_QH).to(torch.int64)
    nrow = b.shape[0]
    out = torch.empty(nrow, PTQ1_0_QK, dtype=torch.int64, device=b.device)
    # qs stages: byte j holds values j + n*c.  Build [row, byte, n] trits,
    # transpose to [row, n, byte], flatten -> element order.  (reshape copies
    # are fine here; only in-place writes into strided views would alias.)
    t16 = torch.stack(
        [((((b[:, 0:16] * (3 ** n)) & 0xFF) * 3) >> 8) for n in range(5)],
        dim=-1)
    out[:, 0:80] = t16.transpose(1, 2).reshape(nrow, 80)
    t8 = torch.stack(
        [((((b[:, 16:24] * (3 ** n)) & 0xFF) * 3) >> 8) for n in range(5)],
        dim=-1)
    out[:, 80:120] = t8.transpose(1, 2).reshape(nrow, 40)
    # qh: byte h holds values 120 + h + 2*m for m = 0..3 (n = 0..3 only).
    th = torch.stack(
        [((((h * (3 ** n)) & 0xFF) * 3) >> 8) for n in range(4)], dim=-1)
    out[:, 120:128] = th.transpose(1, 2).reshape(nrow, 8)
    return (out.reshape(*pre, PTQ1_0_QK).to(torch.int8) - 1)


def repack_bank_ptq1_0(codes: torch.Tensor, scales: torch.Tensor,
                       group: int = PTQ1_0_QK
                       ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Repack a 2-bit ternary bank into PTQ1_0 blocks.

    ``codes`` is ``[..., ng, group // 4]`` uint8 (from ``pack_ternary_codes``),
    ``scales`` ``[..., ng]`` fp16 — the exact :class:`CompactBank` buffers.
    Returns ``(qs [..., ng, 24], qh [..., ng, 2], scales)`` with the input
    scales passed through untouched (same tensor).
    """
    from qwen4exp_proxy import unpack_ternary_codes
    if group != PTQ1_0_QK:
        raise ValueError(f"PTQ1_0 is defined at group {PTQ1_0_QK}, got {group}")
    if codes.shape[-1] != group // 4 or codes.shape[:-1] != scales.shape:
        raise ValueError(
            f"codes {tuple(codes.shape)} / scales {tuple(scales.shape)} "
            f"do not form {group}-groups")
    trits = unpack_ternary_codes(codes)
    qs, qh = pack_trits_ptq1_0(trits)
    return qs, qh, scales


def decode_ptq1_0(qs: torch.Tensor, qh: torch.Tensor, scales: torch.Tensor,
                  dtype: torch.dtype | None = None) -> torch.Tensor:
    """Decode PTQ1_0 blocks to dense, with the same op order as decode_ternary.

    ``qs``/``qh`` are ``[..., ng, 24]``/``[..., ng, 2]``, ``scales``
    ``[..., ng]`` fp16; the result is ``[..., ng * 128]``.  int8 -> float32,
    one multiply by the fp32 scale — so the output is bit-identical to the
    2-bit decode of the same trits+scales.
    """
    t = unpack_trits_ptq1_0(qs, qh).float()
    deq = t * scales.float().unsqueeze(-1)
    deq = deq.reshape(*deq.shape[:-2], -1)
    return deq if dtype is None else deq.to(dtype)


def pack_block_bytes(qs: torch.Tensor, qh: torch.Tensor,
                     scales: torch.Tensor) -> torch.Tensor:
    """Interleave one PTQ1_0 block per group: ``[..., ng, 28]`` uint8.

    Layout matches the C struct exactly: ``qs[24] | qh[2] | d[2]`` with the
    fp16 scale in native little-endian bytes.  This is what the GGUF export
    writer (item 3) serialises per row.
    """
    if (qs.shape[-1] != PTQ1_0_QS or qh.shape[-1] != PTQ1_0_QH
            or qs.shape[:-1] != qh.shape[:-1] or qs.shape[:-1] != scales.shape):
        raise ValueError("qs/qh/scales shapes do not form PTQ1_0 blocks")
    d16 = scales.to(torch.float16).contiguous().view(torch.uint8).reshape(
        *scales.shape, 2)
    return torch.cat([qs, qh, d16.reshape(*qs.shape[:-1], 2)], dim=-1)


def repack_bytes_per_param() -> float:
    """1.75 — the deployed PTQ1_0 line rate (values + scales)."""
    return PTQ1_0_BPW
