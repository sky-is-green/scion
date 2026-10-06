"""Tests for the PTQ1_0 base-3 repack (``moe/ptq1_0.py``, Tier 1).

What is pinned here:
  - the byte codec replicates ``quantize_row_ptq1_0_ref`` exactly (stage
    order, base-3 field order, ``ceil(Q*256/243)`` scaling, qh shift);
  - the reader transcribes ``dequantize_row_ptq1_0`` 1:1 (including the
    load-bearing ``uint8_t`` truncation), proved by exhaustive round-trips
    over every representable qs byte (243) and qh byte (81);
  - the repack is lossless: ``decode_ptq1_0`` is BIT-EQUAL to the 2-bit
    ``decode_ternary`` on random banks and on real mirror weights;
  - the deployed line rate is 1.75 bpw with the input fp16 scales untouched.

The C++-side truth arrives in item 3 (the export writer loads our blocks
with the real fork); these tests prove the Python side emits exactly what
that reader expects.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest
import torch

import ptq1_0
from ptq1_0 import (decode_ptq1_0, pack_block_bytes, pack_trits_ptq1_0,
                    repack_bank_ptq1_0, unpack_trits_ptq1_0)


def base3_digits(q: int, n: int):
    return [(q // (3 ** (n - 1 - i))) % 3 for i in range(n)]


def test_exhaustive_byte_roundtrip():
    """Every qs byte value (243) and qh byte value (81) round-trips exactly,
    with no crosstalk into neighbouring positions."""
    torch.manual_seed(0)
    for Q in range(243):
        d5 = torch.tensor(base3_digits(Q, 5), dtype=torch.int8) - 1
        d4 = torch.tensor(base3_digits(Q % 81, 4), dtype=torch.int8) - 1
        # background: all-zero trits (xi = 1 everywhere -> byte 121).
        t = torch.zeros(128, dtype=torch.int8)
        t[[0, 16, 32, 48, 64]] = d5          # stage-c16 byte 0
        t[[80, 88, 96, 104, 112]] = d5       # stage-c8 byte 16
        t[[120, 122, 124, 126]] = d4         # qh byte 0
        qs, qh = pack_trits_ptq1_0(t)
        back = unpack_trits_ptq1_0(qs, qh)
        assert torch.equal(back, t), f"round-trip failed for Q={Q}"
        # the untouched bytes are the all-zero-trit encoding (121 scaled).
        zero_byte = (121 * 256 + 242) // 243
        assert bool((qs[1:16] == zero_byte).all())
        assert bool((qh[1:2] == ((40 * 3 * 256 + 242) // 243)).all())


def test_random_roundtrip_and_determinism():
    torch.manual_seed(7)
    t = torch.randint(-1, 2, (3, 5, 128), dtype=torch.int8)
    qs, qh = pack_trits_ptq1_0(t)
    assert qs.shape == (3, 5, 24) and qs.dtype == torch.uint8
    assert qh.shape == (3, 5, 2) and qh.dtype == torch.uint8
    assert torch.equal(unpack_trits_ptq1_0(qs, qh), t)
    qs2, qh2 = pack_trits_ptq1_0(t)
    assert torch.equal(qs, qs2) and torch.equal(qh, qh2)


def test_repack_decode_bitequal_random():
    """decode_ptq1_0 == decode_ternary bit-for-bit (same trits, same scales,
    same op order) — the Tier-1 'quality cost ~0' claim."""
    from qwen4exp_proxy import decode_ternary, pack_ternary_codes
    torch.manual_seed(11)
    trits = torch.randint(-1, 2, (2, 3, 4, 128), dtype=torch.int8)
    codes = pack_ternary_codes(trits)
    scales = (torch.randn(2, 3, 4) * 2).to(torch.float16)
    scales[0, 0, 0] = torch.zeros((), dtype=torch.float16)          # zero scale edge
    scales[0, 0, 1] = torch.full((), 6.1035156e-05, dtype=torch.float16)  # fp16 subnormal-min edge
    scales[0, 0, 2] = torch.full((), 65504.0, dtype=torch.float16)      # fp16 max edge
    scales[0, 0, 3] = torch.full((), -1.25, dtype=torch.float16)        # sign is the codes' job
    qs, qh, s_out = repack_bank_ptq1_0(codes, scales)
    assert s_out is scales  # scales pass through untouched
    assert qs.shape == (2, 3, 4, 24) and qh.shape == (2, 3, 4, 2)
    for dt in (None, torch.bfloat16):
        a = decode_ternary(codes, scales, dt)
        b = decode_ptq1_0(qs, qh, scales, dt)
        assert torch.equal(a, b), f"decode mismatch at dtype={dt}"
    # line rate: 24 + 2 + 2 bytes per 128 params = 1.75 bpw.
    nparams = float(trits.numel())
    nbytes = float(qs.numel() + qh.numel()) + scales.numel() * 2
    assert nbytes * 8 / nparams == pytest.approx(1.75)
    assert ptq1_0.repack_bytes_per_param() == pytest.approx(1.75)


def test_pack_block_bytes_matches_c_struct():
    from qwen4exp_proxy import pack_ternary_codes
    torch.manual_seed(3)
    trits = torch.randint(-1, 2, (2, 128), dtype=torch.int8)
    codes = pack_ternary_codes(trits)
    scales = torch.tensor([1.5, 0.25], dtype=torch.float16)
    qs, qh, _ = repack_bank_ptq1_0(codes, scales)
    blk = pack_block_bytes(qs, qh, scales)
    assert blk.shape == (2, 28) and blk.dtype == torch.uint8
    assert torch.equal(blk[:, :24], qs) and torch.equal(blk[:, 24:26], qh)
    d0 = struct.unpack("<e", bytes(blk[0, 26:28].tolist()))[0]
    d1 = struct.unpack("<e", bytes(blk[1, 26:28].tolist()))[0]
    assert d0 == pytest.approx(1.5) and d1 == pytest.approx(0.25)


def test_invalid_inputs_raise():
    from qwen4exp_proxy import pack_ternary_codes
    t = torch.zeros(127, dtype=torch.int8)
    with pytest.raises(ValueError):
        pack_trits_ptq1_0(t)
    bad = torch.zeros(128, dtype=torch.int8)
    bad[0] = 2
    with pytest.raises(ValueError):
        pack_trits_ptq1_0(bad)
    with pytest.raises(ValueError):
        unpack_trits_ptq1_0(torch.zeros(3, 23, dtype=torch.uint8),
                            torch.zeros(3, 2, dtype=torch.uint8))
    codes = pack_ternary_codes(torch.zeros(4, 128, dtype=torch.int8))
    with pytest.raises(ValueError):
        repack_bank_ptq1_0(codes, torch.zeros(5, dtype=torch.float16))
    with pytest.raises(ValueError):
        repack_bank_ptq1_0(codes, torch.zeros(4, dtype=torch.float16), group=64)


def test_mirror_bank_repack_bitequal():
    """The Tier-1 proof on real weights: one layer-0 gate_up + down_proj
    expert through the real encode path, repacked, decoded both ways."""
    import json
    safetensors = pytest.importorskip("safetensors")
    import qwen4exp_proxy as q4
    model_dir = Path(q4.MODEL)
    index = model_dir / "model.safetensors.index.json"
    if not index.exists():
        pytest.skip("local fp8 mirror not present")
    weight_map = json.loads(index.read_text())["weight_map"]
    shard_dir = model_dir / "shards"
    prefix = "model.language_model.layers.0.mlp.experts.0."
    cache = {}

    def get(key):
        shard = weight_map[key]
        h = cache.get(shard)
        if h is None:
            h = safetensors.safe_open(shard_dir / shard, framework="pt",
                                      device="cpu")
            cache[shard] = h
        return h.get_tensor(key)

    try:
        g = get(prefix + "gate_proj.weight")
        g_si = get(prefix + "gate_proj.weight_scale_inv")
        u = get(prefix + "up_proj.weight")
        u_si = get(prefix + "up_proj.weight_scale_inv")
        d = get(prefix + "down_proj.weight")
        d_si = get(prefix + "down_proj.weight_scale_inv")
    finally:
        for h in cache.values():
            h.__exit__(None, None, None)
    gu = torch.cat([q4.dequant_fp8_block(g, g_si),
                    q4.dequant_fp8_block(u, u_si)], dim=0)
    dn = q4.dequant_fp8_block(d, d_si)
    assert gu.shape[1] % 128 == 0 and dn.shape[1] % 128 == 0
    for name, bank in (("gate_up", gu), ("down", dn)):
        codes, scales = q4._encode_slice(bank, 128, "lloyd")
        qs, qh, _ = repack_bank_ptq1_0(codes, scales)
        for dt in (None, torch.bfloat16):
            a = q4.decode_ternary(codes, scales, dt)
            b = decode_ptq1_0(qs, qh, scales, dt)
            assert torch.equal(a, b), f"{name} decode mismatch at {dt}"
        nbytes = qs.numel() + qh.numel() + 2 * scales.numel()
        assert nbytes * 8 / bank.numel() == pytest.approx(1.75)
