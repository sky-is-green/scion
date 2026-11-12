"""Tests for the FlashNext P2 compact-bank port on the 35B proxy.

The frozen expert banks are stored in their *deployed* form -- 2-bit ternary
codes + an fp16 scale per 128-group -- and the training forward decodes only
the expert slices a token hits (``--compact-banks``), with the decode and the
decoder layers recomputed in backward (``--grad-checkpoint``).  CPU-only: the
tiny random ``Qwen3_5MoeTextModel`` is the plumbing rig; the real-weights
bit-exactness gate runs on the 4-layer prefix (v2/MEMORY-PORT.md).

Pinned here:
  - pack/unpack round-trip and the Q2_0_g128 byte layout;
  - encode/decode is **bit-equal** to the in-place ``ternary_lloyd`` banker;
  - ``CompactBank.from_blocks`` reproduces the GGUF block layout (the v2
    verbatim container);
  - the compact forward reproduces the STE training forward;
  - a tiny model trains (LM + top-k KD) over the compact banks;
  - decode checkpointing (block/expert/group) is exact and backprops;
  - the chunked CE matches the full-vocab CE in value and gradient;
  - the new flags default off and ``group``/``--expert-group-size`` parse.
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import qwen35_moe_proxy as q35  # noqa: E402
from moe_proxy import ternary_lloyd  # noqa: E402


def tiny_config(**over):
    from transformers import Qwen3_5MoeTextConfig
    cfg = dict(
        vocab_size=1024, eos_token_id=1000, bos_token_id=1000,
        hidden_size=64, num_hidden_layers=3, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, attn_output_gate=True,
        linear_conv_kernel_dim=4, linear_key_head_dim=16,
        linear_value_head_dim=16, linear_num_key_heads=4,
        linear_num_value_heads=8,
        moe_intermediate_size=32, shared_expert_intermediate_size=32,
        num_experts=8, num_experts_per_tok=2,
        layer_types=["linear_attention", "linear_attention", "full_attention"],
        full_attention_interval=4,
        partial_rotary_factor=0.25,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default",
                         "partial_rotary_factor": 0.25,
                         "mrope_section": [3, 3, 2], "mrope_interleaved": True},
        rms_norm_eps=1e-6,
    )
    cfg.update(over)
    return Qwen3_5MoeTextConfig(**cfg)


@pytest.fixture(scope="module")
def device():
    """The Qwen3_5Moe GDN path calls Triton kernels even as its 'fallback'
    (transformers 5.17), so a model forward needs a real device."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA/ROCm device (GDN forward is Triton-backed)")
    return "cuda"


@pytest.fixture(scope="module")
def tiny(device):
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeTextModel)
    import torch.nn as nn
    torch.manual_seed(0)
    model = Qwen3_5MoeTextModel(tiny_config()).to(device)
    # the text prefix has no LM head; attach one so ``model_logits`` runs
    model.lm_head = nn.Linear(model.config.hidden_size,
                              model.config.vocab_size, bias=False).to(device)
    model.eval()
    return model


@pytest.fixture()
def args_ns():
    class A:
        group = 128
        quant = "lloyd"
        bank_gguf = ""
        compact_banks = True
        rank = 4
        branch_quant = "fp32"
        branch_target = "both"
        branch_gate = "none"
    return A()


# ------------------------------------------------------------- code packing --

def test_pack_unpack_roundtrip():
    torch.manual_seed(0)
    q = torch.randint(-1, 2, (5, 3, 32), dtype=torch.int8)
    packed = q35.pack_ternary_codes(q)
    assert packed.dtype == torch.uint8
    assert packed.shape == (5, 3, 8)
    assert torch.equal(q35.unpack_ternary_codes(packed), q)
    with pytest.raises(ValueError):
        q35.pack_ternary_codes(torch.zeros(4, 30, dtype=torch.int8))  # 30 % 4 != 0


def test_pack_layout_is_the_gguf_layout():
    """Q2_0_g128 stores code j at bits 2*(j%4) of byte j//4, value q+1."""
    torch.manual_seed(1)
    q = torch.randint(-1, 2, (8,), dtype=torch.int8)
    packed = q35.pack_ternary_codes(q)
    for j in range(8):
        byte = int(packed[j // 4])
        stored = (byte >> (2 * (j % 4))) & 0b11
        assert stored == int(q[j]) + 1


# ------------------------------------------------------- encode / decode -----

@pytest.mark.parametrize("shape,group", [
    ((4, 128, 256), 128),        # exact groups
    ((2, 3, 64), 128),           # group > last -> fallback to the row
    ((2, 3, 96), 32),            # explicit smaller group
])
def test_encode_decode_bit_equal_to_ternary_lloyd(shape, group):
    torch.manual_seed(0)
    w = torch.randn(*shape) * 0.2
    bank = q35.CompactBank.from_tensor(w, group, "lloyd")
    ref = ternary_lloyd(w.float(), group).to(w.dtype)
    got = bank.decode_all()
    assert got.shape == w.shape
    assert torch.equal(ref, got)     # bit-equal, not allclose


def test_decode_expert_is_a_slice():
    torch.manual_seed(3)
    w = torch.randn(5, 4, 128)
    bank = q35.CompactBank.from_tensor(w, 128, "lloyd")
    for e in range(5):
        assert torch.equal(bank.decode_expert(e), bank.decode_all()[e])


def test_decode_matches_the_gguf_dequantisation():
    """The container format IS Q2_0_g128: codes*q+1 packed 4/byte + fp16 scale.

    The block -> [E, out, ng, 32] reshape (``read_bank_blocks``) must decode to
    exactly the values the training-time seeding writes (this is the property
    the real-weights prefix gate checks on the deployed model).
    """
    import sys as _sys
    _sys.path.insert(0, str(HERE))
    from seed_banks_from_gguf import _decode_ternary_blocks
    torch.manual_seed(4)
    w = torch.randn(3, 8, 256) * 0.1
    bank = q35.CompactBank.from_tensor(w, 128, "lloyd")
    # repack the bank's buffers back into 34-byte Q2_0_g128 blocks
    ng = 2
    scale_bytes = bank.scales.reshape(-1).numpy().astype(np.float16).tobytes()
    code_bytes = bank.codes.reshape(-1).numpy().tobytes()
    blocks = np.frombuffer(
        b"".join(scale_bytes[i * 2:i * 2 + 2] + code_bytes[i * 32:(i + 1) * 32]
                 for i in range(bank.scales.numel())), dtype=np.uint8)
    codes, scales = _decode_ternary_blocks(blocks.tobytes())
    vals = codes.astype(np.float32) * scales[:, None]
    assert np.array_equal(vals.reshape(w.shape), bank.decode_all().numpy())


def test_from_blocks_verbatim_container():
    """The GGUF splits gate/up/down; the HF bank fuses gate+up along rows.

    Groups run along the last dim, so concatenating the per-tensor containers
    along axis 1 must equal encoding the fused bank directly.
    """
    torch.manual_seed(5)
    gate = torch.randn(3, 4, 256) * 0.1
    up = torch.randn(3, 4, 256) * 0.1
    down = torch.randn(3, 8, 128) * 0.1
    fused = torch.cat([gate, up], dim=1)
    direct = q35.CompactBank.from_tensor(fused, 128, "lloyd")
    g_bank = q35.CompactBank.from_tensor(gate, 128, "lloyd")
    u_bank = q35.CompactBank.from_tensor(up, 128, "lloyd")
    d_bank = q35.CompactBank.from_tensor(down, 128, "lloyd")
    fused_gu, fused_dn = q35.compact_banks_from_blocks(
        g_bank.codes, g_bank.scales, d_bank.codes, d_bank.scales, 128,
        fused.dtype, u_bank.codes, u_bank.scales)
    assert torch.equal(fused_gu.decode_all(), direct.decode_all())
    assert torch.equal(fused_gu.decode_all(),
                       ternary_lloyd(fused.float(), 128).to(fused.dtype))
    assert torch.equal(fused_dn.decode_all(),
                       ternary_lloyd(down.float(), 128).to(down.dtype))


# --------------------------------------------------------- compact forward ---

def test_compact_forward_bit_equal_to_inplace_banker(tiny, args_ns):
    """The compact decode must reproduce the current training path exactly.

    The current path freezes the banks in place (``ternarize_banks``) and runs
    the model's own forward over the dense ternary bank; the compact forward
    decodes the same per-expert slices and runs the same loop.
    """
    src = copy.deepcopy(tiny)
    ids = torch.randint(0, 1024, (1, 9), device=next(tiny.parameters()).device)
    q35.ternarize_banks(src, args_ns)
    with torch.no_grad():
        ref = q35.model_logits(src, ids)

    dst = copy.deepcopy(tiny)
    q35.compact_banks(dst, args_ns, verify=True)
    with torch.no_grad():
        got = q35.model_logits(dst, ids)
    # the banks are bit-equal (verify=True above); the forward matches to fp32
    # reduction-order noise (different matmul memory layout)
    assert torch.allclose(ref, got, atol=1e-5, rtol=1e-4)
    assert float((ref - got).abs().max()) < 1e-5


def test_compact_banks_free_the_masters(tiny, args_ns):
    model = copy.deepcopy(tiny)
    n = q35.compact_banks(model, args_ns)
    assert n == 2 * len(q35.text_layers(model))
    for layer in q35.text_layers(model):
        m = layer.mlp.experts
        assert getattr(m, "_compact", False)
        assert m.gate_up_proj.numel() == 0 and m.down_proj.numel() == 0
        assert m._compact_gu.param_numel() > 0


def test_build_student_compact_flag(tiny, args_ns):
    from olmoe_corrections import MoEWithCorrection
    model = copy.deepcopy(tiny)
    n = q35.build_student(model, args_ns)
    assert n > 0
    for layer in q35.text_layers(model):
        assert isinstance(layer.mlp, MoEWithCorrection)
        assert getattr(q35.moe_block(layer).experts, "_compact", False)


def test_compact_survives_the_ternary_class_patch(tiny, args_ns):
    """patch_experts runs first in stage_train; the compact dispatch must
    layer on top of it and keep dense ternary modules working."""
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeExperts)
    q35.patch_experts(args_ns.group)
    model = copy.deepcopy(tiny)
    q35.compact_banks(model, args_ns)
    assert getattr(Qwen3_5MoeExperts, "_q35_compact_patch", False)
    # compact modules decode; a dense module still ternarises
    dense = copy.deepcopy(tiny)
    q35.ternarize_banks(dense, args_ns)
    with torch.no_grad():
        dev = next(model.parameters()).device
        a = q35.model_logits(model, torch.randint(0, 1024, (1, 6), device=dev))
        b = q35.model_logits(dense, torch.randint(0, 1024, (1, 6), device=dev))
    assert torch.isfinite(a).all() and torch.isfinite(b).all()


def test_rebind_compact_forwards_restores_binding(tiny, args_ns):
    """Accelerate's hook removal restores a stale instance forward; the rebind
    must put ``compact_experts_forward`` back (P2 finding)."""
    model = copy.deepcopy(tiny)
    q35.compact_banks(model, args_ns)
    for layer in q35.text_layers(model):
        layer.mlp.experts.forward = lambda *a, **k: None   # stale restore
    n = q35.rebind_compact_forwards(model)
    assert n == len(q35.text_layers(model))
    for layer in q35.text_layers(model):
        assert getattr(layer.mlp.experts.forward, "__func__", None) \
            is q35.compact_experts_forward


def test_compact_class_dispatch_survives_lost_instance_forward(tiny, args_ns):
    """Even if the instance ``forward`` is lost, the class patch dispatches on
    ``_compact`` so the compact banks are still decoded."""
    model = copy.deepcopy(tiny)
    q35.compact_banks(model, args_ns)
    for layer in q35.text_layers(model):
        m = layer.mlp.experts
        m.__dict__.pop("forward", None)      # simulate a lost binding
    with torch.no_grad():
        out = q35.model_logits(model, torch.randint(0, 1024, (1, 6), device=next(model.parameters()).device))
    assert torch.isfinite(out).all()


# -------------------------------------------------------------- training -----

def test_compact_tiny_training_smoke(tiny, args_ns):
    """LM + a top-k KD term over the compact banks: one finite backward."""
    import torch.nn as nn
    model = copy.deepcopy(tiny)
    q35.build_student(model, args_ns)
    model.train()
    ids = torch.randint(0, 1024, (1, 12), device=next(model.parameters()).device)
    teacher = torch.randn(11, 8, device=next(model.parameters()).device)
    params = [p for p in model.parameters() if p.requires_grad]
    assert params
    opt = torch.optim.SGD(params, lr=1e-3)
    out = model(input_ids=ids, use_cache=False)
    logits = model.lm_head(out.last_hidden_state)
    lm = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                         ids[:, 1:].reshape(-1))
    sel = logits[:, :-1].topk(8, dim=-1)
    s = logits[:, :-1].gather(-1, sel.indices).reshape(-1, 8)
    kd = F.kl_div(F.log_softmax(s, dim=-1),
                  F.log_softmax(teacher, dim=-1),
                  log_target=True, reduction="batchmean")
    (lm + 2.0 * kd).backward()
    branch_grads = [p for n, p in model.named_parameters()
                    if ".branch." in n and p.grad is not None]
    assert branch_grads and all(bool(torch.isfinite(p.grad).all())
                                for p in branch_grads)
    opt.step()


# ------------------------------------------------------------- accounting ----

def test_memory_accounting_is_the_deployed_container(tiny, args_ns):
    model = copy.deepcopy(tiny)
    q35.compact_banks(model, args_ns)
    rep = q35.memory_report(model, args_ns, "test")
    numel = rep["expert_dense_numel"]
    assert numel > 0
    # 2 bits/param codes; 2 B per fp16 group scale
    assert rep["expert_codes"] == numel // 4
    assert rep["expert_bf16_equivalent"] == numel * 2
    expected_scales = sum(b.scales.numel() * 2
                          for layer in q35.text_layers(model)
                          for b in (layer.mlp.experts._compact_gu,
                                    layer.mlp.experts._compact_dn))
    assert rep["expert_scales"] == expected_scales
    assert rep["branches"] == 0        # no branches attached yet
    # the compact banks are strictly smaller than the bf16 master they replace
    assert rep["expert_codes"] + rep["expert_scales"] < rep["expert_bf16_equivalent"]


def test_deployed_container_bytes_on_128_groups():
    torch.manual_seed(0)
    w = torch.randn(4, 128, 256)
    bank = q35.CompactBank.from_tensor(w, 128, "lloyd")
    numel = w.numel()
    assert bank.codes.numel() == numel // 4          # 2 bits/param
    assert bank.scales.numel() == numel // 128       # fp16 per 128-group
    assert bank.nbytes() == numel // 4 + 2 * (numel // 128)


# --------------------------------------------------- decode checkpointing ----

@pytest.mark.parametrize("mode", ["block", "expert", "group"])
def test_compact_forward_grad_checkpoint_exact_and_finite(tiny, args_ns, mode):
    """The decode checkpoint is exact and still backprops (all granularities)."""
    import torch.nn as nn
    model = copy.deepcopy(tiny)
    q35.build_student(model, args_ns)
    ids = torch.randint(0, 1024, (1, 10), device=next(model.parameters()).device)
    with torch.no_grad():
        ref = model(input_ids=ids, use_cache=False).last_hidden_state.clone()
    n = q35.enable_grad_checkpointing(model, mode)
    assert n == len(q35.text_layers(model))
    model.train()
    out = model(input_ids=ids, use_cache=False).last_hidden_state
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4)
    logits = model.lm_head(out)
    F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                    ids[:, 1:].reshape(-1)).backward()
    grads = [p for n_, p in model.named_parameters()
             if ".branch." in n_ and p.grad is not None]
    assert grads and all(bool(torch.isfinite(p.grad).all()) for p in grads)


@pytest.mark.parametrize("group_size", [1, 2, 3, 8])
def test_compact_group_batched_matches_reference(tiny, args_ns, group_size):
    """The batched expert-group forward (padded bmm) must match the reference
    forward and keep finite branch grads for every group size (padding bugs
    only show at non-divisible sizes)."""
    import torch.nn as nn
    model = copy.deepcopy(tiny)
    q35.build_student(model, args_ns)
    ids = torch.randint(0, 1024, (1, 10), device=next(model.parameters()).device)
    model.eval()
    with torch.no_grad():
        ref = model(input_ids=ids, use_cache=False).last_hidden_state.clone()
    model.train()
    n = q35.enable_grad_checkpointing(model, "group")
    assert n == len(q35.text_layers(model))
    for layer in q35.text_layers(model):
        m = q35.moe_block(layer).experts
        m._ckpt_group = group_size
    out = model(input_ids=ids, use_cache=False).last_hidden_state
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4)
    logits = model.lm_head(out)
    F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                    ids[:, 1:].reshape(-1)).backward()
    grads = [p for n_, p in model.named_parameters()
             if ".branch." in n_ and p.grad is not None]
    assert grads and all(bool(torch.isfinite(p.grad).all()) for p in grads)


def test_enable_layer_checkpointing_flags_and_backprops(tiny):
    """The decoder layers get checkpointed (P2 L40S fit path) and a training
    step still backprops through the recompute."""
    model = copy.deepcopy(tiny)
    assert not any(getattr(l, "gradient_checkpointing", False)
                   for l in q35.text_layers(model))
    assert q35.enable_layer_checkpointing(model)
    assert all(getattr(l, "gradient_checkpointing", False)
               for l in q35.text_layers(model))
    model.train()
    ids = torch.randint(0, 1024, (1, 6), device=next(model.parameters()).device)
    out = model(input_ids=ids, use_cache=False).last_hidden_state
    F.cross_entropy(model.lm_head(out)[:, :-1].reshape(-1, 1024),
                    ids[:, 1:].reshape(-1)).backward()
    assert model.lm_head.weight.grad is not None


# ------------------------------------------------------------- chunked CE ----

def test_chunked_cross_entropy_matches_full_value_and_grad():
    """Pin the fit lever: the chunked (recomputed) CE is the same loss and
    gradient as the full-vocab CE it replaces."""
    torch.manual_seed(0)
    logits = torch.randn(23, 4096, requires_grad=True)
    tgt = torch.randint(0, 4096, (23,))
    full = F.cross_entropy(logits.float(), tgt)
    g_full = torch.autograd.grad(full, logits, retain_graph=True)[0]
    chunk = q35.chunked_cross_entropy(logits, tgt, chunk=8)
    g_chunk = torch.autograd.grad(chunk, logits)[0]
    assert torch.allclose(full, chunk, atol=1e-5, rtol=1e-4)
    assert torch.allclose(g_full, g_chunk, atol=1e-5, rtol=1e-4)


# ------------------------------------------------------------------ CLI ------

def test_compact_flag_default_off():
    ap = q35.build_parser()
    a = ap.parse_args(["train", "--cache-file", "/tmp/x.pt"])
    assert a.compact_banks is False
    assert a.grad_checkpoint is False
    assert a.chunked_ce is False
    b = ap.parse_args(["train", "--cache-file", "/tmp/x.pt", "--compact-banks"])
    assert b.compact_banks is True


def test_group_checkpoint_mode_parses():
    ap = q35.build_parser()
    a = ap.parse_args(["train", "--cache-file", "/tmp/x.pt",
                       "--checkpoint-mode", "group"])
    assert a.checkpoint_mode == "group"
    assert a.expert_group_size == 32
    b = ap.parse_args(["train", "--cache-file", "/tmp/x.pt",
                       "--checkpoint-mode", "group", "--expert-group-size", "64"])
    assert b.expert_group_size == 64
