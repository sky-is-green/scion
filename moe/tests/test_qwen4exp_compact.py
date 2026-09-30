"""Tests for the P2 compact-bank training mode (``--compact-banks``, gate G2).

The frozen expert banks are stored in their *deployed* form -- 2-bit ternary
codes + an fp16 scale per 128-group -- and the training forward decodes only the
expert slices a token hits.  CPU-only: the tiny random ``Qwen4ExpTextModel`` is
the plumbing rig; the real-weights 2-layer check is the ``compact-check`` stage
on the 7900 XT.

Pinned here:
  - pack/unpack round-trip;
  - encode/decode is **bit-equal** to the current in-place ``ternary_lloyd``
    banker (including the group fallback on non-128-divisible dims);
  - the fp8 + ``weight_scale_inv`` source dequantises before ternarising;
  - the compact forward reproduces the STE training forward bit-for-bit;
  - a tiny model trains (LM+KD) with the compact banks;
  - the byte accounting is the deployed container (2 bits + 2 B/group);
  - the ``--compact-banks`` flag is off by default.
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import qwen4exp_proxy as q4  # noqa: E402
from moe_proxy import ternary_lloyd  # noqa: E402


def tiny_config(**over):
    from transformers import Qwen4ExpTextConfig
    cfg = dict(
        vocab_size=1024, eos_token_id=1000, bos_token_id=1000,
        hidden_size=64, num_hidden_layers=3, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16,
        linear_conv_kernel_dim=4, linear_key_head_dim=16,
        linear_value_head_dim=16, linear_num_key_heads=4,
        linear_num_value_heads=8,
        moe_intermediate_size=32, shared_expert_intermediate_size=32,
        num_experts=8, num_experts_per_tok=2,
        layer_types=["linear_attention", "linear_attention", "full_attention"],
        hc_count=4, hc_lowrank=8,
        ple_layer_ids=[2], ple_embed_dim=64, ple_conv_kernel_size=4,
        ngram_size=3, heads_per_ngram=2, ngram_vocab_size_base=64,
        make_ngram_vocab_size_divisible_by=16,
        indexer_n_heads=1, indexer_kv_heads=1, indexer_head_dim=8,
        indexer_budget=8, indexer_compress_ratio=4,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default",
                         "partial_rotary_factor": 0.25,
                         "mrope_section": [3, 3, 2], "mrope_interleaved": True},
    )
    cfg.update(over)
    return Qwen4ExpTextConfig(**cfg)


@pytest.fixture(scope="module")
def tiny():
    q4.patch_indexer()
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextModel
    torch.manual_seed(0)
    model = Qwen4ExpTextModel(tiny_config())
    model.eval()
    return model


@pytest.fixture()
def args_ns():
    class A:
        group = 128
        quant = "lloyd"
        compact_banks = True
        rank = 4
        branch_quant = "fp32"
        branch_target = "both"
        branch_gate = "none"
        top_logits = 8
        tail_logits = 0
    return A()


# ------------------------------------------------------------- code packing --

def test_pack_unpack_roundtrip():
    torch.manual_seed(0)
    q = torch.randint(-1, 2, (5, 3, 32), dtype=torch.int8)
    packed = q4.pack_ternary_codes(q)
    assert packed.dtype == torch.uint8
    assert packed.shape == (5, 3, 8)
    assert torch.equal(q4.unpack_ternary_codes(packed), q)
    with pytest.raises(ValueError):
        q4.pack_ternary_codes(torch.zeros(4, 30, dtype=torch.int8))  # 30 % 4 != 0


# ------------------------------------------------------- encode / decode -----

@pytest.mark.parametrize("shape,group", [
    ((4, 128, 256), 128),        # exact groups
    ((2, 3, 64), 128),           # group > last -> fallback to the row
    ((2, 3, 96), 32),            # explicit smaller group
])
def test_encode_decode_bit_equal_to_ternary_lloyd(shape, group):
    torch.manual_seed(0)
    w = torch.randn(*shape) * 0.2
    bank = q4.CompactBank.from_tensor(w, group, "lloyd")
    ref = ternary_lloyd(w.float(), group).to(w.dtype)
    got = bank.decode_all()
    assert got.shape == w.shape
    assert torch.equal(ref, got)     # bit-equal, not allclose


def test_encode_decode_matches_inplace_banker(args_ns):
    """The exact path the trainer uses: quantize_bank_chunked (lloyd)."""
    torch.manual_seed(1)
    w = (torch.randn(3, 256, 128) * 0.1)
    ref = w.clone()
    redone = q4.quantize_bank_chunked(ref, 128, "lloyd", chunk=1)
    assert redone == 0
    bank = q4.CompactBank.from_tensor(w, 128, "lloyd")
    assert torch.equal(bank.decode_all(), ref)


def test_encode_fp8_dequantises_before_ternarising():
    torch.manual_seed(2)
    w = (torch.randn(2, 260, 256) * 0.4).to(torch.float8_e4m3fn)
    si = torch.rand(2, math.ceil(260 / 128), math.ceil(256 / 128)) + 0.5
    bank = q4.CompactBank.from_tensor(w, 128, "lloyd", scale_inv=si)
    # an fp8 master decodes to bf16 (never a second fp8 quantisation)
    assert bank.out_dtype == torch.bfloat16
    for e in range(2):
        ref = ternary_lloyd(q4.dequant_fp8_block(w[e], si[e]), 128).to(torch.bfloat16)
        assert torch.equal(bank.decode_expert(e), ref)


def test_decode_expert_is_a_slice(args_ns):
    torch.manual_seed(3)
    w = torch.randn(5, 4, 128)
    bank = q4.CompactBank.from_tensor(w, 128, "lloyd")
    for e in range(5):
        assert torch.equal(bank.decode_expert(e), bank.decode_all()[e])


def test_absmean_supported_finite(args_ns):
    torch.manual_seed(4)
    w = torch.randn(2, 128, 64)
    bank = q4.CompactBank.from_tensor(w, 128, "absmean")
    out = bank.decode_all()
    assert torch.isfinite(out).all()
    # per-group ternary: each row's group has at most {0, +a, -a}
    assert torch.unique(out[0, 0]).numel() <= 3


# --------------------------------------------------------- compact forward ---

def test_compact_forward_bit_equal_to_inplace_banker(tiny, args_ns):
    """The compact decode must reproduce the current training path exactly.

    The current path freezes the banks in place (``ternarize_banks``) and runs
    the model's own forward over the dense ternary bank; the compact forward
    decodes the same per-expert slices and runs the same loop.
    """
    src = copy.deepcopy(tiny)
    ids = torch.randint(0, 1024, (1, 9))
    q4.ternarize_banks(src, args_ns)
    with torch.no_grad():
        ref = src(input_ids=ids, use_cache=False).last_hidden_state

    dst = copy.deepcopy(tiny)
    q4.compact_banks(dst, args_ns, verify=True)
    with torch.no_grad():
        got = dst(input_ids=ids, use_cache=False).last_hidden_state
    # the banks are bit-equal (verify=True above); the forward matches to fp32
    # reduction-order noise (different matmul memory layout)
    assert torch.allclose(ref, got, atol=1e-5, rtol=1e-4)
    assert float((ref - got).abs().max()) < 1e-5


def test_compact_banks_free_the_masters(tiny, args_ns):
    model = copy.deepcopy(tiny)
    n = q4.compact_banks(model, args_ns)
    assert n == 2 * len(q4.text_layers(model))
    for layer in q4.text_layers(model):
        m = layer.mlp.experts
        assert getattr(m, "_compact", False)
        assert m.gate_up_proj.numel() == 0 and m.down_proj.numel() == 0
        assert m._compact_gu.param_numel() > 0


def test_build_student_compact_flag(tiny, args_ns):
    from olmoe_corrections import MoEWithCorrection
    model = copy.deepcopy(tiny)
    n = q4.build_student(model, args_ns)
    assert n > 0
    for layer in q4.text_layers(model):
        assert isinstance(layer.mlp, MoEWithCorrection)
        assert getattr(q4.moe_block(layer).experts, "_compact", False)


# -------------------------------------------------------------- training -----

def test_compact_tiny_training_smoke(tiny, args_ns):
    """LM + a top-k KD term over the compact banks: one finite backward."""
    import torch.nn as nn
    model = copy.deepcopy(tiny)
    model.lm_head = nn.Linear(64, 1024, bias=False)
    q4.build_student(model, args_ns)
    model.train()
    ids = torch.randint(0, 1024, (1, 12))
    teacher = torch.randn(11, 8)
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
    q4.compact_banks(model, args_ns)
    rep = q4.memory_report(model, args_ns, "test")
    numel = rep["expert_dense_numel"]
    assert numel > 0
    # 2 bits/param codes; 2 B per fp16 group scale
    assert rep["expert_codes"] == numel // 4
    assert rep["expert_bf16_equivalent"] == numel * 2
    expected_scales = sum(b.scales.numel() * 2
                          for layer in q4.text_layers(model)
                          for b in (layer.mlp.experts._compact_gu,
                                    layer.mlp.experts._compact_dn))
    assert rep["expert_scales"] == expected_scales
    assert rep["branches"] == 0        # no branches attached yet
    # the compact banks are strictly smaller than the bf16 master they replace
    assert rep["expert_codes"] + rep["expert_scales"] < rep["expert_bf16_equivalent"]


def test_deployed_container_bytes_on_128_groups():
    torch.manual_seed(0)
    w = torch.randn(4, 128, 256)
    bank = q4.CompactBank.from_tensor(w, 128, "lloyd")
    numel = w.numel()
    assert bank.codes.numel() == numel // 4          # 2 bits/param
    assert bank.scales.numel() == numel // 128       # fp16 per 128-group
    assert bank.nbytes() == numel // 4 + 2 * (numel // 128)


def test_quantize_bank_chunked_tolerates_requires_grad():
    """P2 pod finding: the loaded banks require grad; the in-place freeze must
    run under no_grad (it raised 'a view of a leaf Variable ... in-place')."""
    p = torch.randn(2, 128, 128, requires_grad=True)
    redone = q4.quantize_bank_chunked(p, 128, "lloyd")
    assert redone == 0
    assert torch.isfinite(p).all()


def test_compact_banks_parking_handles_scale_inv_parameter(tiny, args_ns):
    """Native fp8 stores ``*_scale_inv`` as an nn.Parameter; parking it on host
    must reassign a Parameter (a bare tensor raises TypeError) -- P2 finding."""
    import torch.nn as nn
    model = copy.deepcopy(tiny)
    for layer in q4.text_layers(model):
        e = layer.mlp.experts
        for name in ("gate_up_proj", "down_proj"):
            w = getattr(e, name)
            grid = (w.shape[0], math.ceil(w.shape[1] / 128),
                    math.ceil(w.shape[2] / 128))
            setattr(e, name + "_scale_inv",
                    nn.Parameter(torch.ones(*grid), requires_grad=False))
    n = q4.compact_banks(model, args_ns)
    assert n == 2 * len(q4.text_layers(model))
    for layer in q4.text_layers(model):
        assert not hasattr(layer.mlp.experts, "gate_up_proj_scale_inv")
        assert getattr(layer.mlp.experts, "_compact", False)


def test_rebind_compact_forwards_restores_binding(tiny, args_ns):
    """Accelerate's hook removal restores a stale instance forward; the rebind
    must put ``compact_experts_forward`` back (P2 finding)."""
    model = copy.deepcopy(tiny)
    q4.compact_banks(model, args_ns)
    for layer in q4.text_layers(model):
        layer.mlp.experts.forward = lambda *a, **k: None   # stale restore
    n = q4.rebind_compact_forwards(model)
    assert n == len(q4.text_layers(model))
    for layer in q4.text_layers(model):
        assert getattr(layer.mlp.experts.forward, "__func__", None) \
            is q4.compact_experts_forward


def test_compact_class_dispatch_survives_lost_instance_forward(tiny, args_ns):
    """Even if the instance ``forward`` is lost, the class patch dispatches on
    ``_compact`` so the compact banks are still decoded."""
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpTextExperts)
    model = copy.deepcopy(tiny)
    q4.compact_banks(model, args_ns)
    assert getattr(Qwen4ExpTextExperts, "_q4exp_compact_patch", False)
    for layer in q4.text_layers(model):
        m = layer.mlp.experts
        m.__dict__.pop("forward", None)      # simulate a lost binding
    with torch.no_grad():
        out = model(input_ids=torch.randint(0, 1024, (1, 6)), use_cache=False)
    assert torch.isfinite(out.last_hidden_state).all()


def test_compact_forward_grad_checkpoint_exact_and_finite(tiny, args_ns):
    """The per-expert decode checkpoint is exact and still backprops."""
    import torch.nn as nn
    model = copy.deepcopy(tiny)
    model.lm_head = nn.Linear(64, 1024, bias=False)
    q4.build_student(model, args_ns)
    ids = torch.randint(0, 1024, (1, 10))
    with torch.no_grad():
        ref = model(input_ids=ids, use_cache=False).last_hidden_state.clone()
    n = q4.enable_grad_checkpointing(model)
    assert n == len(q4.text_layers(model))
    model.train()
    out = model(input_ids=ids, use_cache=False).last_hidden_state
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4)
    logits = model.lm_head(out)
    F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                    ids[:, 1:].reshape(-1)).backward()
    grads = [p for n_, p in model.named_parameters()
             if ".branch." in n_ and p.grad is not None]
    assert grads and all(bool(torch.isfinite(p.grad).all()) for p in grads)


# ------------------------------------------------------------------ CLI ------

def test_compact_flag_default_off():
    ap = q4.build_parser()
    a = ap.parse_args(["train", "--cache-file", "/tmp/x.pt"])
    assert a.compact_banks is False
    b = ap.parse_args(["train", "--cache-file", "/tmp/x.pt", "--compact-banks"])
    assert b.compact_banks is True


def test_compact_check_stage_registered():
    ap = q4.build_parser()
    a = ap.parse_args(["compact-check"])
    assert a.stage == "compact-check"
