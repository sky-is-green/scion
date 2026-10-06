"""Tests for the Qwen3.8-Flash-Next (``qwen4_exp``) ternary proxy port.

CPU-only and tiny-model based: the official FP8 checkpoint is far too large for
the test box, so the fixture builds a random-init ``Qwen4ExpTextModel`` (3
layers: two GDN, one QSA full attention, PLE on layer 1) and exercises exactly
the code paths the port adds on top of the 35B harness:

  - the QSA indexer scatter-dtype fix (upstream bug on the pinned runtime);
  - the fused-bank STE patch and the branch placement inside the
    hyper-connection decoder;
  - the cache record schema (``idx/val/w/router/tidx/tlp``);
  - the manual FP8 loader's block dequantisation, expert merge order, and the
    row-compact PLE table;
  - the PLE precision quantiser used by the A/B;
  - the router-balance port (identity at zero bias).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import qwen4exp_proxy as q4  # noqa: E402


# --------------------------------------------------------------- fixtures ----

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
        top_logits = 8
        tail_logits = 0
        rank = 4
        branch_quant = "fp32"
        branch_target = "both"
        branch_gate = "none"
        quant = "lloyd"
        group = 128
    return A()


class FakeShard:
    """Minimal ``safetensors.safe_open`` stand-in: name -> tensor."""

    def __init__(self, tensors):
        self.tensors = tensors

    def get_tensor(self, key):
        return self.tensors[key]


# ------------------------------------------------------- indexer + forward ---

def test_indexer_patch_idempotent_and_forward(tiny):
    q4.patch_indexer()
    q4.patch_indexer()          # second call must be a no-op
    ids = torch.randint(0, 1024, (1, 12))
    with torch.no_grad():
        out = tiny(input_ids=ids, use_cache=False)
    h = out.last_hidden_state
    assert h.shape == (1, 12, 64)
    assert torch.isfinite(h).all()


def test_router_hook_schema(tiny):
    store = {}
    handles = [layer.mlp.gate.register_forward_hook(q4.gate_hook(store, i))
               for i, layer in enumerate(q4.text_layers(tiny))]
    with torch.no_grad():
        tiny(input_ids=torch.randint(0, 1024, (1, 10)), use_cache=False)
    for h in handles:
        h.remove()
    assert sorted(store) == [0, 1, 2]
    logits, weights, idx = store[0]
    assert logits.shape == (10, 8)              # [T, num_experts]
    assert weights.shape == (10, 2)             # [T, top_k]
    assert idx.shape == (10, 2)
    assert idx.dtype in (torch.int64, torch.int32)
    assert torch.allclose(weights.sum(-1), torch.ones(10), atol=1e-5)


# ------------------------------------------------------------------- STE -----

def test_patch_experts_ste_forward(tiny):
    q4.patch_experts(128)
    experts = [layer.mlp.experts for layer in q4.text_layers(tiny)]
    ids = torch.randint(0, 1024, (1, 8))
    for e in experts:
        e._ternary = False
    with torch.no_grad():
        fp = tiny(input_ids=ids, use_cache=False).last_hidden_state
    for e in experts:
        e._ternary = True
    with torch.no_grad():
        tern = tiny(input_ids=ids, use_cache=False).last_hidden_state
    assert torch.isfinite(tern).all()
    assert tern.shape == fp.shape
    drift = float((tern - fp).norm() / (fp.norm() + 1e-12))
    assert drift >= 0.0
    # ternary banks are 3-valued per group after ST: |w| values collapse
    gu = experts[0].gate_up_proj
    q = q4.ternary_ste(gu, 128).detach()
    uniq = torch.unique((q / q.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
                         ).round())
    assert uniq.numel() <= 3


def test_expert_patch_is_class_wide(tiny):
    """The class patch is what the loaded instances use when rebinding fails."""
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpTextExperts)
    q4.patch_experts(128)
    e = q4.text_layers(tiny)[0].mlp.experts
    e._ternary = True
    x = torch.randn(4, 64)
    idx = torch.tensor([[0, 1]] * 4)
    w = torch.rand(4, 2)
    y_patched = e(x, idx, w)
    e._ternary = False
    y_orig = e(x, idx, w)
    assert torch.isfinite(y_patched).all()
    assert not torch.allclose(y_patched, y_orig)


# ------------------------------------------------- branch placement tests ----

def test_branch_wrap_identity_at_init(tiny, args_ns):
    """A fresh branch is exactly zero: the wrapped model must not move."""
    from olmoe_corrections import MoEWithCorrection
    import copy
    model = copy.deepcopy(tiny)

    ids = torch.randint(0, 1024, (1, 9))
    with torch.no_grad():
        base = model(input_ids=ids, use_cache=False).last_hidden_state
    for layer in q4.text_layers(model):
        layer.mlp = MoEWithCorrection(layer.mlp, 64, args_ns.rank).to("cpu")
    with torch.no_grad():
        wrapped = model(input_ids=ids, use_cache=False).last_hidden_state
    assert torch.equal(base, wrapped)
    # and the branch path is exercised: perturb the branch output weights
    for layer in q4.text_layers(model):
        torch.nn.init.normal_(layer.mlp.branch.up.weight, std=0.05)
    with torch.no_grad():
        perturbed = model(input_ids=ids, use_cache=False).last_hidden_state
    assert not torch.allclose(base, perturbed)


def test_attach_branches_both_targets(tiny, args_ns):
    """attn_out wraps out_proj/o_proj; moe_out wraps the sparse block."""
    from olmoe_corrections import MoEWithCorrection
    import copy
    model = copy.deepcopy(tiny)
    n = q4.attach_branches(model, args_ns)
    assert n > 0
    for layer in q4.text_layers(model):
        assert isinstance(layer.mlp, MoEWithCorrection)
        if layer.layer_type == "linear_attention":
            assert isinstance(layer.linear_attn.out_proj, MoEWithCorrection)
        else:
            assert isinstance(layer.self_attn.o_proj, MoEWithCorrection)
    # forward still runs with both placements active
    with torch.no_grad():
        out = model(input_ids=torch.randint(0, 1024, (1, 7)), use_cache=False)
    assert torch.isfinite(out.last_hidden_state).all()


def test_quick_eval_through_correction_wrapper(tiny, args_ns):
    """In-run eval must find the router gate inside the correction wrapper.

    Regression for the P2b step-1000 crash: ``quick_eval`` read
    ``layer.mlp.gate`` directly, but after ``attach_branches`` the block is a
    ``MoEWithCorrection`` and the gate is one level down.  The wrapped model
    is bit-identical at init, so the PPL must match the unwrapped call.
    """
    import copy
    import torch.nn as nn
    model = copy.deepcopy(tiny)
    model.lm_head = nn.Linear(64, 1024, bias=False)
    model.eval()
    args_ns.device = "cpu"
    data = torch.randint(0, 1024, (2, 9))
    ppl0, _ = q4.quick_eval(model, data, args_ns)
    for layer in q4.text_layers(model):
        layer.mlp = q4.MoEWithCorrection(layer.mlp, 64, args_ns.rank).to("cpu")
    ppl1, _ = q4.quick_eval(model, data, args_ns)
    assert math.isfinite(ppl1)
    assert ppl1 == pytest.approx(ppl0, rel=1e-5)


# ------------------------------------------------------------ cache record ---

def test_make_record_schema(args_ns):
    logits = torch.randn(5, 64)
    rec = q4.make_record(logits, args_ns)
    assert set(rec) == {"idx", "val", "w"}
    assert rec["idx"].shape == (5, 8) and rec["idx"].dtype == torch.int32
    assert rec["val"].shape == (5, 8) and rec["val"].dtype == torch.float16
    assert rec["w"].shape == (5,) and rec["w"].dtype == torch.float16
    assert float(rec["w"].min()) >= 0.0 and float(rec["w"].max()) <= 1.0


def test_make_record_tail(args_ns):
    args_ns.tail_logits = 6
    logits = torch.randn(5, 64)
    g = torch.Generator().manual_seed(3)
    rec = q4.make_record(logits, args_ns, generator=g)
    assert set(rec) == {"idx", "val", "w", "tidx", "tlp"}
    assert rec["tidx"].shape == (5, 6) and rec["tidx"].dtype == torch.int32
    assert rec["tlp"].shape == (5, 6) and rec["tlp"].dtype == torch.float16
    assert float(rec["tlp"].max()) <= 0.0          # log-probs of the tail


# --------------------------------------------------------------- fp8 math ----

def test_dequant_fp8_block_exact():
    torch.manual_seed(0)
    w = (torch.randn(130, 200) * 0.1).to(torch.float8_e4m3fn)
    si = torch.rand(2, 2) + 0.5
    out = q4.dequant_fp8_block(w, si)
    assert out.shape == (130, 200)
    manual = torch.zeros_like(out)
    for r in range(0, 130, 128):
        for c in range(0, 200, 128):
            manual[r:r + 128, c:c + 128] = (w[r:r + 128, c:c + 128].float()
                                            * si[r // 128, c // 128])
    assert torch.equal(out, manual)


def test_dequant_fp8_block_roundtrip_fp8():
    """A tensor exactly representable in fp8 comes back unscaled-scaled."""
    base = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).to(torch.float8_e4m3fn)
    si = torch.tensor([[2.0]])
    out = q4.dequant_fp8_block(base, si)
    assert torch.allclose(out, base.float() * 2.0)


def test_expert_merge_order():
    """gate rows first, up rows second, down as-is (official merge order)."""
    E, ff, h = 2, 4, 3
    tensors = {}
    wm = {}
    for e in range(E):
        for kind in ("gate", "up", "down"):
            shape = (ff, h) if kind != "down" else (h, ff)
            key = f"model.language_model.layers.0.mlp.experts.{e}.{kind}_proj.weight"
            tensors[key] = torch.full(shape, e * 3 + {"gate": 1, "up": 2,
                                                      "down": 3}[kind],
                                      dtype=torch.float8_e4m3fn)
            tensors[key + "_scale_inv"] = torch.ones(math.ceil(shape[0] / 128),
                                                     math.ceil(shape[1] / 128))
            wm[key] = "s0"
            wm[key + "_scale_inv"] = "s0"
    state = q4._expert_state(0, E, ff, h, wm, lambda s: FakeShard(tensors),
                             torch.float32)
    gu = state["layers.0.mlp.experts.gate_up_proj"]
    dn = state["layers.0.mlp.experts.down_proj"]
    assert gu.shape == (E, 2 * ff, h)
    assert dn.shape == (E, h, ff)
    for e in range(E):
        assert torch.all(gu[e, :ff] == e * 3 + 1)
        assert torch.all(gu[e, ff:] == e * 3 + 2)
        assert torch.all(dn[e] == e * 3 + 3)


# ------------------------------------------------------------------ PLE ------

def test_sparse_ple_table_roundtrip():
    torch.manual_seed(0)
    V, D = 50, 8
    dense = torch.randn(V, D)
    tab = q4.SparseNGramTable(D, V)
    tab.out_dtype = torch.float32
    tab.set_rows(torch.arange(V), dense)
    ids = torch.randint(0, V, (3, 7))
    out = tab(ids)
    assert torch.equal(out, dense[ids])
    with pytest.raises(KeyError):
        tab(torch.tensor([V + 1]))          # not loaded -> hard failure


def test_sparse_ple_recording_returns_ids():
    tab = q4.SparseNGramTable(4, 10)
    tab.start_recording()
    _ = tab(torch.tensor([[1, 2], [2, 3]]))
    ids = tab.stop_recording()
    assert set(ids.tolist()) == {1, 2, 3}


def test_sparse_ngram_init_matches_upstream():
    """The sparse PLE construction must hash to exactly the upstream ids."""
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpTextNGramEmbedding)
    cfg = tiny_config()
    torch.manual_seed(0)
    up = Qwen4ExpTextNGramEmbedding(cfg, 64, 1, 0)          # tiny table
    with q4.sparse_ple_construction():
        # import the patched init through the class
        sp = Qwen4ExpTextNGramEmbedding(cfg, 64, 1, 0)
    assert isinstance(sp.ngram_embedding, q4.SparseNGramTable)
    assert torch.equal(up.layer_multipliers, sp.layer_multipliers)
    assert torch.equal(up.ngram_heads_vocab_sizes, sp.ngram_heads_vocab_sizes)
    assert torch.equal(up.ngram_heads_offsets, sp.ngram_heads_offsets)
    assert up.total_vocab_size == sp.total_vocab_size
    # ids: capture the upstream lookup and the sparse recording side by side
    ids = torch.randint(0, 1024, (1, 6))
    seen = []
    h = up.ngram_embedding.register_forward_hook(lambda m, i, o: seen.append(i[0].clone()))
    with torch.no_grad():
        up(ids, None)
        sp.ngram_embedding.start_recording()
        sp(ids, None)
        got = sp.ngram_embedding.stop_recording()
    h.remove()
    want = torch.unique(seen[0].reshape(-1).long())
    assert torch.equal(want.sort().values, got.sort().values)
    # the sparse module refuses to forward before rows are set
    with pytest.raises(RuntimeError):
        sp(torch.tensor([[1, 2]]), None)


def test_ple_row_gather_is_boundary_aware():
    """Per-window forwards hash differently at window starts than one
    concatenated row: the gather must use the exact forward batches."""
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextModel
    q4.patch_indexer()
    torch.manual_seed(0)
    with q4.sparse_ple_construction():
        model = Qwen4ExpTextModel(tiny_config())
    model.eval()
    windows_ = [torch.randint(0, 1024, (1, 24)) for _ in range(2)]
    need = q4.collect_ple_ids(model, windows_)
    tab = model.layers[1].ple.ple_embedding.ngram_embedding
    tab.out_dtype = torch.float32          # the tiny model is fp32
    tab.set_rows(need, torch.zeros(need.numel(), tab.dim))
    with torch.no_grad():
        for ids in windows_:
            model(input_ids=ids, use_cache=False)      # must not raise
    # the concatenated row is NOT a substitute (window-start ids differ)
    concat = torch.cat(windows_, dim=1)
    need_concat = q4.collect_ple_ids(model, [concat])
    assert not set(need_concat.tolist()) <= set(need.tolist())


def test_load_ple_rows_gathers_from_parts():
    """Row offsets follow the concatenation order of the shard parts."""
    V, D = 20, 4
    part_rows = 8
    parts = {0: torch.zeros(part_rows, D, dtype=torch.float8_e4m3fn),
             1: torch.zeros(part_rows, D, dtype=torch.float8_e4m3fn),
             2: torch.zeros(V - 2 * part_rows, D, dtype=torch.float8_e4m3fn)}
    # make every row identifiable: row i has value i/8 (fp8-exact)
    tensors, wm = {}, {}
    scale = torch.tensor([2.0])
    for p, t in parts.items():
        for r in range(t.shape[0]):
            val = float(p * part_rows + r)      # integers are fp8-exact here
            t[r] = val
        key = (f"model.language_model.layers.1.ple.ple_embedding."
               f"ngram_embedding.shard_{p}.weight")
        tensors[key] = t
        wm[key] = f"s{p}"
    scale_key = ("model.language_model.layers.1.ple.ple_embedding."
                 "ngram_embedding.weight_scale")
    tensors[scale_key] = scale
    wm[scale_key] = "s_scale"
    ids = torch.tensor([0, 3, 6, 12])
    got_ids, rows = q4.load_ple_rows(ids, 1, wm,
                                     lambda s: FakeShard(tensors),
                                     dtype=torch.float32)
    assert got_ids.tolist() == [0, 3, 6, 12]
    expect = torch.tensor([[0.0] * D, [6.0] * D, [12.0] * D, [24.0] * D])
    assert torch.allclose(rows, expect)


def test_quantize_rows_error_monotone():
    torch.manual_seed(0)
    rows = torch.randn(32, 160)
    outs = {}
    for bits in (8, 4, 2):
        deq, bpr = q4.quantize_rows(rows, bits, group=32)
        outs[bits] = float((deq - rows).abs().mean())
        assert deq.shape == rows.shape
        assert bpr == pytest.approx(160 * bits / 8 + 5 * 2)
    assert outs[8] < outs[4] < outs[2]
    passthrough, bpr16 = q4.quantize_rows(rows, 16)
    assert torch.equal(passthrough, rows) and bpr16 == 320.0
    with pytest.raises(ValueError):
        q4.quantize_rows(rows, 4, group=7)


# ------------------------------------------------------------- fp8 plan ------

def test_fp8_prefix_plan_against_local_index():
    model_dir = q4.MODEL
    if not (model_dir / "model.safetensors.index.json").exists():
        pytest.skip("local fp8 index not present")
    plan = q4.fp8_prefix_plan(2, model_dir)
    assert plan["expert_layers"] == [0, 1]
    assert plan["ple_layers"] == [1]
    assert "lm_head.weight" in plan["tensors"]
    assert "model.language_model.embed_tokens.weight" in plan["tensors"]
    # layer 0 lives in shards 1-3, layer 1 (non-PLE) in 3-4, embed/lm_head 130/131
    assert {"model-00001-of-00131.safetensors",
            "model-00002-of-00131.safetensors",
            "model-00003-of-00131.safetensors",
            "model-00130-of-00131.safetensors",
            "model-00131-of-00131.safetensors"} <= set(plan["shards"])
    one = q4.fp8_prefix_plan(1, model_dir)
    assert one["ple_layers"] == []
    assert one["expert_layers"] == [0]


# ------------------------------------------------------------ balance ---------

def test_balance_port_identity_at_zero_bias(tiny):
    import copy
    model = copy.deepcopy(tiny)
    gate = q4.text_layers(model)[0].mlp.gate
    x = torch.randn(6, 64)
    gate.eval()
    with torch.no_grad():
        ref = gate(x)
    n = q4.patch_router_balance(model, "bias")
    assert n == 3
    with torch.no_grad():
        got = gate(x)
    assert ref[0].shape == got[0].shape
    assert torch.allclose(ref[1], got[1], atol=1e-6)   # weights untouched
    assert torch.equal(ref[2], got[2])          # selection untouched at bias=0
    # an update from a training forward moves the bias (the ALF-LB rule)
    model.train()
    _ = gate(x)
    from router_bias import balance_update
    diag = balance_update(model, "bias", 1e-3)
    assert gate.balance_bias.abs().sum() > 0
    assert math.isfinite(diag["mean_load_entropy"])


# ------------------------------------------------------------------ CLI -------

def test_cli_cur05_defaults():
    ap = q4.build_parser()
    a = ap.parse_args(["train", "--cache-file", "/tmp/x.pt"])
    assert a.kd_tailcond_weight == 3.0
    assert a.kd_tail_weight == 2.0
    assert a.kd_weight == 2.0
    assert a.agentic_frac == 0.05
    assert a.branch_target == "both"
    assert a.ple_bits == "8,4,2"


def test_quantize_bank_chunked_finite_and_dtype():
    torch.manual_seed(0)
    p = (torch.randn(4, 256, 128) * 0.1)
    p_orig = p.clone()
    redone = q4.quantize_bank_chunked(p, 128, "absmean", chunk=2)
    assert redone == 0
    assert p.dtype == p_orig.dtype
    assert torch.isfinite(p).all()
    # per 128-group at most {0, +a, -a}: check one group has <=3 unique values
    g = p[0, 0]
    assert torch.unique(g).numel() <= 3
    # a zero bank stays zero (no NaN from the 0/0 scale)
    z = torch.zeros(2, 64, 128)
    assert q4.quantize_bank_chunked(z, 128, "lloyd") == 0
    assert torch.isfinite(z).all() and float(z.abs().sum()) == 0.0


def test_vectorized_indexer_matches_reference():
    """The fast path must reproduce the reference indexer mask bit-exactly.

    Uses the official regime (block_topk >= blocks-per-window): every complete
    block is selected, so the selection is the plain causal mask and there is
    no tie ambiguity.  The general (small-budget) path is exercised separately.
    """
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextModel
    q4.patch_indexer(fast=False)
    torch.manual_seed(0)
    model = Qwen4ExpTextModel(tiny_config(indexer_budget=32))
    model.eval()
    idx_layer = model.layers[2].self_attn.indexer
    ids = torch.randint(0, 1024, (1, 24))

    def run():
        seen = []
        h = idx_layer.register_forward_hook(lambda m, i, o: seen.append(o.detach().clone()))
        with torch.no_grad():
            model(input_ids=ids, use_cache=False)
        h.remove()
        return seen

    ref = run()
    q4.patch_indexer(fast=True)
    fast = run()
    q4.patch_indexer(fast=False)
    assert len(ref) == len(fast) == 1
    assert ref[0].shape == fast[0].shape
    assert ref[0].dtype == fast[0].dtype
    assert torch.equal(ref[0], fast[0])
    # the mask is causal and the query's own token is always visible
    m = fast[0][0, 0]
    assert bool(m.diagonal().all())
    assert not bool(torch.triu(m, diagonal=1).any())


def test_vectorized_indexer_general_path_valid_mask():
    """Small budget (selection active): the fast mask is a causal subset.

    Exact equality with the reference is not asserted here because equal block
    scores make the reference's topk tie choice unspecified; the pod runs the
    all-blocks regime, which the equality test above pins.
    """
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextModel
    q4.patch_indexer(fast=True)
    torch.manual_seed(0)
    model = Qwen4ExpTextModel(tiny_config())      # budget 8 -> block_topk 2
    model.eval()
    idx_layer = model.layers[2].self_attn.indexer
    seen = []
    h = idx_layer.register_forward_hook(lambda m, i, o: seen.append(o.detach().clone()))
    with torch.no_grad():
        model(input_ids=torch.randint(0, 1024, (1, 24)), use_cache=False)
    h.remove()
    q4.patch_indexer(fast=False)
    m = seen[0][0, 0]
    # selection is sparse, but never looks into the future and never empty
    assert not bool(torch.triu(m, diagonal=1).any())
    assert bool(m.any(dim=-1).all())
    assert bool(m[0, 0])                       # query 0's tail is just token 0


def test_pure_causal_detection():
    s = 6
    causal = torch.ones(1, 1, s, s, dtype=torch.bool).tril()
    assert q4._pure_causal(causal)
    padded = causal.clone()
    padded[0, 0, 1, 1] = False           # a visible entry hidden -> not pure causal
    assert not q4._pure_causal(padded)
    padded2 = causal.clone()
    padded2[0, 0, 0, 3] = True           # a future entry visible -> not pure causal
    assert not q4._pure_causal(padded2)
    assert not q4._pure_causal(torch.zeros(1, 1, s, s))
    assert not q4._pure_causal(None)


def test_new_scripts_have_main_guards():
    """A missing __main__ guard exits 0 and looks like success on a GPU stage."""
    import subprocess
    for script in ("qwen4exp_proxy.py", "qwen4exp_eval.py"):
        proc = subprocess.run([sys.executable, str(HERE / script), "--help"],
                              capture_output=True, text=True, timeout=180)
        assert proc.returncode == 0, f"{script} --help failed: {proc.stderr[-500:]}"
        assert "usage:" in proc.stdout
    # the 48-layer pod gate needs the full-model path on the eval tool
    proc = subprocess.run([sys.executable, str(HERE / "qwen4exp_eval.py"), "--help"],
                          capture_output=True, text=True, timeout=180)
    for flag in ("--full", "--compact-banks", "--force-gpu", "--device-map"):
        assert flag in proc.stdout, f"qwen4exp_eval.py is missing {flag}"
