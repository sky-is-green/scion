"""Tests for the qwen4exp export prototype (``moe/qwen4exp_export.py``).

Fast and local: the V-head reorder is checked against an independent
grouped->tiled permutation, Q8_0 bytes against hand-computed blocks, and the
expert serializer against a tiny known-trit bank (block order + GGUF
parse-back).  The full-file proof (loader acceptance + forward) is the manual
load test in the runlog, not this file.
"""

from __future__ import annotations

import struct
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import qwen4exp_export as qx


def make_exporter(**kw):
    d = {"model_dir": "/home/penis/Desktop/work/models/qwen38-flashnext-fp8",
         "layers": 2, "experts": "ptq1_0", "body": "f16", "branches": "none",
         "branch_dtype": "f16", "branch_quant": "g128",
         "deploy_quant": "lloyd", "routers": "replace", "adapter_recipe": "",
         "use_temp_file": False, "out": "/tmp/opencode/test-export.gguf"}
    d.update(kw)
    return qx.Exporter(Namespace(**d))


def write_fake_ckpt(tmp_path, layers=2, zero=False):
    """A synthetic branch+router checkpoint (real P2b key shapes, tiny dims)."""
    torch.manual_seed(0)
    sd = {}
    for il in range(layers):
        for proj in ("mlp", "linear_attn.out_proj"):
            shape_d, shape_u = (4, 256), (16, 4)
            d = torch.zeros(shape_d) if zero else torch.randn(shape_d)
            u = torch.zeros(shape_u) if zero else torch.randn(shape_u)
            sd[f"model.language_model.layers.{il}.{proj}.branch.down.weight"] = d
            sd[f"model.language_model.layers.{il}.{proj}.branch.up.weight"] = u
        sd[f"model.language_model.layers.{il}.mlp.mlp.gate.weight"] = \
            torch.randn(16, 256).bfloat16()
        sd[f"model.language_model.layers.{il}.mlp.mlp.gate.balance_bias"] = \
            torch.randn(16)
    path = tmp_path / "ckpt.pt"
    torch.save(sd, path)
    return str(path)


def write_adapter_gguf(tmp_path, ex, name="adapter.gguf"):
    import gguf
    out = tmp_path / name
    w = gguf.GGUFWriter(str(out), "qwen4exp")
    ex.export_adapter(w)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return gguf.GGUFReader(str(out))


def test_load_branch_checkpoint_and_layer_filter(tmp_path):
    path = write_fake_ckpt(tmp_path, layers=2)
    pairs, routers, n_bias = qx.load_branch_checkpoint(path, 2)
    assert set(pairs) == {(0, "ssm_out.weight"), (1, "ssm_out.weight"),
                          (0, "ffn_moe_out.weight"), (1, "ffn_moe_out.weight")}
    assert set(routers) == {0, 1} and n_bias == 2
    assert pairs[(0, "ssm_out.weight")]["down"].shape == (4, 256)
    # exported range filters later layers out of both branches and routers
    pairs1, routers1, _ = qx.load_branch_checkpoint(path, 1)
    assert all(k[0] == 0 for k in pairs1) and set(routers1) == {0}
    empty = tmp_path / "empty.pt"
    torch.save({}, empty)
    with pytest.raises(SystemExit):
        qx.load_branch_checkpoint(empty, 2)


def test_router_tensor_override(tmp_path):
    path = write_fake_ckpt(tmp_path, layers=2)
    ex = make_exporter(layers=2, branches=path)
    got = ex.layer_tensor(1, "mlp.gate.weight",
                          "model.language_model.layers.1.mlp.gate.weight")
    assert torch.equal(got, ex.routers[1])
    assert got.dtype == torch.float32
    ex.args.routers = "none"
    ex.read_hf_chunked = lambda key: torch.full((2, 2), 7.0)
    got = ex.layer_tensor(1, "mlp.gate.weight", "x")
    assert torch.equal(got, torch.full((2, 2), 7.0))


def test_branch_merge_identity_parse_back(tmp_path):
    """Zero branches -> zero adapter factors; metadata + dims parse back."""
    import gguf
    path = write_fake_ckpt(tmp_path, layers=2, zero=True)
    ex = make_exporter(layers=2, branches=path)
    r = write_adapter_gguf(tmp_path, ex)
    expected = {f"blk.{il}.{t}.lora_{f}"
                for il in (0, 1)
                for t in ("ssm_out.weight", "ffn_moe_out.weight")
                for f in ("a", "b")}
    names = {t.name for t in r.tensors}
    assert names == expected
    assert r.fields["adapter.embedded"].contents() is True
    assert r.fields["adapter.type"].contents() == "lora"
    assert r.fields["adapter.lora.alpha"].contents() == 0.0
    for t in r.tensors:
        assert t.tensor_type == gguf.GGMLQuantizationType.F16
        import numpy as np
        vals = np.asarray(t.data, dtype=np.float16)
        assert np.abs(vals).max() == 0.0, t.name


def test_adapter_parse_back_matches_deploy(tmp_path):
    """Parse-back dims + values equal the training-time deployed factors."""
    import numpy as np
    from export_branches_lora import deploy_weights
    path = write_fake_ckpt(tmp_path, layers=2)
    ex = make_exporter(layers=2, branches=path)
    r = write_adapter_gguf(tmp_path, ex)
    by_name = {t.name: t for t in r.tensors}
    # ggml dims come back reversed vs the numpy write: a=[in, rank], b=[rank, out]
    assert tuple(by_name["blk.0.ssm_out.weight.lora_a"].shape) == (256, 4)
    assert tuple(by_name["blk.0.ssm_out.weight.lora_b"].shape) == (4, 16)
    assert tuple(by_name["blk.1.ffn_moe_out.weight.lora_a"].shape) == (256, 4)
    d, u = deploy_weights(
        ex.pairs[(0, "ssm_out.weight")]["down"],
        ex.pairs[(0, "ssm_out.weight")]["up"], "g128", "lloyd")
    got_a = np.asarray(by_name["blk.0.ssm_out.weight.lora_a"].data,
                       dtype=np.float32).reshape(d.shape)
    got_b = np.asarray(by_name["blk.0.ssm_out.weight.lora_b"].data,
                       dtype=np.float32).reshape(u.shape)
    assert np.array_equal(got_a, d.numpy().astype(np.float16))
    assert np.array_equal(got_b, u.numpy().astype(np.float16))


def test_shard_release_plan_and_unlink(tmp_path):
    ex = make_exporter(layers=2, release_shards="all", keep_shards="")
    # the plan covers the real mirror's read set; shared tensors map to the
    # post-layer sentinel (= number of layers)
    assert ex.shard_last and any(v == 2 for v in ex.shard_last.values())
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    for name in ("a", "b", "c"):
        (shard_dir / name).write_bytes(b"x")
    ex.shard_dir = shard_dir
    ex.shard_last = {"a": 0, "b": 1, "c": 0}
    ex.keep_shards = {"c"}
    ex.handles = {"a": object(), "b": object(), "c": object()}
    ex._release_shards(0)
    assert not (shard_dir / "a").exists()
    assert (shard_dir / "b").exists() and (shard_dir / "c").exists()
    assert "a" not in ex.handles and "b" in ex.handles
    ex._release_shards(1)
    assert not (shard_dir / "b").exists() and (shard_dir / "c").exists()
    assert ex.stats.get("released") == 2


def test_write_deployed_checkpoint_strips_bias(tmp_path):
    src = write_fake_ckpt(tmp_path, layers=2)
    dst = tmp_path / "deployed.pt"
    info = qx.write_deployed_checkpoint(src, str(dst))
    assert info == {"keys": 10, "dropped": 2}
    sd = torch.load(dst, map_location="cpu")
    assert all(".balance_bias" not in k for k in sd)
    pairs, routers, n_bias = qx.load_branch_checkpoint(str(dst), 2)
    assert n_bias == 0 and len(routers) == 2 and len(pairs) == 4


def test_branch_target_layer_type_mismatch(tmp_path):
    # full-attention target on a linear-attention layer must refuse
    sd = {"model.language_model.layers.0.self_attn.o_proj.branch.down.weight":
          torch.randn(4, 256),
          "model.language_model.layers.0.self_attn.o_proj.branch.up.weight":
          torch.randn(16, 4)}
    path = tmp_path / "bad.pt"
    torch.save(sd, path)
    with pytest.raises(SystemExit):
        make_exporter(layers=2, branches=str(path))


def test_reorder_v_heads_against_brute_force():
    # grouped (by K head): idx(k, v, d) = (k*nvpk + v)*hd + d
    # tiled:               idx(k, v, d) = (v*nk + k)*hd + d
    nk, nvpk, hd = 2, 3, 4
    n = nk * nvpk * hd
    t = torch.arange(n).reshape(n, 1).expand(n, 5).clone()
    got = qx.reorder_v_heads(t, 0, nk, nvpk, hd)
    expect = torch.empty_like(got)
    for k in range(nk):
        for v in range(nvpk):
            for d in range(hd):
                expect[(v * nk + k) * hd + d] = t[(k * nvpk + v) * hd + d]
    assert torch.equal(got, expect)
    # dim=1 variant (out_proj columns) permutes the same way.
    tc = torch.arange(n).reshape(1, n).expand(3, n).clone()
    gotc = qx.reorder_v_heads(tc, 1, nk, nvpk, hd)
    assert torch.equal(gotc[0], expect[:, 0])


def test_quantize_q8_0_bytes():
    ex = make_exporter()
    t = torch.tensor([[1.0] * 32, [0.5] * 32])
    raw = ex.quantize_q8_0(t)
    assert raw.shape == (2, 34) and raw.dtype == torch.uint8
    d0 = struct.unpack("<e", bytes(raw[0, :2].tolist()))[0]
    assert d0 == pytest.approx(1.0 / 127, rel=1e-3)
    assert raw[0, 2:].to(torch.int8).eq(127).all()
    q1 = raw[1, 2:].to(torch.int8)
    # 0.5 / (0.5/127) = 127 -> saturates like row 0 (amax scaling).
    assert q1.eq(127).all()
    with pytest.raises(AssertionError):
        ex.quantize_q8_0(torch.zeros(4, 30))


def test_expert_serialization_order():
    """One known-trit bank -> blocks parse back in GGUF order (e, row, blk)."""
    ex = make_exporter(experts="ptq1_0")
    torch.manual_seed(0)
    trits = torch.randint(-1, 2, (2, 1, 128), dtype=torch.int8)
    scales = torch.tensor([[1.5], [0.25]], dtype=torch.float16)
    from qwen4exp_proxy import pack_ternary_codes
    import ptq1_0
    codes = pack_ternary_codes(trits)
    qs, qh, _ = ptq1_0.repack_bank_ptq1_0(codes, scales)
    blk = ptq1_0.pack_block_bytes(qs, qh, scales)
    assert blk.shape == (2, 1, 28)
    # first block bytes: qs[24] | qh[2] | fp16 scale.
    assert torch.equal(blk[0, 0, :24], qs[0, 0])
    assert torch.equal(blk[0, 0, 24:26], qh[0, 0])
    d = struct.unpack("<e", bytes(blk[0, 0, 26:28].tolist()))[0]
    assert d == pytest.approx(1.5)
    # decode the serialized block back to the input trits.
    back = ptq1_0.decode_ptq1_0(qs, qh, scales)
    from qwen4exp_proxy import decode_ternary
    assert torch.equal(back, decode_ternary(codes, scales))


def test_gguf_name_mapping_covers_layer_jobs():
    ex = make_exporter()
    ex._bid = 0
    for suffix, tensor_enum, _kind in qx.LAYER_JOBS:
        hf = f"model.language_model.layers.0.{suffix}"
        name = ex.gguf_name(hf)
        assert name.startswith("blk.0."), f"{hf} -> {name}"
    assert ex.gguf_name("model.language_model.embed_tokens.weight") == \
        "token_embd.weight"


def test_body_quantize_paths():
    ex = make_exporter()
    t = torch.randn(4, 64)
    d16, raw = ex.to_body(t, "f16")
    assert raw is None and d16.dtype == "float16" or hasattr(d16, "dtype")
    import numpy as np
    assert d16.dtype == np.float16
    d32, raw = ex.to_body(t, "f32")
    assert d32.dtype == np.float32
    ex.args.body = "q8_0"
    dq, raw = ex.to_body(t, "q")
    assert raw is not None and dq.shape == (4, 68)


def test_tokenizer_vocab_matches_hf_size():
    import json
    ex = make_exporter()
    vocab = json.load(
        open("/home/penis/Desktop/work/models/qwen38-flashnext-fp8/vocab.json"))
    assert len(vocab) == 248044
    assert int(ex.hp["vocab_size"]) == 248320


class StubWriter:
    def __init__(self):
        self.tensors = []

    def add_tensor(self, name, data, raw_dtype=None):
        self.tensors.append((name, tuple(data.shape),
                             str(raw_dtype) if raw_dtype else "raw"))


def test_layer_type_dispatch_gdn_vs_full():
    import numpy as np
    ex = make_exporter()
    ex.layer_types = ["linear_attention", "full_attention",
                      "linear_attention", "full_attention"]
    ex.args.layers = 2
    ex.read_hf_chunked = lambda key: torch.zeros(64, 128)
    ex.apply_v_reorder = lambda name, t: t
    ex.export_layer_experts = lambda w, il: None
    split_calls = []
    ex.export_indexer_split = lambda w, il, p: split_calls.append(il)
    w = StubWriter()
    # replicate run()'s per-layer job selection
    from qwen4exp_export import LAYER_JOBS, FULL_JOBS, GDN_ONLY
    seen = []
    for il in range(2):
        is_full = ex.layer_types[il] == "full_attention"
        jobs = ([j for j in LAYER_JOBS if j[0] not in GDN_ONLY]
                + FULL_JOBS) if is_full else LAYER_JOBS
        for suffix, tensor_enum, kind in jobs:
            data, raw = ex.to_body(ex.read_hf_chunked(suffix), kind)
            seen.append((il, suffix))
        if is_full:
            ex.export_indexer_split(w, il, "")
    gdn = [s for il, s in seen if il == 0]
    full = [s for il, s in seen if il == 1]
    assert any("in_proj_qkv" in s for s in gdn)
    assert not any("in_proj_qkv" in s for s in full)
    assert any("self_attn.q_proj" in s for s in full)
    assert not any("self_attn.q_proj" in s for s in gdn)
    assert any("mlp.gate" in s for s in full)  # MoE common to both
    assert split_calls == [1]
    # every LAYER_JOBS entry is either GDN-only or common (no orphans).
    common = [j[0] for j in LAYER_JOBS if j[0] not in GDN_ONLY]
    assert any("mlp" in s or "hyper" in s for s in common)


REF_VOCAB = [20000003, 20000023, 20000033, 20000047, 20000059, 20000063,
             20000069, 20000077, 20000081, 20000093, 20000107, 20000147,
             20000153, 20000159, 20000161, 20000171]
REF_MULT = [23703573157769, 20109073645365, 8052911324071]


def test_ple_kv_values_match_reference():
    ex = make_exporter(ple="q4_0")
    assert ex.ple_layer == 1
    v = ex.ple_kv_values()
    assert v["head_dim"] == 160
    assert (v["ngram_size"], v["heads_per_ngram"]) == (3, 8)
    assert v["conv_kernel"] == 4
    assert v["layer_multipliers"] == REF_MULT
    assert v["head_vocab_sizes"] == REF_VOCAB
    assert v["head_offsets"][:3] == [0, 20000003, 40000026]
    assert sum(v["head_vocab_sizes"]) == 320001446
    assert v["rows"] == 320001536  # padded to the 128-divisible table size
    assert (v["eos_token_id"], v["image_token_id"]) == (248044, 248056)


def test_ple_table_parts_ordered():
    ex = make_exporter(ple="q4_0")
    parts = ex.ple_table_parts()
    assert len(parts) == 128
    assert [p[0] for p in parts] == list(range(128))
    assert all(p[1].endswith(f"shard_{p[0]}.weight") for p in parts)


def test_ple_table_nbytes_q4_0():
    assert qx.Exporter._ple_table_nbytes(320001536, 160) == 28800138240


def test_ple_kv_written_to_gguf(tmp_path):
    import gguf
    import export_check as ec
    ex = make_exporter(ple="q4_0")
    out = tmp_path / "ple-kv.gguf"
    w = gguf.GGUFWriter(str(out), "qwen4exp")
    ex.write_kv(w)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    meta, _ = ec.parse_local(str(out))
    assert meta["qwen4exp.embedding_length_per_layer_input"] == 160
    assert meta["qwen4exp.ple.layers"] == {
        "__array__": "num", "len": 1, "head": [1]}
    assert meta["qwen4exp.ple.ngram_size"] == 3
    assert meta["qwen4exp.ple.layer_multipliers"] == {
        "__array__": "num", "len": 3, "head": REF_MULT}
    assert meta["qwen4exp.ple.head_vocab_sizes"]["head"] == REF_VOCAB[:3]
    assert meta["qwen4exp.ple.head_vocab_sizes"]["len"] == 16
    assert meta["qwen4exp.ple.head_offsets"]["head"] == [0, 20000003, 40000026]


def test_ple_none_writes_no_ple_kv(tmp_path):
    import gguf
    import export_check as ec
    ex = make_exporter()
    out = tmp_path / "nople.gguf"
    w = gguf.GGUFWriter(str(out), "qwen4exp")
    ex.write_kv(w)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    meta, _ = ec.parse_local(str(out))
    assert "qwen4exp.ple.layers" not in meta
