"""N-layer qwen4exp GGUF export (closing-work item 3).

Writes a loadable ``qwen4exp`` GGUF for the N-layer prefix (default 2) from
the official FP8 mirror: dequantized body + ternary experts (PQ2_0/PTQ1_0 via
:mod:`ptq1_0`) + tokenizer + arch KV.  PLE is optional: the default (``--ple
none``) omits the ``ple.*`` key group and n-gram table (the calibration-floor
prototype); ``--ple q4_0`` writes the 6 small PLE tensors, the PLE KV group
(multipliers/head ranges built exactly as the runtime class builds them) and
the flat ``per_layer_token_embd`` table, streamed Q4_0 (the table is ~28.8 GB
at 48 layers and never materialised in RAM).

``--branches PATH`` embeds the trained correction branches from a P2b
checkpoint as the fork's single-file adapter (``aa96b8c12``):
``.lora_a``/``.lora_b`` tensors + ``adapter.embedded=true``, deployed with the
training-time quantizer (g128 lloyd), exactly as ``export_branches_lora.py``
writes them.  The trained routers (``ffn_gate_inp``) are replaced numerically
in the body; ``balance_bias`` is dropped (training-time shaping only, per the
35B deployed re-gates).

Purpose: (1) measure the export write throughput (quantize + serialize) that
the cost sheet still lists as unmeasured, extrapolated to 48 layers; (2) prove
the merged fork loads our blocks and runs a forward.

Name mapping reuses the merged gguf-py's ``TensorNameMap`` (HF key minus the
``language_model.`` multimodal prefix, as the converter does); the linear-
attention V-head reorder replicates ``_LinearAttentionVReorderBase`` on the
dequantized float weights *before* quantization (no scale bookkeeping needed).

Usage:
    PYTHONPATH=.../shadow-tf-5.17 python moe/qwen4exp_export.py \\
        --model-dir .../models/qwen38-flashnext-fp8 --layers 2 \\
        --experts ptq1_0 --body f16 --out /tmp/qwen4exp-2l.gguf
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import os
import struct
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ptq1_0  # noqa: E402
from qwen4exp_proxy import (MODEL as DEFAULT_MODEL_DIR,  # noqa: E402
                            dequant_fp8_block, unpack_ternary_codes,
                            _encode_slice)
from scion_paths import QWEN4EXP_GGUF_PY  # noqa: E402

GGUF_PY = str(QWEN4EXP_GGUF_PY)
sys.path.insert(0, str(GGUF_PY))
import gguf  # noqa: E402
from gguf.tensor_mapping import TensorNameMap  # noqa: E402

ARCH = "qwen4exp"
HF_PREFIX = "model.language_model."

# (HF suffix after the layer prefix, GGUF MODEL_TENSOR, storage kind)
# kind: f32 = exact float32, f16 = body type, q = body type, router = f16.
LAYER_JOBS = [
    ("linear_attn.in_proj_qkv.weight", "ATTN_QKV", "q"),
    ("linear_attn.in_proj_z.weight", "ATTN_GATE", "q"),
    ("linear_attn.in_proj_a.weight", "SSM_ALPHA", "q"),
    ("linear_attn.in_proj_b.weight", "SSM_BETA", "q"),
    ("linear_attn.out_proj.weight", "SSM_OUT", "q"),
    ("linear_attn.conv1d.weight", "SSM_CONV1D", "f32"),
    ("linear_attn.dt_bias", "SSM_DT", "f32"),
    ("linear_attn.A_log", "SSM_A", "f32"),
    ("linear_attn.norm.weight", "SSM_NORM", "f32"),
    ("mlp.gate.weight", "FFN_GATE_INP", "router"),
    ("mlp.shared_expert.gate_proj.weight", "FFN_GATE_SHEXP", "q"),
    ("mlp.shared_expert.up_proj.weight", "FFN_UP_SHEXP", "q"),
    ("mlp.shared_expert.down_proj.weight", "FFN_DOWN_SHEXP", "q"),
    ("mlp.shared_expert_gate.weight", "FFN_GATE_INP_SHEXP", "router"),
    ("attn_hyper_connection.hc_norm.weight", "HC_ATTN_NORM", "f32"),
    ("attn_hyper_connection.input_mix_weight_down.weight", "HC_ATTN_DOWN", "f16"),
    ("attn_hyper_connection.input_mix_weight_up.weight", "HC_ATTN_UP", "f16"),
    ("attn_hyper_connection.block_inject_weight.weight", "HC_ATTN_INJECT", "f16"),
    ("mlp_hyper_connection.hc_norm.weight", "HC_FFN_NORM", "f32"),
    ("mlp_hyper_connection.input_mix_weight_down.weight", "HC_FFN_DOWN", "f16"),
    ("mlp_hyper_connection.input_mix_weight_up.weight", "HC_FFN_UP", "f16"),
    ("mlp_hyper_connection.block_inject_weight.weight", "HC_FFN_INJECT", "f16"),
]

# Full-attention layers (e.g. every 4th): no SSM/GDN tensors; the indexer
# projection is split into q/k like the converter does.
FULL_JOBS = [
    ("self_attn.q_proj.weight", "ATTN_Q", "q"),
    ("self_attn.k_proj.weight", "ATTN_K", "q"),
    ("self_attn.v_proj.weight", "ATTN_V", "q"),
    ("self_attn.o_proj.weight", "ATTN_OUT", "q"),
    ("self_attn.q_norm.weight", "ATTN_Q_NORM", "f32"),
    ("self_attn.k_norm.weight", "ATTN_K_NORM", "f32"),
    ("self_attn.indexer.q_layernorm.weight", "INDEXER_Q_NORM", "f32"),
    ("self_attn.indexer.k_layernorm.weight", "INDEXER_K_NORM", "f32"),
]

# LAYER_JOBS entries that only exist on linear-attention (GDN) layers.
GDN_ONLY = {
    "linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight",
    "linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight",
    "linear_attn.out_proj.weight", "linear_attn.conv1d.weight",
    "linear_attn.dt_bias", "linear_attn.A_log", "linear_attn.norm.weight",
}

SHARED_JOBS = [
    ("model.language_model.embed_tokens.weight", "TOKEN_EMBD", "q"),
    ("lm_head.weight", "OUTPUT", "q"),
    ("model.language_model.hyper_connection_mixer.hc_norm.weight", "HC_HEAD_NORM", "f32"),
    ("model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
     "HC_HEAD_DOWN", "f16"),
    ("model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
     "HC_HEAD_UP", "f16"),
]

# PLE (per-layer embeddings): one PLE layer carries 6 small tensors + one flat
# [rows, head_dim] n-gram table.  The table is the one tensor too large to
# materialise; export_ple registers it and stream_ple_table appends its
# quantized bytes after the spooled tensors are written.
PLE_JOBS = [
    ("ple.conv1d.weight", "PLE_CONV1D", "f16"),
    ("ple.key_proj.weight", "PLE_KEY", "f16"),
    ("ple.value_proj.weight", "PLE_VALUE", "f16"),
    ("ple.norm_conv.weight", "PLE_NORM_CONV", "f32"),
    ("ple.norm_key.weight", "PLE_NORM_KEY", "f32"),
    ("ple.norm_query.weight", "PLE_NORM_QUERY", "f32"),
]
PLE_TABLE = "per_layer_token_embd.weight"
PLE_CHUNK_ROWS = 65536


def reorder_v_heads(tensor: torch.Tensor, dim: int, num_k_heads: int,
                    num_v_per_k: int, head_dim: int) -> torch.Tensor:
    """Grouped (by K head) -> tiled V-head order (converter's static, dim 0/1)."""
    shape = list(tensor.shape)
    if dim < 0:
        dim += len(shape)
    new_shape = shape[:dim] + [num_k_heads, num_v_per_k, head_dim] + shape[dim + 1:]
    tensor = tensor.reshape(*new_shape)
    perm = list(range(len(new_shape)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return tensor.permute(*perm).contiguous().reshape(*shape)


def load_branch_checkpoint(path: str, layers: int):
    """Load a trained branch+router checkpoint for layers ``0..layers-1``.

    Returns ``(pairs, routers, n_ignored_bias)``: ``pairs[(il, gguf_target)]``
    holds the fp32 ``down``/``up`` masters, ``routers[il]`` the trained
    ``ffn_gate_inp`` weight.  ``balance_bias`` is training-time shaping only
    and is dropped (35B deployed-gate finding).
    """
    from export_branches_lora import collect_branch_pairs, layer_index
    sd = torch.load(path, map_location="cpu")
    sd = {k.replace(".doctor.", ".branch."): v for k, v in sd.items()}
    wanted = {"attn_output.weight", "ssm_out.weight", "ffn_moe_out.weight"}
    pairs, _gates = collect_branch_pairs(sd, wanted)
    pairs = {k: v for k, v in pairs.items() if k[0] < layers}
    routers = {}
    n_bias = 0
    for k, v in sd.items():
        if k.endswith(".balance_bias"):
            n_bias += 1
        elif k.endswith((".mlp.gate.weight", ".mlp.mlp.gate.weight")):
            il = layer_index(k)
            if il < layers:
                routers[il] = v.float()
    if not pairs and not routers:
        raise SystemExit(f"--branches {path}: no branch/router tensors found")
    return pairs, routers, n_bias


def write_deployed_checkpoint(src: str, dst: str) -> dict:
    """Strip training-only ``balance_bias`` keys -> the served checkpoint form.

    The 35B deployed re-gates showed the per-expert bias is a training-time
    shaping effect: the runtime carries neither the bias nor its update, and
    the trained router weights keep the gain.  The gate/eval load then passes
    its strict missing/unexpected check.
    """
    sd = torch.load(src, map_location="cpu")
    out = {k: v for k, v in sd.items() if not k.endswith(".balance_bias")}
    dropped = len(sd) - len(out)
    torch.save(out, dst)
    return {"keys": len(out), "dropped": dropped}


class Exporter:
    def __init__(self, args):
        self.args = args
        self.model_dir = Path(args.model_dir)
        self.weight_map = json.loads(
            (self.model_dir / "model.safetensors.index.json").read_text()
        )["weight_map"]
        cfg = json.loads((self.model_dir / "config.json").read_text())
        self.hp = cfg.get("text_config", cfg)
        self.layer_types = list(self.hp["layer_types"])
        self.shard_dir = self.model_dir / "shards"
        self.handles: dict[str, object] = {}
        self.tmap = TensorNameMap(gguf.MODEL_ARCH.QWEN4EXP, args.layers)
        self.stats = {"read_s": 0.0, "dequant_s": 0.0, "quant_s": 0.0,
                      "write_s": 0.0, "in_bytes": 0, "out_bytes": 0}
        # linear-attention geometry (HF config names).
        self.nk = int(self.hp["linear_num_key_heads"])
        self.nv = int(self.hp["linear_num_value_heads"])
        self.hk = int(self.hp["linear_key_head_dim"])
        self.hv = int(self.hp["linear_value_head_dim"])
        self.nv_per_k = self.nv // self.nk
        self.q_dim = self.hk * self.nk
        self.pairs, self.routers, self.n_ignored_bias = {}, {}, 0
        if args.branches != "none":
            self.pairs, self.routers, self.n_ignored_bias = \
                load_branch_checkpoint(args.branches, args.layers)
            self._check_branch_targets()
        self.keep_shards = {s.strip() for s in
                            (getattr(args, "keep_shards", "") or "").split(",")
                            if s.strip()}
        self.ple = getattr(args, "ple", "none")
        self.ple_layer = None
        self.ple_table_nbytes = 0
        if self.ple != "none":
            ple_layers = sorted({int(k.split(".layers.")[1].split(".")[0])
                                 for k in self.weight_map if ".ple." in k})
            if len(ple_layers) != 1:
                raise SystemExit(
                    f"--ple: expected exactly one PLE layer, got {ple_layers}")
            self.ple_layer = ple_layers[0]
            ple_dir = getattr(args, "ple_shard_dir", "") or ""
            self.ple_shard_dir = Path(ple_dir) if ple_dir else self.shard_dir
            self.ple_handles: dict[str, object] = {}
        self.shard_last: dict[str, int] = {}
        if getattr(args, "release_shards", "none") != "none":
            self.shard_last = self._plan_shard_release()

    def _handle(self, shard: str):
        from safetensors import safe_open
        h = self.handles.get(shard)
        if h is None:
            h = safe_open(self.shard_dir / shard, framework="pt", device="cpu")
            self.handles[shard] = h
        return h

    def read_hf_chunked(self, key: str, rows: int = 32768) -> torch.Tensor:
        """Read one HF tensor dequantized, in row chunks (bounded RAM).

        embed_tokens/lm_head are 635M fp8 params each: a full fp32
        materialization is 2.5 GB transient, which OOMs a 30 GB host when
        other work is resident.  Chunks keep the transient ~0.5 GB.
        """
        from safetensors import safe_open  # noqa: F401 (handle cache warms it)
        h = self._handle(self.weight_map[key])
        w = h.get_tensor(key)
        si_key = key + "_scale_inv"
        if w.dim() != 2:
            # tiny non-matrix tensors (conv kernels): no chunking needed.
            t0 = time.time()
            out = w.float()
            self.stats["read_s"] += time.time() - t0
            self.stats["in_bytes"] += w.numel() * w.element_size()
            return out
        out = torch.empty(w.shape[0], w.shape[1], dtype=torch.float16)
        t0 = time.time()
        if si_key in self.weight_map:
            si = self._handle(self.weight_map[si_key]).get_tensor(si_key)
            for a in range(0, w.shape[0], rows):
                b = min(a + rows, w.shape[0])
                r0 = a // 128
                r1 = (b + 127) // 128
                out[a:b] = dequant_fp8_block(
                    w[a:b], si[r0:r1]).to(torch.float16)
            t2 = time.time()
            self.stats["read_s"] += t2 - t0
            self.stats["dequant_s"] += t2 - t0
        else:
            for a in range(0, w.shape[0], rows):
                b = min(a + rows, w.shape[0])
                out[a:b] = w[a:b].float().to(torch.float16)
            self.stats["read_s"] += time.time() - t0
        self.stats["in_bytes"] += w.numel() * w.element_size()
        import gc
        gc.collect()
        return out

    def layer_tensor(self, il: int, suffix: str, hf: str) -> torch.Tensor:
        """Layer tensor source: the trained router replaces the frozen one."""
        if (suffix == "mlp.gate.weight" and il in self.routers
                and self.args.routers == "replace"):
            return self.routers[il].to(torch.float32)
        return self.read_hf_chunked(hf)

    def quantize_q8_0(self, t: torch.Tensor) -> torch.Tensor:
        """Rowwise Q8_0 bytes [..., nblocks, 34] (fp16 scale + 32 int8)."""
        r = t.shape[-1]
        assert r % 32 == 0, f"Q8_0 needs rows % 32 == 0, got {tuple(t.shape)}"
        f = t.float().reshape(-1, r // 32, 32)
        amax = f.abs().amax(-1, keepdim=True).clamp_min(1e-12)
        d = (amax / 127.0).to(torch.float16)
        q = torch.clamp(torch.round(f / d.float()), -127, 127).to(torch.int8)
        d8 = d.view(torch.uint8).reshape(-1, r // 32, 2)
        return torch.cat([d8, q.view(torch.uint8)], dim=-1).reshape(
            *t.shape[:-1], r // 32 * 34)

    def to_body(self, t: torch.Tensor, kind: str):
        t0 = time.time()
        if kind == "f32":
            out = (t.float().numpy(), None)
        elif kind in ("f16", "router"):
            out = (t.to(torch.float16).numpy(), None)
        elif kind == "q":
            if self.args.body == "f16":
                out = (t.to(torch.float16).numpy(), None)
            else:
                out = (self.quantize_q8_0(t).numpy(),
                       gguf.GGMLQuantizationType.Q8_0)
        else:
            raise ValueError(kind)
        self.stats["quant_s"] += time.time() - t0
        return out

    def apply_v_reorder(self, name: str, t: torch.Tensor) -> torch.Tensor:
        """Converter-side weight transforms: V-head reorder and norm offsets."""
        if name.endswith(".linear_attn.in_proj_qkv.weight"):
            q, k, v = t[:self.q_dim], t[self.q_dim:2 * self.q_dim], t[2 * self.q_dim:]
            v = reorder_v_heads(v, 0, self.nk, self.nv_per_k, self.hv)
            return torch.cat([q, k, v], dim=0)
        if name.endswith(".linear_attn.conv1d.weight"):
            # the conv channels are the mixed qkv dims, so the V block follows
            # the same head permutation as in_proj_qkv (reference parity)
            q = t[:2 * self.q_dim]
            v = reorder_v_heads(t[2 * self.q_dim:], 0, self.nk,
                                self.nv_per_k, self.hv)
            return torch.cat([q, v], dim=0)
        if name.endswith(".linear_attn.dt_bias"):
            return reorder_v_heads(t.reshape(-1, 1), 0, self.nk,
                                   self.nv_per_k, 1).reshape(-1)
        if name.endswith(".linear_attn.A_log"):
            # the runtime stores -exp(A_log) in V-head order (reference parity)
            r = reorder_v_heads(t.reshape(-1, 1), 0, self.nk,
                                self.nv_per_k, 1).reshape(-1)
            return -torch.exp(r.float())
        if name.endswith((".attn_hyper_connection.hc_norm.weight",
                          ".mlp_hyper_connection.hc_norm.weight",
                          ".hyper_connection_mixer.hc_norm.weight",
                          ".self_attn.q_norm.weight",
                          ".self_attn.k_norm.weight",
                          ".self_attn.indexer.q_layernorm.weight",
                          ".self_attn.indexer.k_layernorm.weight",
                          ".ple.norm_conv.weight",
                          ".ple.norm_key.weight",
                          ".ple.norm_query.weight")):
            # every RMS norm except ssm_norm is stored as 1 + w (reference)
            return t + 1.0
        if name.endswith(".linear_attn.in_proj_z.weight"):
            return reorder_v_heads(t, 0, self.nk, self.nv_per_k, self.hv)
        if name.endswith((".linear_attn.in_proj_a.weight",
                          ".linear_attn.in_proj_b.weight")):
            return reorder_v_heads(t, 0, self.nk, self.nv_per_k, 1)
        if name.endswith(".linear_attn.out_proj.weight"):
            perm = reorder_v_heads(
                torch.arange(self.nv * self.hv).unsqueeze(0), 1,
                self.nk, self.nv_per_k, self.hv).squeeze(0)
            return t.index_select(1, perm)
        return t

    def gguf_name(self, hf_key: str) -> str:
        stripped = hf_key.replace("language_model.", "")
        # same rename other converters apply: the map knows dt_proj, not dt_bias.
        if stripped.endswith(".linear_attn.dt_bias"):
            stripped = stripped.removesuffix(".dt_bias") + ".dt_proj.bias"
        name = self.tmap.get_name(stripped, try_suffixes=(".weight", ".bias"))
        if name is None:
            raise KeyError(f"no GGUF mapping for {hf_key}")
        return name.format(bid=self._bid) if "{bid}" in name else name

    def _ple_handle(self, shard: str):
        from safetensors import safe_open
        h = self.ple_handles.get(shard)
        if h is None:
            h = safe_open(self.ple_shard_dir / shard, framework="pt",
                          device="cpu")
            self.ple_handles[shard] = h
        return h

    def _ple_read_keys(self) -> list[str]:
        prefix = f"{HF_PREFIX}layers.{self.ple_layer}."
        return [prefix + suffix for suffix, _, _ in PLE_JOBS]

    def ple_kv_values(self) -> dict:
        """PLE KV values, built exactly as the runtime class builds them."""
        hp = self.hp
        ngram = int(hp["ngram_size"])
        per = int(hp["heads_per_ngram"])
        n_heads = (ngram - 1) * per
        head_dim = int(hp["ple_embed_dim"]) // n_heads
        from transformers.models.qwen4_exp.modeling_qwen4_exp import (
            _build_layer_multipliers, _find_nth_prime_after)
        mult = [int(x) for x in _build_layer_multipliers(
            int(hp["vocab_size"]), ngram, 0, int(hp.get("seed", 1234)))]
        pref = f"{HF_PREFIX}layers.{self.ple_layer}.ple.ple_embedding."
        sizes_key, off_key = (pref + "ngram_heads_vocab_sizes",
                              pref + "ngram_heads_offsets")
        if sizes_key in self.weight_map:
            sizes = [int(x) for x in self._handle(
                self.weight_map[sizes_key]).get_tensor(sizes_key).tolist()]
            offsets = [int(x) for x in self._handle(
                self.weight_map[off_key]).get_tensor(off_key).tolist()]
        else:
            sizes, offsets, total = [], [], 0
            for h in range(n_heads):
                s = int(_find_nth_prime_after(
                    int(hp["ngram_vocab_size_base"]) - 1, h + 1))
                sizes.append(s)
                offsets.append(total)
                total += s
        eos = hp.get("eos_token_id", 248044)
        eos = int(eos[0]) if isinstance(eos, list) else int(eos)
        divisor = int(hp.get("make_ngram_vocab_size_divisible_by", 128))
        total = sum(sizes)
        rows = ((total + divisor - 1) // divisor) * divisor
        return {"layers": [self.ple_layer], "ngram_size": ngram,
                "heads_per_ngram": per,
                "conv_kernel": int(hp.get("ple_conv_kernel_size", 4)),
                "eos_token_id": eos,
                "image_token_id": int(hp.get("image_token_id", 248056)),
                "layer_multipliers": mult, "head_offsets": offsets,
                "head_vocab_sizes": sizes, "head_dim": head_dim,
                "rows": rows}

    def ple_table_parts(self) -> list[tuple[int, str, str]]:
        """(index, tensor key, shard file) for the n-gram table, index order."""
        pref = f"{HF_PREFIX}layers.{self.ple_layer}.ple.ple_embedding."
        parts = []
        for key, shard in self.weight_map.items():
            if key.startswith(pref + "ngram_embedding.shard_"):
                idx = int(key.split(".shard_")[1].split(".")[0])
                parts.append((idx, key, shard))
        parts.sort()
        return parts

    def _iter_ple_table_bytes(self):
        from gguf.quants import quantize
        pref = f"{HF_PREFIX}layers.{self.ple_layer}.ple.ple_embedding."
        scale_key = pref + "ngram_embedding.weight_scale"
        scale = self._ple_handle(
            self.weight_map[scale_key]).get_tensor(scale_key).float().reshape(())
        qtype = gguf.GGMLQuantizationType.Q4_0
        rows_seen = 0
        for _, key, shard in self.ple_table_parts():
            t = self._ple_handle(shard).get_tensor(key)
            rows_seen += t.shape[0]
            for a in range(0, t.shape[0], PLE_CHUNK_ROWS):
                b = min(a + PLE_CHUNK_ROWS, t.shape[0])
                rows = (t[a:b].float() * scale).numpy()
                yield quantize(rows, qtype).tobytes()
            del t
        if rows_seen != getattr(self, "ple_rows", rows_seen):
            raise RuntimeError(
                f"PLE table rows {rows_seen} != expected {self.ple_rows}")

    @staticmethod
    def _ple_table_nbytes(rows: int, head_dim: int) -> int:
        from gguf.quants import quantize
        per32 = quantize(np.zeros((32, head_dim), dtype=np.float32),
                         gguf.GGMLQuantizationType.Q4_0).nbytes
        return rows // 32 * per32

    def export_ple(self, writer):
        """Small PLE tensors now; register the table for stream_ple_table."""
        il = self.ple_layer
        prefix = f"{HF_PREFIX}layers.{il}."
        for suffix, enum, kind in PLE_JOBS:
            t = self.read_hf_chunked(prefix + suffix)
            if suffix == "ple.conv1d.weight":
                t = t.squeeze(1)
            t = self.apply_v_reorder(prefix + suffix, t)
            data, raw = self.to_body(t, kind)
            name = gguf.TENSOR_NAMES[getattr(gguf.MODEL_TENSOR, enum)]
            name = name.format(bid=il) + ".weight"
            t0 = time.time()
            if raw is None:
                writer.add_tensor(name, data)
            else:
                writer.add_tensor(name, data, raw_dtype=raw)
            self.stats["write_s"] += time.time() - t0
            self.stats["out_bytes"] += data.nbytes
            del t, data
        v = self.ple_kv_values()
        self.ple_rows = v["rows"]
        self.ple_head_dim = v["head_dim"]
        self.ple_table_nbytes = self._ple_table_nbytes(v["rows"], v["head_dim"])
        print(f"PLE: layer {il}, table {v['rows']} x {v['head_dim']} Q4_0 "
              f"-> {self.ple_table_nbytes / 1e9:.2f} GB", flush=True)

    def register_ple_table(self, writer):
        """Register the table last: its data is appended after the spool copy."""
        writer.add_tensor_info(PLE_TABLE, (self.ple_rows, self.ple_head_dim),
                               np.float16, self.ple_table_nbytes,
                               raw_dtype=gguf.GGMLQuantizationType.Q4_0)

    def stream_ple_table(self, writer):
        """Append the table bytes; the header already holds its info/offset."""
        if not self.ple_table_nbytes:
            return
        # the table must be the last registered tensor or every offset after
        # it shifts by 28.8 GB (the adapter read table bytes as NaN, 2026-10-03)
        last_keys = list(writer.tensors[-1])
        if not last_keys or last_keys[-1] != PLE_TABLE:
            raise RuntimeError("PLE table is not the last registered tensor")
        fout = writer.fout[0] if isinstance(writer.fout, list) else writer.fout
        written = 0
        for buf in self._iter_ple_table_bytes():
            fout.write(buf)
            written += len(buf)
        if written != self.ple_table_nbytes:
            raise RuntimeError(f"PLE table wrote {written} bytes, expected "
                               f"{self.ple_table_nbytes}")
        self.stats["out_bytes"] += written

    def export_indexer_split(self, writer, il: int, prefix: str):
        """Split index_qk_proj into q/k (converter parity: first n_q rows)."""
        n_q = (int(self.hp["indexer_n_heads"])
               * int(self.hp["indexer_head_dim"]))
        t = self.read_hf_chunked(prefix + "self_attn.indexer.index_qk_proj.weight")
        q, k = t[:n_q], t[n_q:]
        for mat, enum in ((q, "INDEXER_Q_PROJ"), (k, "INDEXER_K_PROJ")):
            data, raw = self.to_body(mat, "q")
            name = gguf.TENSOR_NAMES[getattr(gguf.MODEL_TENSOR, enum)]
            name = name.format(bid=il) + ".weight"
            t0 = time.time()
            if raw is None:
                writer.add_tensor(name, data)
            else:
                writer.add_tensor(name, data, raw_dtype=raw)
            self.stats["write_s"] += time.time() - t0
            self.stats["out_bytes"] += data.nbytes
        del t, q, k

    def export_layer_experts(self, writer, il: int):
        """MoE banks for one layer, streamed one expert at a time.

        Peak is one expert's fp8 (~25 MB) + one repack (~14 MB) + the
        accumulated output bytes (~0.55 GB/layer total).  Stacking all 512
        experts as fp32 (the first version) peaked ~20 GB and OOMed a
        30 GB host twice — never again.
        """
        import gc

        import numpy as np
        from qwen4exp_proxy import dequant_fp8_block as _dq
        prefix = f"{HF_PREFIX}layers.{il}.mlp.experts."
        n_exp = int(self.hp["num_experts"])
        kind = self.args.experts
        bufs = {"gate": [], "up": [], "down": []}
        t_exp = 0.0
        for e in range(n_exp):
            gn = f"{prefix}{e}.gate_proj.weight"
            h = self._handle(self.weight_map[gn])
            g = _dq(h.get_tensor(gn),
                    self._handle(self.weight_map[gn + "_scale_inv"])
                    .get_tensor(gn + "_scale_inv"))
            un = f"{prefix}{e}.up_proj.weight"
            h = self._handle(self.weight_map[un])
            u = _dq(h.get_tensor(un),
                    self._handle(self.weight_map[un + "_scale_inv"])
                    .get_tensor(un + "_scale_inv"))
            dn = f"{prefix}{e}.down_proj.weight"
            h = self._handle(self.weight_map[dn])
            d = _dq(h.get_tensor(dn),
                    self._handle(self.weight_map[dn + "_scale_inv"])
                    .get_tensor(dn + "_scale_inv"))
            self.stats["in_bytes"] += (g.numel() + u.numel() + d.numel())
            for proj, mat in (("gate", g), ("up", u), ("down", d)):
                t0 = time.time()
                codes, scales = _encode_slice(mat, 128, "lloyd")
                if kind == "ptq1_0":
                    q, hh, s = ptq1_0.repack_bank_ptq1_0(codes, scales)
                    raw = ptq1_0.pack_block_bytes(q, hh, s)
                else:
                    d8 = scales.view(torch.uint8).reshape(
                        *scales.shape, 2)
                    raw = torch.cat(
                        [d8, codes.reshape(*codes.shape[:-1], 32)],
                        dim=-1)
                t_exp += time.time() - t0
                bufs[proj].append(raw.numpy())
                del codes, scales, raw
            del g, u, d
        self.stats["quant_s"] += t_exp
        gc.collect()
        for proj in ("gate", "up", "down"):
            raw = np.stack(bufs[proj], axis=0)
            # gguf-py reads the last dim as nblocks*type_size: flatten the
            # (ngroups, blockbytes) axes or the file gains a phantom dim.
            raw = raw.reshape(*raw.shape[:-2], raw.shape[-2] * raw.shape[-1])
            qt = (gguf.GGMLQuantizationType.PTQ1_0 if kind == "ptq1_0"
                  else gguf.GGMLQuantizationType.PQ2_0)
            gguf_key = {"gate": "FFN_GATE_EXP", "up": "FFN_UP_EXP",
                        "down": "FFN_DOWN_EXP"}[proj]
            name = gguf.TENSOR_NAMES[getattr(gguf.MODEL_TENSOR, gguf_key)]
            name = name.format(bid=il) + ".weight"
            t0 = time.time()
            writer.add_tensor(name, raw, raw_dtype=qt)
            self.stats["write_s"] += time.time() - t0
            self.stats["out_bytes"] += raw.nbytes
            del raw
        gc.collect()

    def layer_jobs(self, il: int):
        """(prefix, is_full, jobs) for one layer; shared by run and the plan."""
        prefix = f"{HF_PREFIX}layers.{il}."
        is_full = self.layer_types[il] == "full_attention"
        jobs = ([j for j in LAYER_JOBS if j[0] not in GDN_ONLY]
                + FULL_JOBS) if is_full else LAYER_JOBS
        return prefix, is_full, jobs

    def _read_keys(self, il: int):
        """Every HF key (incl. ``_scale_inv``) the export reads for layer il."""
        prefix, is_full, jobs = self.layer_jobs(il)
        keys = [prefix + suffix for suffix, _, _ in jobs]
        if is_full:
            keys.append(prefix + "self_attn.indexer.index_qk_proj.weight")
        for e in range(int(self.hp["num_experts"])):
            for p in ("gate_proj", "up_proj", "down_proj"):
                keys.append(f"{prefix}mlp.experts.{e}.{p}.weight")
        return keys

    def _plan_shard_release(self) -> dict:
        """Last layer that reads each source shard (shared tensors -> layers)."""
        last: dict[str, int] = {}
        for il in range(self.args.layers):
            for k in self._read_keys(il):
                for kk in (k, k + "_scale_inv"):
                    s = self.weight_map.get(kk)
                    if s is not None:
                        last[s] = il
        for hf, _, _ in SHARED_JOBS:
            for kk in (hf, hf + "_scale_inv"):
                s = self.weight_map.get(kk)
                if s is not None:
                    last[s] = self.args.layers
        if self.ple != "none":
            for k in self._ple_read_keys():
                for kk in (k, k + "_scale_inv"):
                    s = self.weight_map.get(kk)
                    if s is not None:
                        last[s] = self.args.layers
            # n-gram shards are read while streaming, after every release pass
            for _, _, s in self.ple_table_parts():
                last.pop(s, None)
        return last

    def _release_shards(self, done_through: int):
        """Close + unlink source shards whose last use is <= done_through.

        The 48-layer mirror + output + spool do not fit this box's disk at
        once; releasing each shard after its last tensor keeps the mirror
        footprint bounded.  ``--keep-shards`` protects basenames (the prep
        passes the pre-existing prefix/PLE set so local work survives).
        """
        if not self.shard_last:
            return
        import gc
        for s in [s for s, last in self.shard_last.items()
                  if last <= done_through]:
            self.handles.pop(s, None)
            gc.collect()
            if s in self.keep_shards:
                continue
            try:
                os.unlink(self.shard_dir / s)
                self.stats["released"] = self.stats.get("released", 0) + 1
            except OSError:
                pass

    def _check_branch_targets(self):
        """Refuse a checkpoint whose attention branches do not match the layers."""
        for (il, target) in self.pairs:
            is_full = self.layer_types[il] == "full_attention"
            if target == "attn_output.weight" and not is_full:
                raise SystemExit(
                    f"branches: layer {il} is not full attention but carries "
                    f"{target}")
            if target == "ssm_out.weight" and is_full:
                raise SystemExit(
                    f"branches: layer {il} is full attention but carries "
                    f"{target}")

    def branch_factors(self, down, up):
        from export_branches_lora import deploy_weights
        return deploy_weights(down, up, self.args.branch_quant,
                              self.args.deploy_quant)

    def export_adapter(self, writer):
        """Write the deployed branch factors + adapter metadata (single file).

        Tensor names follow the LoRA container conventions: ``lora_a`` is the
        down factor (numpy [rank, in] -> ggml ne [in, rank]) and ``lora_b``
        the up factor.  ``blk.N.ffn_moe_out.weight`` is the fork's virtual
        MoE-output target (anchored at ``ffn_gate_inp``); ``attn_output`` /
        ``ssm_out`` are ordinary LoRA targets of the two attention kinds.
        """
        from export_branches_lora import add_factor, pack_q1_0_g128  # noqa: F401
        # general.type stays MODEL: the embedded loader requires that.
        writer.add_bool("adapter.embedded", True)
        writer.add_string("adapter.type", "lora")
        writer.add_float32("adapter.lora.alpha", 0.0)
        writer.add_string("adapter.recipe",
                          self.args.adapter_recipe or "qwen4exp corrections")
        writer.add_string("adapter.source", str(self.args.branches))
        raw_qtype = None
        if self.args.branch_dtype == "q1_0_g128":
            raw_qtype = gguf.GGMLQuantizationType.Q1_0_g128
        n_pairs = 0
        for (il, target), t in sorted(self.pairs.items()):
            down, up = self.branch_factors(t["down"], t["up"])
            add_factor(writer, f"blk.{il}.{target}.lora_a", down,
                       self.args.branch_dtype, raw_qtype)
            add_factor(writer, f"blk.{il}.{target}.lora_b", up,
                       self.args.branch_dtype, raw_qtype)
            n_pairs += 1
        if self.n_ignored_bias:
            print(f"adapter: dropped {self.n_ignored_bias} balance_bias "
                  f"tensor(s) (training-time shaping)")
        print(f"adapter: {n_pairs} branch pairs embedded "
              f"({self.args.branch_dtype}, deploy "
              f"{self.args.branch_quant}/{self.args.deploy_quant}); "
              f"{len(self.routers)} trained routers in the body")

    def write_kv(self, writer):
        hp = self.hp
        L = self.args.layers
        w = writer
        w.add_type(gguf.GGUFType.MODEL)
        w.add_name(f"qwen4exp {L}-layer export prototype ({self.args.experts})")
        ftype = {"ptq1_0": gguf.LlamaFileType.MOSTLY_PTQ1_0,
                 "pq2_0": gguf.LlamaFileType.MOSTLY_PQ2_0}[self.args.experts]
        w.add_file_type(ftype.value if hasattr(ftype, "value") else int(ftype))
        w.add_vocab_size(int(hp["vocab_size"]))
        w.add_context_length(int(hp.get("max_position_embeddings", 262144)))
        w.add_embedding_length(int(hp["hidden_size"]))
        w.add_block_count(L)
        w.add_head_count(int(hp["num_attention_heads"]))
        w.add_head_count_kv(int(hp["num_key_value_heads"]))
        w.add_key_length(int(hp.get("head_dim", 256)))
        w.add_value_length(int(hp.get("head_dim", 256)))
        w.add_layer_norm_rms_eps(float(hp.get("rms_norm_eps", 1e-6)))
        w.add_rope_dimension_count(int(hp["head_dim"] * 0.25))
        w.add_rope_freq_base(10000000.0)
        w.add_rope_dimension_sections([11, 11, 10, 0])
        w.add_expert_count(int(hp["num_experts"]))
        w.add_expert_used_count(int(hp["num_experts_per_tok"]))
        w.add_expert_feed_forward_length(int(hp["moe_intermediate_size"]))
        w.add_expert_shared_feed_forward_length(
            int(hp["shared_expert_intermediate_size"]))
        w.add_expert_gating_func(gguf.ExpertGatingFuncType.SOFTMAX)
        w.add_ssm_conv_kernel(int(hp["linear_conv_kernel_dim"]))
        w.add_ssm_inner_size(6144)
        w.add_ssm_state_size(128)
        w.add_ssm_time_step_rank(48)
        w.add_ssm_group_count(16)
        w.add_full_attention_interval(int(hp.get("full_attention_interval", 4)))
        ratios = [0 if self.layer_types[i] != "full_attention"
                  else int(hp.get("indexer_compress_ratio", 4))
                  for i in range(L)]
        w.add_attention_compress_ratios(ratios)
        w.add_hyper_connection_count(int(hp["hc_count"]))
        w.add_hyper_connection_low_rank(int(hp["hc_lowrank"]))
        w.add_indexer_head_count(int(hp["indexer_n_heads"]))
        w.add_indexer_key_length(int(hp["indexer_head_dim"]))
        w.add_indexer_top_k(int(hp["indexer_budget"]))
        if self.ple != "none":
            v = self.ple_kv_values()
            w.add_key_value("qwen4exp.embedding_length_per_layer_input",
                            v["head_dim"], gguf.GGUFValueType.UINT32)
            w.add_key_value("qwen4exp.ple.layers", v["layers"],
                            gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.INT32)
            for key, val in (("ngram_size", v["ngram_size"]),
                             ("heads_per_ngram", v["heads_per_ngram"]),
                             ("conv_kernel", v["conv_kernel"]),
                             ("eos_token_id", v["eos_token_id"]),
                             ("image_token_id", v["image_token_id"])):
                w.add_key_value(f"qwen4exp.ple.{key}", val,
                                gguf.GGUFValueType.UINT32)
            for key in ("layer_multipliers", "head_offsets",
                        "head_vocab_sizes"):
                w.add_key_value(f"qwen4exp.ple.{key}", v[key],
                                gguf.GGUFValueType.ARRAY,
                                gguf.GGUFValueType.UINT64)

    def write_tokenizer(self, writer):
        vocab = json.loads((self.model_dir / "vocab.json").read_text())
        merges_txt = (self.model_dir / "merges.txt").read_text().splitlines()
        merges = [l for l in merges_txt
                  if l and not l.startswith("#version")]  # real merges only
        tokens = [None] * len(vocab)
        for tok, i in vocab.items():
            tokens[i] = tok.encode("utf-8")
        # HF vocab_size (248320) exceeds vocab.json (248044): the reference
        # file appends the 33 added_tokens then [PAD<id>] placeholders, all
        # CONTROL type.  Reproduce verbatim (added tokens from tokenizer.json).
        tjson = json.loads((self.model_dir / "tokenizer.json").read_text())
        added = sorted(tjson.get("added_tokens", []), key=lambda d: d["id"])
        n_vocab = int(self.hp["vocab_size"])
        assert len(tokens) + 0 == len(vocab)
        for d in added:
            assert d["id"] == len(tokens), (d["id"], len(tokens))
            tokens.append(d["content"].encode("utf-8"))
        while len(tokens) < n_vocab:
            tokens.append(f"[PAD{len(tokens)}]".encode("utf-8"))
        assert len(tokens) == n_vocab, (len(tokens), n_vocab)
        types = [1] * len(vocab) + [3] * (n_vocab - len(vocab))
        writer.add_tokenizer_model("gpt2")
        writer.add_tokenizer_pre("qwen35")
        writer.add_token_list(tokens)
        writer.add_token_types(types)
        writer.add_token_merges([m.encode("utf-8") for m in merges])
        writer.add_bos_token_id(248044)
        writer.add_eos_token_id(248046)
        writer.add_pad_token_id(248044)
        writer.add_add_bos_token(False)
        sv = gguf.SpecialVocab(self.model_dir, load_merges=False)
        sv.add_to_gguf(writer)

    def run(self):
        t_start = time.time()
        writer = gguf.GGUFWriter(
            str(self.args.out), ARCH,
            use_temp_file=getattr(self.args, "use_temp_file", False))
        self.write_kv(writer)
        self.write_tokenizer(writer)
        for il in range(self.args.layers):
            self._bid = il
            prefix, is_full, jobs = self.layer_jobs(il)
            for suffix, tensor_enum, kind in jobs:
                hf = prefix + suffix
                t = self.layer_tensor(il, suffix, hf)
                if suffix == "linear_attn.conv1d.weight" and t.dim() == 3:
                    t = t.squeeze(1)
                if (suffix == "mlp.shared_expert_gate.weight"
                        and t.dim() == 2 and t.shape[0] == 1):
                    t = t.squeeze(0)
                t = self.apply_v_reorder(hf, t)
                data, raw = self.to_body(t, kind)
                gguf_key = getattr(gguf.MODEL_TENSOR, tensor_enum)
                name = gguf.TENSOR_NAMES[gguf_key].format(bid=il)
                name += ".bias" if suffix == "linear_attn.dt_bias" else (
                    "" if suffix == "linear_attn.A_log" else ".weight")
                t0 = time.time()
                if raw is None:
                    writer.add_tensor(name, data)
                else:
                    writer.add_tensor(name, data, raw_dtype=raw)
                self.stats["write_s"] += time.time() - t0
                self.stats["out_bytes"] += data.nbytes
                del t, data
            if is_full:
                self.export_indexer_split(writer, il, prefix)
            self.export_layer_experts(writer, il)
            self._release_shards(il)
        for hf, tensor_enum, kind in SHARED_JOBS:
            t = self.read_hf_chunked(hf)
            t = self.apply_v_reorder(hf, t)
            data, raw = self.to_body(t, kind)
            name = gguf.TENSOR_NAMES[getattr(gguf.MODEL_TENSOR, tensor_enum)]
            name += ".weight"
            t0 = time.time()
            if raw is None:
                writer.add_tensor(name, data)
            else:
                writer.add_tensor(name, data, raw_dtype=raw)
            self.stats["write_s"] += time.time() - t0
            self.stats["out_bytes"] += data.nbytes
        if self.ple != "none":
            t0 = time.time()
            self.export_ple(writer)
            self.stats["write_s"] += time.time() - t0
        self._release_shards(self.args.layers)
        if self.pairs:
            t0 = time.time()
            self.export_adapter(writer)
            self.stats["write_s"] += time.time() - t0
        if self.ple != "none":
            # last registration: the table data is appended after the spool copy
            self.register_ple_table(writer)
        t0 = time.time()
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
        if self.ple != "none":
            t1 = time.time()
            self.stream_ple_table(writer)
            self.stats["write_s"] += time.time() - t1
        writer.close()
        self.stats["write_s"] += time.time() - t0
        total = time.time() - t_start
        s = self.stats
        import resource
        peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        print(f"export done in {total:.1f}s "
              f"(read {s['read_s']:.1f} / dequant {s['dequant_s']:.1f} / "
              f"quant {s['quant_s']:.1f} / write {s['write_s']:.1f})")
        rel = f"; released {s['released']} shard(s)" if s.get("released") else ""
        print(f"in {s['in_bytes'] / 1e9:.2f} GB fp8 -> "
              f"out {s['out_bytes'] / 1e9:.2f} GB on disk; "
              f"peak RSS {peak_gb:.1f} GB{rel}")
        per_layer = total / self.args.layers
        proj = per_layer * 48
        print(f"per-layer {per_layer:.1f}s -> 48-layer projection ~{proj / 60:.0f} min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--experts", choices=["ptq1_0", "pq2_0"], default="ptq1_0")
    ap.add_argument("--body", choices=["f16", "q8_0"], default="f16")
    ap.add_argument("--branches", default="none",
                    help="path to a trained branch+router checkpoint (.pt) to "
                         "embed as a single-file adapter; 'none' = no "
                         "corrections")
    ap.add_argument("--branch-dtype", choices=["f16", "q1_0_g128"],
                    default="f16",
                    help="adapter factor storage (q1_0_g128 needs the fork "
                         "gguf-py, which is already on the path)")
    ap.add_argument("--branch-quant", choices=["g128", "rank", "none"],
                    default="g128", help="deployed branch format")
    ap.add_argument("--deploy-quant", choices=["lloyd", "absmean"],
                    default="lloyd", help="scale rule for ternarising")
    ap.add_argument("--routers", choices=["replace", "none"], default="replace",
                    help="replace ffn_gate_inp with the checkpoint's trained "
                         "router (exact numeric merge)")
    ap.add_argument("--adapter-recipe", default="",
                    help="provenance string written to adapter.recipe")
    ap.add_argument("--use-temp-file", action="store_true",
                    help="spool tensor data to a spooled temp file (TMPDIR) "
                         "instead of holding it in RAM; required past ~40 "
                         "layers (point TMPDIR at a disk path, not tmpfs)")
    ap.add_argument("--release-shards", choices=["none", "all"], default="none",
                    help="unlink each source shard after its last tensor is "
                         "read; needed when mirror + output + spool exceed the "
                         "disk (re-download to restore; --keep-shards protects)")
    ap.add_argument("--ple", choices=["none", "q4_0"], default="none",
                    help="export the PLE layer's small tensors + n-gram table "
                         "(Q4_0 table, ~28.8 GB at 48 layers); default none "
                         "keeps the calibration-floor no-PLE file")
    ap.add_argument("--ple-shard-dir", default="",
                    help="directory holding the n-gram table shards (default: "
                         "model-dir/shards; point it at a mirror that keeps "
                         "the table when the body mirror releases shards)")
    ap.add_argument("--keep-shards", default="",
                    help="comma-separated shard basenames never unlinked "
                         "(the prep passes the pre-existing prefix/PLE set)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    Exporter(args).run()


if __name__ == "__main__":
    main()
