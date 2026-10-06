"""Reverse-load a Clef-Flash GGUF (Qwen3.5-9B dense) into a transformers text model.

The fork's converter (`llama.cpp/conversion/qwen.py`) maps HF -> GGUF with a few
invertible transforms.  This module inverts them so the *deployed* PQ2_0 body
(and the f16 reference) can run in torch with the deployment quantizer in the
training loop.  No HF checkpoint download: the GGUF already holds the weights,
and for PQ2_0 the dequantised values are exactly what the runtime multiplies
(verified byte-identical to the fork's ``dequantize_row_pq2_0``).

Memory discipline (a naive build OOM'd the host):
  * never materialise a full state dict -- the iterator yields one tensor at a
    time and the streamed loader assigns it straight onto the model;
  * never use fp32 tensors for a 9B model (36 GB); use bf16/f16 (18 GB);
  * ``load_text_model_streamed`` builds the model on CPU in bf16, adopts each
    tensor, then moves to the device, so the host peak is one model, not two.

Inverted converter transforms (symbols in ``conversion/qwen.py``):
  * GGUF ``.data`` / ``gguf.quants.dequantize`` are already torch ``[out, in]``
    orientation (gguf-py reverses the ggml dims on read), so no transpose.
  * RMSNorm weights are stored as ``w + 1`` for every ``norm.weight`` except
    the GDN ``linear_attn.norm`` (``Qwen3NextModel.modify_tensors``).
  * GDN ``A_log`` is stored as ``-exp(A_log)``; ``dt_bias`` as ``ssm_dt.bias``;
    the depthwise ``conv1d`` is squeezed on write.
  * GDN V heads are reordered grouped -> tiled for ggml broadcast
    (``_LinearAttentionVReorderBase``); the inverse swaps the head counts.
  * full-attention ``q_proj`` keeps its fused output gate
    (``head_dim * n_heads * 2`` output rows) and is stored as ``attn_q``.

The custom PQ2_0 container (ggml type 142) is parsed here: per 128 weights, 2
little-endian fp16 scale bytes then 32 bytes of 2-bit codes (code = q + 1, at
bits ``2*(j % 4)``).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

try:  # fork's gguf-py on PYTHONPATH
    from gguf import GGUFReader
    from gguf import quants as _quants
except Exception as _e:  # pragma: no cover - import error is surfaced by callers
    GGUFReader = None
    _quants = None
    _gguf_import_error = _e

PQ2_0 = 142  # GGMLQuantizationType value; the fork's Q1_0_g128 container

#: Clef-Flash GDN geometry (16 K heads, 32 V heads, 128/128 head dims).
NK, NV, HK, HV = 16, 32, 128, 128

_RMS_NORM_LAYERS = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
}
_LINEAR = {
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}
_GDN = {
    "attn_qkv.weight": (0, "qkv"),
    "attn_gate.weight": (0, "rows"),
    "ssm_out.weight": (1, "cols"),
    "ssm_alpha.weight": (0, "heads1"),
    "ssm_beta.weight": (0, "heads1"),
}
_GDN_HF = {
    "attn_qkv.weight": "in_proj_qkv.weight",
    "attn_gate.weight": "in_proj_z.weight",
    "ssm_out.weight": "out_proj.weight",
    "ssm_alpha.weight": "in_proj_a.weight",
    "ssm_beta.weight": "in_proj_b.weight",
}


# ------------------------------------------------------------------ reorder --

def reorder_v_heads(t: torch.Tensor, dim: int, num_a: int, num_b: int,
                    head_dim: int) -> torch.Tensor:
    """Grouped -> tiled V-head reorder (``_reorder_v_heads`` from the converter).

    The inverse is the same function with the two head counts swapped: the
    operation is only an involution when the reorder axis is the trailing one.
    """
    shape = list(t.shape)
    if dim < 0:
        dim += len(shape)
    new_shape = shape[:dim] + [num_a, num_b, head_dim] + shape[dim + 1:]
    t = t.reshape(*new_shape)
    perm = list(range(len(new_shape)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return t.permute(*perm).contiguous().reshape(*shape)


# ------------------------------------------------------------------- dequant --

def dequant_pq2_0(data: np.ndarray) -> np.ndarray:
    """Dequantise the fork's PQ2_0 raw bytes to float32.

    ``data`` is ``[rows, bytes_per_row]`` uint8; 34 bytes per 128-weight group:
    fp16 scale then 32 bytes of 2-bit codes.  Byte-identical to the fork's
    ``dequantize_row_pq2_0`` (verified against a C harness).
    """
    data = np.ascontiguousarray(data)
    if data.dtype != np.uint8:
        data = data.view(np.uint8)
    rows, nbytes = data.shape
    if nbytes % 34 != 0:
        raise ValueError(f"PQ2_0 row has {nbytes} bytes, not a multiple of 34")
    groups = nbytes // 34
    b = data.reshape(rows, groups, 34)
    scale = b[:, :, :2].copy().view(np.float16).reshape(rows, groups).astype(np.float32)
    codes = b[:, :, 2:]                      # [rows, groups, 32]
    shifts = np.arange(0, 8, 2, dtype=np.uint8).reshape(1, 1, 1, 4)
    q = (codes[:, :, :, None] >> shifts) & 3  # [rows, groups, 32, 4]
    q = q.reshape(rows, groups, 128).astype(np.float32) - 1.0
    return (q * scale[:, :, None]).reshape(rows, groups * 128)


def _torch_tensor(t) -> np.ndarray:
    """Raw array for one GGUF tensor (dequantised when quantised)."""
    qtype = int(t.tensor_type)
    if qtype == PQ2_0:
        return dequant_pq2_0(t.data)
    if _quants is None:  # pragma: no cover
        raise RuntimeError(f"gguf-py unavailable: {_gguf_import_error}")
    if qtype in (0, 1):  # f32, f16 -- gguf-py already gives [out, in]
        return np.array(t.data, copy=True)  # writable, detached from the mmap
    # dequantize() already returns a fresh array; callers must not copy again.
    return _quants.dequantize(t.data, t.tensor_type)


# ---------------------------------------------------------------- iterator ----

def _undo_gdn(suffix: str, mode: str, dim: int, ten: torch.Tensor) -> torch.Tensor:
    rep = NV // NK

    def inv(t, dim, head_dim):
        return reorder_v_heads(t, dim, rep, NK, head_dim)

    if mode == "qkv":
        qd = kd = HK * NK
        q, k, v = ten[:qd], ten[qd:qd + kd], ten[qd + kd:]
        return torch.cat([q, k, inv(v, 0, HV)], dim=0)
    if mode == "rows":
        return inv(ten, 0, HV)
    if mode == "cols":
        return inv(ten, 1, HV)
    if mode == "heads1":
        return inv(ten, 0, 1)
    if mode == "conv":
        qk = HK * NK * 2
        qk_part, v_part = ten[:qk], ten[qk:]
        return torch.cat([qk_part, inv(v_part, 0, HV)], dim=0).unsqueeze(1)
    raise ValueError(mode)


def iter_hf_tensors(path: str | Path, *, dtype=torch.bfloat16,
                    include_lm_head: bool = True, layer_limit: int | None = None,
                    meta: dict | None = None) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(hf_name, tensor)`` one at a time, already cast to ``dtype``.

    ``meta`` (optional) is filled with ``vocab``/``hidden``.  This is the single
    source of the reverse mapping; the dict and streamed loaders both use it.
    """
    if meta is None:
        meta = {}
    r = GGUFReader(str(path))
    for t in r.tensors:
        name = t.name
        if name.startswith("blk.") and layer_limit is not None:
            if int(name.split(".", 2)[1]) >= layer_limit:
                continue
        # skip before reading: output.weight is a full 2 GiB tensor we do not
        # need when lm_head is excluded (host RAM is the binding constraint).
        if name == "output.weight" and not include_lm_head:
            continue
        # _torch_tensor already returns a fresh writable array (the mmap copy or
        # a dequant result), so no second copy here.
        ten = torch.from_numpy(_torch_tensor(t)).to(dtype)

        if name == "token_embd.weight":
            meta["vocab"] = int(ten.shape[0])
            meta["hidden"] = int(ten.shape[1])
            yield "embed_tokens.weight", ten
            continue
        if name == "output_norm.weight":
            yield "norm.weight", ten - 1.0
            continue
        if name == "output.weight":
            yield "lm_head.weight", ten
            continue
        if not name.startswith("blk."):
            continue
        _, idx_s, suffix = name.split(".", 2)
        base = f"layers.{int(idx_s)}."

        if suffix in _RMS_NORM_LAYERS:
            yield base + _RMS_NORM_LAYERS[suffix], ten - 1.0
        elif suffix in _LINEAR:
            yield base + _LINEAR[suffix], ten
        elif suffix in _GDN:
            dim, mode = _GDN[suffix]
            yield base + f"linear_attn.{_GDN_HF[suffix]}", _undo_gdn(suffix, mode, dim, ten)
        elif suffix == "ssm_norm.weight":
            yield base + "linear_attn.norm.weight", ten
        elif suffix == "ssm_conv1d.weight":
            yield base + "linear_attn.conv1d.weight", _undo_gdn(suffix, "conv", 0, ten)
        elif suffix == "ssm_a":
            ten = torch.log((-ten.float()).clamp_min(1e-30)).to(dtype)
            yield base + "linear_attn.A_log", _undo_gdn(suffix, "heads1", 0, ten)
        elif suffix == "ssm_dt.bias":
            yield base + "linear_attn.dt_bias", _undo_gdn(suffix, "heads1", 0, ten)
        else:
            raise KeyError(f"unmapped GGUF tensor {name}")


def load_gguf_state(path: str | Path, *, include_lm_head: bool = True,
                    dtype=torch.bfloat16, layer_limit: int | None = None) -> tuple[dict, dict]:
    """Dict form (small models / tests only -- for 9B use the streamed loader)."""
    meta: dict = {}
    state = {k: v for k, v in iter_hf_tensors(
        path, dtype=dtype, include_lm_head=include_lm_head,
        layer_limit=layer_limit, meta=meta)}
    if "hidden" not in meta:
        raise RuntimeError("GGUF had no token embedding")
    return state, meta


# --------------------------------------------------------------------- build --

def _text_config(path: str | Path, layer_limit: int | None):
    from transformers import Qwen3_5TextConfig
    cfg = json.loads(Path(path).read_text())["text_config"]
    if layer_limit is not None:
        cfg = dict(cfg)
        cfg["num_hidden_layers"] = layer_limit
        cfg["layer_types"] = list(cfg["layer_types"])[:layer_limit]
    tc = Qwen3_5TextConfig(**cfg)
    tc.mtp_num_hidden_layers = 0
    return tc


def load_text_model_streamed(path: str | Path, device="cpu",
                             dtype=torch.bfloat16,
                             text_config_path: str | Path | None = None,
                             layer_limit: int | None = None):
    """Build the text tower and adopt the reversed weights one tensor at a time.

    The model is constructed on ``device`` in ``dtype`` (never fp32, never a
    second full copy), then each parameter is overwritten in place.  Host peak
    is one model plus one tensor; there is no full state dict.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    if text_config_path is None:
        text_config_path = Path(path).parent / "hf-head" / "config.json"
    cfg = _text_config(text_config_path, layer_limit)

    dev = torch.device(device)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        # Construct directly on the target device so the host never holds a
        # second full copy of the 9B weights (RSS is the binding constraint).
        with torch.device(dev):
            model = Qwen3_5TextModel(cfg)
    finally:
        torch.set_default_dtype(prev)
    model = model.to(device=dev, dtype=dtype)  # buffers created with device=None

    meta: dict = {}
    loaded = 0
    with torch.no_grad():
        for hf_name, ten in iter_hf_tensors(
                path, dtype=dtype, include_lm_head=False,
                layer_limit=layer_limit, meta=meta):
            param = model.get_parameter(hf_name)
            param.data = ten.to(dev).contiguous()
            loaded += 1
    if "hidden" not in meta:
        raise RuntimeError("GGUF had no token embedding")
    model.eval()
    return model, meta, loaded


def patch_recurrent_gdn(model) -> int:
    """Use the *recurrent* GDN prefill, matching the deployed llama.cpp CPU path.

    transformers defaults to ``torch_chunk_gated_delta_rule``; the fork's CPU
    runtime is the recurrent form.  Chunked diverges over position (f16
    per-token cos 1.0 -> 0.82 by token 39) while recurrent holds cos ~1.0, so
    every training/eval forward must be patched.  Returns the layer count.
    """
    from transformers.models.qwen3_5 import modeling_qwen3_5 as _m
    patched = 0
    for mod in model.modules():
        if type(mod).__name__ == "Qwen3_5GatedDeltaNet":
            mod.chunk_gated_delta_rule = _m.torch_recurrent_gated_delta_rule
            patched += 1
    return patched


def patch_truncated_gdn(model, chunk: int = 64) -> int:
    """Recurrent GDN forward with **truncated BPTT** (state detached every chunk).

    The deployed CPU runtime is the recurrent form, so this keeps the forward
    exactly right -- but the recurrent fallback unrolls the whole sequence and
    its backward NaN's after one step on this 9B (the recurrence amplifies, per
    the dense forensics F4).  Detaching the carried state every ``chunk``
    tokens bounds the backward horizon to ``chunk`` while leaving the forward
    bit-for-bit the recurrent one.  Returns the layer count.
    """
    from transformers.models.qwen3_5 import modeling_qwen3_5 as _m
    rec = _m.torch_recurrent_gated_delta_rule

    def fn(query, key, value, g, beta, initial_state=None,
           output_final_state=False, use_qk_l2norm_in_kernel=False, **kw):
        total = query.shape[1]
        outs, state = [], initial_state
        for s in range(0, total, chunk):
            e = min(s + chunk, total)
            out, state = rec(query[:, s:e], key[:, s:e], value[:, s:e], g[:, s:e],
                             beta[:, s:e], state, True, use_qk_l2norm_in_kernel)
            outs.append(out)
            state = state.detach() if state is not None else None  # truncated BPTT
        out = torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]
        return out, (state if output_final_state else None)

    patched = 0
    for mod in model.modules():
        if type(mod).__name__ == "Qwen3_5GatedDeltaNet":
            mod.chunk_gated_delta_rule = fn
            patched += 1
    return patched


def build_text_model(state: dict, text_config: dict, dtype=torch.bfloat16):
    """Small-model/test helper: construct on meta and adopt a state dict."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    cfg = _text_config_from_dict(text_config)
    with torch.device("meta"):
        model = Qwen3_5TextModel(cfg)
    state = {k: v for k, v in state.items() if k != "lm_head.weight"}
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    model = model.to(dtype=dtype)
    return model, list(missing), list(unexpected)


def _text_config_from_dict(text_config: dict):
    from transformers import Qwen3_5TextConfig
    cfg = dict(text_config)
    tc = Qwen3_5TextConfig(**cfg)
    tc.mtp_num_hidden_layers = 0
    return tc


def load_text_model_from_gguf(path: str | Path, *, dtype=torch.bfloat16,
                              text_config_path: str | Path | None = None,
                              layer_limit: int | None = None):
    """Small/prefix helper (dict path).  Full model -> ``load_text_model_streamed``."""
    state, meta = load_gguf_state(
        path, include_lm_head=False, dtype=dtype, layer_limit=layer_limit)
    if text_config_path is None:
        text_config_path = Path(path).parent / "hf-head" / "config.json"
    cfg = json.loads(Path(text_config_path).read_text())["text_config"]
    if layer_limit is not None:
        cfg = dict(cfg)
        cfg["num_hidden_layers"] = layer_limit
        cfg["layer_types"] = list(cfg["layer_types"])[:layer_limit]
    model, missing, unexpected = build_text_model(state, cfg, dtype=dtype)
    return model, meta, missing, unexpected


if __name__ == "__main__":
    p = sys.argv[1]
    meta: dict = {}
    n = 0
    for name, ten in iter_hf_tensors(p, layer_limit=int(sys.argv[2]) if len(sys.argv) > 2 else None, meta=meta):
        n += ten.numel()
    print(json.dumps(meta), f"{n/1e9:.3f}B params, {n*2/2**30:.2f} GiB at bf16")
