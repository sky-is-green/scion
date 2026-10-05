"""Numerically verify a PLE-inclusive export against the Python source path.

Reads the GGUF's PLE group (Q4_0 n-gram table rows, the 6 small tensors, the
ple.* KV) and compares against the official shards through the same code the
Python gates use (``load_ple_rows`` / ``ple_kv_values``).  This is the
export-path proof that does not need a loadable fork model depth.

Usage:
    python moe/ple_export_verify.py <file.gguf> <model_dir> [n_rows]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/penis/Desktop/work/llama-qwen4exp/gguf-py")
import gguf  # noqa: E402
from gguf.quants import dequantize  # noqa: E402
from safetensors import safe_open  # noqa: E402
import qwen4exp_proxy as q4  # noqa: E402
import qwen4exp_export as qx  # noqa: E402
from argparse import Namespace  # noqa: E402


def main():
    path, model_dir = sys.argv[1], sys.argv[2]
    n_rows = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    reader = gguf.GGUFReader(path)
    by_name = {t.name: t for t in reader.tensors}
    meta = {f.name: f.contents() for f in reader.fields.values()}

    # 1. PLE KV vs the runtime builder
    ex = qx.Exporter(Namespace(
        model_dir=model_dir, layers=48, experts="ptq1_0", body="f16",
        branches="none", branch_dtype="f16", branch_quant="g128",
        deploy_quant="lloyd", routers="replace", adapter_recipe="",
        use_temp_file=False, release_shards="none", keep_shards="",
        out="/dev/null", ple="q4_0"))
    v = ex.ple_kv_values()
    errs = []
    for key, expect in (("qwen4exp.ple.layers", [v["layers"]]),
                        ("qwen4exp.ple.layer_multipliers",
                         [v["layer_multipliers"]]),
                        ("qwen4exp.ple.head_offsets", [v["head_offsets"]]),
                        ("qwen4exp.ple.head_vocab_sizes",
                         [v["head_vocab_sizes"]])):
        got = list(meta[key])
        if got != expect[0]:
            errs.append(f"KV {key}: got {got[:4]}... want {expect[0][:4]}...")
    for key, val in (("qwen4exp.ple.ngram_size", v["ngram_size"]),
                     ("qwen4exp.ple.heads_per_ngram", v["heads_per_ngram"]),
                     ("qwen4exp.ple.conv_kernel", v["conv_kernel"]),
                     ("qwen4exp.ple.eos_token_id", v["eos_token_id"]),
                     ("qwen4exp.embedding_length_per_layer_input",
                      v["head_dim"])):
        if int(meta[key]) != int(val):
            errs.append(f"KV {key}: got {meta[key]} want {val}")
    print(f"PLE KV: {'OK' if not errs else errs}")

    # 2. Table rows: GGUF Q4_0 vs load_ple_rows (same ids)
    idx = ex.weight_map
    ids = np.linspace(0, v["rows"] - 1, n_rows, dtype=np.int64)
    handles = {}

    def open_shard(shard):
        if shard not in handles:
            handles[shard] = safe_open(str(Path(model_dir) / "shards" / shard),
                                       framework="pt", device="cpu")
        return handles[shard]

    _, py_rows = q4.load_ple_rows(
        __import__("torch").from_numpy(ids), ex.ple_layer, idx, open_shard)
    py = py_rows.float().numpy()
    t = by_name["per_layer_token_embd.weight"]
    head_dim = v["head_dim"]
    row_bytes = head_dim // 32 * 18
    qtype = t.tensor_type
    with open(path, "rb") as f:
        gg = np.empty_like(py)
        for i, rid in enumerate(ids):
            f.seek(int(t.data_offset) + int(rid) * row_bytes)
            raw = np.frombuffer(f.read(row_bytes), dtype=np.uint8)
            gg[i] = dequantize(raw.reshape(row_bytes // 18, 18), qtype
                               ).reshape(-1)[:head_dim]
    diff = np.abs(py - gg)
    print(f"table: {n_rows} rows, type {qtype.name}, "
          f"max|d| {diff.max():.4f} mean|d| {diff.mean():.4f}")
    if diff.max() > 1.5:
        errs.append(f"table row mismatch {diff.max():.3f}")
    # spot check: rows must come from the right shard offsets
    if not np.isfinite(gg).all():
        errs.append("table has non-finite dequant")

    # 3. Small PLE tensors vs HF (transpose rules)
    checks = {
        "blk.1.ple_key.weight": ("model.language_model.layers.1.ple.key_proj.weight", "D"),
        "blk.1.ple_value.weight": ("model.language_model.layers.1.ple.value_proj.weight", "D"),
        "blk.1.ple_conv1d.weight": ("model.language_model.layers.1.ple.conv1d.weight", "S"),
        "blk.1.ple_norm_conv.weight": ("model.language_model.layers.1.ple.norm_conv.weight", "O"),
        "blk.1.ple_norm_key.weight": ("model.language_model.layers.1.ple.norm_key.weight", "O"),
        "blk.1.ple_norm_query.weight": ("model.language_model.layers.1.ple.norm_query.weight", "O"),
    }
    for gname, (hf_key, mode) in checks.items():
        t = by_name[gname]
        if t.tensor_type in (gguf.GGMLQuantizationType.Q4_0,
                             gguf.GGMLQuantizationType.Q6_K,
                             gguf.GGMLQuantizationType.Q8_0):
            g = dequantize(t.data, t.tensor_type).astype(np.float32)
        else:
            g = t.data.astype(np.float32)
        h = open_shard(idx[hf_key]).get_tensor(hf_key).float().numpy()
        if mode == "S":
            h = h.squeeze(1)
        elif mode == "O":
            h = h + 1.0
        d = np.abs(g - h)
        rel = d.max() / (np.abs(h).max() + 1e-9)
        status = "OK" if rel < 0.02 else "MISMATCH"
        if status != "OK":
            errs.append(f"{gname}: rel err {rel:.4f}")
        print(f"{gname}: shape {g.shape} vs {h.shape} max|d| {d.max():.4f} "
              f"rel {rel:.5f} {status}")

    # 4. adapter factors must be finite: a table registered before the adapter
    #    shifts offsets and the runtime reads table bytes as the adapter (NaN)
    n_bad = 0
    for t in reader.tensors:
        if ".lora_" not in t.name:
            continue
        a = t.data
        if a.dtype == np.uint8:
            a = a.view(np.float16)
        if not np.isfinite(a.astype(np.float32)).all():
            n_bad += 1
            errs.append(f"{t.name}: non-finite adapter values")
    print(f"adapter tensors non-finite: {n_bad}")

    print("VERIFY", "OK" if not errs else f"FAILED: {errs}")
    sys.exit(0 if not errs else 1)


if __name__ == "__main__":
    main()
