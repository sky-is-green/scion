"""Parse back an exported qwen4exp GGUF: inventory, types, KV, dims.

Usage: python moe/export_check.py <file.gguf>
Compares tensor names/shapes/types against the ISTA reference conventions
(experts [cols, ff, E] PTQ1_0/PQ2_0; the PLE group is optional -- when the
ple.layers KV is present the 6 blk.N.ple_* tensors and the
per_layer_token_embd table are required, otherwise absent) and fails loudly
on any mismatch the loader would reject.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gguf_header as G


def parse_local(path, budget=128 << 20):
    with open(path, "rb") as f:
        b = f.read(budget)
    o = 4
    assert b[:4] == b"GGUF", "not a GGUF file"
    version, o = G._u32(b, o)
    n_tensors, o = G._u64(b, o)
    n_kv, o = G._u64(b, o)
    meta = {}
    for _ in range(n_kv):
        k, o = G._s(b, o)
        t, o = G._u32(b, o)
        v, o = G._val(b, o, t)
        meta[k] = v
    tensors = []
    for _ in range(n_tensors):
        name, o = G._s(b, o)
        nd, o = G._u32(b, o)
        dims = []
        for _ in range(nd):
            d, o = G._u64(b, o)
            dims.append(d)
        t, o = G._u32(b, o)
        off, o = G._u64(b, o)
        n = 1
        for d in dims:
            n *= d
        tensors.append({"name": name, "dims": dims,
                        "type": G.GGML_TYPES.get(t, str(t)), "n": n})
    return meta, tensors


REQUIRED_KV = ["general.architecture", "qwen4exp.block_count",
               "qwen4exp.embedding_length", "qwen4exp.expert_count",
               "qwen4exp.expert_used_count",
               "qwen4exp.attention.head_count",
               "qwen4exp.ssm.inner_size", "qwen4exp.hyper_connection.count",
               "tokenizer.ggml.tokens", "tokenizer.ggml.merges"]


def main(path: str):
    meta, tensors = parse_local(path)
    errs = []
    for k in REQUIRED_KV:
        if k not in meta:
            errs.append(f"missing KV {k}")
    if meta.get("general.architecture") != "qwen4exp":
        errs.append(f"arch is {meta.get('general.architecture')}")
    names = {t["name"] for t in tensors}
    table = next((t for t in tensors
                  if t["name"] == "per_layer_token_embd.weight"), None)
    if "qwen4exp.ple.layers" in meta:
        ple_layers = meta["qwen4exp.ple.layers"]
        if isinstance(ple_layers, dict):
            ple_layers = ple_layers["head"]
        il = ple_layers[0]
        for k in ("ngram_size", "heads_per_ngram", "conv_kernel",
                  "eos_token_id", "layer_multipliers", "head_offsets",
                  "head_vocab_sizes"):
            if f"qwen4exp.ple.{k}" not in meta:
                errs.append(f"PLE build missing KV qwen4exp.ple.{k}")
        for suffix in ("ple_conv1d", "ple_key", "ple_value", "ple_norm_conv",
                       "ple_norm_key", "ple_norm_query"):
            if f"blk.{il}.{suffix}.weight" not in names:
                errs.append(f"PLE build missing blk.{il}.{suffix}.weight")
        if table is None:
            errs.append("PLE build missing per_layer_token_embd.weight")
        else:
            if table["type"] not in ("Q4_0", "IQ4_NL", "F16", "Q8_0"):
                errs.append(f"PLE table type {table['type']}")
            if table["dims"][0] != 160:
                errs.append(f"PLE table head dim {table['dims'][0]} != 160")
            total_rows = int(table["dims"][1])
            vs = meta.get("qwen4exp.ple.head_vocab_sizes")
            os_ = meta.get("qwen4exp.ple.head_offsets")
            if isinstance(vs, dict) and isinstance(os_, dict):
                if vs["len"] != 16 or os_["len"] != 16:
                    errs.append(f"PLE head arrays len {vs['len']}/"
                                f"{os_['len']} != 16")
                used = max(o + s for o, s in zip(os_["head"], vs["head"]))
                if used > total_rows:
                    errs.append(f"PLE head ranges {used} > table rows "
                                f"{total_rows}")
    elif table is not None:
        errs.append("PLE table present but no ple.layers KV")
    n_layer = int(meta.get("qwen4exp.block_count", -1))
    for il in range(n_layer):
        for suffix in ("ffn_gate_exps.weight", "ffn_up_exps.weight",
                       "ffn_down_exps.weight", "ffn_gate_inp.weight"):
            if f"blk.{il}.{suffix}" not in names:
                errs.append(f"missing blk.{il}.{suffix}")
    for t in tensors:
        if t["name"].endswith("_exps.weight") and t["type"] not in (
                "PTQ1_0", "PQ2_0"):
            errs.append(f"{t['name']} has type {t['type']}")
    if meta.get("adapter.embedded", False):
        if meta.get("adapter.type") != "lora":
            errs.append(f"adapter.type is {meta.get('adapter.type')}")
        stems = {}
        for t in tensors:
            for suffix, slot in ((".lora_a", "a"), (".lora_b", "b")):
                if t["name"].endswith(suffix):
                    stems.setdefault(t["name"][:-len(suffix)], {})[slot] = t
        if not stems:
            errs.append("adapter.embedded set but no .lora_a/.lora_b tensors")
        by_name = {t["name"]: t for t in tensors}
        for stem, slots in sorted(stems.items()):
            if set(slots) != {"a", "b"}:
                errs.append(f"{stem}: incomplete lora pair")
                continue
            a, b = slots["a"]["dims"], slots["b"]["dims"]
            if a[1] != b[0]:
                errs.append(f"{stem}: rank mismatch {a} x {b}")
            if any(slots[s]["type"] not in ("F16", "Q1_0_g128")
                   for s in slots):
                errs.append(f"{stem}: factor type "
                            f"{slots['a']['type']}/{slots['b']['type']}")
            target = stem.split(".", 2)[-1]
            if target not in ("attn_output.weight", "ssm_out.weight",
                              "ffn_moe_out.weight"):
                errs.append(f"{stem}: unexpected lora target")
            elif target == "ffn_moe_out.weight":
                # activation target: both sides are n_embd; the anchor only
                # supplies the device (llama-adapter.cpp is_moe_corr check)
                n_embd = int(meta.get("qwen4exp.embedding_length", -1))
                if a[0] != n_embd or b[-1] != n_embd:
                    errs.append(f"{stem}: dims {a} x {b} not n_embd {n_embd}")
                base_name = stem.replace(".ffn_moe_out.weight",
                                         ".ffn_gate_inp.weight")
                if base_name not in by_name:
                    errs.append(f"{stem}: anchor {base_name} missing")
            else:
                base = by_name.get(stem)
                if base is None:
                    errs.append(f"{stem}: base tensor missing")
                elif base["dims"] != [a[0], b[-1]]:
                    errs.append(f"{stem}: base dims {base['dims']} do not "
                                f"match ({a[0]}, {b[-1]})")
        print(f"adapter: {len(stems)} embedded lora pair(s), "
              f"type={meta.get('adapter.type')}")
    total = sum(t["n"] for t in tensors)
    print(f"{path}: {len(tensors)} tensors, {total / 1e9:.2f} B params, "
          f"{len(meta)} KV")
    by_type: dict[str, int] = {}
    for t in tensors:
        by_type[t["type"]] = by_type.get(t["type"], 0) + 1
    for ty, c in sorted(by_type.items()):
        print(f"  {ty}: {c}")
    if errs:
        print("ERRORS:")
        for e in errs:
            print(f"  - {e}")
        sys.exit(1)
    print("parse-back OK")


if __name__ == "__main__":
    main(sys.argv[1])
