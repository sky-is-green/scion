"""MoE role map + ternary projection for the Qwen3.5/3.6-MoE family.

Joins:
  - the llama.cpp tensor inventory (from a remote GGUF header census), and
  - an APEX per-tensor precision config (optional),

and emits the Bonsai-style role map: which tensors are ternary+rotated, which
are precision-exempt but basis-absorbed, which stay raw F32, and the projected
artifact size.

Usage:
  python role_map.py artifacts/ternary/moe/empero-bf16-header.json \
      --apex artifacts/ternary/moe/apex-carnice-qwen36-mtp-micro.txt \
      --out artifacts/ternary/moe/role-map.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter

# --- role assignment -------------------------------------------------------

# Ternary, input-axis rotated (Prism: rotate ne0/last axis of every packed tensor)
TERNARY_INPUT = {
    "ffn_gate_exps", "ffn_up_exps",              # routed experts, in=2048
    "attn_qkv", "attn_gate", "attn_q", "attn_k", "attn_v",  # in=2048
}
# Ternary, input-axis rotated but small input (down projections rotate their 512 in-features)
TERNARY_INPUT_SMALL = {"ffn_down_exps"}          # in=512
# Ternary, output-side projections (still rotate their input axis = 4096)
TERNARY_OUTPUT = {"attn_output", "ssm_out"}      # in=4096
# Ternary with inverse-rotation path
TERNARY_EMBED = {"token_embd.weight", "output.weight"}
# Precision-exempt, but hidden-axis consumers -> must absorb R (F1b rule)
ABSORB_EXEMPT = {"ffn_gate_inp", "ffn_gate_inp_shexp", "ssm_alpha", "ssm_beta",
                 "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp"}
# Raw F32, no rotation (1-D controls and norms)
RAW_F32 = {"ssm_a", "ssm_dt", "ssm_conv1d", "attn_norm", "post_attention_norm",
           "output_norm", "ssm_norm", "attn_q_norm", "attn_k_norm"}
EXCLUDE = ("visual", "mmproj", "mtp", "nextn")


def suffix_of(name: str) -> tuple[str, int]:
    m = re.match(r"blk\.(\d+)\.(.+)", name)
    if not m:
        return name, -1
    return ".".join(m.group(2).split(".")[:-1]), int(m.group(1))


def role(name: str) -> str:
    if any(p in name for p in EXCLUDE):
        return "excluded"
    s, _ = suffix_of(name)
    if s in TERNARY_INPUT or s in TERNARY_INPUT_SMALL or s in TERNARY_OUTPUT:
        return "ternary_rot"
    if s in TERNARY_EMBED:
        return "ternary_embed"
    if s in ABSORB_EXEMPT:
        return "absorb_exempt_fp16"
    if s in RAW_F32:
        return "raw_f32"
    return "unknown"


def rot_block(width: int) -> int:
    if width <= 0:
        return 0
    b = 1
    while b * 2 <= min(1024, width):
        b *= 2
    return b


def rot_width(t: dict) -> int:
    """ne0 in ggml = in_features in HF (row-major [out,in] -> fastest = in)."""
    return t["dims"][0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("header_json")
    ap.add_argument("--apex", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = json.load(open(args.header_json))
    tensors = d["tensors"]

    apex = {}
    if args.apex:
        for line in open(args.apex):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            k, _, v = line.partition("=")
            apex[k] = v

    rows = []
    for t in tensors:
        r = role(t["name"])
        s, layer = suffix_of(t["name"])
        w = rot_width(t)
        rows.append({"name": t["name"], "suffix": s, "layer": layer, "role": r,
                     "dims": t["dims"], "n": t["n"], "type": t["type"],
                     "rot_width": w if r.startswith("ternary") else 0,
                     "rot_block": rot_block(w) if r.startswith("ternary") else 0,
                     "apex_type": apex.get(t["name"], apex.get(t["name"][:-7], ""))})

    # --- ternary projection -------------------------------------------------
    q_bpw = 2.125          # PQ2_0: 34 bytes / 128 weights
    fp16_bpw = 16.0
    f32_bpw = 32.0
    proj = Counter()
    for row in rows:
        if row["role"] in ("ternary_rot", "ternary_embed"):
            proj["ternary_params"] += row["n"]
            proj["ternary_bits"] += row["n"] * q_bpw
        elif row["role"] == "absorb_exempt_fp16":
            proj["fp16_params"] += row["n"]
            proj["fp16_bits"] += row["n"] * fp16_bpw
        elif row["role"] == "raw_f32":
            proj["f32_params"] += row["n"]
            proj["f32_bits"] += row["n"] * f32_bpw
        else:
            proj["unknown_params"] += row["n"]

    total_bytes = (proj["ternary_bits"] + proj["fp16_bits"] + proj["f32_bits"]) / 8
    total_params = sum(t["n"] for t in tensors)
    summary = {
        "tensors": len(tensors),
        "total_params": total_params,
        "ternary_params": proj["ternary_params"],
        "fp16_params": proj["fp16_params"],
        "f32_params": proj["f32_params"],
        "unknown_params": proj["unknown_params"],
        "projected_bytes": int(total_bytes),
        "projected_gib": round(total_bytes / (1 << 30), 3),
        "effective_bpw": round(total_bytes * 8 / total_params, 3),
        "roles": dict(Counter(r["role"] for r in rows)),
    }

    # per-layer rotation-width census
    widths = Counter(r["rot_width"] for r in rows if r["role"].startswith("ternary"))
    summary["rotation_widths"] = dict(sorted(widths.items()))
    summary["rotation_blocks"] = {str(w): rot_block(w) for w in widths}

    # --- APEX per-role precision table (if provided) ------------------------
    if apex:
        by_role = {}
        for row in rows:
            if not row["apex_type"]:
                continue
            key = (row["role"], row["apex_type"])
            by_role.setdefault(key, {"tensors": 0, "n": 0})
            by_role[key]["tensors"] += 1
            by_role[key]["n"] += row["n"]
        summary["apex_by_role_type"] = {
            f"{k[0]}:{k[1]}": v for k, v in sorted(by_role.items())
        }

    print(json.dumps(summary, indent=2))
    if args.out:
        json.dump({"summary": summary, "rows": rows}, open(args.out, "w"), indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
