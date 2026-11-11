#!/usr/bin/env python3
"""Code-level export audit for the Scion ternary expert banks.

The gate from v2/PLAN.md section 13.1: the deployed GGUF must carry the same
ternary *codes* the corrections were trained against.  The training pipeline
quantises the HF BF16 body with the Python Lloyd g128 rule
(``moe/moe_proxy.py:ternary_lloyd``); the release body was packed by the C++
``GGUF_PQ2_0_LLOYD`` path.  This script decodes the deployed type-142 banks,
recomputes the Python training-time codes from the BF16 source GGUF, and
reports mismatches per tensor and per million weights.

It also validates the embedded branch factors (type 43, the legacy id that is
byte-identical to Q2_0_g128): every code must be in {0,1,2} (the packer never
emits the fourth state) and every group scale must equal max|dequant|.

Usage:
  python v2/scripts/export_audit.py \
      --deployed models-p310/scion-strata/Scion-35B-A3B-PQ2_0-corr.gguf \
      --source   models-p310/bf16/Qwen3.8-35B-A3B-Distill-BF16.gguf \
      --out v2/export-audit/audit.json --device cuda:1

  # branch-factor validation only (no source needed)
  python v2/scripts/export_audit.py --deployed <model.gguf> --branches-only
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "moe"))

# ggml type -> (elements per block, bytes per block)
TYPE_BLOCK = {
    0: (1, 4),        # F32
    1: (1, 2),        # F16
    8: (32, 34),      # Q8_0
    30: (1, 2),       # BF16
    43: (128, 34),    # legacy Q1_0_g128 == Q2_0_g128 layout
    142: (128, 34),   # Q2_0_g128 (formerly PQ2_0)
    143: (128, 28),   # TQ1_0_g128
}
TYPE_NAME = {0: "F32", 1: "F16", 8: "Q8_0", 30: "BF16",
             43: "Q1_0_g128", 142: "Q2_0_g128", 143: "TQ1_0_g128"}


# --------------------------------------------------------------- gguf parse --
def _rd(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]


def _string(f):
    return f.read(_rd(f, "Q")).decode("utf-8", "replace")


def _value(f, t):
    if t == 0:
        return _rd(f, "B")
    if t == 1:
        return _rd(f, "b")
    if t == 2:
        return _rd(f, "H")
    if t == 3:
        return _rd(f, "h")
    if t == 4:
        return _rd(f, "I")
    if t == 5:
        return _rd(f, "i")
    if t == 6:
        return _rd(f, "f")
    if t == 7:
        return bool(_rd(f, "B"))
    if t == 8:
        return _string(f)
    if t == 9:
        et = _rd(f, "I")
        return [_value(f, et) for _ in range(_rd(f, "Q"))]
    if t == 10:
        return _rd(f, "Q")
    if t == 11:
        return _rd(f, "q")
    if t == 12:
        return _rd(f, "d")
    raise ValueError(f"GGUF KV type {t}")


def read_header(path: Path) -> dict:
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF", path
        version = _rd(f, "I")
        n_tensors, n_kv = _rd(f, "Q"), _rd(f, "Q")
        kv = {}
        for _ in range(n_kv):
            k = _string(f)
            kv[k] = _value(f, _rd(f, "I"))
        tensors = []
        for _ in range(n_tensors):
            name = _string(f)
            nd = _rd(f, "I")
            dims = [_rd(f, "Q") for _ in range(nd)]
            ttype = _rd(f, "I")
            offset = _rd(f, "Q")
            n = int(np.prod(dims)) if dims else 1
            tensors.append({"name": name, "dims": dims, "type": ttype,
                            "offset": offset, "n": n})
        align = int(kv.get("general.alignment", 32))
        data_start = (f.tell() + align - 1) // align * align
    return {"path": str(path), "version": version, "kv": kv,
            "tensors": tensors, "data_start": data_start}


def tensor_nbytes(t: dict) -> int:
    if t["type"] not in TYPE_BLOCK:
        raise SystemExit(f"unsupported type {t['type']} on {t['name']}")
    be, bb = TYPE_BLOCK[t["type"]]
    assert t["n"] % be == 0, (t["name"], t["n"], be)
    return t["n"] // be * bb


def read_tensor(path: Path, h: dict, t: dict) -> bytes:
    with open(path, "rb") as f:
        f.seek(h["data_start"] + t["offset"])
        b = f.read(tensor_nbytes(t))
    if len(b) != tensor_nbytes(t):
        raise SystemExit(f"short read on {t['name']}")
    return b


# ------------------------------------------------------------- codec helpers --
def decode_q2_0_g128(raw: bytes) -> tuple[np.ndarray, np.ndarray, int]:
    """Decode fp16 scale + 32-byte 2-bit blocks -> (codes, scales, n_code3)."""
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 34)
    scales = blocks[:, :2].copy().view("<f2").reshape(-1).astype(np.float32)
    qb = blocks[:, 2:]
    codes = np.empty((blocks.shape[0], 128), dtype=np.int8)
    for j in range(4):
        codes[:, j::4] = ((qb >> (2 * j)) & 0x03).astype(np.int8) - 1
    n_code3 = int((codes == 2).sum())
    return codes, scales, n_code3


def source_to_float(raw: bytes, ttype: int) -> np.ndarray:
    if ttype == 30:      # BF16
        u = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
        return u.view(np.float32)
    if ttype == 1:       # F16
        return np.frombuffer(raw, dtype="<f2").astype(np.float32)
    if ttype == 0:       # F32
        return np.frombuffer(raw, dtype="<f4").copy()
    raise SystemExit(f"source type {ttype} not supported for code recompute")


def lloyd_codes(w: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Training-time Python Lloyd g128 codes for a flat weight array."""
    import torch
    from moe_proxy import _lloyd_scale

    x = torch.from_numpy(np.ascontiguousarray(w)).to(device).float()
    g = x.reshape(-1, 128)
    a = _lloyd_scale(g, g.abs().mean(-1))
    a = a.half().float()
    q = torch.clamp(torch.round(g / a[:, None].clamp_min(1e-12)), -1, 1)
    return q.to(torch.int8).cpu().numpy(), a.half().cpu().numpy()


# ------------------------------------------------------------------- audits --
def audit_banks(deployed: Path, source: Path, layers: list[int] | None,
                device: str) -> dict:
    hd = read_header(deployed)
    hs = read_header(source)
    sd = {t["name"]: t for t in hs["tensors"]}
    rows = []
    totals = {"tensors": 0, "weights": 0, "groups": 0, "code3": 0,
              "code_mismatch": 0, "nonzero_weights": 0,
              "code_mismatch_nonzero": 0, "scale_mismatch_groups": 0,
              "scale_mismatch_nonzero_groups": 0,
              "zero_state_disagreement_groups": 0,
              "code_mismatch_scale_equal": 0, "code_mismatch_scale_diff": 0,
              "weights_scale_equal": 0, "weights_scale_diff": 0,
              "sum_abs_err": 0.0, "max_abs_err": 0.0}
    names = [t["name"] for t in hd["tensors"]
             if t["name"].endswith("_exps.weight")]
    if layers is not None:
        names = [n for n in names
                 if int(n.split(".")[1]) in set(layers)]
    for name in sorted(names):
        t = next(x for x in hd["tensors"] if x["name"] == name)
        s = sd.get(name)
        if s is None:
            raise SystemExit(f"source is missing {name}")
        if t["dims"] != s["dims"]:
            raise SystemExit(f"dim mismatch {name}: {t['dims']} vs {s['dims']}")
        raw_d = read_tensor(deployed, hd, t)
        codes_d, scales_d, code3 = decode_q2_0_g128(raw_d)
        raw_s = read_tensor(source, hs, s)
        w = source_to_float(raw_s, s["type"])
        codes_s, scales_s = lloyd_codes(w, device)
        if codes_d.shape != codes_s.shape:
            raise SystemExit(f"shape mismatch {name}")
        mismatch = (codes_d != codes_s)
        # scales_d is float32 (decoded from fp16); scales_s is fp16.  Compare
        # the exact fp16 bit patterns after the lossless round-trip.
        scale_mis = (scales_d.astype(np.float16).view(np.uint16)
                     != scales_s.view(np.uint16))
        # Groups whose source is BF16-denormal / zero get fp16 scale 0 from
        # both quantizers; they are not informative for code fidelity.
        nonzero = (scales_d != 0) | (scales_s != 0)
        zero_state = (scales_d == 0) != (scales_s == 0)
        same_scale = ~scale_mis
        err = np.abs(codes_d.astype(np.float32) * scales_d[:, None] - w.reshape(-1, 128))
        row = {
            "name": name,
            "type": TYPE_NAME.get(t["type"], str(t["type"])),
            "dims": t["dims"],
            "weights": int(t["n"]),
            "groups": int(codes_d.shape[0]),
            "zero_scale_groups": int((~nonzero).sum()),
            "zero_scale_weights": int((~nonzero).sum() * 128),
            "code_mismatch": int(mismatch.sum()),
            "mismatch_ppm": float(mismatch.sum() * 1e6 / t["n"]),
            "nonzero_weights": int(nonzero.sum() * 128),
            "code_mismatch_nonzero": int(mismatch[nonzero].sum()),
            "mismatch_nonzero_ppm": float(mismatch[nonzero].sum() * 1e6
                                          / max(int(nonzero.sum() * 128), 1)),
            "scale_mismatch_groups": int(scale_mis.sum()),
            "scale_mismatch_ppm_groups": float(scale_mis.sum() * 1e6
                                               / codes_d.shape[0]),
            "scale_mismatch_nonzero_groups": int(scale_mis[nonzero].sum()),
            "scale_mismatch_nonzero_ppm": float(scale_mis[nonzero].sum() * 1e6
                                                / max(int(nonzero.sum()), 1)),
            "zero_state_disagreement_groups": int(zero_state.sum()),
            "zero_group_code_mismatch": int(mismatch[~nonzero].sum()),
            "code_mismatch_scale_equal": int(mismatch[same_scale].sum()),
            "code_mismatch_scale_diff": int(mismatch[~same_scale].sum()),
            "weights_scale_equal": int(same_scale.sum() * 128),
            "weights_scale_diff": int((~same_scale).sum() * 128),
            "code3": code3,
            "mean_abs_dequant_err": float(err.mean()),
            "max_abs_dequant_err": float(err.max()),
            "max_abs_err": float(err.max()),
        }
        rows.append(row)
        totals["tensors"] += 1
        totals["weights"] += int(t["n"])
        totals["groups"] += row["groups"]
        totals["code_mismatch"] += row["code_mismatch"]
        totals["nonzero_weights"] += row["nonzero_weights"]
        totals["code_mismatch_nonzero"] += row["code_mismatch_nonzero"]
        totals["scale_mismatch_groups"] += row["scale_mismatch_groups"]
        totals["scale_mismatch_nonzero_groups"] += row["scale_mismatch_nonzero_groups"]
        totals["zero_state_disagreement_groups"] += row["zero_state_disagreement_groups"]
        totals["code_mismatch_scale_equal"] += row["code_mismatch_scale_equal"]
        totals["code_mismatch_scale_diff"] += row["code_mismatch_scale_diff"]
        totals["weights_scale_equal"] += row["weights_scale_equal"]
        totals["weights_scale_diff"] += row["weights_scale_diff"]
        totals["code3"] += code3
        totals["sum_abs_err"] += float(err.sum())
        totals["max_abs_err"] = max(totals["max_abs_err"], row["max_abs_err"])
        print(f"{name:32s} ppm={row['mismatch_ppm']:10.2f} "
              f"nz_ppm={row['mismatch_nonzero_ppm']:10.2f} "
              f"scale_nz_ppm={row['scale_mismatch_nonzero_ppm']:8.2f} "
              f"code3={code3}", flush=True)
        del raw_d, raw_s, w, codes_d, codes_s, err
    totals["mismatch_ppm"] = (totals["code_mismatch"] * 1e6
                              / max(totals["weights"], 1))
    totals["mismatch_nonzero_ppm"] = (
        totals["code_mismatch_nonzero"] * 1e6
        / max(totals["nonzero_weights"], 1))
    totals["mismatch_scale_equal_ppm"] = (
        totals["code_mismatch_scale_equal"] * 1e6
        / max(totals["weights_scale_equal"], 1))
    totals["mismatch_scale_diff_ppm"] = (
        totals["code_mismatch_scale_diff"] * 1e6
        / max(totals["weights_scale_diff"], 1))
    totals["scale_mismatch_ppm_groups"] = (
        totals["scale_mismatch_groups"] * 1e6 / max(totals["groups"], 1))
    totals["scale_mismatch_nonzero_ppm"] = (
        totals["scale_mismatch_nonzero_groups"] * 1e6
        / max(totals["groups"]
              - (totals["weights"] - totals["nonzero_weights"]) // 128, 1))
    totals["mean_abs_dequant_err"] = (totals["sum_abs_err"]
                                      / max(totals["weights"], 1))
    return {"deployed": str(deployed), "source": str(source),
            "rows": rows, "totals": totals}


def selfcheck_banks(deployed: Path, layers: list[int] | None,
                    device: str) -> dict:
    """Re-quantise the deployed dequantised banks with the training rule.

    This is the remediation proof: if the frozen body handed to correction
    training is seeded from the deployed GGUF (``moe/seed_banks_from_gguf.py``),
    the training-time Python Lloyd must return the deployed codes.  We test
    both the exact float32 dequant and the bf16 round-trip the HF model
    actually stores.
    """
    import torch

    hd = read_header(deployed)
    names = [t["name"] for t in hd["tensors"]
             if t["name"].endswith("_exps.weight")]
    if layers is not None:
        names = [n for n in names if int(n.split(".")[1]) in set(layers)]
    totals = {"tensors": 0, "weights": 0, "f32_mismatch": 0,
              "bf16_mismatch": 0, "f32_scale_mismatch_groups": 0,
              "bf16_scale_mismatch_groups": 0}
    for name in sorted(names):
        t = next(x for x in hd["tensors"] if x["name"] == name)
        codes_d, scales_d, _ = decode_q2_0_g128(read_tensor(deployed, hd, t))
        w = codes_d.astype(np.float32) * scales_d[:, None]
        # f32: exact deployed values
        c32, s32 = lloyd_codes(w.reshape(-1), device)
        # bf16: what the HF checkpoint stores
        wb = (torch.from_numpy(w).to(torch.bfloat16).float().numpy())
        cb, sb = lloyd_codes(wb.reshape(-1), device)
        totals["tensors"] += 1
        totals["weights"] += int(t["n"])
        totals["f32_mismatch"] += int((c32 != codes_d).sum())
        totals["bf16_mismatch"] += int((cb != codes_d).sum())
        totals["f32_scale_mismatch_groups"] += int(
            (s32.view(np.uint16)
             != scales_d.astype(np.float16).view(np.uint16)).sum())
        totals["bf16_scale_mismatch_groups"] += int(
            (sb.view(np.uint16)
             != scales_d.astype(np.float16).view(np.uint16)).sum())
        print(f"selfcheck {name:32s} f32={int((c32 != codes_d).sum())} "
              f"bf16={int((cb != codes_d).sum())}", flush=True)
        del w, wb, c32, s32, cb, sb, codes_d, scales_d
    totals["f32_mismatch_ppm"] = (totals["f32_mismatch"] * 1e6
                                  / max(totals["weights"], 1))
    totals["bf16_mismatch_ppm"] = (totals["bf16_mismatch"] * 1e6
                                   / max(totals["weights"], 1))
    return totals


def validate_branches(deployed: Path) -> dict:
    h = read_header(deployed)
    rows = []
    totals = {"tensors": 0, "weights": 0, "code3": 0,
              "groups": 0, "scale_not_max": 0, "bad_scale": 0}
    for t in h["tensors"]:
        if t["type"] not in (43, 142) or ".lora_" not in t["name"]:
            continue
        if t["type"] == 142:
            continue
        raw = read_tensor(deployed, h, t)
        codes, scales, code3 = decode_q2_0_g128(raw)
        # packed values are q*scale; the packer chooses scale = max|w| of the
        # group, so every group must contain a +/-1 code and the fp16 scale
        # must equal the group max of the deployed values.
        rowmax = np.abs(codes).max(axis=1)
        # scale == max|q*scale| iff the group has at least one +/-1
        not_max = (rowmax == 0) & (scales != 0)
        bad_scale = (scales < 0) | ~np.isfinite(scales)
        rows.append({"name": t["name"], "weights": int(t["n"]),
                     "groups": int(codes.shape[0]), "code3": code3,
                     "groups_without_pm1": int(not_max.sum())})
        totals["tensors"] += 1
        totals["weights"] += int(t["n"])
        totals["groups"] += int(codes.shape[0])
        totals["code3"] += code3
        totals["scale_not_max"] += int(not_max.sum())
        totals["bad_scale"] += int(bad_scale.sum())
    return {"rows": rows, "totals": totals}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deployed", required=True)
    ap.add_argument("--source", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cpu",
                    help="torch device for the Python Lloyd recompute")
    ap.add_argument("--layers", default="",
                    help="comma-separated layer indices (default: all)")
    ap.add_argument("--branches-only", action="store_true")
    ap.add_argument("--self-check", action="store_true",
                    help="also re-quantise the deployed dequantised banks "
                         "with the training rule (remediation proof)")
    args = ap.parse_args()

    out = {"branches": validate_branches(Path(args.deployed))}
    print("branches:", json.dumps(out["branches"]["totals"]), flush=True)
    if not args.branches_only:
        if not args.source:
            raise SystemExit("--source is required unless --branches-only")
        layers = ([int(x) for x in args.layers.split(",") if x] or None)
        out["banks"] = audit_banks(Path(args.deployed), Path(args.source),
                                   layers, args.device)
        print("banks:", json.dumps(out["banks"]["totals"]), flush=True)
    if args.self_check:
        layers = ([int(x) for x in args.layers.split(",") if x] or None)
        out["selfcheck"] = selfcheck_banks(Path(args.deployed), layers,
                                           args.device)
        print("selfcheck:", json.dumps(out["selfcheck"]), flush=True)
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=1))
        print("wrote", p)


if __name__ == "__main__":
    main()
