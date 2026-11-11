#!/usr/bin/env python3
"""Literal C++ transcription of the PQ2_0 Lloyd g128 quantizer.

Purpose: prove which quantizer produced the deployed expert banks.  This is a
vectorised NumPy transcription of ``quantize_row_pq2_0_lloyd_ref`` and
``ggml_pq2_0_lloyd_scale`` from ``prism-moe-corr/ggml/src/ggml-quants.c``
(float32 sequential group sums, four starts, eight refinement iterations,
round-half-away codes, fp16 scale round-trip).  Compare its codes/scales with
the deployed GGUF:

  python v2/scripts/cpp_lloyd_check.py <deployed.gguf> <source-bf16.gguf> \
      --layers 0 --out v2/export-audit/cpp-parity.json

Expected result on the shipped 35B body: scales match bit-for-bit and code
mismatches are ~1 per million or fewer (float ties), i.e. the body is the C++
quantizer's output.  That is the baseline the Python training rule is compared
against in export_audit.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import export_audit as E  # noqa: E402


def cpp_lloyd_g128(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (codes int8 [ng,128], fp16 scales [ng]) for float32 groups."""
    n_groups, qk = x.shape
    assert qk == 128
    sum_abs = np.zeros(n_groups, np.float32)
    for j in range(qk):
        sum_abs += np.abs(x[:, j])                    # C: sequential float sum
    mean = (sum_abs / np.float32(qk)).astype(np.float32)

    best_a = mean.copy()
    best_obj = np.full(n_groups, -1.0, np.float32)
    for init in (0.5, 0.7, 0.9, 1.1):
        th = (np.float32(init) * mean).astype(np.float32)
        s1 = np.zeros(n_groups, np.float32)
        sw = np.zeros(n_groups, np.int32)
        for j in range(qk):
            a = np.abs(x[:, j])
            m = a > th
            s1 += np.where(m, a, 0.0).astype(np.float32)
            sw += m
        a = np.where(sw > 0, s1 / np.maximum(sw, 1), 0.0).astype(np.float32)
        for _ in range(8):
            th = (np.float32(0.5) * a).astype(np.float32)
            s1 = np.zeros(n_groups, np.float32)
            sw = np.zeros(n_groups, np.int32)
            for j in range(qk):
                aa = np.abs(x[:, j])
                m = aa > th
                s1 += np.where(m, aa, 0.0).astype(np.float32)
                sw += m
            a = np.where(sw > 0, s1 / np.maximum(sw, 1), 0.0).astype(np.float32)
        obj = np.where(sw > 0, s1 * s1 / np.maximum(sw, 1), 0.0).astype(np.float32)
        take = obj > best_obj
        best_obj = np.where(take, obj, best_obj)
        best_a = np.where(take, a, best_a)

    dq = best_a.astype(np.float16)
    ratio = x / np.maximum(dq.astype(np.float32)[:, None], np.float32(1e-30))
    q = np.clip(np.sign(ratio) * np.floor(np.abs(ratio) + np.float32(0.5)),
                -1, 1).astype(np.int8)
    return q, dq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("deployed")
    ap.add_argument("source")
    ap.add_argument("--layers", default="0")
    ap.add_argument("--kinds", default="gate,up,down")
    ap.add_argument("--out", default="")
    ap.add_argument("--max-experts", type=int, default=0,
                    help="limit to the first N experts (0 = all)")
    args = ap.parse_args()

    hd = E.read_header(Path(args.deployed))
    hs = E.read_header(Path(args.source))
    src = {t["name"]: t for t in hs["tensors"]}
    rows = []
    totals = {"weights": 0, "code_mismatch": 0,
              "scale_mismatch_groups": 0, "groups": 0}
    for layer in [int(x) for x in args.layers.split(",") if x]:
        for kind in args.kinds.split(","):
            name = f"blk.{layer}.ffn_{kind}_exps.weight"
            t = next(x for x in hd["tensors"] if x["name"] == name)
            s = src[name]
            cd, sd, _ = E.decode_q2_0_g128(E.read_tensor(Path(args.deployed), hd, t))
            w = E.source_to_float(E.read_tensor(Path(args.source), hs, s),
                                  s["type"]).reshape(-1, 128)
            if args.max_experts:
                groups = args.max_experts * (t["n"] // (256 * 128))
                w = w[:groups]
                cd = cd[:groups]
                sd = sd[:groups]
            t0 = time.time()
            q, d = cpp_lloyd_g128(np.ascontiguousarray(w))
            mm = int((q != cd).sum())
            sm = int((d.view(np.uint16) != sd.astype(np.float16).view(np.uint16)).sum())
            row = {"name": name, "weights": int(w.size), "groups": int(len(w)),
                   "code_mismatch": mm, "mismatch_ppm": mm * 1e6 / w.size,
                   "scale_mismatch_groups": sm,
                   "scale_mismatch_ppm": sm * 1e6 / len(w),
                   "elapsed_s": round(time.time() - t0, 2)}
            rows.append(row)
            totals["weights"] += int(w.size)
            totals["code_mismatch"] += mm
            totals["scale_mismatch_groups"] += sm
            totals["groups"] += int(len(w))
            print(json.dumps(row), flush=True)
            del w, cd, sd, q, d
    totals["mismatch_ppm"] = totals["code_mismatch"] * 1e6 / max(totals["weights"], 1)
    totals["scale_mismatch_ppm"] = (totals["scale_mismatch_groups"] * 1e6
                                    / max(totals["groups"], 1))
    out = {"deployed": args.deployed, "source": args.source,
           "rows": rows, "totals": totals}
    print("totals:", json.dumps(totals), flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
