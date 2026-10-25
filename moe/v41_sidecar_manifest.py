#!/usr/bin/env python3
"""Emit a role/type manifest for a V4.1 DSpark sidecar GGUF.

This reads GGUF metadata only; it does not load the model or allocate tensors.

  .venv-rocm/bin/python moe/v41_sidecar_manifest.py \
      --gguf path/to/V41-MTP-Vision.Q2_K.gguf \
      --gguf-py ~/llama.cpp/gguf-py --out sidecar-manifest.json
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path


def role_of(name: str) -> str:
    if ".ffn.experts." in name:
        return "expert"
    if ".ffn.shared_experts." in name:
        return "shared_expert"
    if ".ffn.gate." in name:
        return "router"
    if ".attn." in name:
        return "attention"
    if ".markov_head." in name:
        return "markov"
    if ".confidence_head." in name:
        return "confidence"
    if name.endswith("norm.weight") or ".norm." in name:
        return "norm"
    if ".hc_" in name:
        return "hyper_connection"
    if ".main_proj." in name:
        return "tap_projection"
    if ".main_norm." in name:
        return "tap_norm"
    return "other"


def tensor_row(t) -> dict:
    role = role_of(t.name)
    return {
        "name": t.name,
        "shape": [int(x) for x in t.shape],
        "type": t.tensor_type.name,
        "n_bytes": int(t.n_bytes),
        "role": role,
        "first_quant_candidate": role == "expert",
        "recommended": "ternary-candidate" if role == "expert" else "keep",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gguf", required=True, help="V4.1 DSpark sidecar GGUF")
    ap.add_argument("--gguf-py", default="", help="gguf-py directory")
    ap.add_argument("--out", default="", help="write JSON here; stdout when omitted")
    a = ap.parse_args()
    if a.gguf_py:
        sys.path.insert(0, a.gguf_py)
    try:
        from gguf import GGUFReader
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "gguf-py is not importable; pass --gguf-py <llama.cpp>/gguf-py"
        ) from exc

    path = Path(a.gguf)
    reader = GGUFReader(str(path))
    rows = [tensor_row(t) for t in reader.tensors]
    by_role = collections.defaultdict(lambda: {"tensors": 0, "bytes": 0})
    by_type = collections.Counter()
    for row in rows:
        entry = by_role[row["role"]]
        entry["tensors"] += 1
        entry["bytes"] += row["n_bytes"]
        by_type[row["type"]] += 1
    payload = {
        "schema": 1,
        "file": str(path),
        "file_bytes": path.stat().st_size,
        "tensors": len(rows),
        "summary": {
            "types": dict(sorted(by_type.items())),
            "roles": dict(sorted(by_role.items())),
            "ternary_candidates": sum(1 for r in rows if r["first_quant_candidate"]),
            "ternary_candidate_bytes": sum(
                r["n_bytes"] for r in rows if r["first_quant_candidate"]
            ),
        },
        "tensors_detail": rows,
    }
    text = json.dumps(payload, indent=2) + "\n"
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text, encoding="utf-8")
        print(f"wrote {a.out}: {len(rows)} tensors")
        print(json.dumps(payload["summary"], indent=2))
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
