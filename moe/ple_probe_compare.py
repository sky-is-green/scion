"""Compare the Python-loader probe and the fork llama-server probe top-k lists.

Usage: python moe/ple_probe_compare.py <py.json> <cpp.json> [k]
Prints the two top-k lists, the shared-token logprob deltas, top-1 agreement
and the top-k overlap.  A correct PLE export matches on the shared tokens (the
2-layer model is near-uniform, so the tail order can reshuffle at Q4_0 noise).
"""
from __future__ import annotations

import json
import sys


def cpp_top(data: dict) -> list[dict]:
    probs = data.get("completion_probabilities") or []
    if not probs:
        raise SystemExit(f"C++ response has no completion_probabilities: "
                         f"{list(data)[:8]}")
    first = probs[0]
    entries = (first.get("top_logprobs") or first.get("probs")
               or first.get("top_probs") or [])
    out = []
    for e in entries:
        out.append({"token": int(e.get("id", e.get("token_id", -1))),
                    "logprob": float(e["logprob"]),
                    "text": e.get("token", "")})
    return out


def main():
    py_path, cpp_path = sys.argv[1], sys.argv[2]
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    py = json.loads(open(py_path).read())
    cpp = json.loads(open(cpp_path).read())
    pt = py["top"][:k]
    ct = cpp_top(cpp)[:k]
    print(f"prompt: {py['prompt']!r}")
    print(f"{'#':>2} {'python':>28} {'cpp':>28}")
    for i in range(max(len(pt), len(ct))):
        p = f"{pt[i]['token']:7d} {pt[i]['logprob']:8.3f} {pt[i]['text']!r}" \
            if i < len(pt) else ""
        c = f"{ct[i]['token']:7d} {ct[i]['logprob']:8.3f} {ct[i]['text']!r}" \
            if i < len(ct) else ""
        print(f"{i:2d} {p:>28} {c:>28}")
    pids = {t["token"]: t["logprob"] for t in pt}
    cids = {t["token"]: t["logprob"] for t in ct}
    shared = sorted(set(pids) & set(cids))
    print(f"\ntop-1 match: {pt[0]['token'] == ct[0]['token']} "
          f"(py {pt[0]['token']} vs cpp {ct[0]['token']})")
    print(f"top-{k} overlap: {len(shared)}/{k}")
    if shared:
        deltas = [abs(pids[t] - cids[t]) for t in shared]
        print(f"shared logprob |delta|: mean {sum(deltas)/len(deltas):.4f} "
              f"max {max(deltas):.4f}")
        for t in shared:
            print(f"  tok {t:7d} py {pids[t]:9.4f} cpp {cids[t]:9.4f} "
                  f"d {pids[t]-cids[t]:+8.4f}")
    # rough rank correlation over the shared set
    py_rank = {t["token"]: i for i, t in enumerate(pt)}
    cpp_rank = {t["token"]: i for i, t in enumerate(ct)}
    if len(shared) > 1:
        inv = sum(1 for a, b in [(shared[i], shared[j])
                                 for i in range(len(shared))
                                 for j in range(i + 1, len(shared))]
                  if (py_rank[a] - py_rank[b]) * (cpp_rank[a] - cpp_rank[b]) < 0)
        print(f"rank inversions over shared set: {inv}")


if __name__ == "__main__":
    main()
