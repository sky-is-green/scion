"""Export a trained DSpark-shaped multi-token head to the extended sidecar GGUF.

Tensors (GGUF ne is the reverse of the torch layout, as gguf-py writes it):

  mtp2.pos            [k, 2H]      per-position input bias
  mtp2.gru.weight_ih  [3S, 2H]     GRUCell input weights (r,z,n order, torch)
  mtp2.gru.weight_hh  [3S, S]
  mtp2.gru.bias_ih    [3S]
  mtp2.gru.bias_hh    [3S]
  mtp2.proj.weight    [H, S]       state -> hidden
  mtp2.fc1.weight     [2H, 2H]     direct residual branch (the k=1 MLP form)
  mtp2.fc2.weight     [H, 2H]
  metadata: mtp2.format/version/hidden_size/k/state_dim/source + acceptance

The head reuses the target's own tok_embd / output_norm / output, exactly like
the k=1 sidecar.

Usage:
  python moe/mtp_dspark_export.py --head DIR/mtp-dspark-head-v2-cur05.pt \
    --out DIR/mtp-dspark-v2-cur05.gguf --gguf-py /home/penis/llama.cpp/gguf-py \
    --source qwen35-release-v2-cur05-soup --verify
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

KEYS = [
    ("mtp2.pos", "dspark.pos", np.float32),
    ("mtp2.gru.weight_ih", "dspark.gru.weight_ih", np.float16),
    ("mtp2.gru.weight_hh", "dspark.gru.weight_hh", np.float16),
    ("mtp2.gru.bias_ih", "dspark.gru.bias_ih", np.float32),
    ("mtp2.gru.bias_hh", "dspark.gru.bias_hh", np.float32),
    ("mtp2.proj.weight", "dspark.proj.weight", np.float16),
    ("mtp2.fc1.weight", "dspark.fc1.weight", np.float16),
    ("mtp2.fc2.weight", "dspark.fc2.weight", np.float16),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gguf-py", default="/home/penis/llama.cpp/gguf-py")
    ap.add_argument("--source", default="")
    ap.add_argument("--accept", type=float, default=None, help="generation tokens/forward")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, a.gguf_py)
    from gguf import GGUFWriter

    sd = torch.load(a.head, map_location="cpu")
    meta = sd["meta"]
    H, K, S = int(meta["hidden"]), int(meta["k"]), int(meta["state_dim"])

    w = GGUFWriter(a.out, arch="mtp2")
    w.add_string("mtp2.format", "scion-dspark-head")
    w.add_uint32("mtp2.version", 1)
    w.add_uint32("mtp2.hidden_size", H)
    w.add_uint32("mtp2.k", K)
    w.add_uint32("mtp2.state_dim", S)
    if a.source:
        w.add_string("mtp2.source", a.source)
    if a.accept is not None:
        w.add_float32("mtp2.accept_tokens_per_forward", float(a.accept))
    arrs = {}
    for gname, tname, dt in KEYS:
        arr = sd[tname].float().numpy().astype(dt)
        arrs[gname] = arr
        w.add_tensor(gname, arr)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {a.out}: H {H}, k {K}, state {S}; tensors {len(arrs)}")

    if a.verify:
        from gguf import GGUFReader
        r = GGUFReader(a.out)
        got = {t.name: np.asarray(t.data).copy() for t in r.tensors}
        bad = []
        for gname, tname, _ in KEYS:
            src = sd[tname].float().numpy()
            ref = src
            g = got[gname].astype(np.float32)
            # gguf-py reverses dims on write, so the reader returns the original layout
            if g.shape != ref.shape:
                bad.append(f"{gname}: {g.shape} vs {ref.shape}")
                continue
            md = float(np.abs(g - ref).max())
            if md > 2e-2:
                bad.append(f"{gname}: maxdiff {md:.3e}")
        if bad:
            raise SystemExit("round-trip mismatch: " + "; ".join(bad))
        print("verify OK")


if __name__ == "__main__":
    main()
