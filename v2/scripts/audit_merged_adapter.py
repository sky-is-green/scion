#!/usr/bin/env python3
"""Audit a merged (adapter-embedded) GGUF against its sources (PLAN 13a D).

Three checks, all of which must PASS before any eval number is recorded:

  1. merge integrity: every body tensor in the merged file is byte-identical
     to the body file's tensor (nothing re-quantised in the merge), and every
     adapter tensor is byte-identical to the adapter file's.
  2. factor fidelity: the packed Q1_0_g128 lora factors decode to exactly the
     ``deploy_weights`` output of the training checkpoint's branch tensors
     (the 0-ppm-vs-training-codes gate; the export audit proved this matters
     for the banks - the branch factors get the same gate here).
  3. router fidelity: the dense f16 router deltas equal trained - base for
     every exported layer.

With ``--replace`` merges the body's superseded tensors are simply absent;
they are checked in the adapter file instead.

Usage:
  python v2/scripts/audit_merged_adapter.py --merged merged.gguf \
      --body body.gguf --adapter adapter.gguf --ckpt ckpt.pt \
      --base-model empero-hf [--layers 0,1,2,3]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFReader

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE / "moe"))
from export_branches_lora import (deploy_weights, load_base_routers,  # noqa: E402
                                  router_canon)


def decode_q1_0_g128(raw: np.ndarray, shape: tuple[int, ...]) -> torch.Tensor:
    """Inverse of pack_q1_0_g128 for the torch-logical ``shape`` [rank, cols].

    The packer writes ``[rank, ng, 34]`` blocks contiguous (rank-major,
    group-fastest); the gguf dims are reversed on write but the byte order is
    unchanged, so decoding straight from the flat payload in the torch layout
    reproduces the values exactly.
    """
    g = 128
    rank, cols = shape
    blocks = np.frombuffer(bytes(raw), dtype=np.uint8).reshape(rank, cols // g, 34)
    scales = blocks[:, :, :2].copy().view("<f2").astype(np.float32)[..., 0]
    qb = blocks[:, :, 2:]
    codes = np.empty((rank, cols // g, g), dtype=np.int8)
    for j in range(4):
        codes[:, :, j::4] = ((qb >> (2 * j)) & 0x03).astype(np.int8) - 1
    vals = codes.astype(np.float32) * scales[..., None]
    return torch.from_numpy(vals.reshape(rank, cols))


def tensor_bytes(reader: GGUFReader, name: str) -> bytes:
    for t in reader.tensors:
        if t.name == name:
            return bytes(t.data)
    raise KeyError(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True)
    ap.add_argument("--body", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--ckpt", required=True, help="the training checkpoint (.pt)")
    ap.add_argument("--base-model", required=True,
                    help="HF dir with the frozen routers (as exported against)")
    ap.add_argument("--deploy-quant", default="lloyd", choices=["lloyd", "absmean"])
    ap.add_argument("--branch-quant", default="g128", choices=["g128", "rank", "none"])
    args = ap.parse_args()

    merged, body, adapter = (GGUFReader(p) for p in
                             (args.merged, args.body, args.adapter))
    merged_names = {t.name: t for t in merged.tensors}
    adapter_names = {t.name for t in adapter.tensors}

    # ---- 1. merge integrity ------------------------------------------------
    n_body = n_adapter = 0
    mism = 0
    for t in body.tensors:
        if t.name in adapter_names:
            continue          # --replace site: the adapter's tensor is checked
        if bytes(t.data) != bytes(merged_names[t.name].data):
            mism += 1
            print(f"MISMATCH body tensor {t.name}")
        n_body += 1
    for t in adapter.tensors:
        if bytes(t.data) != bytes(merged_names[t.name].data):
            mism += 1
            print(f"MISMATCH adapter tensor {t.name}")
        n_adapter += 1
    print(f"merge integrity: {n_body} body + {n_adapter} adapter tensors "
          f"byte-identical, {mism} mismatches")

    # ---- 2+3. factor / router fidelity vs the training checkpoint ----------
    # (run on the ADAPTER file's tensors; check 1 proved they are byte-identical
    # in the merged file, and only the adapter's sites come from this ckpt)
    sd = torch.load(args.ckpt, map_location="cpu")
    sd = {k.replace(".doctor.", ".branch."): v for k, v in sd.items()}
    base_routers = load_base_routers(args.base_model)

    import re
    fac_mism = fac_tot = 0
    router_mism = router_tot = 0
    for name, t in sorted(((t.name, t) for t in adapter.tensors)):
        if not name.endswith(".lora_a") and not name.endswith(".lora_b"):
            continue
        layer = int(name.split(".")[1])
        target = name.split(".")[2]
        if target == "ffn_gate_inp":
            continue                       # routers, handled below
        # find the matching checkpoint factors
        suffix = {"ffn_moe_out": ".mlp.branch.", "ssm_out": ".linear_attn.out_proj.branch.",
                  "attn_output": ".self_attn.o_proj.branch."}[target]
        down = up = None
        for k, v in sd.items():
            if f"layers.{layer}." in k and k.endswith(suffix + "down.weight"):
                down = v
            if f"layers.{layer}." in k and k.endswith(suffix + "up.weight"):
                up = v
        if down is None or up is None:
            print(f"no checkpoint factors for {name} (layer {layer}, {target})")
            fac_mism += 1
            continue
        if name.endswith(".lora_a"):
            ref, _ = deploy_weights(down, up, args.branch_quant, args.deploy_quant)
        else:
            _, ref = deploy_weights(down, up, args.branch_quant, args.deploy_quant)
        # the packer writes [rank, ng, 34] contiguous; decode in the same
        # logical layout ([rank, cols], groups along cols)
        got = decode_q1_0_g128(t.data, ref.shape)
        if not torch.equal(got, ref.float()):
            fac_mism += int((got != ref.float()).sum())
        fac_tot += ref.numel()

    # routers: dense f16 pairs == (trained - base), layer by layer
    for name, t in sorted(((t.name, t) for t in adapter.tensors)):
        if not name.startswith("blk.") or not name.endswith(".ffn_gate_inp.weight.lora_a"):
            continue
        layer = int(name.split(".")[1])
        b_name = name.replace(".lora_a", ".lora_b")
        a = torch.from_numpy(t.data.copy()).float()          # [rank, in]
        b = torch.from_numpy(next(x.data for x in adapter.tensors
                                  if x.name == b_name).copy()).float()
        key_trained = None
        key_base = None
        for k in sd:
            if f"layers.{layer}." in k and k.endswith(".mlp.gate.weight"):
                key_trained = k
        for k in base_routers:
            if f"layers.{layer}." in k:
                key_base = k
        if key_trained is None or key_base is None:
            print(f"no router pair for layer {layer}")
            router_mism += 1
            continue
        delta = sd[key_trained].float() - base_routers[key_base]
        # a = vh [rank, in], b = (u*s) [out, rank]; b @ a reconstructs the
        # delta exactly at full rank
        approx = b @ a
        if not torch.allclose(approx.float(), delta.float(), atol=2e-2, rtol=2e-2):
            router_mism += int((~torch.isclose(approx.float(), delta.float(),
                                               atol=2e-2, rtol=2e-2)).sum())
        router_tot += delta.numel()

    ok = mism == 0 and fac_mism == 0 and router_mism == 0
    print(f"factor fidelity: {fac_tot} weights, {fac_mism} mismatches vs "
          f"deploy_weights(checkpoint)")
    print(f"router fidelity: {router_tot} weights, {router_mism} mismatches vs "
          f"trained-base (atol 2e-2, f16 SVD pairs)")
    print("AUDIT " + ("PASS" if ok else "FAIL"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
