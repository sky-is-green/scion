#!/usr/bin/env python3
"""Real-weights bit-exactness gate for the 35B compact-bank port (PLAN 13a A).

On the 4-layer local prefix (real empero HF shards, real deployed Scion GGUF):

  1. Seed the dense banks from the deployed GGUF (the v1 ``--bank-gguf`` path)
     and keep a host copy.
  2. Build the compact banks from the same GGUF *verbatim* (codes + fp16
     scales, no re-quantisation) with the trainer's own
     ``compact_banks(..., verify=True)`` -- which per layer asserts decode ==
     seeded values bit-for-bit and re-derives the codes with ``ternary_lloyd``
     (0 mismatches expected; the exported-body property).
  3. Independently compare every decoded expert against the step-1 host copy
     (all 512 experts x 2 banks x 4 layers, ``torch.equal``).
  4. Forward comparison on a real window: the v1 dense path (seeded banks,
     in-place ``ternary_lloyd`` + STE forward) vs the compact decode path --
     hidden drift and router top-8 agreement.  These two SHOULD differ by a
     small amount: the in-place quantiser re-derives fp16 scales from bf16
     values and lands one ulp away from the deployed scale in most groups
     (the export audit measured 86.3% scale-bit changes, 0 code changes), so
     the compact-from-GGUF decode is *closer to deployment* than the dense
     training path.  Drift is reported, not asserted.

Output: one JSON line "GATE {...}"; exit 1 on any bit-equality failure.

Usage:
  SCION_GGUF=models-p310/scion-strata/Scion-35B-A3B-PQ2_0-corr.gguf \
  MOE_ARTIFACTS=models-p310/empero-prefix \
  python v2/scripts/prefix_gate.py --device cuda:1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "moe"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--layers", type=int, default=4)
    cli = ap.parse_args()

    import qwen35_moe_proxy as q35
    from moe_proxy import ternary_lloyd
    from olmoe_proxy import gate_hook
    from seed_banks_from_gguf import seed_model_banks

    gguf = Path(os.environ.get(
        "SCION_GGUF", "models-p310/scion-strata/Scion-35B-A3B-PQ2_0-corr.gguf"))
    assert gguf.exists(), gguf

    class A:
        group = 128
        quant = "lloyd"
        bank_gguf = str(gguf)
        compact_banks = True
        device = cli.device
        catq_steps = catq_lr = catq_gamma = catq_s0 = 0
        alloc_file = ""

    args = A()

    # ---- 1. dense seeded reference on an independent copy ------------------
    q35.patch_experts(args.group)
    ref_model, tok, missing, unexpected = q35.load_prefix(cli.layers, cli.device)
    print(f"prefix: {cli.layers} layers; missing={len(missing)} "
          f"unexpected={len(unexpected)}", flush=True)
    n_seeded = seed_model_banks(ref_model, str(gguf))
    print(f"seeded {n_seeded} dense banks from {gguf.name} (v1 path)", flush=True)
    host_copy = []
    for layer in q35.text_layers(ref_model):
        e = layer.mlp.experts
        host_copy.append((e.gate_up_proj.detach().to("cpu"),
                          e.down_proj.detach().to("cpu")))

    # v1 training path: banks quantised in place; the stock experts forward
    # then runs over the quantised dense banks (STE is only for training the
    # banks themselves, which are frozen here)
    q35.ternarize_banks(ref_model, args)

    # ---- 2+3. compact banks on the working copy, trainer's own gate --------
    model, tok, _, _ = q35.load_prefix(cli.layers, cli.device)
    n_banks = q35.compact_banks(model, args, verify=True)
    print(f"compact banks: {n_banks} encoded (verify=True passed inside "
          f"the builder)", flush=True)

    mism = 0
    checked = 0
    for li, layer in enumerate(q35.text_layers(model)):
        m = q35.moe_block(layer).experts
        gu_host, dn_host = host_copy[li]
        for bank, host in ((m._compact_gu, gu_host), (m._compact_dn, dn_host)):
            chunk = 16
            n_exp = host.shape[0]
            for e0 in range(0, n_exp, chunk):
                dec = bank.decode_experts(list(range(e0, min(e0 + chunk, n_exp))))
                ref = host[e0:e0 + chunk].to(dec.device)
                mism += int((dec != ref).sum())
                checked += ref.numel()
    print(f"independent check: {checked} weights, {mism} element mismatches "
          f"vs the seeded host copy", flush=True)

    # ---- 4. forward comparison --------------------------------------------
    ids = tok("Explain the proof that there are infinitely many primes.",
              return_tensors="pt").input_ids[:, :cli.seq].to(cli.device)
    routes_ref, routes_cmp = {}, {}
    h_ref = [l.mlp.gate.register_forward_hook(gate_hook(routes_ref, i))
             for i, l in enumerate(q35.text_layers(ref_model))]
    h_cmp = [l.mlp.gate.register_forward_hook(gate_hook(routes_cmp, i))
             for i, l in enumerate(q35.text_layers(model))]
    with torch.no_grad():
        out_ref = q35.model_logits(ref_model, ids)
        out_cmp = q35.model_logits(model, ids)
    for h in h_ref + h_cmp:
        h.remove()
    drift = float((out_cmp - out_ref).norm() / (out_ref.norm() + 1e-12))
    agree = float(torch.stack([
        (routes_ref[i][2].unsqueeze(-1) == routes_cmp[i][2].unsqueeze(-2))
        .any(-1).float().mean() for i in routes_ref]).mean())
    maxdiff = float((out_cmp - out_ref).abs().max())
    print(f"forward (v1 dense-in-place vs compact-from-GGUF): logit drift "
          f"{drift:.6f}, max|diff| {maxdiff:.6f}, router top-8 agreement "
          f"{agree:.4f}", flush=True)

    # ---- accounting --------------------------------------------------------
    rep = q35.memory_report(model, args, "gate")

    ok = mism == 0
    print("GATE " + json.dumps({
        "gguf": gguf.name, "layers": cli.layers, "banks": n_banks,
        "weights_checked": checked, "bitexact_mismatches": mism,
        "logit_drift_dense_vs_compact": round(drift, 6),
        "logit_maxdiff": round(maxdiff, 6),
        "router_top8_agreement": round(agree, 4),
        "expert_codes_gib": round(rep["expert_codes"] / 2**30, 3),
        "expert_scales_gib": round(rep["expert_scales"] / 2**30, 3),
        "expert_bf16_equivalent_gib": round(rep["expert_bf16_equivalent"] / 2**30, 2),
        "pass": ok,
    }), flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
