"""Checkpoint trajectory + router diagnostics for the MoTE proxy."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from moe_proxy import BASE, OUT, ternary_absmean, upcycle, windows  # noqa: E402


@torch.no_grad()
def eval_ppl(model, data, device):
    total, ntok = 0.0, 0
    for i in range(len(data)):
        ids = data[i:i + 1].to(device)
        logits = model(ids).logits[:, :-1]
        total += F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                                 ids[:, 1:].reshape(-1), reduction="sum").item()
        ntok += ids[:, 1:].numel()
    return math.exp(total / ntok)


@torch.no_grad()
def router_stats(model, data, device):
    probs_all, idx_all = [], []
    for i in range(min(2, len(data))):
        ids = data[i:i + 1].to(device)
        model(ids)
        for layer in model.model.layers:
            p = layer.mlp._last_probs
            probs_all.append(p.mean(dim=1))                 # [T, E]
            idx_all.append(p.argmax(-1))
    probs = torch.cat(probs_all, dim=0)                     # [L*T, E]
    idx = torch.cat(idx_all, dim=0)
    share = torch.stack([(idx == e).float().mean() for e in range(probs.shape[-1])])
    return {"mean_p_max": float(probs.max(-1).values.mean()),
            "mean_p_entropy": float(-(probs * (probs + 1e-9).log()).sum(-1).mean()),
            "expert_share": [round(float(s), 4) for s in share]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=2)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ckpts", nargs="*", default=[])
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    data = windows(tok, args.windows, args.seq, 999, "wikitext")

    model = upcycle(args.experts, args.group, args.device)
    rows = []

    def snap(tag):
        ppl = eval_ppl(model, data, args.device)
        stats = router_stats(model, data, args.device)
        rows.append({"tag": tag, "ppl": round(ppl, 4), **stats})
        print(f"{tag:28s} ppl {ppl:8.4f}  p_max {stats['mean_p_max']:.3f} "
              f"share {stats['expert_share']}", flush=True)

    # expert weight stats at init (masters from FFN)
    w0 = model.model.layers[0].mlp.experts[0].gate_proj.weight
    print(f"init expert |w| mean {w0.abs().mean().item():.5f} "
          f"ternary zero-share {(ternary_absmean(w0, args.group) == 0).float().mean().item():.3f}")

    snap("init_fp_masters")
    with torch.no_grad():
        for layer in model.model.layers:
            for e in layer.mlp.experts:
                for lin in (e.gate_proj, e.up_proj, e.down_proj):
                    lin.weight.copy_(ternary_absmean(lin.weight, args.group))
    snap("rtn_no_training")

    for ck in args.ckpts:
        sd = torch.load(ck, map_location="cpu")
        model.load_state_dict(sd, strict=False)
        snap(Path(ck).stem)

    # ablation: trained router + experts reset to init masters, re-ternarised
    if args.ckpts:
        sd = torch.load(args.ckpts[-1], map_location="cpu")
        model.load_state_dict(sd, strict=False)
        with torch.no_grad():
            for layer in model.model.layers:
                for e in layer.mlp.experts:
                    e.gate_proj.weight.copy_(layer.mlp.shared.gate_proj.weight)
                    e.up_proj.weight.copy_(layer.mlp.shared.up_proj.weight)
                    e.down_proj.weight.copy_(layer.mlp.shared.down_proj.weight)
                    for lin in (e.gate_proj, e.up_proj, e.down_proj):
                        lin.weight.copy_(ternary_absmean(lin.weight, args.group))
        snap("trained_router_rtn_experts")

    (OUT / "trajectory.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
