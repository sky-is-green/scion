"""Standalone drafter-acceptance evaluation (Phase E item 11).

The in-run MTP eval measures 2 windows (~1 K positions, ±1.5% noise), which is
fine for watching a training curve but thin for a shipping decision.  This
instrument loads the correction checkpoint *and* its `_mtp_head.*` tensors and
measures greedy draft acceptance over N eval windows, reporting top-1 (and
optionally top-k) acceptance plus the ideal tokens-per-step for a 1-token
speculative step.

The head is part of the checkpoint (`save()` keeps `_mtp_head.*`), so the run
that produced the arm needs no extra artifacts.

Usage:
  python moe/mtp_eval.py --prefix-layers 4 \
    --load $MOE/qwen35/qwen35-corr-r512-g128-step4096-mtp-self.pt \
    --windows 16 --out $MOE/qwen35/mtp-accept-mtp-self.json

CPU-testable parser pieces; the run itself needs the card (same as kld_eval).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="",
                    help="FP base checkpoint dir (default: the proxy's MODEL)")
    ap.add_argument("--prefix-layers", type=int, default=4,
                    help="N-layer prefix; must match the run that produced --load")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--windows", type=int, default=16,
                    help="eval windows (16 = ~8 K positions, ±0.5%% acceptance)")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--split", default="wikitext")
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--quant", choices=["absmean", "lloyd", "catq"], default="lloyd")
    ap.add_argument("--catq-steps", type=int, default=200)
    ap.add_argument("--catq-lr", type=float, default=0.05)
    ap.add_argument("--catq-gamma", type=float, default=0.8)
    ap.add_argument("--catq-s0", type=float, default=30.0)
    ap.add_argument("--rank", type=int, default=512)
    ap.add_argument("--branch-quant", choices=["fp32", "g128", "rank"], default="g128")
    ap.add_argument("--branch-target", choices=["moe_out", "attn_out", "both"],
                    default="both")
    ap.add_argument("--load", required=True,
                    help="correction checkpoint that carries the MTP head")
    ap.add_argument("--mtp-head-layers", type=int, choices=(1, 2), default=1,
                    help="head capacity of the checkpoint (default 1)")
    ap.add_argument("--topk", type=int, default=1,
                    help="also report acceptance when the main token is anywhere "
                         "in the draft's top-k (default 1 = top-1 only)")
    ap.add_argument("--chain", type=int, default=0,
                    help="autoregressively chain the head for K drafted positions "
                         "(stale hidden; the naive multi-token approximation) and "
                         "report per-position acceptance + ideal tokens/step")
    ap.add_argument("--out", default="", help="write the stats JSON here")
    return ap


def main() -> None:
    args = build_parser().parse_args()

    from transformers import AutoConfig, AutoTokenizer

    from olmoe_corrections import load_branch_state
    from olmoe_proxy import windows
    from qwen35_moe_proxy import (MODEL, MTPHead, build_student,  # noqa: E402
                                  load_prefix, mtp_acceptance, patch_experts)

    model_dir = Path(args.model_dir) if args.model_dir else MODEL
    patch_experts(args.group)
    tok = AutoTokenizer.from_pretrained(model_dir)
    cfg = AutoConfig.from_pretrained(model_dir)
    tcfg = getattr(cfg, "text_config", cfg)

    model, _, _, _ = load_prefix(args.prefix_layers, args.device, model_dir=model_dir)
    n_tr = build_student(model, args)
    head = MTPHead(int(tcfg.hidden_size), layers=args.mtp_head_layers).to(args.device)
    model._mtp_head = head
    missing, unexpected = load_branch_state(model, args.load)
    head_missing = [k for k in missing if k.startswith("_mtp_head.")]
    if head_missing:
        raise SystemExit(
            f"{args.load} carries no MTP head for --mtp-head-layers "
            f"{args.mtp_head_layers} ({len(head_missing)} head tensors missing); "
            f"pass the capacity the run used")
    unexpected = [k for k in unexpected if not k.startswith("_mtp_head.")]
    want = {k for k in model.state_dict() if ".branch." in k or ".gate." in k}
    got = want & set(missing)
    print(f"loaded {args.load}: branches {len(want) - len(got)}/{len(want)}, "
          f"head OK, unexpected={len(unexpected)} (frozen body in `missing` is "
          f"expected)", flush=True)
    if got or unexpected:
        raise SystemExit("checkpoint does not match this prefix/placement")

    data = windows(tok, args.windows, args.seq, args.seed, args.split)
    acc = mtp_acceptance(model, head, data, args, topk=1)
    res = {"checkpoint": args.load, "window": args.windows, "seq": args.seq,
           "seed": args.seed, "split": args.split,
           "prefix_layers": args.prefix_layers,
           "head_layers": args.mtp_head_layers,
           "greedy_top1": round(acc, 4),
           "tokens_per_step_k1_ideal": round(1.0 + acc, 3)}
    print(f"greedy draft acceptance (top-1): {acc:.4f} over "
          f"{args.windows} windows; ideal 1-token spec step = "
          f"{1.0 + acc:.2f} tokens/step", flush=True)
    if args.topk > 1:
        acc_k = mtp_acceptance(model, head, data, args, topk=args.topk)
        res["greedy_topk"] = {"k": args.topk, "acc": round(acc_k, 4)}
        print(f"top-{args.topk} acceptance: {acc_k:.4f}", flush=True)
    if args.chain > 0:
        from mtp import chained_acceptance, mtp_targets
        from qwen35_moe_proxy import model_hidden_logits

        accs = []
        model.eval()
        with torch.no_grad():
            for i in range(len(data)):
                ids = data[i:i + 1].to(args.device)
                h, logits = model_hidden_logits(model, ids)
                h_in, e_in, _ = mtp_targets(h, ids)
                accs.append(chained_acceptance(model, head, logits, h_in, e_in,
                                               args.chain))
        n = min(len(a) for a in accs)
        per_pos = [sum(a[j] for a in accs) / len(accs) for j in range(n)]
        tps, prod = 1.0, 1.0
        for a in per_pos:
            prod *= a
            tps += prod
        res["chain"] = {"k": args.chain,
                        "per_position": [round(a, 4) for a in per_pos],
                        "ideal_tokens_per_step": round(tps, 3)}
        print(f"chained draft acceptance per position: "
              f"{[round(a, 4) for a in per_pos]} (ideal {tps:.2f} tokens/step)",
              flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2) + "\n")
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
