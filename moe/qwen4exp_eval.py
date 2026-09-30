"""Full-vocabulary KLD gate for the qwen4_exp port (Flash-Next prefix).

The instrument is ``kld_eval``: its pure math (``log_probs``,
``kld_from_logprobs``, ``kld_decompose``, ``tail_stats``,
``tail_sample_estimate``, ``host_memory_guard``, ``top1_agreement``) is
imported untouched from ``kld_eval.py``; only the model wiring differs because
``kld_eval.main`` imports ``qwen35_moe_proxy`` by name.  Keeping this wiring in
a separate module means the 35B gate file stays single-writer while the two
instruments share one source of truth for the math.

Same protocol as the 35B gate, with one box-specific change: the teacher
log-probs are parked **per window on disk** (``--tcache-dir``) instead of in
RAM, because a 2-layer FP8 prefix already takes 13.5 GB host and the second
load plus a 4 GB park swap-thrashed the 31 GB box.  ``--stage teacher`` and
``--stage student`` run the two halves as separate processes; ``both`` keeps
the single-process flow for small prefixes.

Example (2-layer real-weights prefix, free card, wikitext seed 999):

    PYTHONPATH=<shadow> HIP_VISIBLE_DEVICES=1 python moe/qwen4exp_eval.py \\
        --prefix-layers 2 --device cuda:0 --quant lloyd --branch-quant g128 \\
        --branch-target both --rank 512 --decompose-topk 512 \\
        --eval-windows 8 --out $MOE/qwen4exp/kld-body-2l.json
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from kld_eval import (kld_decompose, kld_from_logprobs, log_probs,  # noqa: E402
                      tail_stats, top1_agreement)


@torch.no_grad()
def teacher_pass(model, logits_fn, data, device, chunk, tcache_dir):
    """Park fp32 teacher log-probs per window **on disk** (+ entropy/peak).

    The 31 GB box cannot hold a 2-layer FP8 prefix (13.5 GB host) plus a second
    load plus a 4 GB in-RAM park at once; one file per window keeps the teacher
    stage at ~model+one-window and lets the student stage run in a fresh
    process (``--stage student``) with the parked files read one at a time.
    """
    tcache_dir = Path(tcache_dir)
    tcache_dir.mkdir(parents=True, exist_ok=True)
    ents, tops = [], []
    for i in range(len(data)):
        ids = data[i:i + 1].to(device)
        raw = logits_fn(model, ids)[0, :-1]
        lp = log_probs(raw, chunk).cpu()
        torch.save(lp, tcache_dir / f"w{i}.pt")
        ents.append(float(-(lp.exp() * lp).sum(-1).mean()))
        tops.append(float(lp.exp().max(-1).values.mean()))
        print(f"  teacher {i+1}/{len(data)} entropy {ents[-1]:.3f} "
              f"peak {tops[-1]:.4f}", flush=True)
        del raw, lp
    return sum(ents) / len(ents), sum(tops) / len(tops)


@torch.no_grad()
def student_pass(model, logits_fn, data, device, tcache_dir, chunk,
                 decompose_topk=0):
    """Per-token KLD, top-1 agreement, entropy and peak (kld_eval port)."""
    tcache_dir = Path(tcache_dir)
    per_token, agrees, ents, tops = [], [], [], []
    dec_parts = []
    for i in range(len(data)):
        ids = data[i:i + 1].to(device)
        slp = log_probs(logits_fn(model, ids)[0, :-1], chunk).cpu()
        tlp = torch.load(tcache_dir / f"w{i}.pt", map_location="cpu")
        kld = kld_from_logprobs(tlp, slp, chunk)
        per_token.append(kld)
        agrees.append(top1_agreement(tlp, slp))
        ents.append(float(-(slp.exp() * slp).sum(-1).mean()))
        tops.append(float(slp.exp().max(-1).values.mean()))
        if decompose_topk:
            dec = kld_decompose(tlp, slp, decompose_topk, chunk)
            dec_parts.append(dec)
        del slp, tlp
        print(f"  student {i+1}/{len(data)} mean kld {kld.mean():.4f} "
              f"max {kld.max():.4f} top1 {agrees[-1]:.4f} "
              f"H {ents[-1]:.3f} peak {tops[-1]:.4f}", flush=True)
    result = (torch.cat(per_token), sum(agrees) / len(agrees),
              sum(ents) / len(ents), sum(tops) / len(tops))
    if decompose_topk:
        dec = {name: torch.cat([d[name] for d in dec_parts])
               for name in dec_parts[0]}
        return result + (dec,)
    return result


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prefix-layers", type=int, default=2)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--student-device", default="",
                    help="where the student model runs; default = --device. "
                         "Use 'cpu' when the card is shared/busy: the banks are "
                         "still quantized on the GPU (small peak) and the "
                         "forward is only 8 windows")
    ap.add_argument("--eval-windows", type=int, default=8)
    ap.add_argument("--worst", type=int, default=16)
    ap.add_argument("--stage", choices=["both", "teacher", "student"],
                    default="both",
                    help="both = one process (small prefixes); teacher/student "
                         "split lets the student run in a fresh process with "
                         "the parked per-window log-probs read from disk")
    ap.add_argument("--tcache-dir", default="",
                    help="per-window teacher log-prob park (w{i}.pt + meta.json); "
                         "default $MOE_ARTIFACTS/qwen4exp/tcache-<N>l")
    ap.add_argument("--decompose-topk", type=int, default=0)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--quant", choices=["absmean", "lloyd"], default="lloyd")
    ap.add_argument("--rank", type=int, default=512)
    ap.add_argument("--branch-quant", choices=["fp32", "g128", "rank"],
                    default="g128")
    ap.add_argument("--branch-target", choices=["moe_out", "attn_out", "both"],
                    default="both")
    ap.add_argument("--branch-gate", choices=["none", "rw"], default="none")
    ap.add_argument("--load", default="", help="branch checkpoint (omit = body)")
    ap.add_argument("--balance", choices=["none", "bias", "quantile", "zloss",
                                          "cb", "cbqb"], default="none")
    ap.add_argument("--balance-cb-eta", type=float, default=0.05)
    ap.add_argument("--balance-qb-damp", type=float, default=1.0)
    ap.add_argument("--split", default="wikitext")
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--out", default="")
    ap.add_argument("--model-dir", default="")
    ap.add_argument("--shard-dir", default="")
    ap.add_argument("--ple", choices=["rows", "none"], default="rows")
    ap.add_argument("--no-fp8", dest="fp8", action="store_false", default=True)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    from olmoe_corrections import load_branch_state
    from olmoe_proxy import windows
    import qwen4exp_proxy as q4

    model_dir = Path(args.model_dir) if args.model_dir else q4.MODEL
    if not (model_dir / "config.json").exists():
        raise SystemExit(f"no checkpoint at {model_dir}; pass --model-dir")
    q4.patch_indexer()

    from transformers import AutoConfig, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    cfg = AutoConfig.from_pretrained(str(model_dir))
    tcfg = getattr(cfg, "text_config", cfg)
    vocab = int(tcfg.vocab_size)
    data = windows(tok, args.eval_windows, args.seq, args.seed, args.split)
    print(f"KLD eval: {args.prefix_layers}-layer prefix, vocab {vocab}, "
          f"{len(data)} windows x {args.seq} tokens, quant={args.quant} "
          f"branch={args.branch_target}/{args.branch_quant} "
          f"gate={args.branch_gate} load={args.load or 'none'}", flush=True)

    need = sum(d.numel() - 1 for d in data) * vocab * 4
    print(f"teacher park: {need/1e9:.2f} GB on disk "
          f"({len(data)} windows x {vocab} vocab, per-window files)", flush=True)
    tcache_dir = (Path(args.tcache_dir) if args.tcache_dir
                  else q4.OUT / f"tcache-{args.prefix_layers}l")

    def make(device=None):
        ple_ids = torch.stack([data[i] for i in range(len(data))])
        model, _, _, _ = q4.load_fp8_prefix(
            args.prefix_layers, device or args.device, model_dir=model_dir,
            shard_dir=Path(args.shard_dir) if args.shard_dir else None,
            ple=args.ple, ple_ids=ple_ids)
        return model

    t_ent = t_top = None
    if args.stage in ("both", "teacher"):
        # ---- teacher: FP prefix, banks untouched, no branches
        model = make()
        t_ent, t_top = teacher_pass(model, q4.model_logits, data,
                                    args.device, args.chunk, tcache_dir)
        (Path(tcache_dir) / "meta.json").write_text(json.dumps(
            {"teacher_entropy_nats": t_ent, "teacher_top1_prob": t_top,
             "eval_windows": args.eval_windows, "seq": args.seq,
             "split": args.split, "seed": args.seed}))
        del model
        gc.collect()
        torch.cuda.empty_cache()
        if args.stage == "teacher":
            print("teacher stage done", flush=True)
            return
    else:
        meta = json.loads((Path(tcache_dir) / "meta.json").read_text())
        t_ent, t_top = meta["teacher_entropy_nats"], meta["teacher_top1_prob"]

    # ---- student: same prefix, ternarised banks + branches + checkpoint.
    # Built on the **CPU**: the 20 GB local card cannot hold the 13.5 GB model
    # and the Lloyd temporaries together (see ternarize_banks); the banks are
    # quantized on the GPU while they are the only resident tensors, then the
    # finished model moves over (or stays on CPU when --student-device cpu and
    # the card is shared with the other session's local work).
    student_device = args.student_device or args.device
    model = make(device="cpu")
    if args.balance != "none":
        n_gates = q4.patch_router_balance(model, args.balance,
                                          args.balance_cb_eta, args.balance_qb_damp)
        print(f"router balance: {args.balance} on {n_gates} gates", flush=True)
    quant_dev = args.device if str(args.device).startswith("cuda") else None
    n_tr = q4.build_student(model, args, quant_work_device=quant_dev)
    print(f"student built: trainable {n_tr/1e6:.2f}M", flush=True)
    if args.load:
        missing, unexpected = load_branch_state(model, args.load)
        unexpected = [k for k in unexpected if not k.startswith("_mtp_head.")]
        want = {k for k in model.state_dict() if ".branch." in k or ".gate." in k}
        got = want & set(missing)
        print(f"loaded {args.load}: branch tensors {len(want) - len(got)}/{len(want)}, "
              f"unexpected={len(unexpected)}", flush=True)
        if got or unexpected:
            raise SystemExit(
                f"checkpoint does not match this prefix/placement: "
                f"{len(got)} wanted tensor(s) did not load, "
                f"{len(unexpected)} did not belong")
    if str(student_device) != "cpu":
        model.to(student_device)
    print(f"student on {next(model.parameters()).device}", flush=True)
    if args.decompose_topk:
        per_token, agree, ent, top1p, dec = student_pass(
            model, q4.model_logits, data, student_device, tcache_dir, args.chunk,
            args.decompose_topk)
    else:
        per_token, agree, ent, top1p = student_pass(
            model, q4.model_logits, data, student_device, tcache_dir, args.chunk)
        dec = None

    res = tail_stats(per_token)
    if not res["p999_resolved"]:
        print(f"WARNING: at {res['n_tokens']} tokens, p99.9 is only the "
              f"{res['p999_rank']}-worst token -- treat 'max' and 'worst_tokens' "
              f"as the tail signal.", flush=True)
    flat = per_token.reshape(-1).float()
    k = min(args.worst, flat.numel())
    top = torch.topk(flat, k)
    seq = args.seq - 1
    res["worst_tokens"] = [
        {"token": int(t), "window": int(t) // seq, "pos": int(t) % seq,
         "kld": float(v)} for v, t in zip(top.values, top.indices)]
    res["teacher_entropy_nats"] = round(float(t_ent), 4)
    res["teacher_top1_prob"] = round(float(t_top), 4)
    res["student_entropy_nats"] = round(float(ent), 4)
    res["student_top1_prob"] = round(float(top1p), 4)
    res["sharper_than_teacher"] = bool(ent < t_ent - 0.5)
    if dec is not None:
        pieces = ("marginal", "support", "tail")
        total_mean = float(dec["full"].mean())
        res["kld_decomposition"] = {
            "top_k": args.decompose_topk,
            "mean_full": total_mean,
            "pieces": {name: {
                "mean": float(dec[name].mean()),
                "share": float(dec[name].mean() / total_mean) if total_mean else None,
                "max": float(dec[name].max())} for name in pieces},
            "mean_support_mass_teacher": float(dec["w_t"].mean()),
            "mean_support_mass_student": float(dec["w_s"].mean()),
        }
        for name in pieces:
            p = res["kld_decomposition"]["pieces"][name]
            print(f"KLD piece {name:8}: mean {p['mean']:.4f} "
                  f"({p['share']:.1%} of {total_mean:.4f}) max {p['max']:.4f}",
                  flush=True)
        print(f"support mass: teacher {dec['w_t'].mean():.4f} "
              f"student {dec['w_s'].mean():.4f}", flush=True)
    res.update({"top1_agreement": round(agree, 6), "checkpoint": args.load,
                "prefix_layers": args.prefix_layers, "quant": args.quant,
                "branch_target": args.branch_target,
                "branch_quant": args.branch_quant,
                "branch_gate": args.branch_gate, "rank": args.rank,
                "eval_windows": args.eval_windows, "seq": args.seq})
    print(json.dumps(res, indent=2), flush=True)
    out = Path(args.out) if args.out else q4.OUT / "kld-eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
