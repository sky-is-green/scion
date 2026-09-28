"""Full-vocabulary KLD between the FP prefix teacher and the corrected student.

The W1 gate instrument (``docs/TAIL-EXPERIMENT-PLAN.md`` step 1).  Correction
training only ever matches the teacher's **top-50** logits, so the rest of the
student distribution is unconstrained -- and PPL cannot see that, because a
ternary body can hold mean likelihood while the tail is wrong.  The gate is
written against per-token ``KL(teacher || student)`` over the *whole*
vocabulary, reported as mean / p99 / p99.9 / max.

Both sides are the same N-layer prefix, so the vocabularies line up token for
token:

  teacher : the prefix exactly as loaded (FP banks, no branches)
  student : banks ternarised in place (``--quant``) + correction branches
            (``--branch-*``) + ``--load`` checkpoint

The student is built by ``qwen35_moe_proxy.build_student``, the same helper the
train/eval stages use, so this instrument cannot drift from the stage it
measures.

The teacher pass runs first and parks its log-probabilities on the CPU, because
ternarisation rewrites the banks in place and the two passes cannot share one
resident model.  That costs ``n_windows * (seq-1) * vocab * 4`` bytes of host
memory -- about 4 GB for the default 8 windows at seq 512 with a 248k vocab --
so the estimate is printed before the pass starts.

**Read the tail stats with the sample size in hand.**  ``tail_stats`` reports
``p999_rank``, the number of tokens at or above the p99.9 position, because that
percentile is only n/1000-th from the top by construction.  At the default 8
windows that is the 5th-worst token, so ``p99.9`` and ``max`` are nearly the same
measurement and ``p99_resolved`` is false.  The gate should lean on ``max`` and
``worst_tokens`` (which record window and position, so a spike is inspectable
rather than merely reported) until the instrument runs at >= 1e5 tokens, which
needs more eval windows than host memory allows for a 248k vocab.

Example (4-layer prefix on the free card):

    HIP_VISIBLE_DEVICES=1 python moe/kld_eval.py --prefix-layers 4 \\
        --device cuda:0 --quant lloyd --branch-quant g128 --branch-target both \\
        --rank 512 --load $MOE/qwen35-corr-r512-g128-step4096.pt \\
        --out $MOE/kld-armA.json

The pure math (log-probs, per-token KLD, tail stats) is import-safe without
transformers/datasets so it can be unit-tested on CPU; see
``moe/tests/test_kld_eval.py``.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ART = Path(os.environ.get("MOE_ARTIFACTS", HERE / "artifacts"))
OUT = ART / "qwen35"


# ------------------------------------------------------------------- math ----

def log_probs(logits: torch.Tensor, chunk: int = 32) -> torch.Tensor:
    """fp32 log-softmax over the vocab, chunked over tokens to bound memory."""
    parts = [F.log_softmax(logits[i:i + chunk].float(), dim=-1)
             for i in range(0, logits.shape[0], chunk)]
    return torch.cat(parts)


def kld_from_logprobs(tlp: torch.Tensor, slp: torch.Tensor,
                      chunk: int = 32) -> torch.Tensor:
    """Per-token ``KL(teacher || student)`` from two [T, V] log-prob blocks.

    ``tlp`` is assumed already normalised; ``slp`` is renormalised here so a
    student that has drifted out of the simplex cannot produce a negative KLD.
    """
    out = []
    for i in range(0, tlp.shape[0], chunk):
        t = tlp[i:i + chunk]
        s = slp[i:i + chunk] - torch.logsumexp(slp[i:i + chunk], dim=-1, keepdim=True)
        out.append((t.exp() * (t - s)).sum(-1))
    return torch.cat(out)


def kld_per_token(teacher_logits: torch.Tensor, student_logits: torch.Tensor,
                  chunk: int = 32) -> torch.Tensor:
    """Convenience wrapper: logits in, per-token full-vocab KLD out."""
    return kld_from_logprobs(log_probs(teacher_logits, chunk),
                             log_probs(student_logits, chunk), chunk)


#: percentiles reported by default; p99.9 and max are the gate, the rest is
#: context so a tail move can be told apart from a uniform shift.
PERCENTILES = (50.0, 90.0, 99.0, 99.9)

#: Minimum number of tokens at-or-above the p99.9 position for that percentile
#: to carry information beyond the top few outliers.  8 windows x 511 tokens =
#: 4088 tokens, where p99.9 is the 5th-worst observation -- statistically
#: indistinguishable from the max.  Resolving it properly needs more eval
#: windows (memory-bound; see the module docstring on the teacher cache).
P999_MIN_RANK = 100


def tail_stats(per_token: torch.Tensor) -> dict:
    """Mean / percentiles / max of the per-token KLD, plus the worst token.

    ``p999_above`` is the number of tokens at or beyond the p99.9 value: it is
    the honest order-statistic rank behind that percentile.  When it is small,
    p99.9 and max are the same measurement and only ``max`` is meaningful.
    """
    flat = per_token.reshape(-1).float()
    if flat.numel() == 0:
        raise ValueError("no tokens")
    ordered = torch.sort(flat).values
    worst = int(flat.argmax())
    out = {
        "n_tokens": int(flat.numel()),
        "mean": float(flat.mean()),
        "max": float(ordered[-1]),
        "argmax_token": worst,
        "p50": float(torch.quantile(ordered, 0.50)),
    }
    for p in PERCENTILES:
        if p == 50.0:
            continue
        # exact order statistics; torch.quantile's interpolation would smear the
        # very tail this instrument exists to resolve
        idx = min(int(round(p / 100.0 * (ordered.numel() - 1))), ordered.numel() - 1)
        out[f"p{p:g}"] = float(ordered[idx])
    p999_idx = min(int(round(0.999 * (ordered.numel() - 1))), ordered.numel() - 1)
    # how many tokens sit at or above the p99.9 position, counting from the top.
    # This is the honest resolution of the percentile: at 4088 tokens it is 5,
    # i.e. p99.9 is the 5th-worst token and carries no more information than
    # the max.  (A count of values >= p99.9 would be confounded by ties and by
    # however many background samples land there.)
    out["p999_rank"] = int(ordered.numel() - p999_idx)
    out["p999_resolved"] = bool(out["p999_rank"] >= P999_MIN_RANK)
    return out


def top1_agreement(tlp: torch.Tensor, slp: torch.Tensor) -> float:
    """Fraction of tokens where teacher and student agree on the argmax token.

    Reported next to the KLD because it is the quantity the 1.7B canary failure
    mode (gold answer stuck at rank 2 behind near-ties) actually showed.
    """
    return float((tlp.argmax(-1) == slp.argmax(-1)).float().mean())


# ------------------------------------------------------------------- run -----

@torch.no_grad()
def teacher_pass(model, data, device, chunk: int, vocab: int):
    """Park fp32 teacher log-probs per window, plus the teacher's own entropy.

    The teacher entropy/peak are the reference the student's numbers are read
    against, so they are measured here rather than recomputed later.
    """
    from qwen35_moe_proxy import model_logits

    # ``windows()`` hands back one 1-D row per window, so the token count is the
    # number of elements, not shape[1] (which raises on a 1-D row).
    need = sum(d.numel() - 1 for d in data) * vocab * 4
    print(f"teacher cache: {need/1e9:.2f} GB of host memory "
          f"({len(data)} windows x {vocab} vocab)", flush=True)
    out, ents, tops = [], [], []
    for i in range(len(data)):
        ids = data[i:i + 1].to(device)
        lp = log_probs(model_logits(model, ids)[0, :-1], chunk).cpu()
        out.append(lp)
        ents.append(float(-(lp.exp() * lp).sum(-1).mean()))
        tops.append(float(lp.exp().max(-1).values.mean()))
        print(f"  teacher {i+1}/{len(data)} entropy {ents[-1]:.3f} peak {tops[-1]:.4f}",
              flush=True)
    return out, sum(ents) / len(ents), sum(tops) / len(tops)


@torch.no_grad()
def student_pass(model, data, device, tcache, chunk: int):
    """Per-token KLD, top-1 agreement, mean entropy and mean top-1 mass.

    Entropy and peak mass come along for free and are what make the KLD number
    readable: see the note in ``main``.
    """
    from qwen35_moe_proxy import model_logits

    per_token, agrees = [], []
    ents, tops = [], []
    for i in range(len(data)):
        ids = data[i:i + 1].to(device)
        slp = log_probs(model_logits(model, ids)[0, :-1], chunk).cpu()
        kld = kld_from_logprobs(tcache[i], slp, chunk)
        per_token.append(kld)
        agrees.append(top1_agreement(tcache[i], slp))
        ents.append(float(-(slp.exp() * slp).sum(-1).mean()))
        tops.append(float(slp.exp().max(-1).values.mean()))
        del slp
        print(f"  student {i+1}/{len(data)} mean kld {kld.mean():.4f} "
              f"max {kld.max():.4f} top1 {agrees[-1]:.4f} "
              f"H {ents[-1]:.3f} peak {tops[-1]:.4f}", flush=True)
    return (torch.cat(per_token), sum(agrees) / len(agrees),
            sum(ents) / len(ents), sum(tops) / len(tops))


def build_parser() -> argparse.ArgumentParser:
    """The CLI, as a function so tests can check flags without running anything."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prefix-layers", type=int, default=4,
                    help="N-layer prefix; must match the run that produced --load")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--eval-windows", type=int, default=8,
                    help="held-out windows (seed 999, wikitext) -- same as stage_eval")
    ap.add_argument("--worst", type=int, default=16,
                    help="how many of the worst per-token KLDs to record with their "
                         "window/position, so the tail is inspectable")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=32, help="token chunk for log_softmax")
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--quant", choices=["absmean", "lloyd", "catq"], default="lloyd")
    ap.add_argument("--catq-steps", type=int, default=200)
    ap.add_argument("--catq-lr", type=float, default=0.05)
    ap.add_argument("--catq-gamma", type=float, default=0.8)
    ap.add_argument("--catq-s0", type=float, default=30.0)
    ap.add_argument("--rank", type=int, default=512)
    ap.add_argument("--branch-quant", choices=["fp32", "g128", "rank"], default="g128")
    ap.add_argument("--branch-target", choices=["moe_out", "attn_out", "both"], default="both")
    ap.add_argument("--load", default="", help="branch checkpoint (omit = uncorrected body)")
    ap.add_argument("--split", default="wikitext")
    ap.add_argument("--out", default="", help="write the stats JSON here")
    ap.add_argument("--model-dir", default="", help="override $MOE_ARTIFACTS/empero-hf")
    return ap


def main() -> None:
    args = build_parser().parse_args()

    from transformers import AutoConfig, AutoTokenizer

    from olmoe_corrections import load_branch_state
    from olmoe_proxy import windows
    from qwen35_moe_proxy import MODEL, build_student, load_prefix, patch_experts

    model_dir = Path(args.model_dir) if args.model_dir else MODEL
    if not (model_dir / "config.json").exists():
        raise SystemExit(f"no checkpoint at {model_dir}; pass --model-dir or set "
                         f"MOE_ARTIFACTS (the FP empero-hf shards are needed for the "
                         f"teacher pass)")
    patch_experts(args.group)

    tok = AutoTokenizer.from_pretrained(model_dir)
    cfg = AutoConfig.from_pretrained(model_dir)
    tcfg = getattr(cfg, "text_config", cfg)
    vocab = int(tcfg.vocab_size)
    data = windows(tok, args.eval_windows, args.seq, 999, args.split)
    print(f"KLD eval: {args.prefix_layers}-layer prefix, vocab {vocab}, "
          f"{len(data)} windows x {args.seq} tokens, quant={args.quant} "
          f"branch={args.branch_target}/{args.branch_quant} load={args.load or 'none'}",
          flush=True)

    # ---- teacher: FP prefix, banks untouched, no branches
    model, _, _, _ = load_prefix(args.prefix_layers, args.device, model_dir=model_dir)
    tcache, t_ent, t_top = teacher_pass(model, data, args.device, args.chunk, vocab)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ---- student: same prefix, ternarised banks + branches + checkpoint
    model, _, _, _ = load_prefix(args.prefix_layers, args.device, model_dir=model_dir)
    n_tr = build_student(model, args)
    print(f"student built: trainable {n_tr/1e6:.2f}M", flush=True)
    if args.load:
        missing, unexpected = load_branch_state(model, args.load)
        # The checkpoint only ever holds branch/router tensors, so
        # load_state_dict reports the whole frozen body as "missing" -- that is
        # expected and not a signal.  What matters is that every branch/gate
        # tensor this model *wants* actually came from the file.
        want = {k for k in model.state_dict() if ".branch." in k or ".gate." in k}
        got = want & set(missing)
        print(f"loaded {args.load}: branch tensors {len(want) - len(got)}/{len(want)}, "
              f"unexpected={len(unexpected)} (frozen body keys in `missing` "
              f"({len(missing)}) are expected)", flush=True)
        if got or unexpected:
            raise SystemExit(
                f"checkpoint does not match this prefix/placement: "
                f"{len(got)} wanted branch/router tensor(s) did not load, "
                f"{len(unexpected)} did not belong. Re-run with the same "
                f"--prefix-layers/--branch-target/--rank as training.")
    per_token, agree, ent, top1p = student_pass(model, data, args.device, tcache,
                                                args.chunk)

    res = tail_stats(per_token)
    if not res["p999_resolved"]:
        print(f"WARNING: at {res['n_tokens']} tokens, p99.9 is only the "
              f"{res['p999_rank']}-worst token -- treat 'max' and 'worst_tokens' "
              f"as the tail signal, not p99.9. Raise --eval-windows to resolve it.",
              flush=True)
    # the worst individual tokens, with where they came from, so a tail move can
    # be inspected rather than taken on faith
    flat = per_token.reshape(-1).float()
    k = min(args.worst, flat.numel())
    top = torch.topk(flat, k)
    seq = args.seq - 1
    res["worst_tokens"] = [
        {"token": int(t), "window": int(t) // seq, "pos": int(t) % seq,
         "kld": float(v)}
        for v, t in zip(top.values, top.indices)
    ]
    # Entropy and peak mass are what make a KLD number interpretable.  A
    # correction branch that sharpens the student (entropy well below the
    # teacher's) can cut PPL while *raising* full-vocab KLD, because KL charges
    # for every token the teacher spreads mass onto that the student has since
    # crushed.  Without these two numbers, "KLD got worse" is uninterpretable.
    res["teacher_entropy_nats"] = round(float(t_ent), 4)
    res["teacher_top1_prob"] = round(float(t_top), 4)
    res["student_entropy_nats"] = round(float(ent), 4)
    res["student_top1_prob"] = round(float(top1p), 4)
    # a student markedly *sharper* than the teacher has been pushed away from
    # the teacher's distribution, which is the usual reason KLD rises while PPL
    # falls; say so in the record rather than leaving a bare number
    res["sharper_than_teacher"] = bool(ent < t_ent - 0.5)
    res.update({
        "top1_agreement": round(agree, 6),
        "checkpoint": args.load,
        "prefix_layers": args.prefix_layers,
        "quant": args.quant,
        "branch_target": args.branch_target,
        "branch_quant": args.branch_quant,
        "rank": args.rank,
        "eval_windows": args.eval_windows,
        "seq": args.seq,
    })
    print(json.dumps(res, indent=2), flush=True)
    out = Path(args.out) if args.out else OUT / "kld-eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
