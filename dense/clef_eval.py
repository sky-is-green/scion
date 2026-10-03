"""Decision-parity / hidden-cosine evaluation for the dense Clef corrections.

Loads the deployed PQ2_0 student (optionally with a trained branch checkpoint),
runs the joint head on the student's post-norm states over the cached decision
records, and reports against the recorded bf16 ``p_correct`` and the programmatic
checker labels at threshold 0.5 -- on train and the held-out test split.

Also reports the per-sequence final-hidden cosine to the f16 teacher, so the
"did the body get closer" question is separable from the decision metric.

Usage:
    python dense/clef_eval.py --cache <dir> [--checkpoint branches.pt]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from clef_dense_load import (load_text_model_streamed, patch_recurrent_gdn,  # noqa: E402
                             patch_truncated_gdn)
import clef_head as H  # noqa: E402
from clef_corrections import TeacherCache, attach_branches  # noqa: E402

HIVE = Path("/home/penis/Desktop/work/hivebench/experiments/cascade")
EVAL_JSON = {
    "train": HIVE / "results/clef-flash-judge-eval-train.json",
    "test": HIVE / "results/clef-flash-judge-eval-test.json",
}
DEFAULT_MODEL = "/home/penis/Desktop/work/models/clef-flash-ternary"


def load_reference(split: str, refs_dir: Path | None = None) -> dict:
    path = (refs_dir / EVAL_JSON[split].name) if refs_dir else EVAL_JSON[split]
    d = json.loads(path.read_text())
    model = next(iter(d["models"].values()))
    ref = {k: v["scion_ok"] for k, v in d["labels"].items()}
    return {"p_bf16": model["p_correct"], "gold_ok": ref}


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--rank", type=int, default=512)
    ap.add_argument("--branch-quant", choices=["g128", "rank", "fp32"], default="g128")
    ap.add_argument("--target", choices=["attn_out", "mlp_out", "both"], default="both")
    ap.add_argument("--gdn", choices=["recurrent", "chunked"], default="recurrent",
                    help="evaluation forward; use the deployment form (recurrent) for "
                         "the honest number, or the training form for a like-for-like check")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--refs", default="", help="dir with the bf16 judge-eval JSONs")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    dt = torch.float32 if args.dtype == "float32" else torch.bfloat16
    model, _, _ = load_text_model_streamed(
        Path(args.model) / "clef-flash-PQ2_0.gguf", device=args.device, dtype=dt)
    if args.gdn == "recurrent":
        patch_recurrent_gdn(model)
    attach_branches(model, args.rank, args.branch_quant, args.target)
    if args.checkpoint:
        sd = torch.load(args.checkpoint, map_location="cpu")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"loaded {args.checkpoint}: missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    head, _ = H.load_joint_head(args.model, device="cpu", dtype=torch.float32)
    lm_head = H.load_lm_head(args.model, device="cpu", dtype=torch.float32)
    tok = H.load_tokenizer(args.model)
    cache = TeacherCache(args.cache)

    refs_dir = Path(args.refs) if args.refs else None
    report = {}
    for split in ("train", "test"):
        ref = load_reference(split, refs_dir)
        rows = []
        for i in cache.decision_indices(split):
            rec = cache.load(i, args.device)
            entry = rec["entry"]
            if entry.get("kind") != "bench":
                continue  # negatives are training-only; the metric is the gold set
            ids = rec["input_ids"]
            hidden = model(input_ids=ids.unsqueeze(0),
                           attention_mask=torch.ones_like(ids).unsqueeze(0),
                           use_cache=False).last_hidden_state[0]
            enc = H.encode(tok, entry["prompt"], entry["candidate"])
            logits = H.head_logits(head, hidden.unsqueeze(0).float().cpu(),
                                   ids.cpu().unsqueeze(0),
                                   torch.ones(1, ids.numel(), dtype=torch.long),
                                   [enc], lm_head)[0][0]
            p = float(logits.float().softmax(-1)[0])
            cos = float(F.cosine_similarity(hidden.float(), rec["hidden"], dim=-1).mean())
            rows.append({"id": entry["record_id"], "p": p,
                         "p_bf16": ref["p_bf16"].get(entry["record_id"]),
                         "gold": ref["gold_ok"].get(entry["record_id"]),
                         "cos": cos})
        thr = args.threshold
        acc = [r["p"] >= thr for r in rows]
        correct = sum(1 for a, r in zip(acc, rows) if a == r["gold"])
        fa = sum(1 for a, r in zip(acc, rows) if a and not r["gold"])
        fr = sum(1 for a, r in zip(acc, rows) if not a and r["gold"])
        p_deltas = [r["p"] - r["p_bf16"] for r in rows if r["p_bf16"] is not None]
        report[split] = {
            "n": len(rows), "correct": correct, "false_accepts": fa, "false_rejects": fr,
            "mean_p": round(statistics.mean(r["p"] for r in rows), 4),
            "mean_p_bf16": round(statistics.mean(r["p_bf16"] for r in rows if r["p_bf16"] is not None), 4),
            "mean_p_delta": round(statistics.mean(p_deltas), 4) if p_deltas else None,
            "mean_cos": round(statistics.mean(r["cos"] for r in rows), 5),
            "rows": sorted(rows, key=lambda r: r["p"]),
        }
        s = report[split]
        print(f"[{split}] n {s['n']} correct {correct}/{len(rows)} FA {fa} FR {fr} "
              f"mean_p {s['mean_p']} (bf16 {s['mean_p_bf16']}, d {s['mean_p_delta']}) "
              f"mean_cos {s['mean_cos']}", flush=True)

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
