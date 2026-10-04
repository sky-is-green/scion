"""Calibrate the uncorrected PQ2_0 noul output — free v0 baseline.

The held-out checkpoint (`hivebench/.../TERNARY-NOTES.md` Stage E) showed the
trained corrections flatten ranking (AUC < chance); the uncorrected body keeps
ranking (AUC 0.80 bench-train / 0.57 held-out) but is miscalibrated (useful
threshold ~0.46, not 0.5).  This script fits a monotone recalibration of the
head's ``noul`` output on the uncorrected body and evaluates it with balanced
metrics on the frozen probe.

Calibration (2 parameters, both fit on *train-distribution* data only):

    p_cal = sigmoid(a * (logit(p_raw) - logit(t)))

where ``t`` is the raw-probability threshold that maximises balanced accuracy
on the fit set and ``a`` is a Platt slope (BCE fit).  ``p_cal >= 0.5``
therefore holds exactly when ``p_raw >= t`` — the harness threshold 0.5 lands
on the fitted operating point.

Fit set: bench-train (70) + the 58 constructed training negatives (checked
wrong; a perturbation that stayed correct is dropped).  Frozen evaluation:
bench-test (30, stored bridge run) and the held-out probe (27 correct + 19
constructed negatives, stored harness runs).  The body is unchanged.

Usage:
    .venv-rocm/bin/python dense/clef_calibrate.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/home/penis/Desktop/work/hivebench")  # checker import
import clef_head as H  # noqa: E402
from clef_corrections import TeacherCache  # noqa: E402
from clef_eval import load_reference  # noqa: E402

HIVE = Path("/home/penis/Desktop/work/hivebench/experiments/cascade")
DEFAULT_MODEL = "/home/penis/Desktop/work/models/clef-flash-ternary"
DEFAULT_BODY = f"{DEFAULT_MODEL}/clef-flash-PQ2_0.gguf"
DEFAULT_BRIDGE = "/home/penis/Desktop/work/hivebench/tools/clef-bridge/clef_embed"
DEFAULT_BENCH = f"{DEFAULT_MODEL}/corrections/packaged/bridge-uncorrected.json"
DEFAULT_HOLDOUT = f"{HIVE}/results/clef-flash-validator-20261003"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def balance(pos: list[float], neg: list[float], t: float) -> tuple[float, int, int]:
    """(balanced acc, false accepts, false rejects) at threshold t."""
    tp = sum(p >= t for p in pos)
    tn = sum(p < t for p in neg)
    tpr = tp / len(pos) if pos else 0.0
    tnr = tn / len(neg) if neg else 0.0
    return 0.5 * (tpr + tnr), len(neg) - tn, len(pos) - tp


def fit_platt(xs: list[float], ys: list[float]) -> float:
    """Monotone slope a on logit space (Platt with intercept), LBFGS.

    The intercept is fitted for a proper slope but discarded: the shipped
    calibration fixes its own boundary (raw threshold t) afterwards.
    """
    X = torch.tensor(xs, dtype=torch.float32)
    Y = torch.tensor(ys, dtype=torch.float32)
    raw = torch.tensor(0.0, requires_grad=True)  # a = softplus(raw) + eps
    bias = torch.tensor(0.0, requires_grad=True)

    def slope() -> torch.Tensor:
        return torch.nn.functional.softplus(raw) + 1e-3

    opt = torch.optim.LBFGS([raw, bias], lr=0.5, max_iter=200)

    def closure() -> torch.Tensor:
        opt.zero_grad()
        a = slope()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(a * X + bias, Y)
        loss = loss + 1e-3 * (a - 1.0) ** 2
        loss.backward()
        return loss

    opt.step(closure)
    return float(slope().detach())


@torch.no_grad()
def head_p(head, lm_head, tok, js, entry: dict, hidden: np.ndarray) -> float:
    request = {
        "model": "clef-flash",
        "state": {"task": entry["prompt"], "candidate_answer": entry["candidate"]},
        "questions": {"verdict": {"type": "noul",
                                  "instructions": H.DEFAULT_INSTRUCTION}},
    }
    enc = js.encode_record(tok, request, max_length=16384)
    ids = torch.tensor(enc.input_ids, dtype=torch.long)
    logits = H.head_logits(head, torch.from_numpy(hidden).unsqueeze(0),
                           ids.unsqueeze(0), torch.ones(1, ids.numel(), dtype=torch.long),
                           [enc], lm_head)[0][0]
    return float(logits.float().softmax(-1)[0])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--cache", default=f"{DEFAULT_MODEL}/corrections/cache-smoke")
    ap.add_argument("--body", default=DEFAULT_BODY)
    ap.add_argument("--bridge", default=DEFAULT_BRIDGE)
    ap.add_argument("--bench-json", default=DEFAULT_BENCH)
    ap.add_argument("--holdout-dir", default=DEFAULT_HOLDOUT)
    ap.add_argument("--out", default=f"{DEFAULT_MODEL}/corrections/packaged/calibration-uncorrected.json")
    ap.add_argument("--work", default=f"{DEFAULT_MODEL}/corrections/packaged/calibration-work")
    ap.add_argument("--n-ctx", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from experiments.cascade.checkers import check

    tasks = {t["id"]: t for t in json.loads(
        (HIVE / "tasks-bench.json").read_text())["tasks"]}

    cache = TeacherCache(args.cache)
    neg_idx = [i for i, e in enumerate(cache.entries) if e.get("kind") == "negative"]
    if args.limit:
        neg_idx = neg_idx[:args.limit]
    print(f"fit negatives to score: {len(neg_idx)}", flush=True)

    # 1. score the training negatives through the uncorrected bridge + head
    work = Path(args.work)
    (work / "in").mkdir(parents=True, exist_ok=True)
    for i in neg_idx:
        z = np.load(cache.root / cache.entries[i]["file"])
        ids = z["input_ids"].astype(np.int64)
        (work / "in" / f"{i:05d}.tok").write_text(" ".join(str(int(t)) for t in ids))

    log_path = work / "bridge.log"
    print(f"bridge {Path(args.body).name} ...", flush=True)
    t0 = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.run([args.bridge, args.body, str(work / "in"),
                               str(work / "out"), str(args.n_ctx)],
                              stdout=log, stderr=log, text=True)
    if proc.returncode != 0:
        print("\n".join(log_path.read_text().splitlines()[-5:]), flush=True)
        return proc.returncode
    print(f"bridge done in {time.time()-t0:.0f}s", flush=True)

    head, js = H.load_joint_head(args.model, device="cpu", dtype=torch.float32)
    lm_head = H.load_lm_head(args.model, device="cpu", dtype=torch.float32)
    tok = H.load_tokenizer(args.model)

    fit_x: list[float] = []
    fit_y: list[float] = []
    dropped = 0
    for i in neg_idx:
        e = cache.entries[i]
        raw = np.fromfile(work / "out" / f"{i:05d}.bin", dtype=np.float32)
        hidden = raw.reshape(-1, 4096)
        p = head_p(head, lm_head, tok, js, e, hidden)
        task_id = e["record_id"][:-len("-neg")]
        gold = bool(check(tasks[task_id], e["candidate"]))
        if gold:
            dropped += 1
            continue
        fit_x.append(logit(p))
        fit_y.append(0.0)
    print(f"negatives scored, usable {len(fit_y)} (dropped {dropped} still-correct)",
          flush=True)

    # 2. bench-train positives/negatives from the stored uncorrected run
    bench = json.loads(Path(args.bench_json).read_text())
    train_rows = bench["train"]["rows"]
    for r in train_rows:
        fit_x.append(logit(r["p"]))
        fit_y.append(1.0 if r["gold"] else 0.0)
    n_pos = sum(fit_y)
    n_neg = len(fit_y) - n_pos
    fit_auc = auc([x for x, y in zip(fit_x, fit_y) if y],
                  [x for x, y in zip(fit_x, fit_y) if not y])
    print(f"fit set: n {len(fit_y)} ({int(n_pos)} pos, {int(n_neg)} neg)  AUC {fit_auc:.3f}",
          flush=True)

    # 3. slope, then the BA-optimal raw threshold; b follows so p_cal=0.5 there
    a = fit_platt(fit_x, fit_y)
    pos_x = [x for x, y in zip(fit_x, fit_y) if y]
    neg_x = [x for x, y in zip(fit_x, fit_y) if not y]
    best = None
    for k in range(1, 100):
        t = k / 100.0
        bacc, fa, fr = balance([sigmoid(x) for x in pos_x], [sigmoid(x) for x in neg_x], t)
        if best is None or bacc > best[1]:
            best = (t, bacc, fa, fr)
    t_star, fit_ba, fit_fa, fit_fr = best
    print(f"platt a {a:.4f}  threshold t {t_star:.2f}  fit bal-acc {fit_ba:.3f} "
          f"(FA {fit_fa}, FR {fit_fr})", flush=True)

    def calibrate(p: float) -> float:
        return sigmoid(a * (logit(p) - logit(t_star)))

    # 4. evaluate on the frozen sets (stored, bit-exact same pipeline)
    def evaluate(name: str, pos: list[float], neg: list[float]) -> dict:
        raw_ba, raw_fa, raw_fr = balance(pos, neg, 0.5)
        cal_pos = [calibrate(p) for p in pos]
        cal_neg = [calibrate(p) for p in neg]
        cal_ba, cal_fa, cal_fr = balance(cal_pos, cal_neg, 0.5)
        correct_raw = len(pos) - raw_fr + (len(neg) - raw_fa)
        correct_cal = len(pos) - cal_fr + (len(neg) - cal_fa)
        out = {
            "n": len(pos) + len(neg), "pos": len(pos), "neg": len(neg),
            "auc": round(auc(pos, neg), 4),
            "raw@0.5": {"correct": correct_raw, "false_accepts": raw_fa,
                        "false_rejects": raw_fr, "balanced_acc": round(raw_ba, 4)},
            "cal@0.5": {"correct": correct_cal, "false_accepts": cal_fa,
                        "false_rejects": cal_fr, "balanced_acc": round(cal_ba, 4)},
        }
        print(f"[{name}] n {out['n']} AUC {out['auc']:.3f} | raw@0.5 correct "
              f"{correct_raw}/{out['n']} FA {raw_fa} FR {raw_fr} BA {raw_ba:.3f} | "
              f"cal@0.5 correct {correct_cal}/{out['n']} FA {cal_fa} FR {cal_fr} "
              f"BA {cal_ba:.3f}", flush=True)
        return out

    test_rows = bench["test"]["rows"]
    report = {
        "body": str(args.body), "body_sha256": sha256(Path(args.body)),
        "protocol": "p_cal = sigmoid(a * (logit(p_raw) - logit(t))); "
                    "a = Platt slope (BCE), t = raw threshold maximizing balanced "
                    "accuracy on the fit set; fit set = bench-train + training negatives",
        "fit": {"n": len(fit_y), "pos": int(n_pos), "neg": int(n_neg),
                "auc": round(fit_auc, 4), "platt_a": round(a, 4),
                "threshold_raw": t_star, "balanced_acc": round(fit_ba, 4),
                "false_accepts": fit_fa, "false_rejects": fit_fr},
        "calibration": {"a": a, "threshold_raw": t_star,
                        "b": -a * logit(t_star)},
        "bench_train": evaluate("bench-train", [r["p"] for r in train_rows if r["gold"]],
                                [r["p"] for r in train_rows if not r["gold"]]),
        "bench_test": evaluate("bench-test", [r["p"] for r in test_rows if r["gold"]],
                               [r["p"] for r in test_rows if not r["gold"]]),
    }
    hold = Path(args.holdout_dir)
    hold_pos = json.loads((hold / "holdout-accept-uncorr.json").read_text())
    hold_neg = json.loads((hold / "holdout-reject-uncorr.json").read_text())
    key = "clef-flash-ternary-uncorr"
    report["holdout"] = evaluate(
        "held-out",
        list(hold_pos["models"][key]["p_correct"].values()),
        list(hold_neg["models"][key]["p_correct"].values()))

    out = Path(args.out)
    out.write_text(json.dumps(report, indent=1))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
