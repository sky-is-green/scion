"""Probe diagnostics for the ternary checkpoint (local, no training).

Two phases, both through the harness bridge + CPU head sidecar:

``teacher``
    Run the f16 GGUF (the teacher states the corrections distil from) over the
    frozen 46-record hold-out probe (27 correct + 19 constructed negatives) and
    score the joint head.  Gives the reference AUC/FA/FR the probe is asking
    for.

``corrected-neg``
    Run the *corrected* merged body over the 58 training negatives and compare
    against the teacher's cached option logits.  If the corrections reject the
    negatives they were trained on but not the held-out ones, the failure is
    overfitting; if they reject neither, the decision KD never worked.

Usage:
    .venv-rocm/bin/python dense/clef_probe_diag.py --phase teacher
    .venv-rocm/bin/python dense/clef_probe_diag.py --phase corrected-neg
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clef_head as H  # noqa: E402
from clef_corrections import TeacherCache  # noqa: E402

HIVE = Path("/home/penis/Desktop/work/hivebench/experiments/cascade")
DEFAULT_MODEL = "/home/penis/Desktop/work/models/clef-flash-ternary"
DEFAULT_BRIDGE = "/home/penis/Desktop/work/hivebench/tools/clef-bridge/clef_embed"
HOLDOUT = HIVE / "results/clef-flash-validator-20261003"


def auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def score_rows(rows: list[dict], work: Path, bridge: str, body: str,
               model: str, n_ctx: int) -> list[dict]:
    """rows: [{id, prompt, candidate, gold, tag}] -> p per row via bridge+head."""
    work.mkdir(parents=True, exist_ok=True)
    head, js = H.load_joint_head(model, device="cpu", dtype=torch.float32)
    lm_head = H.load_lm_head(model, device="cpu", dtype=torch.float32)
    tok = H.load_tokenizer(model)
    for k, r in enumerate(rows):
        enc = H.encode(tok, r["prompt"], r["candidate"])
        r["_enc"] = enc
        (work / "in" / f"{k:05d}.tok").parent.mkdir(exist_ok=True)
        (work / "in" / f"{k:05d}.tok").write_text(
            " ".join(str(int(t)) for t in enc.input_ids))
    log = work / "bridge.log"
    print(f"bridge {Path(body).name} over {len(rows)} records ...", flush=True)
    t0 = time.time()
    with open(log, "w") as f:
        proc = subprocess.run([bridge, body, str(work / "in"), str(work / "out"),
                               str(n_ctx)], stdout=f, stderr=f, text=True)
    if proc.returncode != 0:
        tail = "\n".join(log.read_text().splitlines()[-5:])
        raise SystemExit(f"bridge failed:\n{tail}")
    print(f"bridge done {time.time()-t0:.0f}s", flush=True)
    with torch.no_grad():
        for k, r in enumerate(rows):
            raw = np.fromfile(work / "out" / f"{k:05d}.bin", dtype=np.float32)
            hidden = raw.reshape(-1, 4096)
            ids = torch.tensor(r["_enc"].input_ids, dtype=torch.long)
            logits = H.head_logits(
                head, torch.from_numpy(hidden).unsqueeze(0), ids.unsqueeze(0),
                torch.ones(1, ids.numel(), dtype=torch.long), [r["_enc"]], lm_head)[0][0]
            r["p"] = float(logits.float().softmax(-1)[0])
            del r["_enc"]
    return rows


def tasks_index() -> dict:
    out = {}
    for f in ("tasks.json", "tasks-hard.json"):
        for t in json.loads((HIVE / f).read_text())["tasks"]:
            out[t["id"]] = t
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", choices=["teacher", "corrected-neg", "body"],
                    required=True)
    ap.add_argument("--body", default="", help="merged GGUF for --phase body")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--bridge", default=DEFAULT_BRIDGE)
    ap.add_argument("--n-ctx", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N records")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    work = Path(f"/tmp/opencode/clef-diag-{args.phase}")

    if args.phase == "teacher":
        tasks = tasks_index()
        rows = []
        for f, gold in (("holdout-probe-pos.json", True), ("holdout-probe-neg.json", False)):
            for r in json.loads((HOLDOUT / f).read_text())["records"]:
                task = tasks[r["id"]]
                rows.append({"id": r["id"], "prompt": task["prompt"],
                             "candidate": r["scion_answer"], "gold": gold})
        if args.limit:
            rows = rows[:args.limit]
        rows = score_rows(rows, work, args.bridge, f"{args.model}/clef-flash-f16.gguf",
                          args.model, args.n_ctx)
        pos = [r["p"] for r in rows if r["gold"]]
        neg = [r["p"] for r in rows if not r["gold"]]
        out = {"body": "clef-flash-f16.gguf", "n": len(rows), "auc": round(auc(pos, neg), 4),
               "thresholds": {}, "rows": rows}
        for t in (0.3, 0.4, 0.5, 0.55, 0.6):
            out["thresholds"][f"{t:.2f}"] = {
                "false_accepts": sum(p >= t for p in neg),
                "false_rejects": sum(p < t for p in pos),
                "correct": sum(p >= t for p in pos) + sum(p < t for p in neg)}
        print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1), flush=True)
    elif args.phase == "body":
        body = Path(args.body)
        if not body.is_absolute():
            body = Path(args.model) / body
        tasks = tasks_index()
        rows = []
        for f, gold in (("holdout-probe-pos.json", True),
                        ("holdout-probe-neg.json", False)):
            for r in json.loads((HOLDOUT / f).read_text())["records"]:
                task = tasks[r["id"]]
                rows.append({"id": r["id"], "prompt": task["prompt"],
                             "candidate": r["scion_answer"], "gold": gold})
        if args.limit:
            rows = rows[:args.limit]
        rows = score_rows(rows, work, args.bridge, str(body),
                          args.model, args.n_ctx)
        pos = [r["p"] for r in rows if r["gold"]]
        neg = [r["p"] for r in rows if not r["gold"]]
        out = {"body": body.name, "n": len(rows), "auc": round(auc(pos, neg), 4),
               "thresholds": {}, "rows": rows}
        for t in (0.3, 0.4, 0.5, 0.55, 0.6):
            out["thresholds"][f"{t:.2f}"] = {
                "false_accepts": sum(p >= t for p in neg),
                "false_rejects": sum(p < t for p in pos),
                "correct": sum(p >= t for p in pos) + sum(p < t for p in neg)}
        print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1),
              flush=True)
    else:
        cache = TeacherCache(f"{args.model}/corrections/cache-smoke")
        rows = []
        for i, e in enumerate(cache.entries):
            if e.get("kind") != "negative":
                continue
            z = np.load(cache.root / e["file"])
            teacher = z["option_logits"].astype(np.float32)
            e_t = np.exp(teacher - teacher.max())
            rows.append({"id": e["record_id"], "prompt": e["prompt"],
                         "candidate": e["candidate"], "gold": False,
                         "teacher_p": float((e_t / e_t.sum())[0]),
                         "teacher_finite": bool(np.isfinite(teacher).all())})
        if args.limit:
            rows = rows[:args.limit]
        rows = score_rows(rows, work, args.bridge,
                          f"{args.model}/corrections/packaged/clef-flash-PQ2_0-corr-r512-g128-step78.gguf",
                          args.model, args.n_ctx)
        neg = [r["p"] for r in rows]
        teacher = [r["teacher_p"] for r in rows if r["teacher_finite"]]
        out = {"body": "corrected merged", "n": len(rows),
               "p_range": [round(min(neg), 4), round(max(neg), 4)],
               "p_above_0.5": sum(p >= 0.5 for p in neg),
               "teacher_above_0.5": sum(p >= 0.5 for p in teacher),
               "rows": rows}
        print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1), flush=True)

    dest = Path(args.out) if args.out else HOLDOUT / f"diag-{args.phase}.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
