#!/usr/bin/env python
"""The JEV head's reference test on q36: engine verdicts, parity against the JSON head, AUC, latency.

Starts `strata-q36 --serve` (the 35B) with STRATA_Q36_JEV_HEAD, sends one EMB per judge-bench item,
and checks three things:

  parity   the engine's `EMB <dim> <path> <p>` probability against the same head applied in Python
           to the very vector the engine wrote (the fit path and the serve path are the same number);
  quality  the frozen-split AUC/BA of the engine's verdicts (the head was fit on our q36 states);
  cost     the judge-pass milliseconds per item (the model's own prompt read dominates).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parents[2]
OUT = WORK / "scion" / "decision-heads-work"
WT = WORK / "strata-dh"
PACK35 = Path("/home/penis/models/scion-strata/pack-35b/tokenizer")
GGUF35 = Path("/home/penis/models/scion-strata/Scion-35B-A3B-PQ2_0-corr.gguf")
EXE = WT / "build-hip" / "strata-q36"

sys.path.insert(0, str(OUT))
from head_bin import prob  # noqa: E402


def read_until(proc, want: tuple[str, ...], timeout_s: float) -> str:
    deadline = time.monotonic() + timeout_s
    while True:
        if time.monotonic() > deadline:
            raise TimeoutError(f"engine silent for {timeout_s:.0f} s")
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError(f"engine exited (code {proc.poll()})")
        s = line.decode("utf-8", "replace").rstrip("\n")
        if s.startswith(want):
            return s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rows", type=Path, default=OUT / "judge-prompt-ids.jsonl")
    ap.add_argument("--head-json", type=Path, default=OUT / "scion-35b-a3b-decision-head-q36.json")
    ap.add_argument("--head-bin", type=Path, default=OUT / "head-35b.bin")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--hip", type=int, default=1)
    ap.add_argument("--dump", type=Path, default=None, help="also write a logit dump (byte-distill smoke)")
    ap.add_argument("--table", type=Path, default=WORK / "scion" / "byte-distill" / "byte-table.bin")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.rows.read_text().splitlines()]
    if args.limit:
        rows = rows[: args.limit]
    vec_dir = OUT / "states-verdict"
    vec_dir.mkdir(exist_ok=True)
    env = dict(os.environ)
    env["HIP_VISIBLE_DEVICES"] = str(args.hip)
    env["STRATA_Q36_JEV_HEAD"] = str(args.head_bin)
    if args.dump is not None:
        env["STRATA_Q36_LOGIT_DUMP"] = str(args.dump)
        env["STRATA_Q36_BYTE_TABLE"] = str(args.table)
    log = open(OUT / "dh-verdict.engine.log", "ab")
    cmd = [str(EXE), "--serve", "--model", str(GGUF35), "--max-context", "4096", "--chunk", "512"]
    print("verdict: " + " ".join(cmd) + f"  (HIP {args.hip}, head {args.head_bin.name})")
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                            cwd=str(WT), env=env)
    try:
        print("verdict:", read_until(proc, ("READY",), 600), f"in {time.time()-t0:.0f} s")
        engine_p, py_p, labels, splits, times = [], [], [], [], []
        t0 = time.time()
        for n, row in enumerate(rows, 1):
            path = vec_dir / f"{row['item_id'].replace(':', '_')}.f32"
            ids = ",".join(str(i) for i in row["ids"])
            tb = time.time()
            proc.stdin.write(f"EMB path={path} {ids}\n".encode())
            proc.stdin.flush()
            line = read_until(proc, ("EMB ", "ERR"), 900)
            if line.startswith("ERR"):
                raise RuntimeError(f"engine: {line}")
            times.append((time.time() - tb) * 1000)
            f = line.split()
            if len(f) < 4:
                raise RuntimeError(f"no verdict in {line!r} (is the head loaded?)")
            engine_p.append(float(f[3]))
            vec = np.fromfile(path, dtype=np.float32)
            py_p.append(prob(args.head_bin, vec))          # the same head, same vector, in Python
            labels.append(int(row["label"]))
            splits.append(row["split"])
            if n % 50 == 0 or n == len(rows):
                print(f"verdict: [{n}/{len(rows)}] {time.time()-t0:.0f}s", flush=True)
        y = np.asarray(labels)
        te = np.asarray([s == "test" for s in splits])
        e = np.asarray(engine_p)
        p = np.asarray(py_p)
        parity = float(np.max(np.abs(e - p)))
        sys.path.insert(0, str(WORK / "hivebench" / "experiments" / "cascade"))
        from readout_judge import auc, best_ba
        print(f"verdict: parity engine vs Python head: worst |diff| = {parity:.3e} over {len(e)} items")
        if te.sum() > 0 and len(set(y[te])) > 1:
            ba, thr = best_ba(list(e[te]), list(y[te]))
            auc_te = auc(list(e[te]), list(y[te]))
            print(f"verdict: engine verdicts, test split n={int(te.sum())}: AUC {auc_te:.4f} BA {ba:.4f} @ {thr:.2f}")
        else:
            bit = float(np.mean((e >= 0.5) == (y == 1)))
            ba, thr, auc_te = 0.0, 0.5, None
            print(f"verdict: engine verdicts separate {int(np.sum(y))} pos / {int(len(y) - np.sum(y))} neg "
                  f"at 0.5 with accuracy {bit:.3f} (too few items for AUC)")
        print(f"verdict: judge pass {np.mean(times):.0f} ms/item (min {np.min(times):.0f}, max {np.max(times):.0f})")
        (OUT / "dh-verdict.json").write_text(json.dumps({
            "n": len(e), "parity_worst": parity, "mean_ms": float(np.mean(times)),
            "test_auc": auc_te, "test_ba": ba, "threshold": thr, "head": str(args.head_json)}, indent=1))
        print("verdict: wrote", OUT / "dh-verdict.json")
        ok = parity < 1e-5
        return 0 if ok else 1
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
