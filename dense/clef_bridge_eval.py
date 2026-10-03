"""Honest CPU-bridge decision benchmark for the packaged dense Clef corrections.

For every bench record in the teacher cache this runs the fork bridge
(``hivebench/tools/clef-bridge/clef_embed``, CPU, one model load) on the body
under test, loads the f32 per-token hidden states, runs the joint head (CPU
f32), and scores ``p_correct`` against the recorded bf16 probability and the
programmatic checker at 0.5 -- the same rows and metric as ``dense/clef_eval.py``,
but on the deployed llama.cpp forward instead of the torch proxy.

Run it once per body (uncorrected PQ2_0, then the merged corrected release) and
compare the JSONs.  Bridge decode ms and head ms are recorded per record, so the
report carries the CPU latency as well as parity.

Usage:
    .venv-rocm/bin/python dense/clef_bridge_eval.py \
        --cache .../corrections/cache-smoke \
        --body .../clef-flash-PQ2_0-corr-r512-g128-step78.gguf \
        --bridge hivebench/tools/clef-bridge/clef_embed \
        --tag corrected --out .../corrections/packaged/bridge-corrected.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
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
from clef_eval import load_reference  # noqa: E402

DEFAULT_MODEL = "/home/penis/Desktop/work/models/clef-flash-ternary"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_tokens(cache: TeacherCache, entries: list[dict], tok_dir: Path) -> None:
    tok_dir.mkdir(parents=True, exist_ok=True)
    for i in entries:
        e = cache.entries[i]
        z = np.load(cache.root / e["file"])
        ids = z["input_ids"].astype(np.int64)
        (tok_dir / f"{i:05d}.tok").write_text(" ".join(str(int(t)) for t in ids))


@torch.no_grad()
def head_p(head, lm_head, tok, entry: dict, hidden: np.ndarray) -> tuple[float, float]:
    enc = H.encode(tok, entry["prompt"], entry["candidate"])
    ids = torch.tensor(enc.input_ids, dtype=torch.long)
    t0 = time.time()
    logits = H.head_logits(head, torch.from_numpy(hidden).unsqueeze(0),
                           ids.unsqueeze(0), torch.ones(1, ids.numel(), dtype=torch.long),
                           [enc], lm_head)[0][0]
    ms = (time.time() - t0) * 1000.0
    return float(logits.float().softmax(-1)[0]), ms


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--body", required=True)
    ap.add_argument("--bridge", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="", help="scratch dir (default: alongside --out)")
    ap.add_argument("--n-ctx", type=int, default=1024)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N records")
    ap.add_argument("--refs", default="", help="dir with the bf16 judge-eval JSONs")
    args = ap.parse_args()

    body = Path(args.body)
    cache = TeacherCache(args.cache)
    out = Path(args.out)
    work = Path(args.work) if args.work else out.parent / f"bridge-{args.tag}"
    work.mkdir(parents=True, exist_ok=True)

    # 1. exact head token ids per bench record
    entries = [i for i, e in enumerate(cache.entries)
               if e.get("kind") == "bench" and e.get("has_decision")]
    if args.limit:
        entries = entries[:args.limit]
    print(f"bench records: {len(entries)}", flush=True)
    prepare_tokens(cache, entries, work / "in")

    # 2. bridge: one model load, fresh context per record
    log_path = work / "bridge.log"
    print(f"bridge {body.name} ...", flush=True)
    t0 = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.run(
            [args.bridge, str(body), str(work / "in"), str(work / "out"), str(args.n_ctx)],
            stdout=log, stderr=log, text=True)
    wall = time.time() - t0
    if proc.returncode != 0:
        tail = log_path.read_text().splitlines()[-5:]
        print("bridge failed; last log lines:\n  " + "\n  ".join(tail), flush=True)
        return proc.returncode
    rec_ms: dict[str, float] = {}
    for line in log_path.read_text().splitlines():
        if line.startswith("REC "):
            _, stem, ntok, ms = line.split()
            rec_ms[stem] = float(ms)
    print(f"bridge done in {wall:.0f}s ({len(rec_ms)} records)", flush=True)

    # 3. head sidecar on CPU f32 (the validated path)
    head, _ = H.load_joint_head(args.model, device="cpu", dtype=torch.float32)
    lm_head = H.load_lm_head(args.model, device="cpu", dtype=torch.float32)
    tok = H.load_tokenizer(args.model)

    refs_dir = Path(args.refs) if args.refs else None
    report = {"tag": args.tag, "body": str(body), "body_sha256": sha256(body),
              "bridge": str(args.bridge), "bridge_sha256": sha256(Path(args.bridge)),
              "n_ctx": args.n_ctx, "wall_s": round(wall, 1),
              "bridge_log": str(log_path)}
    for split in ("train", "test"):
        ref = load_reference(split, refs_dir)
        rows = []
        for i in entries:
            e = cache.entries[i]
            if e["split"] != split:
                continue
            z = np.load(cache.root / e["file"])
            raw = np.fromfile(work / "out" / f"{i:05d}.bin", dtype=np.float32)
            hidden = raw.reshape(-1, 4096)
            ntok = int(z["input_ids"].shape[0])
            if hidden.shape[0] != ntok:
                raise SystemExit(f"{e['record_id']}: bridge gave {hidden.shape[0]} tokens, "
                                 f"cache has {ntok}")
            p, head_ms = head_p(head, lm_head, tok, e, hidden)
            teacher = z["hidden"].astype(np.float32)
            cos = float(np.mean(np.sum(hidden * teacher, axis=1) /
                                (np.linalg.norm(hidden, axis=1) * np.linalg.norm(teacher, axis=1) + 1e-9)))
            rows.append({"id": e["record_id"], "p": p,
                         "p_bf16": ref["p_bf16"].get(e["record_id"]),
                         "gold": ref["gold_ok"].get(e["record_id"]),
                         "cos": cos, "n_tokens": ntok,
                         "bridge_ms": rec_ms.get(f"{i:05d}"), "head_ms": round(head_ms, 1)})
        thr = args.threshold
        acc = [r["p"] >= thr for r in rows]
        correct = sum(1 for a, r in zip(acc, rows) if a == r["gold"])
        fa = sum(1 for a, r in zip(acc, rows) if a and not r["gold"])
        fr = sum(1 for a, r in zip(acc, rows) if not a and r["gold"])
        d = [r["p"] - r["p_bf16"] for r in rows if r["p_bf16"] is not None]
        if not rows:
            print(f"[{split}] no records", flush=True)
            report[split] = {"n": 0}
            continue
        report[split] = {
            "n": len(rows), "correct": correct, "false_accepts": fa, "false_rejects": fr,
            "mean_p": round(statistics.mean(r["p"] for r in rows), 4),
            "mean_p_bf16": round(statistics.mean(r["p_bf16"] for r in rows
                                                 if r["p_bf16"] is not None), 4),
            "mean_p_delta": round(statistics.mean(d), 4) if d else None,
            "mean_cos": round(statistics.mean(r["cos"] for r in rows), 5),
            "mean_bridge_ms": round(statistics.mean(r["bridge_ms"] for r in rows), 1),
            "mean_head_ms": round(statistics.mean(r["head_ms"] for r in rows), 1),
            "rows": sorted(rows, key=lambda r: r["p"]),
        }
        s = report[split]
        print(f"[{split}] n {s['n']} correct {correct}/{len(rows)} FA {fa} FR {fr} "
              f"mean_p {s['mean_p']} (bf16 {s['mean_p_bf16']}, d {s['mean_p_delta']}) "
              f"cos {s['mean_cos']} bridge {s['mean_bridge_ms']}ms head {s['mean_head_ms']}ms",
              flush=True)

    out.write_text(json.dumps(report, indent=1))
    print(f"wrote {out} ({wall:.0f}s bridge wall, body {report['body_sha256'][:12]})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
