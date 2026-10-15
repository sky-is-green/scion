#!/usr/bin/env python
"""Resumable embed phase for the community arms (the study's phase_embed redoes everything).

Builds the same items as `community_eval.py --phase embed` (samples then hard, answer and
trace judge prompts per item) and embeds them through a llama.cpp-compatible server
(`/apply-template` + `/embedding`), but writes one vector pair per item into
`{out}/{prefix}-states/{n:05d}.f32` and skips the ones already there: a crash costs one item,
not the whole phase.  The final `{prefix}-states.npz` is byte-compatible with the study's
(`keys, task_id, kind, sample, split, label, tokens, H, HV`).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parents[2]
HIVE = WORK / "hivebench"
sys.path.insert(0, str(HIVE))
sys.path.insert(0, str(HIVE / "experiments" / "cascade"))

from community_eval import embed_prompt, load_jsonl, trace_candidate  # noqa: E402

DEFAULT_TASKS = HIVE / "experiments" / "cascade" / "tasks-community.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--server", default="http://127.0.0.1:8897")
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    ap.add_argument("--tail-chars", type=int, default=1000)
    args = ap.parse_args()

    tasks = json.loads(args.tasks.read_text())["tasks"]
    tasks_by = {t["id"]: t for t in tasks}
    items: list[dict] = []
    for kind, name in (("sample", f"{args.prefix}-samples.jsonl"), ("hard", f"{args.prefix}-hard.jsonl")):
        for rec in load_jsonl(args.out_dir / name):
            items.append({"key": f"{rec['task_id']}|{kind}|{rec['sample']}",
                          "task_id": rec["task_id"], "kind": kind, "sample": rec["sample"],
                          "split": rec["split"], "label": int(bool(rec["ok"])), "tokens": rec["tokens"],
                          "answer": rec["content"],
                          "trace": trace_candidate(rec.get("reasoning", ""), rec["content"], args.tail_chars)})
    if not items:
        print("dh_embed: no samples/hard records; run the gen phases first")
        return 1

    vec_dir = args.out_dir / f"{args.prefix}-states"
    vec_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    done = 0
    for n, item in enumerate(items):
        path = vec_dir / f"{n:05d}.f32"
        if path.exists():
            continue
        task = tasks_by[item["task_id"]]
        a = embed_prompt(args.server, task["prompt"], item["answer"])
        v = embed_prompt(args.server, task["prompt"], item["trace"])
        np.concatenate([a, v]).astype(np.float32).tofile(path)
        done += 1
        if done % 25 == 0 or n + 1 == len(items):
            print(f"dh_embed: [{n + 1}/{len(items)}] {done} new, {time.time() - t0:.0f}s", flush=True)

    pairs = [np.fromfile(vec_dir / f"{n:05d}.f32", dtype=np.float32) for n in range(len(items))]
    dim = pairs[0].size // 2
    for n, p in enumerate(pairs):
        if p.size != 2 * dim:
            print(f"dh_embed: item {n} holds {p.size} floats, expected {2 * dim}")
            return 1
    mat = np.stack(pairs)
    out = args.out_dir / f"{args.prefix}-states.npz"
    np.savez_compressed(
        out, keys=np.array([i["key"] for i in items]), task_id=np.array([i["task_id"] for i in items]),
        kind=np.array([i["kind"] for i in items]), sample=np.array([i["sample"] for i in items]),
        split=np.array([i["split"] for i in items]), label=np.array([i["label"] for i in items]),
        tokens=np.array([i["tokens"] for i in items]),
        H=mat[:, :dim].astype(np.float32), HV=mat[:, dim:].astype(np.float32))
    print(f"dh_embed: wrote {out} ({len(items)} x {dim}) in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
