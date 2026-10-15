#!/usr/bin/env python
"""Replay the JEV-readout judge bench on a Strata engine's EMB endpoint.

Three steps, run in order (they share one JSONL/npz pair):

  tokens  render the study's judge prompt for every judge-bench item with the
          pack's own chat template (enable_thinking false) and encode it
  probe   drive `strata-q36 --serve` and send one EMB per item, collecting the
          last-prompt-token hidden state each time
  score   apply the study's published head (space check: did our engine's
          hidden state land in the same space the head was fit on) and refit a
          fresh head on our states (signal check), reporting both AUCs against
          the published 0.9988 test number

The study: huggingface.co/datasets/SkyIsNotGreen/decision-heads
(`CONCLUSION.md`, `probe_scion.py`, `scion-35b-a3b-decision-head.json`).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parents[2]
DH = WORK / "scion" / "decision-heads"
WT = WORK / "strata-dh"
HIVE = WORK / "hivebench" / "experiments" / "cascade"
PACK35 = Path("/home/penis/models/scion-strata/pack-35b/tokenizer")
GGUF35 = Path("/home/penis/models/scion-strata/Scion-35B-A3B-PQ2_0-corr.gguf")
EXE = WT / "build-hip" / "strata-q36"
OUT = WORK / "scion" / "decision-heads-work"

# the 176B: the served pack + model (strata-q6k-176b.json) through the main engine
PACK176 = Path("/home/penis/models/scion-strata/pack-q6k/tokenizer")
GGUF176 = Path("/home/penis/models/scion-strata/Scion-FlashNext-176B-A6B-Q6K.gguf")
EXE_MAIN = WT / "build-hip" / "strata"
ARGS176 = [
    "--pack", "/home/penis/models/scion-strata/pack-q6k",
    "--native", str(GGUF176),
    "--ple-gguf", str(GGUF176),
    "--embd-gguf", "/home/penis/models/scion-strata/scion-embd-bf16.gguf",
    "--expert-profile", "/home/penis/Desktop/work/strata/data/expert-profile.bin",
    "--expert-cache", "auto",
    "--prefill", "8192",
    "--spec", "4",
    "--spec-min-p", "0.5",
    "--mtp", "/home/penis/Desktop/work/strata-data/mtp/rt",
    "--max-context", "4096",
    "--ple-io", "direct",
    "--mmap-experts",
    "--vram-reserve-mib", "3072",
    "--pool-workers", "15",
    "--pcie-frac", "0",
    "--pipeline-windows", "2",
    "--layer-split", "auto",
]

sys.path.insert(0, str(WT / "tools"))
sys.path.insert(0, str(WT / "serve"))
sys.path.insert(0, str(HIVE))
sys.path.insert(0, str(WORK / "hivebench"))

from frontend import ChatTemplate  # noqa: E402
import strata_tokenizer as ST  # noqa: E402
from probe_judge import platt, ridge_fit, ridge_score  # noqa: E402
from readout_judge import auc, best_ba  # noqa: E402

SYSTEM = "You are a decision function. Reply with the answer label only."
BODY = ("Question: {prompt}\n"
        "Candidate answer: {candidate}\n"
        "Is the candidate answer correct?\n"
        "{a}. Yes\n"
        "{b}. No\n")


def load_bench() -> list[dict]:
    return [json.loads(l) for l in (DH / "judge-bench.jsonl").read_text().splitlines()]


def load_tokenizer(pack: Path):
    vocab = json.loads((pack / "vocab.json").read_text(encoding="utf-8"))
    tokens: list[str | None] = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    merges = (pack / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((pack / "token_type.json").read_text())
    return ST.Tokenizer(tokens, merges, types)


def cmd_tokens(args) -> int:
    pack = PACK35 if args.pack == "35b" else PACK176
    tok = load_tokenizer(pack)
    tpl = ChatTemplate(pack / "chat_template.jinja")
    items = load_bench()
    rows = []
    for item in items:
        body = BODY.format(prompt=item["prompt"], candidate=item["candidate"], a="A", b="B")
        prompt = tpl.render([{"role": "system", "content": args.system},
                             {"role": "user", "content": body}],
                            add_generation_prompt=True, enable_thinking=False)
        ids = tok.encode(prompt, parse_special=True)
        rows.append({"item_id": item["item_id"], "split": item["split"], "label": item["label"],
                     "task_id": item["task_id"], "prompt": prompt, "ids": ids})
    path = OUT / ("judge-prompt-ids.jsonl" if args.pack == "35b" else "judge-prompt-ids-main.jsonl")
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    lens = [len(r["ids"]) for r in rows]
    print(f"tokens: {len(rows)} prompts -> {path}")
    print(f"        ids {min(lens)}..{max(lens)} tokens, mean {sum(lens)/len(lens):.0f}")
    head, tail = rows[0]["prompt"][-120:], rows[0]["ids"][:8]
    print("        first prompt ends: " + json.dumps(head))
    return 0


def read_until(proc, want: tuple[str, ...], timeout_s: float) -> str:
    """Read stdout lines until one starts with a tag in `want`; returns that line."""
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


def cmd_probe(args) -> int:
    rows = [json.loads(l) for l in (OUT / "judge-prompt-ids.jsonl").read_text().splitlines()]
    if args.limit:
        rows = rows[:args.limit]
    vec_dir = OUT / "states"
    vec_dir.mkdir(exist_ok=True)
    log = open(OUT / "dh-probe.engine.log", "ab")
    env = dict(os.environ)
    if args.hip is not None:
        env["HIP_VISIBLE_DEVICES"] = str(args.hip)
    cmd = [str(EXE), "--serve", "--model", str(GGUF35), "--max-context", str(args.ctx), "--chunk", "512"]
    print("probe: " + " ".join(cmd) + (f"  (HIP_VISIBLE_DEVICES={args.hip})" if args.hip is not None else ""))
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                            cwd=str(WT), env=env)
    try:
        ready = read_until(proc, ("READY",), 600)
        print(f"probe: engine ready in {time.time()-t0:.0f} s ({ready})")
        states, dim = [], None
        t0 = time.time()
        for n, row in enumerate(rows, 1):
            path = vec_dir / f"{row['item_id'].replace(':', '_')}.f32"
            ids = ",".join(str(i) for i in row["ids"])
            proc.stdin.write(f"EMB path={path} {ids}\n".encode())
            proc.stdin.flush()
            line = read_until(proc, ("EMB ", "ERR"), 600)
            if line.startswith("ERR"):
                raise RuntimeError(f"engine: {line}")
            dim = int(line.split()[1])
            vec = np.fromfile(path, dtype=np.float32)
            assert vec.size == dim, f"{path}: {vec.size} floats, expected {dim}"
            states.append(vec)
            if n % 25 == 0 or n == len(rows):
                print(f"probe: [{n}/{len(rows)}] {time.time()-t0:.0f}s", flush=True)
        X = np.stack(states)
        np.savez_compressed(OUT / "dh-probe-states.npz",
                            item_ids=[r["item_id"] for r in rows],
                            splits=[r["split"] for r in rows],
                            labels=[r["label"] for r in rows],
                            X=X)
        print(f"probe: {X.shape} states -> {OUT/'dh-probe-states.npz'} in {time.time()-t0:.0f} s")
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
    return 0


def head_published() -> dict:
    return json.loads((DH / "scion-35b-a3b-decision-head.json").read_text(encoding="utf-8"))


def report(name: str, p: np.ndarray, y: np.ndarray) -> dict:
    a, thr = best_ba(list(p), list(y))
    acc87 = float(np.mean((p >= 0.87) == (y == 1)))
    out = {"n": int(len(y)), "auc": auc(list(p), list(y)), "best_ba": a, "best_ba_thr": thr,
           "acc_at_0.87": acc87}
    print(f"score: {name:22s} n={out['n']:3d} AUC {out['auc']:.4f}  BA {a:.4f} @ {thr:.2f}  "
          f"acc@0.87 {acc87:.4f}")
    return out


def cmd_probe_main(args) -> int:
    """The main engine (the 176B): `GEN 1 emb=<path> <ids>` reads the prompt and answers EMB <dim> <path>."""
    rows = [json.loads(l) for l in (OUT / "judge-prompt-ids-main.jsonl").read_text().splitlines()]
    if args.limit:
        rows = rows[:args.limit]
    vec_dir = OUT / "states-main"
    vec_dir.mkdir(exist_ok=True)
    log = open(OUT / "dh-probe-main.engine.log", "ab")
    env = dict(os.environ)
    env["HIP_VISIBLE_DEVICES"] = args.hip
    env["STRATA_PREFILL_MMQ"] = "1"
    env["STRATA_PREFILL_RING"] = "96"
    env["STRATA_IO_THREADS"] = "32"
    cmd = [str(EXE_MAIN), "--serve"] + ARGS176
    print("probe-main: " + " ".join(cmd[:6]) + " ... (" + env["HIP_VISIBLE_DEVICES"] + ")")
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                            cwd=str(WT), env=env)
    try:
        ready = read_until(proc, ("READY",), 900)
        print(f"probe-main: engine ready in {time.time()-t0:.0f} s ({ready})", flush=True)
        states, dim = [], None
        t0 = time.time()
        for n, row in enumerate(rows, 1):
            path = vec_dir / f"{row['item_id'].replace(':', '_')}.f32"
            ids = ",".join(str(i) for i in row["ids"])
            proc.stdin.write(f"GEN 1 emb={path} {ids}\n".encode())
            proc.stdin.flush()
            line = read_until(proc, ("EMB ", "ERR"), 900)
            if line.startswith("ERR"):
                raise RuntimeError(f"engine: {line}")
            dim = int(line.split()[1])
            read_until(proc, ("DONE",), 900)
            vec = np.fromfile(path, dtype=np.float32)
            assert vec.size == dim, f"{path}: {vec.size} floats, expected {dim}"
            states.append(vec)
            if n % 25 == 0 or n == len(rows):
                print(f"probe-main: [{n}/{len(rows)}] {time.time()-t0:.0f}s", flush=True)
        X = np.stack(states)
        np.savez_compressed(OUT / "dh-probe-states-main.npz",
                            item_ids=[r["item_id"] for r in rows],
                            splits=[r["split"] for r in rows],
                            labels=[r["label"] for r in rows],
                            X=X)
        print(f"probe-main: {X.shape} states -> {OUT/'dh-probe-states-main.npz'} in {time.time()-t0:.0f} s")
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()
    return 0


def cmd_score(args) -> int:
    states_path = OUT / ("dh-probe-states.npz" if args.states == "35b" else "dh-probe-states-main.npz")
    z = np.load(states_path, allow_pickle=False)
    X = z["X"].astype(np.float64)
    y = np.asarray(z["labels"], dtype=np.float64)
    splits = np.asarray(z["splits"])
    tr, te = splits == "train", splits == "test"
    print(f"score: {X.shape[0]} states, dim {X.shape[1]}, train {tr.sum()}, test {te.sum()} ({states_path.name})")

    out: dict = {"states": str(states_path), "n": int(X.shape[0]), "dim": int(X.shape[1])}
    corr = None
    if args.published != "none":
        hd = head_published()
        w = np.asarray(hd["weights"]["w"], dtype=np.float64)
        if w.size != X.shape[1]:
            print(f"score: the published 35B head is dim {w.size}, these states are dim {X.shape[1]}: skipped")
        else:
            mu = np.asarray(hd["weights"]["mu"], dtype=np.float64)
            sd = np.asarray(hd["weights"]["sd"], dtype=np.float64)
            b = float(hd["weights"]["b"])
            a_p, c_p = (float(v) for v in hd["weights"]["platt"])
            zp = ((X - mu) / sd) @ w + b
            pz = 1.0 / (1.0 + np.exp(-(a_p * zp + c_p)))
            out["published"] = {"test": report("published head (test)", pz[te], y[te]),
                                "train": report("published head (train)", pz[tr], y[tr])}

    model = ridge_fit(X[tr], y[tr])
    s_tr = ridge_score(X[tr], model)
    a_r, b_r = platt(s_tr, y[tr])
    s_te = ridge_score(X[te], model)
    p_r = 1.0 / (1.0 + np.exp(-(a_r * s_te + b_r)))
    out["refit"] = {"test": report("refit on ours (test)", p_r, y[te]), "platt": [a_r, b_r], "ridge": {"lambda": 1.0}}
    if "published" in out:
        corr = float(np.corrcoef(pz[te], p_r)[0, 1])
        print(f"score: correlation(published, refit) on test = {corr:.4f}")
        out["test_score_corr"] = corr
    out["refit"]["train"] = report("refit on ours (train)", 1.0 / (1.0 + np.exp(-(a_r * s_tr + b_r))), y[tr])
    name = "dh-replay.json" if args.states == "35b" else "dh-replay-main.json"
    (OUT / name).write_text(json.dumps(out, indent=1))
    print("score: wrote", OUT / name)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tokens", help="render and encode the judge prompts")
    t.add_argument("--system", default=SYSTEM)
    t.add_argument("--pack", choices=("35b", "main"), default="35b")
    t.set_defaults(fn=cmd_tokens)
    p = sub.add_parser("probe", help="drive the q36 engine (35B) and collect hidden states")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--hip", type=int, default=None, help="HIP_VISIBLE_DEVICES for the engine")
    p.set_defaults(fn=cmd_probe)
    pm = sub.add_parser("probe-main", help="drive the main engine (176B) and collect hidden states")
    pm.add_argument("--limit", type=int, default=0)
    pm.add_argument("--hip", default="0,1", help="HIP_VISIBLE_DEVICES for the engine")
    pm.set_defaults(fn=cmd_probe_main)
    s = sub.add_parser("score", help="published-head and refit scores")
    s.add_argument("--states", choices=("35b", "main"), default="35b")
    s.add_argument("--published", choices=("35b", "none"), default="35b")
    s.set_defaults(fn=cmd_score)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
