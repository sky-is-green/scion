"""Teacher cache for the dense Clef correction run.

Runs the **f16** backbone in torch (reverse-loaded from the GGUF, recurrent GDN
so it matches the deployed CPU runtime) over the training corpus and stores, per
sequence, the final post-norm hidden states the joint head reads.  Decision
records additionally store the teacher head's option logits, so the trainer has
a soft decision target without the head or teacher resident.

Sequences:
  * bench (train + test) -- the D2 accept/reject gold set; per record, one
    ``noul`` verdict question (prompt + candidate);
  * negatives -- hard negatives derived from the train candidates (the bf16
    teacher, not the gold checker, provides the target);
  * text -- generic windows for backbone hidden-state fidelity.

Stored one ``.npz`` per sequence plus an ``index.json`` (memory-mappable, so
the trainer never holds the whole cache in host RAM).

Usage:
    python dense/clef_cache.py --out /path/cache --generic 0 --negatives 1
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from clef_dense_load import load_text_model_streamed, patch_recurrent_gdn  # noqa: E402
import clef_head as H  # noqa: E402

HIVE = Path("/home/penis/Desktop/work/hivebench/experiments/cascade")
TASKS = HIVE / "tasks-bench.json"
TRAIN_REPORT = HIVE / "results/strata-bench-train-20261003/report.json"
TEST_REPORT = HIVE / "results/strata-bench-test-labels-20261003/report.json"
DEFAULT_MODEL = "/home/penis/Desktop/work/models/clef-flash-ternary"


def load_bench() -> list[dict]:
    tasks = {t["id"]: t for t in json.loads(TASKS.read_text())["tasks"]}
    rows = []
    for split, path in (("train", TRAIN_REPORT), ("test", TEST_REPORT)):
        for r in json.loads(path.read_text())["records"]:
            rows.append({
                "kind": "bench", "split": split, "record_id": r["id"],
                "prompt": tasks[r["id"]]["prompt"], "candidate": r["scion_answer"],
            })
    return rows


_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def make_negatives(rows: list[dict]) -> list[dict]:
    """Perturb the last number in a train candidate (x2 or +7). Cheap and hard."""
    out = []
    for r in rows:
        if r["split"] != "train":
            continue
        cand = r["candidate"]
        nums = list(_NUM.finditer(cand))
        if not nums:
            continue
        m = nums[-1]
        token = m.group(0)
        try:
            val = float(token)
        except ValueError:
            continue
        new = f"{val * 2:g}" if val % 1 == 0 else f"{val + 7:g}"
        out.append({**r, "kind": "negative", "record_id": r["record_id"] + "-neg",
                    "candidate": cand[:m.start()] + new + cand[m.end():]})
    return out


def generic_windows(tok, n: int, seq: int, seed: int = 0) -> list[list[int]]:
    """``n`` tokenised windows of ``seq`` from a local text corpus."""
    if n <= 0:
        return []
    from datasets import load_dataset
    last = None
    for name, config in (("Salesforce/wikitext", "wikitext-103-raw-v1"),
                         ("Salesforce/wikitext", "wikitext-2-raw-v1")):
        try:
            ds = load_dataset(name, config, split="train")
            last = ds
            break
        except Exception as e:  # pragma: no cover - offline fallback path
            print(f"  dataset {name}/{config} unavailable: {type(e).__name__}", flush=True)
    if last is None:
        raise SystemExit("no generic text dataset available locally")
    buf: list[int] = []
    windows: list[list[int]] = []
    for row in last:
        text = row.get("text", "")
        if not text.strip():
            continue
        buf.extend(tok(text, add_special_tokens=False).input_ids)
        while len(buf) >= seq and len(windows) < n:
            windows.append(buf[:seq])
            buf = buf[seq:]
        if len(windows) >= n:
            break
    return windows


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--generic", type=int, default=0, help="number of generic windows")
    ap.add_argument("--generic-only", action="store_true",
                    help="skip bench/negatives and the joint head; text windows only")
    ap.add_argument("--negatives", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-tokens", type=int, default=2048)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("loading f16 teacher ...", flush=True)
    model, _, loaded = load_text_model_streamed(
        Path(args.model) / "clef-flash-f16.gguf", device=args.device, dtype=torch.bfloat16)
    n_gdn = patch_recurrent_gdn(model)
    tok = H.load_tokenizer(args.model)
    head = lm_head = None
    if not args.generic_only:
        head, _ = H.load_joint_head(args.model, device="cpu")
        lm_head = H.load_lm_head(args.model, device="cpu")
    print(f"teacher loaded ({loaded} params, {n_gdn} GDN patched)", flush=True)

    rows = [] if args.generic_only else load_bench()
    if args.negatives and not args.generic_only:
        rows += make_negatives(rows)
    # encode decision records
    entries = []
    for i, r in enumerate(rows):
        enc = H.encode(tok, r["prompt"], r["candidate"], max_length=args.max_tokens)
        if len(enc.input_ids) > args.max_tokens:
            continue
        entries.append({**r, "_enc": enc})
    for w in generic_windows(tok, args.generic, args.seq):
        entries.append({"kind": "text", "split": "none", "record_id": f"text-{len(entries)}",
                        "_ids": w})

    index = []
    for i, e in enumerate(entries):
        enc = e.get("_enc")
        ids = list(enc.input_ids) if enc is not None else list(e["_ids"])
        input_ids = torch.tensor([ids], device=args.device)
        hidden = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                       use_cache=False).last_hidden_state[0].float().cpu()
        rec = {"file": f"{i:05d}.npz", "kind": e["kind"], "split": e["split"],
               "record_id": e["record_id"], "n_tokens": len(ids)}
        arrays = {"input_ids": np.asarray(ids, dtype=np.int32),
                  "hidden": hidden.to(torch.float16).numpy()}
        if enc is not None:
            with torch.no_grad():
                logits = H.head_logits(head, hidden.unsqueeze(0).to(torch.bfloat16),
                                       input_ids.cpu(), torch.ones_like(input_ids.cpu()),
                                       [enc], lm_head)[0][0]
            arrays["option_logits"] = logits.float().numpy()
            rec["has_decision"] = True
            rec["prompt"] = e["prompt"]
            rec["candidate"] = e["candidate"]
        np.savez(out / rec["file"], **arrays)
        index.append(rec)
        if i % 25 == 0 or i == len(entries) - 1:
            print(f"  cached {i+1}/{len(entries)} {e['kind']} {e['record_id']}", flush=True)

    (out / "index.json").write_text(json.dumps({"model": args.model, "seq": args.seq,
                                                 "entries": index}, indent=1))
    size = sum((out / r["file"]).stat().st_size for r in index) / 1e9
    print(f"wrote {out}: {len(index)} sequences, {size:.2f} GB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
