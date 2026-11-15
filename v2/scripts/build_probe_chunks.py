#!/usr/bin/env python3
"""Build the MTP probe's token chunks from the captured teacher rollouts.

Each trace becomes serving-format text
    <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{reasoning}{content}<|im_end|>\n
tokenized with the pack's tokenizer (parse_special), and the token streams are
concatenated into chunks of --per-chunk ids.  For every chunk the script writes
the driver's three files:
    chunkN.txt          all ids, comma separated
    chunkN-prompt.txt   the first id (the 1-token prompt)
    chunkN-cont.txt     the rest (the teacher-forced continuation)

The dump driver teacher-forces the continuation and writes one record per
position; assemble_headin.py then cuts 512-token windows.

Run with the strata venv (needs strata_tokenizer from work/strata/tools):
  cd work/strata && .venv/bin/python ../scion/v2/scripts/build_probe_chunks.py \
      --traces ../scion/v2/traces/rollout360 \
      --tokenizer ../models-p310/scion-strata/pack-q6k/tokenizer \
      --out ../scion/v2/traces/mtp-probe-360/chunks --per-chunk 70656
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def load_tokenizer(tp: Path):
    sys.path.insert(0, str(REPO.parent / "strata" / "tools"))
    import strata_tokenizer as ST
    vocab = json.loads((tp / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    merges = (tp / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((tp / "token_type.json").read_text())
    try:
        tcfg = json.loads((tp / "tokenizer.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        tcfg = {}
    return ST.Tokenizer(tokens, merges, types, tcfg.get("pre") or "qwen35",
                        tcfg.get("special_ids") or None)


def trace_text(rec: dict) -> str:
    body = (rec.get("reasoning") or "") + (rec.get("content") or "")
    return ("<|im_start|>user\n" + rec["prompt"] + "<|im_end|>\n"
            "<|im_start|>assistant\n" + body + "<|im_end|>\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-chunk", type=int, default=70656)
    ap.add_argument("--min-trace-tokens", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tok = load_tokenizer(Path(args.tokenizer))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(Path(args.traces).glob("*.json"))
    if args.limit:
        files = files[: args.limit]

    stream: list[int] = []
    n_tok_total = 0
    n_traces = 0
    n_skipped = 0
    for f in files:
        try:
            rec = json.loads(f.read_text())
        except Exception as e:
            print(f"skip {f.name}: {e}", flush=True)
            n_skipped += 1
            continue
        if not (rec.get("prompt") and ((rec.get("reasoning") or "") or (rec.get("content") or ""))):
            n_skipped += 1
            continue
        ids = tok.encode(trace_text(rec), parse_special=True)
        if len(ids) < args.min_trace_tokens:
            n_skipped += 1
            continue
        stream.extend(ids)
        n_tok_total += len(ids)
        n_traces += 1
        if args.limit and n_tok_total >= args.limit * args.per_chunk:
            break

    n_chunks = n_tok_total // args.per_chunk
    print(f"traces {n_traces} (skipped {n_skipped}), tokens {n_tok_total}, chunks {n_chunks}",
          flush=True)
    if n_chunks == 0:
        return 1

    # the trainer reads the token stream next to the win files (int32 ids, chunk order)
    import numpy as np
    np.asarray(stream[:n_chunks * args.per_chunk], dtype=np.int32).tofile(out / "tokens.bin")

    for c in range(n_chunks):
        ids = stream[c * args.per_chunk:(c + 1) * args.per_chunk]
        (out / f"chunk{c}.txt").write_text(",".join(map(str, ids)) + "\n")
        (out / f"chunk{c}-prompt.txt").write_text(str(ids[0]) + "\n")
        (out / f"chunk{c}-cont.txt").write_text(",".join(map(str, ids[1:])) + "\n")
        print(f"chunk {c}: {len(ids)} ids", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
