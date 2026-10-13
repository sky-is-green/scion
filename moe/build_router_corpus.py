#!/usr/bin/env python3
"""Build the router-usage calibration corpus for the Flash-Next shrink (P5).

Two halves, matching the AYOT calibration population split:

  * the 512 AYOT prompts from ``build_ayot_prompts.py`` (128 math, 128 coding,
    256 fineweb continuations).  These are *generated* through for a bounded
    number of tokens, because reasoning-mode routing is what the reasoning
    prompts elicit (the AYOT claim);
  * raw fineweb-edu passages, chunked and prefilled only (``n_predict: 0``) to
    cover ordinary web-text routing at a low cost per token.

Every row carries the source tag and the requested ``n_predict``, so the pod
client is a dumb loop and the corpus's composition is auditable after the fact.

    python moe/build_router_corpus.py --out /tmp/opencode/router-corpus.jsonl

The fineweb draw starts at ``--fineweb-offset`` rows into the dataset, while
the AYOT builder draws from offset 0, so the two do not share documents.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from build_ayot_prompts import build, clean  # noqa: E402


def fineweb_chunks(want_chars: int, skip_rows: int, chunk_chars: int) -> list[str]:
    """Pack fineweb-edu rows into ~chunk_chars chunks, streamed with datasets.

    The datasets-server /rows API rate-limits (HTTP 429) once a few dozen pages
    are drawn, so the bulk half of the corpus uses the streaming reader that
    ``olmoe_proxy.windows`` uses.  Run this with the torch env
    (``work/.venv-rocm/bin/python``) which has ``datasets``.
    """
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0
    got = 0
    seen = 0
    for row in ds:
        seen += 1
        if seen <= skip_rows:
            continue
        if int(row.get("int_score") or 0) < 4 or int(row.get("token_count") or 0) < 120:
            continue
        t = clean(row.get("text"), 10_000_000)
        if not t:
            continue
        for para in t.split("\n\n"):
            if buf and buf_len + len(para) > chunk_chars:
                chunks.append("\n\n".join(buf))
                got += buf_len + 2
                buf, buf_len = [], 0
            buf.append(para)
            buf_len += len(para) + 2
            if buf_len >= chunk_chars:
                chunks.append("\n\n".join(buf))
                got += buf_len + 2
                buf, buf_len = [], 0
        if got >= want_chars:
            break
        if seen % 500 == 0:
            print(f"  fineweb: {got}/{want_chars} chars, {len(chunks)} chunks "
                  f"({seen} rows)", flush=True)
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--fineweb-chars", type=int, default=10_000_000,
                    help="approx prefill corpus size; ~4 chars/token on web text")
    ap.add_argument("--fineweb-offset", type=int, default=10_000,
                    help="rows skipped before the draw (AYOT fineweb prompts "
                         "come from the first few thousand rows)")
    ap.add_argument("--chunk-chars", type=int, default=6_000,
                    help="fineweb chunk size (keep under the 4096-token slot)")
    ap.add_argument("--ayot-gen", type=int, default=256,
                    help="generated tokens per AYOT prompt (0 = prefill only)")
    ap.add_argument("--ayot-file", default=str(HERE / "artifacts/qwen4exp/ayot-prompts-512.jsonl"),
                    help="AYOT prompt cache; reused when it exists")
    ap.add_argument("--no-ayot", action="store_true")
    args = ap.parse_args()

    rows: list[dict] = []
    if not args.no_ayot:
        ayot_path = Path(args.ayot_file)
        if ayot_path.exists():
            print(f"reusing {ayot_path}", flush=True)
        else:
            ayot_path.parent.mkdir(parents=True, exist_ok=True)
            build({"math": 128, "coding": 128, "fineweb": 256}, 0, 1200, ayot_path)
        for line in ayot_path.read_text().splitlines():
            r = json.loads(line)
            rows.append({"prompt": r["question"], "source": "ayot-" + r["source"],
                         "n_predict": args.ayot_gen})
        print(f"ayot rows: {len(rows)}", flush=True)

    chunks = fineweb_chunks(args.fineweb_chars, args.fineweb_offset, args.chunk_chars)
    for i, c in enumerate(chunks):
        rows.append({"prompt": c, "source": "fineweb", "n_predict": 0})
    print(f"fineweb chunks: {len(chunks)}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    chars = sum(len(r["prompt"]) for r in rows)
    gen_rows = sum(1 for r in rows if r["n_predict"])
    print(f"wrote {out}: {len(rows)} rows, {chars/1e6:.1f}M prompt chars, "
          f"{gen_rows} generate rows x {args.ayot_gen} tokens")


if __name__ == "__main__":
    main()
