#!/usr/bin/env python3
"""Mixed QAT corpus: tokenized 512-token windows from local sources.

Sources (all local; no downloads; cached HF datasets only):
  prose  Salesforce/wikitext (train)
  math   EleutherAI/hendrycks_math (all configs) + openai/gsm8k (train)
  code   permissively-licensed local source trees (*.py,c,cc,cpp,h,hpp,cu,md)

Windows never cross source boundaries.  Writes <out>/windows.npy (int32
[n, seq]) + <out>/corpus.json (provenance, ratios, per-source token counts,
sha256).  Shared by the 0.8B derisk and the v2 9B run.

Usage:
  clef_mixed_corpus.py --tokenizer <snap> --out <dir> --tokens 1048576 --seq 512
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

from clef_paths import REPO, WORKSPACE

DEFAULT_CODE_ROOTS = [
    str(WORKSPACE / "llama.cpp"),
    str(REPO),
    str(WORKSPACE / "hivebench"),
    str(WORKSPACE / "FreeToken"),
    str(WORKSPACE / "llama-qwen4exp"),
    str(WORKSPACE / "prism-ml-llama.cpp"),
]
_code_roots_env = os.environ.get("SCION_CODE_ROOTS")
if _code_roots_env:
    DEFAULT_CODE_ROOTS = _code_roots_env.split(",")
CODE_EXTS = {".py", ".c", ".cc", ".cpp", ".h", ".hpp", ".cu", ".md"}
SKIP_DIRS = {"build", ".git", "__pycache__", "node_modules", ".venv", "vendor",
             "worktrees", "site-packages", ".cache"}
MAX_FILE_BYTES = 200_000


def wikitext_texts():
    from datasets import load_dataset
    for name, config in (("Salesforce/wikitext", "wikitext-103-raw-v1"),
                         ("Salesforce/wikitext", "wikitext-2-raw-v1")):
        try:
            ds = load_dataset(name, config, split="train")
        except Exception as e:
            print(f"  {name}/{config} unavailable: {type(e).__name__}")
            continue
        print(f"  prose: {name}/{config} (train)")
        for row in ds:
            t = row.get("text", "")
            if t.strip():
                yield t


HF_MATH = [("EleutherAI/hendrycks_math", c, "train") for c in
           ("algebra", "counting_and_probability", "geometry",
            "intermediate_algebra", "number_theory", "prealgebra",
            "precalculus")] + [("openai/gsm8k", "main", "train")]


def math_texts():
    from datasets import load_dataset
    for name, config, split in HF_MATH:
        try:
            ds = load_dataset(name, config, split=split)
        except Exception as e:
            print(f"  {name}/{config}:{split} unavailable: {type(e).__name__}")
            continue
        print(f"  math: {name}/{config}:{split}")
        for row in ds:
            if "problem" in row:                       # hendrycks_math
                t = row["problem"] + "\n" + row.get("solution", "")
            else:                                      # gsm8k
                t = row.get("question", "") + "\n" + row.get("answer", "")
            if t.strip():
                yield t


def code_texts(roots: list[str]):
    seen: set[str] = set()
    for root in roots:
        p = Path(root)
        if not p.is_dir():
            print(f"  code: {root} missing")
            continue
        n_files = 0
        for dirpath, dirnames, filenames in os.walk(p):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS
                           and not d.startswith("build-")]
            for fn in filenames:
                if Path(fn).suffix not in CODE_EXTS:
                    continue
                fp = Path(dirpath) / fn
                try:
                    if fp.stat().st_size > MAX_FILE_BYTES:
                        continue
                    text = fp.read_text(errors="replace")
                except OSError:
                    continue
                if not text.strip():
                    continue
                h = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()
                if h in seen:
                    continue
                seen.add(h)
                n_files += 1
                yield f"# file: {fp.relative_to(p)}\n{text}\n\n"
        print(f"  code: {root} ({n_files} files)")


def build_source(texts, tok, quota: int, seq: int, label: str):
    """Tokenize until ``quota`` tokens, then cut into seq-sized windows."""
    buf: list[int] = []
    windows: list[list[int]] = []
    ntok = 0
    for text in texts:
        ids = tok(text, add_special_tokens=False).input_ids
        buf.extend(ids)
        ntok += len(ids)
        while len(buf) >= seq:
            windows.append(buf[:seq])
            buf = buf[seq:]
            if len(windows) * seq >= quota:
                break
        if len(windows) * seq >= quota:
            break
    print(f"  {label}: {ntok} tokens -> {len(windows)} windows "
          f"(quota {quota // seq})")
    return windows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=1_048_576,
                    help="total target tokens (default 2048x512)")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--ratios", default="0.5,0.2,0.3",
                    help="prose,math,code token ratios")
    ap.add_argument("--code-roots", default=",".join(DEFAULT_CODE_ROOTS))
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ratios = [float(x) for x in args.ratios.split(",")]
    assert len(ratios) == 3 and abs(sum(ratios) - 1.0) < 1e-6

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"building mixed corpus: {args.tokens} tokens, ratios {ratios}")

    prose = build_source(wikitext_texts(), tok, int(args.tokens * ratios[0]),
                         args.seq, "prose")
    math = build_source(math_texts(), tok, int(args.tokens * ratios[1]),
                        args.seq, "math")
    code = build_source(code_texts(args.code_roots.split(",")), tok,
                        int(args.tokens * ratios[2]), args.seq, "code")

    windows = prose + math + code
    rng = random.Random(args.seed)
    rng.shuffle(windows)
    arr = np.asarray(windows, dtype=np.int32)
    np.save(out / "windows.npy", arr)
    digest = hashlib.sha256(arr.tobytes()).hexdigest()
    meta = {
        "tokenizer": args.tokenizer, "seq": args.seq, "seed": args.seed,
        "tokens_target": args.tokens, "ratios": ratios,
        "windows": int(arr.shape[0]),
        "tokens_actual": {k: len(v) * args.seq
                          for k, v in (("prose", prose), ("math", math),
                                       ("code", code))},
        "code_roots": args.code_roots.split(","),
        "sha256_windows": digest,
    }
    (out / "corpus.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {out}/windows.npy {arr.shape} sha {digest[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
