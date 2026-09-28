"""Build the AYOT calibration-prompt set (~512 rows) for teacher trace generation.

AYOT (arXiv 2608.01078) claims ternary PTQ of a reasoning LLM collapses on
reasoning tasks unless the calibration inputs contain the model's *own* reasoning
traces.  The inputs are questions; the traces come from
``ayot_gen.py`` running the FP teacher over them.  This builds the question half.

Per the open decision (2026-09-28): **50/50** between the two populations, so one
prompt file feeds arm B without a second run and the split is auditable after
the fact:

  256  reasoning/coding   the population AYOT actually names
        128  math word problems   (GSM8K train)          -> `math`
        128  code-generation asks (Magicoder OSS-Instruct) -> `coding`
  256  fineweb-edu        web passages to continue, the same corpus family the
                          training windows are drawn from -> `fineweb`

Sourcing note: rows come from the HF datasets-server ``/rows`` API rather than
``datasets``, because the only local venv with torch had neither ``datasets`` nor
``pyarrow`` installed.  Same underlying rows, no new dependency.

Every row carries a ``source`` tag, and the reasoning/coding half carries an
explicit chain-of-thought instruction -- eliciting the trace is the mechanism the
paper's claim rests on, so it is stated rather than hoped for.

    python moe/build_ayot_prompts.py --out $MOE_AYOT/ayot-prompts.jsonl
    python moe/ayot_gen.py --prompts $MOE_AYOT/ayot-prompts.jsonl --dry-run
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

API = "https://datasets-server.huggingface.co/rows"
PAGE = 100          # the /rows endpoint caps length at 100

#: Uniform CoT instruction.  Deliberately identical across the reasoning and
#: coding halves so the two stay comparable to each other.
COT_SUFFIX = ("\n\nThink it through step by step, then give the final answer.")

FINEWEB_PROMPT = ("Continue this passage in the same voice, keeping the same level "
                  "of detail.\n\n---\n{passage}\n---")

SOURCES = {
    "math": dict(dataset="openai/gsm8k", config="main", split="train", field="question"),
    "coding": dict(dataset="ise-uiuc/Magicoder-OSS-Instruct-75K", config="default",
                   split="train", field="problem"),
    "fineweb": dict(dataset="HuggingFaceFW/fineweb-edu", config="default",
                    split="train", field="text"),
}


def fetch(dataset: str, config: str, split: str, offset: int, length: int,
          retries: int = 4) -> list[dict]:
    """One page of rows from the datasets-server, retried on transient errors."""
    query = urllib.parse.urlencode({"dataset": dataset, "config": config,
                                    "split": split, "offset": offset,
                                    "length": length})
    url = f"{API}?{query}"
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as fh:
                payload = json.load(fh)
            if "error" in payload:
                raise RuntimeError(payload["error"])
            return [r["row"] for r in payload.get("rows", [])]
        except Exception as exc:                       # noqa: BLE001 - retry anything
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"rows fetch failed for {dataset}/{config}/{split} "
                       f"@{offset}: {last}")


def clean(text: str, max_chars: int) -> str | None:
    """Tidy crawl whitespace and trim to a sane prompt size.

    Line structure is *preserved*: the coding half is markdown with fenced code
    blocks, and collapsing it to a single line makes those prompts markedly
    worse.  Horizontal runs and long blank-line runs are the only things cut.
    """
    if not isinstance(text, str):
        return None
    t = text.replace("\r\n", "\n").replace("\t", "  ")
    # Interior horizontal runs only: the lookbehind/lookahead on \S keeps
    # leading indentation intact, which matters because the coding half is
    # fenced Python.
    t = re.sub(r"(?<=\S)[ ]{2,}(?=\S)", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)        # cap blank-line runs
    t = t.strip()
    if len(t) < 40:
        return None
    if len(t) > max_chars:
        t = t[:max_chars].rsplit("\n", 1)[0] # cut on a line boundary
    return t


def take(spec: dict, field: str, want: int, offset: int, max_chars: int,
         keep=None) -> list[str]:
    """Walk pages from ``offset`` until ``want`` usable prompts are collected."""
    out: list[str] = []
    cur = offset
    while len(out) < want and cur < offset + 40 * PAGE:
        for row in fetch(spec["dataset"], spec["config"], spec["split"], cur, PAGE):
            if keep is not None and not keep(row):
                continue
            t = clean(row.get(field), max_chars)
            if t:
                out.append(t)
                if len(out) == want:
                    break
        cur += PAGE
        print(f"  {spec['dataset']}: {len(out)}/{want}", flush=True)
    if len(out) < want:
        raise RuntimeError(f"{spec['dataset']} yielded {len(out)}/{want} prompts")
    return out


def _fineweb_ok(row: dict) -> bool:
    """Keep only high-scoring edu pages, matching the fineweb-edu training slice."""
    return int(row.get("int_score") or 0) >= 4 and int(row.get("token_count") or 0) >= 120


def build(counts: dict[str, int], offset: int, max_chars: int, out: Path) -> None:
    rows: list[dict] = []

    print("math (GSM8K):", flush=True)
    for q in take(SOURCES["math"], "question", counts["math"], offset, max_chars):
        rows.append({"question": q + COT_SUFFIX, "source": "math"})

    print("coding (Magicoder):", flush=True)
    for q in take(SOURCES["coding"], "problem", counts["coding"], offset, max_chars):
        rows.append({"question": q + COT_SUFFIX, "source": "coding"})

    print("fineweb-edu (continuations):", flush=True)
    for t in take(SOURCES["fineweb"], "text", counts["fineweb"], offset, max_chars,
                  keep=_fineweb_ok):
        rows.append({"question": FINEWEB_PROMPT.format(passage=t), "source": "fineweb"})

    # de-duplicate on the exact prompt text, preserving order
    seen, uniq = set(), []
    for r in rows:
        if r["question"] not in seen:
            seen.add(r["question"])
            uniq.append(r)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for r in uniq:
            fh.write(json.dumps(r) + "\n")

    counts: dict[str, int] = {}
    for r in uniq:
        counts[r["source"]] = counts.get(r["source"], 0) + 1
    reasoning = counts.get("math", 0) + counts.get("coding", 0)
    print(f"\nwrote {out}: {len(uniq)} rows")
    print(f"  by source      : {counts}")
    print(f"  reasoning share: {reasoning}/{len(uniq)} "
          f"({100*reasoning/max(len(uniq),1):.0f}%)")
    print(f"  prompt chars   : min {min(len(r['question']) for r in uniq)}, "
          f"max {max(len(r['question']) for r in uniq)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="JSONL to write")
    ap.add_argument("--count-math", type=int, default=128)
    ap.add_argument("--count-coding", type=int, default=128)
    ap.add_argument("--count-fineweb", type=int, default=256,
                    help="256 keeps the reasoning/fineweb split at 50/50")
    ap.add_argument("--offset", type=int, default=0,
                    help="row offset into each source (raise it for a fresh draw)")
    ap.add_argument("--max-chars", type=int, default=1200,
                    help="truncate each source text to this many characters")
    args = ap.parse_args()
    build({"math": args.count_math, "coding": args.count_coding,
           "fineweb": args.count_fineweb}, args.offset, args.max_chars, Path(args.out))


if __name__ == "__main__":
    main()
