#!/usr/bin/env python3
"""Turn a gate JSON's worst tokens into a hard-window curriculum JSONL.

``kld_eval.py`` records ``worst_tokens`` as ``(window, pos)`` on its eval
split.  For a **training-side probe** (``--split fineweb --seed <disjoint>``)
those windows are safe to add to the correction set; for the wikitext gate they
are not, and this tool refuses the gate split rather than silently
contaminating it (``--force`` overrides, with the warning printed).

It rebuilds the exact windows the gate saw -- same tokenizer, split, seed,
seq -- selects the unique windows that contain the worst tokens, and writes one
JSONL row per window (``{"text": ...}``) for the trainer's ``--corpus-file``
mixer (``ayot.load_traces`` accepts ``text`` rows).

The stop rule still applies: a curriculum arm must rebuild the cache with the
same mixed corpus, or the KD targets desync from the training windows.

usage:
    python moe/curriculum.py kld-probe.json --split fineweb --seed 4242 \\
        --windows 16 --seq 512 --top 32 --out curric.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def worst_window_ids(gate: dict, top: int) -> list[int]:
    """Unique window ids of the ``top`` worst tokens, worst first.

    ``worst_tokens`` is already sorted by KLD (``kld_eval`` uses ``torch.topk``),
    so this preserves that order while dropping duplicates -- a window with
    three bad positions appears once.
    """
    if "worst_tokens" not in gate:
        raise SystemExit("not a kld_eval gate JSON (no worst_tokens)")
    ids, seen = [], set()
    for t in gate["worst_tokens"][:top]:
        w = int(t["window"])
        if w not in seen:
            seen.add(w)
            ids.append(w)
    return ids


def gate_split_is_contaminating(split: str) -> bool:
    """The gate's own split (wikitext-2 test) must not become training data."""
    return split.strip().lower().startswith("wikitext")


def build_rows(data, window_ids: list[int], decode) -> list[dict]:
    """JSONL rows for the selected windows; ``decode`` maps ids -> text.

    ``data`` is the same window tensor the gate ran on (one 1-D row per
    window), injected so this is testable without datasets/tokenizers.
    """
    rows = []
    for w in window_ids:
        if not 0 <= w < len(data):
            raise SystemExit(f"window {w} out of range (gate saw {len(data)})")
        rows.append({"text": decode(data[w].tolist())})
    return rows


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("gate_json", help="kld_eval JSON with worst_tokens")
    ap.add_argument("--split", default="fineweb",
                    help="dataset split the probe ran on (fineweb = training-side)")
    ap.add_argument("--seed", type=int, default=999,
                    help="the seed the probe gate ran with (kld_eval --seed)")
    ap.add_argument("--windows", type=int, default=16,
                    help="how many probe windows the gate ran (kld_eval --eval-windows)")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--top", type=int, default=32,
                    help="take this many worst tokens; unique windows among them")
    ap.add_argument("--out", default="", help="JSONL path (default: next to the gate)")
    ap.add_argument("--force", action="store_true",
                    help="allow the gate's own split (contaminates the gate)")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if gate_split_is_contaminating(args.split) and not args.force:
        raise SystemExit(
            f"refusing --split {args.split!r}: that is the gate's own eval split, "
            f"and training on it would contaminate the W1 gate. Run the probe on "
            f"a training-side split (--split fineweb --seed <disjoint>) instead, "
            f"or pass --force if you really mean it.")

    gate = json.loads(Path(args.gate_json).read_text())
    window_ids = worst_window_ids(gate, args.top)
    if not window_ids:
        raise SystemExit("no worst_tokens to select from")

    from transformers import AutoTokenizer
    from olmoe_proxy import windows
    from qwen35_moe_proxy import MODEL

    tok = AutoTokenizer.from_pretrained(MODEL)
    data = windows(tok, args.windows, args.seq, args.seed, args.split)
    rows = build_rows(data, window_ids, tok.decode)

    out = Path(args.out) if args.out else Path(args.gate_json).with_suffix(".curric.jsonl")
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"wrote {out}: {len(rows)} windows "
          f"({len(set(window_ids))} unique of {len(gate['worst_tokens'])} worst tokens)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
