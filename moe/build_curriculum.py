#!/usr/bin/env python3
"""Build the v2 mix file: worst fineweb windows by student loss + teacher traces.

Why this exists: the recipe's mix file is fineweb-edu plus a hard-window
curriculum, and `cache` refuses to build without it because the cache corpus
must equal the trained corpus.  The faithful KLD instrument (`kld_eval.py`)
parks full-vocabulary teacher log-probs on the host at ~4 GB per 512-token
window, so it cannot produce the 205 windows the mixer needs on this box.

This runs on the pod, before `cache`, and ranks the same windows the cache will
use (`windows()` with the same seed, seq and corpus-chars), so the curriculum is
drawn from exactly the trained corpus.  Ranking is by the student/body's own
mean cross-entropy: a window the model finds hard is the one worth replacing
into the 5% mix.  The teacher traces (``{"text": ...}`` rows) are appended as
the second half of the mix.

Deviation from the recipe, recorded: loss-ranked, not full-vocab-KLD-ranked.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from olmoe_proxy import windows  # noqa: E402
from qwen35_moe_proxy import load_full, model_logits  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--device-map", default="cuda:0",
                    help="passed to from_pretrained; use auto for a 2-card pod")
    ap.add_argument("--max-memory", default="",
                    help='per-device caps for device_map auto, e.g. "0:44GiB,1:44GiB"')
    ap.add_argument("--windows", type=int, default=4096)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--corpus-chars", type=int, default=50_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top", type=int, default=205, help="windows to keep (5%% of 4096)")
    ap.add_argument("--traces", default="", help='JSONL of {"text": ...} teacher traces')
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model, tok = load_full(args)
    data = windows(tok, args.windows, args.seq, args.seed, max_chars=args.corpus_chars)
    print(f"scoring {len(data)} windows at seq {args.seq}", flush=True)

    losses = []
    with torch.no_grad():
        for i in range(len(data)):
            ids = data[i:i + 1].to(args.device)
            logits = model_logits(model, ids)
            ce = F.cross_entropy(logits[0, :-1].float(), ids[0, 1:], reduction="mean")
            losses.append((float(ce), i))
            if (i + 1) % 256 == 0:
                print(f"  {i + 1}/{len(data)}", flush=True)

    losses.sort(key=lambda x: x[0], reverse=True)
    keep = losses[:args.top]
    print(f"loss range: worst {keep[0][0]:.3f}, cut {keep[-1][0]:.3f}, "
          f"best {losses[-1][0]:.3f}", flush=True)

    rows = []
    for ce, i in keep:
        rows.append({"text": tok.decode(data[i].tolist()), "loss": round(ce, 4)})

    if args.traces:
        n_tr = 0
        for line in Path(args.traces).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = rec.get("text") or rec.get("trace")
            if text:
                rows.append({"text": text, "source": "teacher-trace"})
                n_tr += 1
        print(f"appended {n_tr} teacher traces", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"wrote {out}: {len(rows)} rows "
          f"({out.stat().st_size / 1e6:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
