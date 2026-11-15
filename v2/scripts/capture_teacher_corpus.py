#!/usr/bin/env python3
"""Capture teacher rollouts over the 360-prompt corpus (the MTP drafter corpus).

Talks to the resident strata teacher server (the 176B Q6_K).  Each response is
saved as JSON with the reasoning and answer text separately, plus usage/wall
time.  Idempotent: a prompt whose output file already has text is skipped, so
the pass can be resumed after a reboot.

The MTP probe dump consumes this as text: per trace the corpus text is
    <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{reasoning}{content}<|im_end|>
which tokenizes to the serving format (see build_probe_chunks.py).

Usage:
  cd work/strata && .venv/bin/python ../scion/v2/scripts/capture_teacher_corpus.py \
      --port 8092 --max-tokens 2048 --out-dir ../scion/v2/traces/rollout360
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def post(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def response_text(choice: dict) -> tuple[str, str]:
    msg = choice.get("message") or {}
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    content = msg.get("content") or ""
    return reasoning, content


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--model", default="scion-flashnext-176b-a6b-q6k")
    ap.add_argument("--prompts", default=str(REPO / "v2/corpus/rollout-prompts.jsonl"))
    ap.add_argument("--out-dir", default=str(REPO / "v2/traces/rollout360"))
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=1001)
    ap.add_argument("--limit", type=int, default=0, help="0 = all prompts")
    ap.add_argument("--timeout", type=float, default=1200.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = [json.loads(line) for line in open(args.prompts) if line.strip()]
    if args.limit:
        prompts = prompts[: args.limit]

    t_start = time.time()
    done = 0
    for i, p in enumerate(prompts):
        pid = p["id"]
        out = out_dir / f"{pid}.json"
        if out.exists():
            try:
                old = json.loads(out.read_text())
                if (old.get("reasoning") or old.get("content")) and not old.get("error"):
                    done += 1
                    continue
            except Exception:
                pass

        body = {
            "model": args.model,
            "messages": [{"role": "user", "content": p["prompt"]}],
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_tokens": args.max_tokens,
            "seed": args.seed0 + i,
            "stream": False,
        }
        t0 = time.time()
        rec = dict(p)
        try:
            d = post(f"http://{args.host}:{args.port}/v1/chat/completions", body, args.timeout)
            ch = (d.get("choices") or [{}])[0]
            reasoning, content = response_text(ch)
            rec.update(reasoning=reasoning, content=content, usage=d.get("usage"),
                       finish_reason=ch.get("finish_reason"), timings=d.get("timings"))
            ok = bool(reasoning or content)
            if not ok:
                rec["error"] = "empty response"
        except (urllib.error.URLError, TimeoutError, ValueError) as e:
            rec.update(reasoning="", content="", error=f"{type(e).__name__}: {e}")
        rec["wall_s"] = round(time.time() - t0, 1)
        out.write_text(json.dumps(rec, indent=1) + "\n")

        done += 1
        el = time.time() - t_start
        print(f"[{done}/{len(prompts)}] {pid} {rec.get('wall_s')}s "
              f"tok={((rec.get('usage') or {}).get('completion_tokens'))} "
              f"err={rec.get('error')} elapsed={el/60:.1f}m", flush=True)

    print(f"done: {done} traces -> {out_dir} in {(time.time()-t_start)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
