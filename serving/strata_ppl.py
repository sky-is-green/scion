#!/usr/bin/env python3
"""strata_ppl.py - wikitext-2 perplexity through Strata's teacher-forced log-probability path.

How it works: with `STRATA_LOGPOS=<file>` set and the prompt read through the verify windows
(serve mode; the config needs `--short-read` >= the prompt length), the engine appends one line per
prompt position:

    pos <TAB> target <TAB> logprob <TAB> top <TAB> top_logprob <TAB> top==target <TAB> extra <TAB> without

`target` is the prompt's OWN next token, so the logprob column is the teacher-forced score of that
token.  This tool sends consecutive ~512-token slices of the corpus as single user messages to the
resident server, finds each slice's own tokenization in the appended lines by matching the target
sequence, and sums the log-probabilities.

Convention (stated so the number can be compared): the text is tokenized once with the model's
tokenizer; slices are `--chunk` tokens decoded to text and trimmed; every token of a slice except
its first is scored **exactly**.  The first token of a slice has no line (nothing precedes it), the
same convention llama.cpp's perplexity tool uses per chunk.  PPL = exp(-sum(logprob) / scored).

Usage (from the repo root, with its venv so `regex` is present):
    .venv/bin/python /path/strata_ppl.py --corpus <wiki.test.raw> --logpos <file> \
        --out <results.jsonl> [--limit N] [--start I]
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time
import urllib.request

REPO = pathlib.Path(__file__).resolve().parents[2] / "strata"
sys.path.insert(0, str(REPO / "tools"))
from strata_tokenizer import Tokenizer  # noqa: E402

DEFAULT_GGUF = "/home/penis/models/scion-strata/Scion-FlashNext-176B-A6B-Q6K.gguf"


def post_chat(server: str, text: str, timeout: float, assistant_role: bool) -> dict:
    if assistant_role:
        messages = [{"role": "user", "content": "Continue the text."},
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": "Continue."}]
    else:
        messages = [{"role": "user", "content": text}]
    body = json.dumps({
        "messages": messages,
        "max_tokens": 1,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(server.rstrip("/") + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", default="/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/ternary-serve/wiki.test.raw")
    ap.add_argument("--gguf", default=DEFAULT_GGUF)
    ap.add_argument("--server", default="http://127.0.0.1:8081")
    ap.add_argument("--logpos", default="/tmp/opencode/ppl/logpos_serve.txt")
    ap.add_argument("--out", default="/tmp/opencode/ppl/strata-ppl.jsonl")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--start", type=int, default=0, help="first chunk index")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--assistant", action="store_true",
                    help="send each slice as an assistant message (the model reads its own words)")
    ap.add_argument("--timeout", type=float, default=900.0)
    args = ap.parse_args()

    tk = Tokenizer.from_gguf(args.gguf)
    text = pathlib.Path(args.corpus).read_text(encoding="utf-8")
    ids = tk.encode(text)
    n_chunks = (len(ids) + args.chunk - 1) // args.chunk
    print(f"corpus: {len(ids)} tokens -> {n_chunks} chunks of {args.chunk}", flush=True)

    logpos = pathlib.Path(args.logpos)
    logpos.touch(exist_ok=True)
    fh = logpos.open("r", encoding="utf-8")
    fh.seek(0, 2)  # start at the end: only lines this run appends matter

    out = pathlib.Path(args.out)
    done_ids = set()
    total_nll = 0.0
    total_tokens = 0
    if out.exists():
        for line in out.read_text().splitlines():
            try:
                r = json.loads(line)
                if r.get("ok"):
                    done_ids.add(r["i"])
                    total_nll += r.get("nll", 0.0)
                    total_tokens += r.get("tokens", 0)
            except Exception:  # noqa: BLE001
                pass
    done = len(done_ids)
    t0 = time.time()
    limit = args.limit if args.limit > 0 else n_chunks
    for ci in range(args.start, min(n_chunks, args.start + limit)):
        if ci in done_ids:
            continue
        sl = ids[ci * args.chunk:(ci + 1) * args.chunk]
        piece = tk.decode(sl).strip()
        pat = tk.encode(piece)
        if len(pat) < 2:
            continue
        try:
            post_chat(args.server, piece, args.timeout, args.assistant)
        except Exception as e:  # noqa: BLE001
            print(f"chunk {ci}: request failed: {e}", flush=True)
            continue
        new = []
        while True:
            line = fh.readline()
            if not line:
                break
            new.append(line)
        targets = [int(l.split("\t")[1]) for l in new]
        want = pat[1:]
        found = -1
        for j in range(0, max(0, len(targets) - len(want) + 1)):
            if targets[j:j + len(want)] == want:
                found = j
                break
        if found < 0:
            print(f"chunk {ci}: pattern not found in {len(new)} new lines", flush=True)
            with out.open("a", encoding="utf-8") as o:
                o.write(json.dumps({"i": ci, "ok": False, "lines": len(new)}) + "\n")
            continue
        nll = -sum(float(new[found + k].split("\t")[2]) for k in range(len(want)))
        total_nll += nll
        total_tokens += len(want)
        done += 1
        with out.open("a", encoding="utf-8") as o:
            o.write(json.dumps({"i": ci, "ok": True, "tokens": len(want), "nll": nll}) + "\n")
        ppl = math.exp(total_nll / total_tokens) if total_tokens else 0.0
        dt = time.time() - t0
        print(f"chunk {ci:3d}: {len(want):3d} tok, nll {nll:8.2f}, running PPL {ppl:.4f} "
              f"({done} chunks, {total_tokens} tok, {dt:.0f}s)", flush=True)

    if total_tokens:
        print(f"\nStrata PPL ({args.corpus.rsplit('/', 1)[-1]}, {args.chunk}-token slices): "
              f"{math.exp(total_nll / total_tokens):.4f}  over {total_tokens} tokens in {done} chunks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
