#!/usr/bin/env python3
"""Build the coder-variant calibration corpus for the P5 expert-usage map.

Code-dominant, on the 2607/ISTA recipe shape: teacher-forced code solutions
(the dominant mass), on-policy code generations, agentic/tool-calling
dialogues, and a small general ballast slice:

  code-tf      Evol-CodeAlpaca (instruction, output) rendered as a chat and
               prefilled whole: routing on real code solutions, not prompts
  code-gen     the same instructions with a generation prompt, 384 tokens
               generated: the model's own on-policy code traces
  tool-tf/gen  glaive-function-calling-v2 dialogues, the last assistant turn
               either prefilled or left to generate: agentic routing
  ballast      fineweb-edu chunks + GSM8K solutions, kept near 15% of chars

Vision is not represented: the released GGUF carries no vision tower (text
only), so an image slice could not exercise any expert path.

    work/.venv-rocm/bin/python moe/build_coder_corpus.py \
        --out moe/artifacts/qwen4exp/router-corpus-coder.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

FP8_MODEL = Path("/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/"
                 "models/qwen38-flashnext-fp8")


def chat(tok, messages: list[dict], gen: bool) -> str:
    return tok.apply_chat_template(messages, tokenize=False,
                                   add_generation_prompt=gen)


def trunc_line(t: str, max_chars: int) -> str:
    if len(t) <= max_chars:
        return t
    t = t[:max_chars]
    cut = t.rfind("\n")
    return t[:cut] if cut > max_chars // 2 else t


def build_code(tok, rows: int, tf_chars: int, gen_rows: int, gen_tokens: int):
    from datasets import load_dataset
    ds = load_dataset("theblackcat102/evol-codealpaca-v1", split="train",
                      streaming=True)
    out, n_tf = [], 0
    for row in ds:
        instr = (row.get("instruction") or "").strip()
        answer = (row.get("output") or "").strip()
        if len(instr) < 20 or len(answer) < 40:
            continue
        answer = trunc_line(answer, tf_chars)
        msgs = [{"role": "user", "content": instr},
                {"role": "assistant", "content": answer}]
        out.append({"prompt": chat(tok, msgs, gen=False), "n_predict": 0,
                    "source": "code-tf"})
        n_tf += 1
        if n_tf >= rows:
            break
    n_gen = 0
    for row in ds:
        instr = (row.get("instruction") or "").strip()
        answer = (row.get("output") or "").strip()
        if len(instr) < 20 or len(answer) < 40:
            continue
        msgs = [{"role": "user", "content": trunc_line(instr, 4000)}]
        out.append({"prompt": chat(tok, msgs, gen=True), "n_predict": gen_tokens,
                    "source": "code-gen", "temperature": 0.6, "top_p": 0.95})
        n_gen += 1
        if n_gen >= gen_rows:
            break
    return out


_TURN = re.compile(r"^(USER|ASSISTANT|SYSTEM|FUNCTION RESPONSE):", re.M)


def parse_glaive(chattext: str) -> list[dict]:
    """Split the glaive 'chat' string into role/content messages."""
    parts = _TURN.split(chattext)
    msgs = []
    it = iter(parts[1:])
    for role, content in zip(it, it):
        role = {"USER": "user", "ASSISTANT": "assistant",
                "SYSTEM": "system", "FUNCTION RESPONSE": "user"}[role]
        content = content.strip()
        if content:
            msgs.append({"role": role, "content": content})
    # keep only the first exchange pair after the system prompt, cap size
    return msgs[:4]


def build_tool(tok, rows: int, gen_rows: int, gen_tokens: int, max_chars: int):
    from datasets import load_dataset
    ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train",
                      streaming=True)
    out, n_tf, n_gen = [], 0, 0
    for row in ds:
        msgs = parse_glaive(row.get("chat") or "")
        if len(msgs) < 2 or len(msgs[0]["content"]) < 20:
            continue
        msgs = [{"role": m["role"], "content": trunc_line(m["content"], max_chars)}
                for m in msgs]
        if msgs[-1]["role"] == "assistant" and n_tf < rows:
            out.append({"prompt": chat(tok, msgs, gen=False), "n_predict": 0,
                        "source": "tool-tf"})
            n_tf += 1
        elif msgs[-1]["role"] == "user" and n_gen < gen_rows:
            out.append({"prompt": chat(tok, msgs, gen=True),
                        "n_predict": gen_tokens, "source": "tool-gen",
                        "temperature": 0.6, "top_p": 0.95})
            n_gen += 1
        if n_tf >= rows and n_gen >= gen_rows:
            break
    return out


def build_ballast(tok, want_chars: int, gsm_rows: int):
    from build_router_corpus import fineweb_chunks
    out = []
    for c in fineweb_chunks(want_chars, 10_000, 6_000):
        out.append({"prompt": c, "n_predict": 0, "source": "ballast-fineweb"})
    if gsm_rows:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="train", streaming=True)
        n = 0
        for row in ds:
            q, a = row.get("question"), row.get("answer")
            if not q or not a:
                continue
            msgs = [{"role": "user", "content": trunc_line(q.strip(), 2000)}]
            p = chat(tok, msgs, gen=True)
            p += a.strip() + "<|im_end|>"
            out.append({"prompt": p, "n_predict": 0, "source": "ballast-math"})
            n += 1
            if n >= gsm_rows:
                break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-dir", default=str(FP8_MODEL))
    ap.add_argument("--code-tf-rows", type=int, default=2600)
    ap.add_argument("--code-tf-chars", type=int, default=8000)
    ap.add_argument("--code-gen-rows", type=int, default=256)
    ap.add_argument("--code-gen-tokens", type=int, default=384)
    ap.add_argument("--tool-tf-rows", type=int, default=600)
    ap.add_argument("--tool-gen-rows", type=int, default=256)
    ap.add_argument("--tool-gen-tokens", type=int, default=256)
    ap.add_argument("--tool-chars", type=int, default=4000)
    ap.add_argument("--ballast-chars", type=int, default=2_000_000)
    ap.add_argument("--gsm-rows", type=int, default=200)
    ap.add_argument("--max-prompt-chars", type=int, default=11_000,
                    help="hard cap per rendered prompt (the pod slots are 4096 "
                         "tokens; ~3 chars/token on code)")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    print(f"tokenizer {type(tok).__name__}, chat template "
          f"{'present' if tok.chat_template else 'MISSING'}", flush=True)

    rows = []
    part = build_code(tok, args.code_tf_rows, args.code_tf_chars,
                      args.code_gen_rows, args.code_gen_tokens)
    rows += part
    print(f"code: {sum(1 for r in part if r['source']=='code-tf')} tf + "
          f"{sum(1 for r in part if r['source']=='code-gen')} gen", flush=True)
    part = build_tool(tok, args.tool_tf_rows, args.tool_gen_rows,
                      args.tool_gen_tokens, args.tool_chars)
    rows += part
    print(f"tool: {sum(1 for r in part if r['source']=='tool-tf')} tf + "
          f"{sum(1 for r in part if r['source']=='tool-gen')} gen", flush=True)
    part = build_ballast(tok, args.ballast_chars, args.gsm_rows)
    rows += part
    print(f"ballast: {len(part)}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_capped = 0
    with out.open("w") as fh:
        for r in rows:
            if len(r["prompt"]) > args.max_prompt_chars:
                r["prompt"] = trunc_line(r["prompt"], args.max_prompt_chars)
                n_capped += 1
            fh.write(json.dumps(r) + "\n")
    chars = sum(len(r["prompt"]) for r in rows)
    gen = sum(1 for r in rows if r["n_predict"])
    from collections import Counter
    print(f"wrote {out}: {len(rows)} rows, {chars/1e6:.1f}M prompt chars, "
          f"{gen} generate rows, {n_capped} capped", flush=True)
    print("sources:", dict(Counter(r["source"] for r in rows)), flush=True)


if __name__ == "__main__":
    main()
