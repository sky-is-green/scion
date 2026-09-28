"""Generate AYOT reasoning traces with the FP teacher (GPU stage).

Pairs with ``moe/ayot.py``: this produces the JSONL that
``qwen35_moe_proxy.py cache/train --corpus-file`` mixes into the calibration
windows.  Example (when a card frees up):

    python moe/ayot_gen.py --prompts prompts.jsonl --out traces.jsonl \
        --model $MOE_ARTIFACTS/empero-hf --device cuda:0 \
        --count 512 --batch 4 --max-new-tokens 1024

Dry-run validates the prompts file without loading the model.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ayot import load_prompts  # noqa: E402


def generate(args) -> None:
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    prompts = load_prompts(args.prompts)[:args.count]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device)
    model.eval()
    with Path(args.out).open("w") as out:
        done = 0
        for i in range(0, len(prompts), args.batch):
            chunk = prompts[i:i + args.batch]
            texts = [tok.apply_chat_template(
                [{"role": "user", "content": q}], tokenize=False,
                add_generation_prompt=True) for q in chunk]
            enc = tok(texts, return_tensors="pt", padding=True,
                      padding_side="left").to(args.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                     do_sample=True, temperature=args.temperature,
                                     top_p=args.top_p)
            for j, q in enumerate(chunk):
                new = gen[j][enc["input_ids"].shape[1]:]
                trace = tok.decode(new, skip_special_tokens=True)
                out.write(json.dumps({"question": q, "trace": trace}) + "\n")
                done += 1
            out.flush()
            print(f"{done}/{len(prompts)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True, help="JSONL with question/prompt rows")
    ap.add_argument("--out", required=True, help="JSONL traces for --corpus-file")
    ap.add_argument("--model", default=str(HERE / "artifacts" / "empero-hf"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--count", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--dry-run", action="store_true",
                    help="validate the prompts file and exit (no model load)")
    args = ap.parse_args()
    if args.dry_run:
        prompts = load_prompts(args.prompts)
        print(f"prompts ok: {len(prompts)} rows (first: {prompts[0][:80]!r})")
        return
    generate(args)


if __name__ == "__main__":
    main()
