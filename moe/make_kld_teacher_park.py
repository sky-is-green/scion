#!/usr/bin/env python3
"""Materialise the 48-layer KLD teacher park for the GGUF-side KLD gate.

Converts the PyTorch park (``tcache-48l/w{i}.pt``, fp32 log-probs produced by
``qwen4exp_eval.py --stage teacher``) into the flat format ``llama-kld``
reads, and regenerates the exact token windows with the same code path the
park used (``olmoe_proxy.windows``, seed 999, wikitext test, 8 x 512):

    tokens.bin          int32 LE [8 * 512]
    teacher/w{i}.bin    fp32 LE [511, n_vocab] log-softmax rows

The parked entropy / top-1 checks in ``meta.json`` are re-measured from the
files and must match; that catches a stale or misaligned park before it is
uploaded to a pod.

    work/.venv-rocm/bin/python moe/make_kld_teacher_park.py
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

FP8_MODEL = Path("/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/"
                 "models/qwen38-flashnext-fp8")
TCACHE = Path("/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/"
              "hivebench/artifacts/ternary/moe/qwen4exp/pod/remote-w1w2/tcache-48l")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tcache", default=str(TCACHE))
    ap.add_argument("--model-dir", default=str(FP8_MODEL))
    ap.add_argument("--out", default=str(HERE / "artifacts/qwen4exp/kld-park"))
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--split", default="wikitext")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "teacher").mkdir(parents=True, exist_ok=True)
    tc = Path(args.tcache)
    meta = json.loads((tc / "meta.json").read_text())
    print(f"park meta: {meta}")

    from transformers import AutoTokenizer
    from olmoe_proxy import windows

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    data = windows(tok, args.windows, args.seq, args.seed, args.split)
    ids = data.to(torch.int32)
    ids.numpy().astype("<i4").tofile(out / "tokens.bin")
    print(f"tokens: {tuple(ids.shape)} -> {out/'tokens.bin'}")

    ents, peaks = [], []
    n_vocab = None
    for w in range(args.windows):
        lp = torch.load(tc / f"w{w}.pt", map_location="cpu")
        if lp.dim() != 2 or lp.shape[0] != args.seq - 1:
            raise SystemExit(f"w{w}.pt shape {tuple(lp.shape)} != "
                             f"[{args.seq-1}, vocab]")
        if n_vocab is None:
            n_vocab = int(lp.shape[1])
        elif int(lp.shape[1]) != n_vocab:
            raise SystemExit(f"w{w}.pt vocab {lp.shape[1]} != {n_vocab}")
        lp32 = lp.float().numpy().astype("<f4")
        lp32.tofile(out / "teacher" / f"w{w}.bin")
        p = np.exp(lp32.astype(np.float64))
        ents.append(float(-(p * lp32).sum(1).mean()))
        peaks.append(float(p.max(1).mean()))
        print(f"  w{w}: {lp32.shape} entropy {ents[-1]:.4f} "
              f"peak {peaks[-1]:.4f}", flush=True)

    t_ent = sum(ents) / len(ents)
    t_pk = sum(peaks) / len(peaks)
    print(f"teacher entropy {t_ent:.4f} (meta {meta['teacher_entropy_nats']}), "
          f"top1 {t_pk:.4f} (meta {meta['teacher_top1_prob']})")
    if abs(t_ent - meta["teacher_entropy_nats"]) > 1e-3 or \
       abs(t_pk - meta["teacher_top1_prob"]) > 1e-3:
        raise SystemExit("park does not match meta.json; refusing to publish")

    (out / "meta.json").write_text(json.dumps({
        "source": str(tc), "n_vocab": n_vocab, "windows": args.windows,
        "seq": args.seq, "seed": args.seed, "split": args.split,
        "teacher_entropy_nats": t_ent, "teacher_top1_prob": t_pk,
    }, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
