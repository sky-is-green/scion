#!/usr/bin/env python
"""Byte-identity check of the copied containers between the original export
and the Q6_K body repack: adapters fully, experts + PLE table sampled.
Memory-bounded: 1 MiB chunked reads via gguf-py offsets."""
import os
import sys
from pathlib import Path

from scion_paths import HIVE_ARTIFACTS, QWEN4EXP_GGUF_PY

sys.path.insert(0, str(QWEN4EXP_GGUF_PY))
import gguf  # noqa: E402

_ART = HIVE_ARTIFACTS / "ternary" / "moe" / "qwen4exp"
A_PATH = os.environ.get(
    "SCION_REPACK_A", str(_ART / "qwen4exp-48l-ptq1_0-corr-step3000-ple.gguf"))
B_PATH = os.environ.get(
    "SCION_REPACK_B", str(_ART / "qwen4exp-48l-ptq1_0-corr-step3000-ple-q6k.gguf"))
CHUNK = 1 << 20


def read_range(f, off, n):
    f.seek(off)
    return f.read(n)


def main():
    ra = {t.name: t for t in gguf.GGUFReader(A_PATH).tensors}
    rb = {t.name: t for t in gguf.GGUFReader(B_PATH).tensors}
    assert set(ra) == set(rb)
    fa, fb = open(A_PATH, "rb"), open(B_PATH, "rb")
    n_adapter = n_expert = n_table = 0
    bad = 0
    for name, ta in sorted(ra.items()):
        tb = rb[name]
        if ".lora_" in name:
            n_adapter += 1
            a = read_range(fa, ta.data_offset, ta.n_bytes)
            b = read_range(fb, tb.data_offset, tb.n_bytes)
            if a != b:
                bad += 1
                print("DIFF adapter", name, flush=True)
        elif "exps" in name:
            n_expert += 1
            size = ta.n_bytes
            # sample head/mid/tail 1 MiB (or whole if small)
            offs = sorted({0, max(0, size // 2 - CHUNK // 2), max(0, size - CHUNK)})
            for o in offs:
                n = min(CHUNK, size - o)
                if read_range(fa, ta.data_offset + o, n) != read_range(fb, tb.data_offset + o, n):
                    bad += 1
                    print("DIFF expert", name, "at", o, flush=True)
                    break
        elif "per_layer_token_embd" in name:
            n_table += 1
            size = ta.n_bytes
            for i in range(16):
                o = int(size * i / 16)
                n = min(CHUNK, size - o)
                if read_range(fa, ta.data_offset + o, n) != read_range(fb, tb.data_offset + o, n):
                    bad += 1
                    print("DIFF table at", o, flush=True)
                    break
    print(f"adapters full-compared: {n_adapter}, experts sampled: {n_expert}, "
          f"table sampled: {n_table}, diffs: {bad}")
    fa.close(); fb.close()
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
