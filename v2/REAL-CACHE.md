# Real-cache path validation - PLAN 13a step B

Date: 2026-11-12.  Status: **validated end-to-end on a real `_logits_` dump.**
One real dump from the deployed 35B through `llama-perplexity
--save-all-logits`, decoded by `moe/build_trace_cache.py`, checked for
tokenizer and row alignment, and trained on for two prefix steps through the
compact-banks port.  No pod; one 7900 XT.

## What was run

1. **Dump.** `work/prism-qwen4exp-shrink/build-hip/bin/llama-perplexity` on
   `Scion-35B-A3B-PQ2_0-corr.gguf` (the student's own deployed body as a
   stand-in teacher - same 248,320 vocab, real qwen35moe path), 24 KB of
   WikiText-2, `-c 512 --chunks 4`, GPU 1.  PPL 9.62 (consistent with the
   known ~9.0).  Dump: 506,589,172 B.
2. **Decode.** `moe/build_trace_cache.py` -> 4 records.
3. **Alignment checks** (below).
4. **Training.** Two `train` steps on the 4-layer prefix with
   `--compact-banks --grad-checkpoint --checkpoint-mode group
   --expert-group-size 64 --bank-gguf ...`, rank 512 both placements, bias
   balance, `--kd-weight 2.0 --kd-tail-weight 2.0 --temp 2.0`.

## The layout finding (and the fix)

The prism forks' `--save-all-logits` scores **only the second half of each
chunk** (`first = n_ctx/2` -> 255 scored rows per 512-token chunk, not the
511 rows `build_trace_cache.py` assumed).  Verified in three places:
the local dump's byte count is exactly `4 chunks x 255 rows x 496,648 B`;
`prism-qwen4exp-shrink` line 542; and the production teacher fork
(`qwen4exp-proto` of `prism-ml-llama.cpp`, fetched to /tmp/opencode/q4p) has
the identical `first = n_ctx/2` at line 542 - so this IS the layout the
FlashNext teacher pass will produce.

`build_trace_cache.py` now auto-detects the layout from the file size
(`full`: n_ctx-1 rows/chunk, ids = whole chunk; `half`: n_ctx-1-n_ctx/2
rows/chunk, ids = the scored subwindow `tokens[first:]`) and aligns ids so
that **trainer row i of idx/val/w predicts `ids[i+1]` in both layouts** -
no trainer change needed.  Conditioned on this asymmetry: the teacher saw
the chunk's first half as context, the student does not; inherent to the
dump format, applies equally to the pilot, recorded here.

Unit tests: `moe/tests/test_trace_cache_layout.py` (4 tests: full layout,
half layout = the prism behaviour, ambiguous size rejected, and a synthetic
well-calibrated dump whose top-1 must equal the next token - pins the row
alignment).

## Alignment results

| check | result |
|---|---|
| dump tokens vs HF tokenizer (empero) | **2048/2048 exact** (the dump stores only the n_chunk x n_ctx processed tokens; llama.cpp adds no BOS for the Qwen tokenizer) |
| teacher top-1 vs the actual next token | 512/1020 = 0.502 (prose; plausible) |
| targets inside the teacher's top-512 | **98.2%** |
| mean teacher NLL from the cache | **2.232** (lower bound, 18 capped rows) vs the engine's own 2.263 (PPL 9.6165) - consistent |
| support mass (cache `w`) | mean **0.9635** (real signal; the synthetic smoke cache was 0.49) |
| real teacher (FlashNext 176B Q2_0) GGUF tokenizer | 248,320 tokens, 247,587 merges, `<\|im_start\|>` 248045, `<\|im_end\|>` 248046 - matches the student; all 248,077 empero vocab strings found in the teacher's token list |

## Training on the real cache (4-layer prefix, 2 steps)

```
step 1 lm 9.4366 kd 5.8329 tail 2.1855 total 25.4735 mass s 0.1335 t 0.9732
step 2 lm 9.5382 kd 5.3356 tail 2.1430 total 24.4952 mass s 0.1290 t 0.9607
```

- `mass t` 0.973/0.961 = the cache's own support mass, matching the decode
  exactly (the synthetic cache read 0.49).
- kd decreases 5.83 -> 5.33; lm ~9.4 (the prefix's own CE on real text
  through the real embed/lm_head; the synthetic smoke read 14.7 on random
  values).
- Checkpoint written, `--verify-banks`-class gates unaffected; memprobe
  checkpoints cleaned up afterwards.

## Reproduce

```bash
FORK=../prism-qwen4exp-shrink
$FORK/build-hip/bin/llama-perplexity \
  -m ../models-p310/scion-strata/Scion-35B-A3B-PQ2_0-corr.gguf \
  -f ppl-text.txt -c 512 --chunks 4 -ngl 99 \
  --save-all-logits logits.bin          # ~1 min on one 7900 XT
python moe/build_trace_cache.py --logits logits.bin --out cache.pt \
  --top-k 512 --seq 512
../.envs/scion-rocm/bin/python -m pytest moe/tests/test_trace_cache_layout.py -q
# then the 2-step prefix train (command in v2/PROGRESS.md)
```

Artifacts kept: `/tmp/opencode/qwen35-realcache/cache.pt` (4 records,
half layout).  The 500 MB logits dump was deleted (regenerable in ~1 min).
