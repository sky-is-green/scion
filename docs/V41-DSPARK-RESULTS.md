# V4.1 DSpark — first local training result (negative)

**Date: 2026-10-25. Status: stopped at the held-out gate; no engine export.**

## What was run

All work was local, one RX 7900 XT, under `strata-ds41/local/mem-guard.sh`
(`8000` MiB available floor, `4000` MiB GTT ceiling), with no second engine.

Capture:

| run | records | note |
|---|---:|---|
| code-review, greedy | 16,382 | capture aborted at 16k in `argsort_f32_i32_cuda_bitonic`; valid prefix used |
| code-review, temp 1 | 8,265 | degenerate repetition; excluded from training |
| math, temp 0.7 | 2,098 | 2,048 generated tokens |
| prose, temp 0.7 | 2,086 | 2,048 generated tokens |
| systems, temp 0.7 | 2,087 | held out |

The abort was not OOM: host memory and GTT stayed low. The later captures used
`GGML_CUDA_DISABLE_GRAPHS=1` and completed.

Corpus: four prompts, 175 windows at 128 tokens, 159 train / 16 held-out
(systems). The first draft used the **mean of the three target taps** (37/38/39)
as the hidden input. That is a deliberate first smoke, not the final
tap-projection design.

## Result

`DSparkHead`, k=3:

| run | params | held-out chained acceptance | ideal tokens/pass |
|---|---:|---|---:|
| full state, 1000 steps | 419M | 0.202 / 0.0005 / 0.0 | 1.20 |
| state 1024, 1000 steps | 197M | 0.123 / 0.0005 / 0.0 | 1.12 |

k=1 MLP control, 1000 steps: **0.142** held-out acceptance.

Training loss collapsed to ~0.001 while held-out acceptance stayed low: the
head is memorising the training prompts. The earlier 3.17 tokens/pass result
was same-sequence window leakage, not a general result.

## Conclusion

The local capture and training path works mechanically, but this corpus is far
too small and the mean-of-taps input is lossy. Do not export this head or wire
it into the engine; it fails the gate.

The next attempt needs, in order:

1. **Tap projection** (3 × 5120 → 5120), not the mean.
2. **Much more data**: thousands of windows across many prompts, not 159 from
   four prompts. At the observed ~20 tok/s, a 1M-token local capture is ~14
   hours; that is the real cost.
3. A smaller/regularised head or the k=1 control as the data-efficiency
   baseline.
4. Only after held-out acceptance clears the existing DSpark, export and
   benchmark wall time.

No OOM occurred. The limiting factor is data and representation, not local
memory.
