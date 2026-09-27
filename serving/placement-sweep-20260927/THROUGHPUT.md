# Hot-cache payoff sweep (2026-09-27)

Model: `olmoe-q1exp-q8rest.gguf` + `branches-r1024-attnoutlloyd-step8000-q1g128-routers.lora.gguf`
(corrected, PPL 16.2). All runs: `-ncmoe 16 -ngl 99 -t 8 --split-mode none --main-gpu 0 -c 512`,
server `/completion`, 128 tokens, seed 1.

| config | experts on GPU | tg t/s | pp t/s |
|---|---|---:|---:|
| cpu (no sidecar) | 0% | **93.3** | 132.5 |
| + hot16 sidecar | 25% of experts | 89.4 | 110.5 |
| + hot32 sidecar | 50% | 76.9 | 109.6 |
| + hot64 sidecar (control) | 100% | 89.4 | 118.7 |

## Why there is no gain (important)

`mul_mat_id` computes **k experts per token regardless of weight**. Our cold
pass feeds the full 64-expert bank with hot slots' weights zeroed — so the CPU
still performs all 8 expert evaluations per token, while the GPU now also
computes the hot slots. Net: same CPU work + extra GPU work + split overhead.

Reducing CPU work requires the per-token selection to be *compacted* into a
smaller number of slots (variable k per token → sparse dispatch), which is real
kernel work: exactly the "dynamic cache" tier that was deferred.

The control confirms the mechanism is sound but performance-neutral: hot64
(cold pass all-zero, hot pass covers everything) lands at the same 89 t/s as
the baseline — the machine is not bound by CPU expert FLOPs at this size; it is
bound by the GPU-side graph, and the split only adds nodes.

## What this means for the extras

The static hot/cold split as implemented is **not worth keeping for
performance** on OLMoE-class models. The effective levers remain the ones
already measured:

- ternary container: −50% generation cost under full CPU offload vs f16
- layer-level placement (`-ncmoe`): 94 → 188 t/s as layers move to GPU
- single GPU when the model fits; threads = physical cores

If the cache is ever revisited, it must be as **sparse per-token dispatch**
(GPU and CPU each take a variable share of the k selected experts), and the
target must be a model where the CPU expert tail actually dominates AND the
hot-set concentration is steep.
