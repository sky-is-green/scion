# Placement sweep — 2026-09-27

Goal: measure what expert offload costs on this box (2× RX 7900 XT, ROCm 7.2.4;
Ryzen 7 7800X3D 8C/16T; 30 GB RAM), and what an adaptive per-expert cache could
buy, before deciding whether any engine work is justified.

Build: `taardis-llama.cpp` @ `53388d74a` (`moe-corr-runtime2`), `build-hip`.
Artifacts: `bench-*.log`, `ppl.log`, `expert-hits.{json,log}`, `run-bench.sh`.

## 1. Expert-offload cost curve — mixed ternary artifact

`olmoe-q1exp-q8rest.gguf` (2.07 GiB, 6.92 B params, Q1_0_g128 experts + Q8_0 rest),
2 GPUs (layer split), `-p 512 -n 128 -r 3 -t 8`.

| `-ncmoe` | pp512 t/s | tg128 t/s | experts moved to CPU |
|---:|---:|---:|---:|
| 0  | 3144.5 | 188.3 | – |
| 4  | 2600.9 | 145.5 | 25% (~428 MiB) |
| 8  | 2349.7 | 122.7 | 50% (~856 MiB) |
| 12 | 2169.5 | 108.0 | 75% (~1.28 GiB) |
| 16 | 2070.0 |  94.1 | 100% (~1.71 GiB) |

Full offload costs **−34% prefill, −50% generation** — still 94 t/s.

## 2. Threads (first-class knob)

At `-ncmoe 16`: `-t 8` → 95.3 t/s; `-t 16` → **35.2 t/s** (reproduced: 36.8, 35.2).
SMT sibling threads destroy CPU expert matmul throughput. Use **physical cores**.

## 3. Single GPU vs layer split (same model)

| config | pp512 | tg128 |
|---|---:|---:|
| 1 GPU, `-sm none`, ncmoe 0 | 4467.3 | 329.6 |
| 2 GPU layer split, ncmoe 0 | 3144.5 | 188.3 |
| 1 GPU, ncmoe 16 | 2347.5 | 104.8 |

For a model that fits one card, layer-splitting costs ~40% generation. Only
split when VRAM forces it.

## 4. f16 contrast (same active params, expert bytes ×7.5)

| model | ncmoe 0 pp/tg | ncmoe 16 pp/tg | penalty |
|---|---:|---:|---:|
| F16 (12.89 GiB) | 4073.8 / 114.4 | 551.0 / 29.0 | −86% / −75% |
| Q1_0_g128 ternary (2.07 GiB) | 3144.5 / 188.3 | 2070.0 / 94.1 | −34% / −50% |

The 2.125 bpw container roughly **halves the CPU-tail penalty** vs f16. This is
the property that makes GPU→CPU tiering viable on this hardware.

## 5. Placement does not change the math

`llama-perplexity`, 128 chunks: ncmoe 0 and ncmoe 16 both **PPL 533.0294 ± 10.77**
(identical per-chunk values). Prefill cost per pass: 1.09 s → 1.61 s (+48%).

## 6. Expert-hit concentration (the adaptive-cache upper bound)

OLMoE-hf on GPU, router logits over 16,384 tokens of `wiki.test.raw`
(16 layers × 64 experts, top-8). Bytes at Q1_0_g128 (1.67 MiB/expert):

| pinned/layer | cache size | hit coverage (mean, min–max) |
|---:|---:|---:|
| 4 (6%)   | 102 MiB  | 15.4% (13.3–17.9) |
| 8 (13%)  | 204 MiB  | 26.2% (23.9–29.4) |
| 16 (25%) | 408 MiB  | 43.8% (41.0–48.4) |
| 24 (38%) | 612 MiB  | 58.4% (55.3–63.8) |
| 32 (50%) | 816 MiB  | 70.7% (66.9–76.6) |
| 48 (75%) | 1224 MiB | 90.1% (86.0–93.6) |

Concentration is real but not steep: half the experts buy ~71% of the mass.
Note llama.cpp stores a whole expert bank as one tensor, so per-expert residency
is engine work, and this is the ceiling it would chase.

## 7. Conclusions / next options

- The "AMD Strata" design is viable with existing pieces: ternary container,
  `-ncmoe`/overrides, single-vs-dual GPU choice, thread count. No engine work
  needed to get most of the way.
- Tuning rules measured here: **threads = physical cores**; **don't layer-split
  when one card fits**; `ncmoe` as the VRAM-budget dial.
- Adaptive per-expert caching has a measured, bounded upside (top-32 = 71%),
  and is a real project — decide against a 256-expert model (empero) before
  building anything.
- `hivebench/m.gguf` (20 GB) does **not** load in this fork — needs a look
  before it can be a tiering subject.

Options: (a) convert/quantize `empero` (qwen3_5_moe, 256 experts) to Q1_0_g128
and repeat the hit profile — sharper concentration would change the verdict;
(b) build `llama-server` end-to-end and serve a tiered config through the API;
(c) stop here — measurements only, nothing new to maintain.
