# AirLLM review — what it does, and what (if anything) transfers (2026-10-14)

Studied from source at `lyogavin/airllm` commit `0f679cc` (main, 2026-10-14); the published claims were
cross-checked against independent reports.  The question: can AirLLM's techniques speed up the Scion releases
in Strata?  **Short answer: no for the warm decode path** — it is a fit-at-any-speed design, and our serve is
already 3-4 orders of magnitude faster there.  One idea applies only to the disk-bound cold start, and its
other lever (batching) Strata already ships.

## How it actually works (verified in the code)

- The real transformers model is built on the `meta` device (no memory); forward hooks stream each module's
  weights: the pre-hook loads that module's per-layer safetensors shard disk -> pinned CPU (`pin_memory`, only
  for modules <= 2 GB, else pageable) -> GPU; the post-hook frees them back to meta.  A one-worker
  `ThreadPoolExecutor` prefetches the *next* module while the current one computes; the H2D copy itself is
  synchronous at the module boundary (`airllm_base.py` lines 429-457, 872-909).
- The disk read is the bottleneck, so the two published speed levers are: (a) block-wise 4/8-bit shard
  compression ("3x speed up" — the load is smaller, which is why the paper is cited), and (b) the prefetch
  overlap (v2.5, "+10%").
- Special cases: for **Qwen3.8-Flash-Next / Qwen4Exp — the 176B's own family** — the packed MoE experts are
  3D tensors, so per-expert hooks do not apply and whole decoder layers stream; the ~102 GB n-gram (PLE) table
  is file-mapped on the host and peeled out of the streamed layer (`airllm_qwen4_exp.py`).  For Kimi K3 the
  experts are separate modules and stream individually (`airllm_kimi_k3.py`).
- Independent numbers: 70B on a 4 GB card runs at ~5-35 s/token (warm file cache), commonly far below
  1 tok/s; a community batching fork reports 7x by amortizing one layer's streamed weights over many
  requests.  The headline claims are about **fit**, and they are real — at that speed.

## Against our measured state

| AirLLM lever | Strata's equivalent | our measurement |
| --- | --- | --- |
| One layer in VRAM at a time, streamed per token | dense weights resident, 11.74 GiB expert cache, `--mmap-experts` | **70-77 tok/s** warmed; AirLLM-class is 5-35 **s**/token on a 70B |
| Prefetch the next module while the current computes | adaptive VRAM tier + ForesightSwap + pinned bounce slots; `--pipeline-windows` overlap (0.1.44) | expert misses 0.04-0.05 entries per layer-window (99.9% hit) |
| Block-wise compression of the shards (the disk is the wall) | experts already 1.75-2.125 bpw (PTQ1_0 / Q2_0 / PQ2_0) | smaller than AirLLM's 4/8-bit optimum |
| Per-expert streaming (K3) | per-layer doorbell + persistent expert cache + CPU pool for misses | VRAM hits 34-35 per layer-window; the cache is not reloaded per token |
| Giant tables file-mapped on the host | `--ple-io mmap` / `direct` | both measured in P4 |
| Batch requests to amortize the streamed weights | batch windows / `--batch-groups` | upstream BATCHING.md: 166 tok/s aggregate on 4 GPUs vs 86 for one group |

The mechanisms overlap almost one-to-one, and where they overlap Strata's version is the more aggressive one
(persistent residency instead of reload-per-token, 2-bit instead of 4/8-bit, copy-engine prefetch instead of a
Python thread).  Their per-module hook design is also structurally incompatible with our captured-graph
decode: it would put a Python callback and a fresh H2D of every weight on the critical path of every token.

## The one place it touches us: the cold start — measured, and already at the wall

The service journal (2026-10-14, several restarts) showed the whole start costing 35-65 s, but the engine's own
fill line separates it: the primary fill of 11745 slots (11.76 GiB) takes **8.0-9.7 s = 1296-1572 MB/s** with
madvise read-ahead (`STRATA_FILL_AHEAD=256`, already on).  AirLLM's logic — "on a disk-bound path, overlap the
loads and shrink the loads" — was the right lens, so the device wall was measured directly (`odread.c`, O_DIRECT,
1 MiB reads on the served 60 GB GGUF):

| pattern | 1 thread | 4 threads |
| --- | ---: | ---: |
| sequential | 1685 MB/s | 2018 MB/s |
| strided 37 MiB (the profile's scatter) | **1085 MB/s** | **1112 MB/s** |

The fill's 1.3-1.6 GB/s already exceeds the strided single-thread figure and is at the sequential wall; the
strided pattern does not scale with threads (4 threads = 1 thread).  So there is no overlappable read headroom,
and the "shrink the loads" half is already served: our experts are 1.75-2.125 bpw, smaller than AirLLM's 4/8-bit
compression target.  The fill path is saturated; nothing from AirLLM applies to it.

For the **Scion-35B-A3B** this is moot twice over: at 11.34 GB / 2.125 bpw it fits one 20 GB card whole, so
streaming would be a strict downgrade; and it first needs the type-142 container port (see
`SCION-MOE-SPEED.md`, step 3).

## Verdict

- No AirLLM code, dependency or design change is proposed for the decode path: nothing in it beats what the
  engine already does, and the observed speed class is ~4 orders of magnitude behind ours.
- The one transferable idea — overlapped loads on the disk-bound cold start — was measured and closed: the fill
  already runs at the strided device wall (1.3-1.6 GB/s vs 1.09-1.11 GB/s O_DIRECT single/multi-thread for the
  same pattern), and our 1.75-2.125 bpw containers are already smaller than the 4/8-bit compression AirLLM uses
  to shrink that path.  The batching lever is already implemented and measured in our batch path.
- If a "run a model that does not fit at all" need ever appears (e.g. a 671B-class file on this box), AirLLM's
  layer-streaming is the reference design for *fit* — with the measured expectation of seconds per token.
