Adding two-card gfx11 numbers to the thread, because the 2x R9700 result (@Niko1221's code -3.3%) is why the flag is opt-in.

On 2x RX 7900 XT (gfx1100), Linux, ROCm 7.2, display on the second card, engine 0.1.44 plus a local branch that only adds artifact support (PTQ1_0 experts, P5 PLE table types, the embedded correction adapters) and touches no pipeline code: fixed 4-request 128-token protocol, temperature 0, one essay prompt, run 1 warm-up, drafter on the last card.

| arm | runs 2-4 (tok/s) | steady |
| --- | --- | --- |
| off | 41.4 / 41.8 / 41.0 | ~41 |
| `--pipeline-windows 2` | 61.7 / 68.4 / 70.5 | ~70-71 |

Prompt reads 61.5 to 98.4 tok/s, and the warmed verify window is 32.6-34.6 ms at 2.2-2.4 tokens/window (GPU-reach wait ~24-26 ms). Config: `--layer-split auto` (26/22), `--expert-cache auto` with 12007 + 10690 resident slots, 92% of the routed pairs, `--spec 4 --spec-min-p 0.5`, `--ple-io direct --mmap-experts`, `--vram-reserve-mib 3072`, `--pcie-frac 0`.

What differs from the 2x R9700 number:

Nothing spills here, so the windows are GPU-bound, the condition [docs/MULTI_GPU.md](https://github.com/Niko1221/Strata/blob/main/docs/MULTI_GPU.md) says the flag pays in. @Cass67's reading of the 2-card result, where the flag's cache reserve pushes a few experts to the CPU on a tight pair, matches what we see: we lose ~284 MiB of CUDA0's cache to the second verifier and the GDN snapshots and still measure this.

The drafter stays on the last card on this pair. Moving it to the first card (`STRATA_MTP_STAGE0=1`, the placement 0.1.44 defaults to on gfx12) measured 52-59 tok/s against 68-71. The 26-layer card becomes the straggler with the chain on it. If the two-stage auto default is ever widened toward gfx11, it should not carry the drafter move with it.

The warmed greedy text is byte-identical to the same engine without the flag; the first two requests after a start differ in both directions while the adaptive tier swaps (2.9 swaps/window settling to 0.3). @CC-David-CC's #1813 is the rigorous form of that check.

One caveat for a default change: #2245 (open) reports a rare cross-queue stall with `--pipeline-windows 2` on gfx1030 and on Windows/gfx1100, avoided there with `GPU_MAX_HW_QUEUES=2`. On this Linux/ROCm 7.2 pair about 30 requests ran clean with both drafter placements. At the reported rate of about 0.7 per 100 requests, that is one data point and not a claim that the stall is absent.
