# HARVEST: what the Nemotron lane leaves for Flash-Next and the 35B

The Nemotron lane is closed — not because the engineering failed (the engine reached exact greedy parity and the
calibration/drift harnesses work), but because **the base isn't worth it**: Super's published 2-bit tier already
sits where Unsloth's own docs say agentic use breaks, NVIDIA's BF16 numbers show it trails Qwen3.8-Flash-Next on
the agentic/coding axis *before* any quantization, and the only source we can legally quantize from locally is
itself an already-quantized file. That is community-scale work we don't want to own.

Nothing here is Super-specific. Three things transfer, in value order.

## 1. Measurement methodology  (→ both lanes, zero port cost)

**The single-token KL is not a usable metric.** Our first drift reading said the 1024-token imatrix was *worse*
than the 128-token one (KL 0.1808 vs 0.1018). Teacher-forcing 32 tokens (`--force K` dumps K+1 full
distributions) reversed it:

| imatrix | KL p50 | KL p90 | steps > 1 nat | top-1 |
|---|---|---|---|---|
| 128 tokens, one sequence | 0.0935 | 0.99 | 4/33 | 84.8% |
| 1024 tokens, 32-token chunks | **0.0569** | 0.80 | 3/33 | 78.8% |

Report **p50/p90 and "steps > 1 nat", not the mean** — the mean is dominated by 3-4 confident token flips (the
reference entropy at those steps is 0.1-0.2, i.e. a different-but-plausible argmax, not a blurred distribution).

**Chunked imatrix calibration.** Many short sequences in separate processes, each dumping its own imatrix, then
merge (`sums` summed, `ncall` summed — llama-quantize divides one by the other, so summation is exact). Cost is
**superlinear** in sequence length: a 32-token chunk ≈ 60 s, a 128-token single run ≈ 22 min. 1024 tokens in
**~40 min** chunked vs ~6 h single-sequence. Caveat: short chunks distort the deep statistics — the
`ffn_down_exps` energy vectors correlated only 0.60-0.84 with a long run, while the latent-driven
`ffn_up_exps` correlated 0.96-0.99.

Tools, family-agnostic — only the model path and the tensor-name regex change (already committed in strata-nem
`5b781526`): `tools/nemotron/{calib_chunks.sh, merge_imatrix.py, drift.sh, drift_kl.py}`.
The 35B lane already has `tools/qwen35/kl.py`; it needs the `--force`-based multi-step reporting added.

## 2. Host-side expert residency, promoted into the core  (→ Flash-Next)

CachyLLama's `--moe-expert-residency`, adapted: a per-layer **recency+frequency** slot cache that issues
`madvise(MADV_WILLNEED)` on the selected experts' **mmap pages** and `MADV_COLD` on evicted ones
(non-destructive; `MADV_DONTNEED` is destructive on a file mapping and `MADV_FREE` is `EINVAL` on
`MAP_SHARED|PROT_READ`).

- **Now:** `src/qwen35/q36_engine.cu`, `NemRes` / `nem_res_touch` / `madv_range`, env `STRATA_Q36_NEM_RESIDENT`,
  driven only from the Nemotron latent-MoE dispatch.
- **Should be:** `src/core/expert_source.cpp` (`FileExpertSource`). The core already has the primitives —
  `platform::advise_willneed` and the `mincore`-based `resident_sampled` — and `ExpertCache` owns the VRAM side.
  **Flash-Next is served by the core stack** (`src/program/generate.cpp` + `ExpertCache`/`FileExpertSource`), so
  promoting the policy there hands it to Flash-Next *and every other MoE* at once. Port is ~120 lines.

**Sizing rule, measured:** the policy asks for pages, the kernel decides. On Super, 64 experts/block gave a
58.6% software hit rate; 256/block gave 82.4% and **bought no wall time at all**, because the policy requested
~29 GB of hot pages inside a 30 GB box and the kernel evicted them. Keep the resident budget at roughly **40% of
RAM or less**, and instrument with the existing `mincore` sampler — a "hit rate" from the policy is a *software*
number and the CachyLLama doc warns about exactly this confusion.

**Budget against this:** a model larger than RAM reads **0.69-0.91 GB/token** from disk (measured on Super:
32 tokens → 22.1 GiB, 64 tokens → 58.2 GiB). That, not FLOPs, is the throughput ceiling.

`ExpertCache` also has a deliberate **no-eviction** per-layer rule (`admit()` returns `kNotResident` when full) —
worth revisiting alongside the residency policy, since eviction is what the R+F policy is for.

## 3. Expert imatrix accumulator  (→ the 35B lane's quantization work)

`STRATA_Q36_NEM_IMATRIX=<file>` accumulates, per MoE block, the expert input energy (Σa²) and
`nem_imatrix_dump()` writes the legacy `.dat` that `llama-quantize --imatrix` reads
(`[len][name][ncall][nval][sums]`, value = `sums/ncall`). The q36 `qwen35moe` forward has the same shape, so
this is a port, not a design — and it removes the need for a pod to get an imatrix.

Two traps worth carrying over:
- For a 3-D expert tensor `llama-quantize` wants **`n_in × n_expert`** values (one per (channel, expert)), not
  `n_in`; our per-channel vector is tiled `n_expert` times.
- It requires an entry for **every** low-bit tensor, **including the MTP block** — the engine never executes
  blk.88, so its entries have to be copied from the last trunk MoE block or the quantize aborts.
- Repeated at the top level because it cost us hours: **never requantize a requant** if a clean source exists.
  Ours was BF16 → UD-IQ2_XXS (3.07 bpw) → ours (2.05); every expert had already lost 2-3 bits first.

## 4. The writeup (post-able)

The lane produced a genuinely useful, *non-inflated* result: a reproducible harness plus a direct measurement of
what a 2-bit tier does to a 120B/12B-active MoE. Suggested framing — a **quantization-sensitivity teardown**,
not a Scion success story:

1. The tier ladder and the per-layer census we measured (experts 3.07 bpw, dynamic: IQ2_XXS on blk 1/3/5/8,
   IQ2_S on 10-17, IQ3_XXS 3.06 on 28 layers, IQ4_XS on 72; downs Q2_0 2.25 on 32 layers, IQ4_NL 4.50 on 8;
   the MTP block at Q8_0).
2. Our three re-quants and their measured drift, with the methodology above.
3. The requant-of-requant warning.
4. The vendor's own guidance correlated with observation: Unsloth's *"1-bit should not be used for agentic
   use-cases"* (sharp drop below UD-Q2_K_XL: ~25% → <8-10% on 32-token divergence; "tool calling… break down";
   "only general knowledge is retained"), against an agentic task (driving OpenCode to manage a 3D printer) that
   failed immediately.
5. Serving conditions: the engine CLI has **no chat template and no sampling at all** (only `--greedy`), while
   the vendor prescribes thinking mode + `presence_penalty=1.5` for these tiers — i.e. the failure was observed
   under the documented worst case.

**State the limits or it isn't fair:** we have no BF16 anchor (249 GB), so all numbers are relative to the
UD-IQ2_XXS itself; NVIDIA's BF16 table shows part of the agentic gap predates quantization; and 33 steps is one
sample.
