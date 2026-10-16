# HARVEST: what the Nemotron lane leaves for Flash-Next and the 35B

The Nemotron lane is closed — not because the engineering failed (the engine reached exact greedy parity and the
calibration/drift harnesses work), but because **the base isn't worth it**: Super's published 2-bit tier already
sits where Unsloth's own docs say agentic use breaks, NVIDIA's BF16 numbers show it trails Qwen3.8-Flash-Next on
the agentic/coding axis *before* any quantization, and the only source we can legally quantize from locally is
itself an already-quantized file. That is community-scale work we don't want to own.

Nothing here is Super-specific. Two things transfer, and one earlier recommendation turned out to be wrong —
recorded below so nobody repeats it.

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

`tools/qwen35/kl.py` — the harness the other lanes already use — was **already doing this right**: it
teacher-forces llama.cpp's greedy answer (`--force`) and reports mean, median, p99, top-1 and the noise-floor
ratio. So the lesson is not "add multi-step reporting", it is **"never hand-roll a single-token KL again"**,
which is precisely what this lane did before the lesson landed. The one figure it lacked — the count of steps
that flip by more than a nat, which is what the mean is actually dominated by — is now reported.

## 2. Host-side expert residency — **do not port: the core already has it**

**Correction (2026-10-16, after actually reading the core): do not port this. The core already has it, and
better.** `FileExpertSource` owns a `RouterLookahead` (CS-T) that applies layer *l+1*'s router to layer *l*'s MoE
input to *predict* the next layer's experts and warm their pages while layer `l` computes; `--expert-profile`
(`read_expert_profile`, produced by `tools/make_profile.py`) decides residency from a whole-corpus
routing-frequency ranking — with a comment in the header stating outright that a frequency profile beats the
compulsory-miss/LRU policy; and it measures *physical* residency with `mincore` (`resident_sampled` /
`resident_in`). Flash-Next's serve config uses exactly this (`--mmap-experts --expert-cache auto
--expert-profile ...`). Our `NemRes` per-layer recency+frequency cache is a crude rediscovery of one piece of it,
with **zero lookahead** — promoting it would have been a duplicated, inferior mechanism.

What our version does genuinely add, kept for the record: a **non-destructive** release (`MADV_COLD`) where the
core uses `MADV_DONTNEED` to free mapped layers (destructive on a file mapping — it re-faults from disk), and the
sizing rule below.

**Measured, and still worth knowing:** the policy asks for pages; the kernel decides. On Super, 64 experts/block
gave a 58.6% software hit rate and 256/block gave 82.4% (evictions 44,056 → 9,627), and **neither bought any
wall time at all** — at 256 the policy requested ~29 GB of hot pages inside a 30 GB box, and the kernel evicted
them. Keep any resident budget at roughly **40% of RAM or less**, and trust `mincore`, never a policy counter.

**The number that actually bounds throughput:** a model larger than RAM reads **0.69-0.91 GB/token** from disk
(measured on Super: 32 tokens → 22.1 GiB, 64 tokens → 58.2 GiB).

**Where it could still matter:** the q36 engine path (`src/qwen35`: `qwen35moe` + `nemotron_h_moe`) has none of
this — but its consumers are the 35B (11 GB, fits in RAM) and Nemotron (deleted), so nothing needs it today. If a
`qwen35moe` model larger than RAM ever lands, borrow the core's `RouterLookahead`, not this.

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
