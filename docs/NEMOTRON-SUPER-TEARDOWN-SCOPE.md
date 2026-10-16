# Teardown scope: "what a 2-bit tier costs a 120B/12B-active MoE"

The reframe: not "we scionified Super", but a **reproducible quantization-sensitivity study** — one artifact, one
geometry, one harness, honest numbers. Everything below exists already except the three items marked
[GAP].

## The one-sentence version
We built a reproducible harness for measuring what a low-bit tier does to a 120B/12B-active MoE, ran it over
Nemotron 3.5 Super's published 2-bit tier and three of our own re-quants, and found that (a) the tier ladder's
floor is *documented by its own quantizer* as non-agentic, (b) we went 10 GB below that floor and reproduced the
documented failure mode, and (c) the measurement methodology the community uses for this — a single-token KL —
silently reverses the answer.

## Claim / evidence / confidence

| # | claim | evidence | confidence |
|---|---|---|---|
| 1 | The published 2-bit tier is a dynamic mix, not a uniform 2-bit: experts **3.069 bpw** (ups 3.078 over 57.8B params, downs 3.061) | our census of the shipped file, per layer | **solid** — arithmetic on the artifact |
| 2 | The mix is deliberate and per-layer: ups IQ2_XXS 2.06 (blk 1/3/5/8), IQ2_S 2.56 (10–17), IQ3_XXS 3.06 (28 layers), IQ4_XS 4.25 (blk 72); downs Q2_0 2.25 (32 layers), IQ4_NL 4.50 (8 layers); MTP block Q8_0 | same census | **solid** |
| 3 | The tier ladder bottoms out at **46.1 GB** (UD-IQ1_M); ours is **35.7 GB** | model card's own size table | **solid** |
| 4 | Unsloth documents that below UD-Q2_K_XL the model stops being usable for agentic work — "sharp drop off… ~25% to under 8-10%", "tool calling… break down", "only general knowledge is retained" | their Dynamic 3.0 doc, quoted | **solid** (their claim, not ours) |
| 5 | On our reference scale (Gemma 3 27B) they put IQ1_M at KLD 0.800 and IQ2_XXS at 0.521 vs Q4_K_XL 0.024 | same doc's table | **solid as a citation**, **NOT transferable to Super** |
| 6 | Our re-quants: 39.3 GB blind Q2_0 → 35.71 GB IQ1_M/Q2_0 (imatrix) → 39.32 GB uniform Q2_0 | file sizes + `llama-quantize` logs | **solid** |
| 7 | Relative drift vs the shipped tier, 33 teacher-forced steps: imx-1024 **p50 0.0569**, p90 0.80, 3/33 steps > 1 nat, top-1 78.8%; imx-128 p50 0.0935, p90 0.99, 4/33, top-1 84.8% | our harness | **solid but thin** — one prompt, 33 steps |
| 8 | A single-token KL is not a usable metric: it read 0.1018 vs 0.1808 *in favour of the wrong imatrix*; over 33 steps it reverses | both readings, same artifacts | **solid** — and the most transferable finding |
| 9 | A model larger than RAM reads **0.69–0.91 GB/token** from disk (32 tok → 22.1 GiB; 64 tok → 58.2 GiB) | `/proc/diskstats` deltas | **solid** |
| 10 | The host-residency policy's hit rate rose 58.6% → 82.4% from 64 → 256 experts/block and bought **no wall time**, because 256 asks for ~29 GB of hot pages inside a 30 GB box | our runs | **solid** |
| 11 | Chunked imatrix calibration is 5–9× cheaper: 1024 tokens in ~40 min vs ~6 h single-sequence | timed runs | **solid** |
| 12 | The shipped model is not a Flash-Next replacement even at BF16 (SWE-bench Verified 71.1 vs 78.7, Terminal-Bench 2.1 66.7 vs 85.8, BrowseComp 78 vs 89, GDPval 1211 vs 1561) | **NVIDIA's own** benchmark table | **solid as a citation** |
| 13 | Our engine serves with no chat template and no sampling (only `--greedy`), i.e. the vendor-documented worst case for these tiers | code fact | **solid** |

### Cannot support — must not appear as claims
- **"the tier's KLD vs BF16 is X."** We have no BF16 (249 GB). Unsloth's ladder is a *different model*; using it for Super would be fabrication. Cost to close: **249 GB download (~1.5 h at the measured 47 MB/s) + ~1 h** [GAP].
- **"quantization destroyed the model."** Three confounds, and we can't separate them: our re-quant (the most aggressive artifact in the story is *ours*, not theirs), the serving conditions (no template/thinking/penalties), and the base model (claim 12). The honest sentence is: *we reproduced the vendor-documented failure mode of these tiers, and we cannot attribute it to the bits alone.*
- **Anything about model capability.** We ran no benchmark suite; one agentic task failing immediately is an anecdote, not a measurement.
- **The bpw→quality curve for Super.** We have three sizes, two of which are identical (35.71 GB twice), and the third (39.32 GB) is unmeasured.

## Outline
1. **Setup** — the artifact, the tier ladder, the geometry (89 blocks, 512 experts, top-22, latent 1024, FF 2688), the per-layer census (claims 1-3).
2. **Harness** — strata with exact-greedy-parity; chunked imatrix calibration; teacher-forced multi-step KL; disk-read accounting. Include the cost model (claim 11) so others can budget.
3. **The metric is broken** (claim 8) — put this third, not last; it's the reason to read the rest.
4. **Results** — the re-quant table (claim 7) and the disk-read numbers (claim 9), with p50/p90/steps>1nat and no means.
5. **Residency** — a throughput side-study (claim 10): a policy hit rate is not physical residency.
6. **Cross-check** — the vendor's own warnings and ladder (4, 5), and that their tier floor is 46.1 GB.
7. **What we saw** — the agentic task, stated as an observation with the serving caveat (13) and the BF16 context (12).
8. **Limits** — the three confounds, the sample size, the missing anchor.
9. **Reproduce** — exact commands and the committed tools.

## Publication risks (and my read)
- **The model is one day old** (HF release 10/15/2026; today is 10/16). A day-1 teardown risks being both unfair and obsolete — better quants and methods are exactly what's coming. **This is the strongest argument to wait.**
- **Our own artifact is the outlier, not theirs.** Their floor is 46.1 GB *with a warning*; we went to 35.7 GB. A piece that reads as "NVIDIA's model is destroyed by quantization" would be misleading, and would be correctly refuted.
- **The confound is the story.** The genuinely novel, non-inflammatory published claim is the *methodology* (claim 8) plus a reproducible harness. That's a contribution that can't be refuted and doesn't need the model to be bad.
- Posture: measure, cite the vendor, state the limits, and let the reader choose a tier.

## Format
Either (a) a repo doc — cheap, internal, feeds the Flash-Next/35B lanes immediately — or (b) a long-form post
(~2500 words, tables + the harness link). Given the risks above, I'd write it as (a) now and hold (b) until
either the model ages or we get the BF16 anchor.

## Closing the gaps, if we want the stronger piece
| gap | cost | what it unlocks |
|---|---|---|
| BF16 anchor (249 GB, streamed) | ~1.5 h download + ~1 h | absolute KLD for the shipped tier *and* for ours — turns "relative, day-1" into a real number |
| NVFP4 (75 GB) as a cheap near-BF16 proxy | ~27 min + ~1 h | the same, weaker, ~1/3 the disk |
| uniform-Q2_0 measurement (39.32 GB) | ~20 min | the middle point of the curve |
| a 256+ token KLD run | ~1 h | replaces the single 33-step sample |
