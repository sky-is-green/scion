# Coder variant: prior art review and revised approach (2026-10-10)

**Goal.** Build "our own coder variant" of Scion-FlashNext-176B-A6B: a
coding-specialised release that keeps (or nearly keeps) code ability while
shedding the expert capacity that only serves other domains, ideally smaller
than ISTA's Coder GGUF (58.4 GB). The generic P5 shrink (25% prune, general
gates) is the base case; this note records how others have done it and what
we should change **before** spending pod time.

## The direct comparisons

| Work | Method | Selection | Calibration | Result |
| :--- | :--- | :--- | :--- | :--- |
| **ISTA GSQ-RCO Coder** (58.4 GB GGUF, the file to beat) | 50% experts removed (uniform, 256/layer), remaining weights 3.5 bpw | **RCO**: KL between pruned and unpruned, joint across layers, one exact budget per layer | code + agentic tool use + **vision** (a code-only search destroyed vision) | SWE-bench Verified 91.3% retained, LiveCodeBench v6 98.7% retained (xhigh reasoning) |
| **ISTA Qwen3-Coder-Next-RCO-pruned** (bf16, 8 variants) | same RCO, uniform vs nonuniform allocation, coding vs general calibration | RCO | evol-codealpaca (coding) or FineWeb-Edu (general) | coding-50-**nonuniform**: HE 97% retained; coding-50-**uniform**: HE 55% (collapse). At 25%: 100% vs 92%. General-calibrated loses coding entirely (6-8% HE) |
| **Jha 2026, "Half the Experts, All the Code"** (2607.16721) | uniform keep, 5 criteria, 4 ratios; functional gates | REAP, MAN, routing mass, domain-contrastive, random | 2,047 chat samples, 70/30 code/general, 3.15M tokens, on-policy variants | half removable with no detectable HE+/MBPP+ loss on Qwen3.6-35B-A3B; criterion winner flips on Gemma-4; contrastive scoring collapses; perplexity can rank a broken model above intact; smoke tests fail both ways; pruning beats quantisation only below ~3 bpw |
| **REAP** (ICLR 2026) | one-shot pruning, beats expert merging on generative tasks | `S_j = (1/N_j) Σ g_j(x)·‖f_j(x)‖₂` (router gate x expert output L2 norm) | task corpus | near-lossless at 50% on Qwen3-Coder-480B and Kimi-K2 for code; public code (CerebrasResearch/reap) |
| EASY-EP (NeurIPS 2025), EEP, C-PRUNE | output-aware importance, token contribution, clustering | various | few-shot in-domain demos / general | same family of cheap-saliency methods; EASY-EP retains half the experts at comparable in-domain quality |

## Findings that change our plan

1. **Counts/gate-mass alone are not the state of the art.** The magnitude
   family (REAP: gate x activation norm) is the load-bearing signal; the
   2607 paper shows routing mass collapsing at 25% keep on one family while
   REAP survives, and the reverse on another. Our current instrumentation
   collects counts, gate mass and full-softmax mass but **no activation
   norms**. We should extend the stats patch to accumulate, in the same
   pass: REAP sum `Σ g·‖f‖`, norm moments, and keep the current fields.
2. **Criteria do not transfer across model families; per-model validation is
   required.** The operational response is not one criterion but a cheap
   shootout: collect all signals in one pass, build 2-3 candidate plans, and
   evaluate them functionally.
3. **Allocation matters more than criterion at high sparsity, and the GGUF
   single-expert-count is a real constraint.** ISTA's own Coder GGUF used
   uniform allocation and accepted it because "the GGUF format stores a
   single expert count"; their bf16 variants show nonuniform recovers 42
   HE points at 50% on this model family. If we want 50% uniform to hold for
   code, we cannot assume it will (possible collapse). Nonuniform requires
   per-layer expert counts in the fork, the GGUF metadata and Strata.
4. **Code calibration buys code and costs everything else, by design.** Both
   ISTA and 2607 confirm the trade is stark and intended; the artifact card
   must state it, and our gates must report both sides (code retained;
   general/reasoning lost).
5. **For calibration: on-policy and agentic help.** 2607 found profiling on
   the base model's own greedy outputs lifts the 25% point (+3 HE) and a
   general-traffic guard adds a bit more; ISTA included agentic tool-use
   traces. Our current corpus is fineweb-heavy with short AYOT generations.
   A coder variant should profile a **code-dominant on-policy corpus**
   (Evol-CodeAlpaca / Magicoder / tool-calling / the model's own reasoning
   traces) with a separate general reference slice for diagnostics and a
   guard.
6. **Perplexity is not a gate.** Use functional code evaluation (HumanEval+
   / MBPP+ style, greedy) as the primary gate; KLD/PPL only as diagnostics;
   track degeneration rate; smoke tests must not gate.
7. **Pruning vs quantisation crossover:** pruning wins only where
   quantisation would drop below ~3 bpw. Our experts are at 1.75 bpw, so
   pruning is the right axis here; further quantisation is not.
8. **Healing works.** A lightweight fine-tune recovers roughly half of what
   aggressive pruning costs (2607); our correction branches are exactly such
   a lightweight adapter. Budget a correction re-fit for the coder variant,
   especially at 50%.
9. **ISTA keeps the PLE n-gram table out of both pruning and the quant
   search** (fixed IQ4_NL, 4.5 bpw). Precedent for keeping our Q4_0 table
   unchanged in this variant; the Q4_0 -> Q3_0 lever stays a separate,
   later item.

## What it could look like

Base numbers: release 60.17 GB (2.72 bpw); PLE table 28.8 GB, ternary
experts ~26.4 GB, body ~4.1 GB, corrections ~0.68 GB. A 50% expert prune
with the expert container unchanged lands near **47 GB** (experts ~13.2 GB),
i.e. ~11 GB below ISTA's coder with the same table; 25% lands near 53.5 GB.
ISTA's retained experts are 3.5 bpw, ours 1.75 bpw ternary + corrections, so
the quality comparison is not bit-for-bit, but the size story is strong.

## Proposed path

**Phase 1 (near-term, reuses the built pipeline).**
1. Extend the router-stats patch with the REAP signal (gate x down-output
   norm) and norm moments; re-verify zero overhead when off.
2. Build a code-dominant on-policy calibration corpus (plus a general
   reference slice) and re-run the (now two-pass) map collection on the
   A6000; publish nothing yet.
3. Build candidate plans: REAP-uniform-25, routing-mass-uniform-25, and a
   guarded variant; prune + export; gate functionally (HumanEval-164 first,
   then MBPP+ if we add it, KLD/PPL diagnostics, general as the trade
   measurement).
4. Only if 25% holds, test 50% uniform. If 50% uniform collapses for code
   (ISTA's uniform-50 did), that motivates Phase 2 before shipping a 50%
   file.

**Phase 2 (if needed for a competitive 50%).**
- Per-layer (nonuniform) expert allocation: budget per layer from the map /
  a small sensitivity probe; fork + GGUF + Strata support for per-layer
  counts; exporter writes per-layer dims. This is the path ISTA's own
  results say carries the 50% point for this model family.

**Phase 3 (optional, highest fidelity).**
- RCO-style selection (IST-DASLab/RCO is public; Grokking the KL objective
  joint across layers with exact per-layer budgets). Could run against the
  ternary compact-bank student in `qwen4exp_proxy` so the search optimises
  the shipped quantiser, not the bf16 model. Larger effort; only if the
  greedy plans leave quality on the table.

**Decisions wanted before proceeding:** (a) do we extend the patch to REAP
signals and swap the corpus to code-dominant before the first map window;
(b) is a 50% coder target worth the per-layer-count engine work up front,
or do we run 25% uniform first; (c) is vision or tool-calling in scope for
the calibration mixture.

## Decisions taken (2026-10-10)

- **Upgrade now**: patch 4 now also collects the REAP signal (gate x expert
  output norm) plus norm first/second moments; the corpus is rebuilt
  code-dominant (Evol-CodeAlpaca teacher-forced + on-policy generations,
  glaive tool-calling, fineweb/GSM8K ballast). Zero-overhead-when-off
  re-verified after the change.
- **25% uniform first**: the stock-compatible path; per-layer (nonuniform)
  counts stay behind a decision that needs the first measurement.
- **Vision requested but blocked by the artifact**: the released GGUF has no
  vision tower (1416 tensors, all `blk.*`/embeddings; only the metadata
  `qwen4exp.ple.image_token_id` remains), so no calibration slice can
  exercise a vision pathway in the file we would prune. Options: declare the
  variant text-only (inherits the release's scope), or rebuild from the FP8
  checkpoint with the vision tower included, which is a separate, much
  larger pipeline. Not silently dropped; recorded here.


## References

- ISTA-DASLab, *Qwen3.8-Flash-Next GSQ-RCO Coder* (HF model card; RCO pruning
  selection, SWE-bench/LiveCodeBench numbers).
- ISTA-DASLab, *Qwen3-Coder-Next-RCO-pruned* (HF model card; uniform vs
  nonuniform, coding vs general calibration).
- Helcig & Alistarh, *Model Compression with Exact Budget Constraints via
  Riemannian Manifolds* (arXiv:2605.00649; code github.com/IST-DASLab/RCO).
- Lasby et al., *REAP the Experts: Why Pruning Prevails for One-Shot MoE
  Compression* (arXiv:2510.13999, ICLR 2026; code CerebrasResearch/reap).
- Jha, *Half the Experts, All the Code: One-Shot Domain Pruning of
  Mixture-of-Experts LLMs for Coding* (arXiv:2607.16721).
- Dong et al., *Domain-Specific Pruning of Large Mixture-of-Experts Models
  with Few-shot Demonstrations* (EASY-EP, arXiv:2504.06792).
- Liu et al., unified saliency formulation (frequency x gate x norm), cited
  by Jha as the "no single criterion wins" result.
