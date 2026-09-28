# Quant-methods plan — CAT-Q / AYOT / SignRoundV2 into the MoE line

**Status:** CPU side landed 2026-09-28 (tests green); GPU steps queued behind
the AMD/FreeToken window (single-card policy).
**Related:** [TAIL-EXPERIMENT-PLAN.md](TAIL-EXPERIMENT-PLAN.md) (the KLD-tail
path), [MOE-EXTENSION.md](MOE-EXTENSION.md), `../moe/README.md`,
`../moe/catq.py`, `../moe/kd_loss.py`, `../moe/ayot.py`, `autogrid_ext.steer`.

## Sources

| source | what it gives us |
|---|---|
| **CAT-Q** (arXiv 2606.26650, ICML 2026 oral; [BitTern](https://github.com/IntelChina-AI/BitTern), Apache-2.0) | ternary PTQ that learns per-group scale/mean/threshold and softens ternarization, coupled to sliding-window output reconstruction |
| **ScaleQ-1.58 / AYOT** (arXiv 2608.01078) | calibration must include the model's own reasoning traces, else reasoning tasks collapse under ternary PTQ |
| **SignRoundV2** (arXiv 2512.04746v2; intel/auto-round) | DeltaLoss (gradient × perturbation) sensitivity; DP bit allocation; loss filtering |
| **TernaryQuench** ([penk](https://github.com/penk/ternary-quench), Apache-2.0) | independent CAT-Q-style trainer + AYOT-style 10% agentic mixing; Qwen3.8-27B reference release |

## Constraint carried over

Calibration-free stays the definition of this method: no imatrix, no external
corpus, teacher-logits KD only.  AYOT traces are supervision *inputs* (the
teacher's own generations), not quantization statistics; nothing here touches
the frozen v1 recipe.

## CPU side landed this pass (tests green)

| file | what | default |
|---|---|---|
| `moe/kd_loss.py` | per-token KL + top-fraction loss filtering (SignRoundV2 §3.4) | off |
| `moe/catq.py` | LM + ST port, per-group factors, hard-ternarize export, `ternary_catq` for the bank quantizer | `--quant` unchanged (lloyd) |
| `moe/ayot.py` | trace/prompt JSONL loaders, trace→window packing, agentic-fraction mixing | off |
| `moe/ayot_gen.py` | teacher trace generator (GPU stage; `--dry-run` validates prompts) | n/a |
| `moe/qwen35_moe_proxy.py` | flags `--quant catq`, `--catq-*`, `--kd-filter-frac`, `--corpus-file`, `--agentic-frac`; cache and train share `_corpus_windows` | frozen v1 |
| `autogrid_ext/steer.py` | DeltaLoss for a linear weight given calibration inputs | library only |
| tests | `moe/tests/test_quant_methods.py` (10), `autogrid/tests/test_steer.py` (3) | green on CPU |

Measured observations (keep these honest):

- Weight-space CAT-Q beats absmean on mean-shifted/outlier groups, 6/6 seeds
  (seed 0: 0.0578 vs 0.0705 MSE), but **did not beat the Lloyd scale rule** in
  the integration check (0.0528 vs 0.0500).  CAT-Q's published gains come from
  *sliding-window output* reconstruction, so `--quant catq` (weight-space) is a
  cheap probe, not the decisive test; W2b is the decisive one.
- Loss filtering, AYOT mixing and DeltaLoss are implemented but unmeasured on
  the model (GPU).

## GPU workstreams (queued)

### W1 — AYOT supervision distribution (highest value; attacks the KLD tail)

1. Prompts: a question set for trace generation (decision needed: fineweb
   slice vs a reasoning/coding set; TernaryQuench uses agentic traces with
   tool calls).  Trace generation needs the full BF16 teacher; ~70 GB does not
   fit locally, so it rides a rental window (minutes on the pod; bundle with a
   v2 run).
2. Cache arms (existing stage, same seed/windows):
   - A: `--top-logits 50` (v1 reference)
   - B: `--top-logits 512 --corpus-file traces.jsonl --agentic-frac 0.1`
   - C: `--top-logits 512` (isolates top-k expansion from AYOT)
3. Train each arm (prefix-scoped recipe unchanged), eval PPL + router
   agreement + **full-vocab KLD** on the prefix eval windows.
4. Gate: KLD 99.9% and max must improve materially (tail plan's Step 1 gate).

### W2 — CAT-Q body A/B

- **W2a (cheap):** `--quant catq` vs `--quant lloyd`, same corrections;
  PPL/agreement/KLD.  Weight-space only, expectation modest.
- **W2b (the paper's claim):** sliding-window *output* reconstruction:
  optimise codes/factors against the FP window outputs (needs a calibration
  stage over cached hidden states; TernaryQuench's `train.py` is the reference
  implementation).  Build only if W2a or external signals justify it.
- Cross-check: CAT-Q's released Qwen3-30B-A3B checkpoint is a same-family MoE
  ternary baseline for routing agreement and KLD.

### W3 — DeltaLoss steering (cheap diagnostic + AUTOGRID follow-up)

- Rank STEER tensors on a prefix with `deltaloss_linear` and the deployed
  container quantizer; compare with the routing-drift placement findings.
- Next CPU step: a thin `steer_probe.py` runner (hook first-layer inputs,
  iterate modules).  The metric itself is CPU; the model wants a card.

### W4 — loss filtering (free)

- `--kd-filter-frac 0.001` on the winning arm; one variable, one decision.

## Order and gates

1. W1 (A vs B vs C) → does the tail move?
2. W4 folded into the winning W1 arm (free).
3. W2a on the winning setup; W2b only if warranted.
4. W3 anytime (diagnostic).

Trace generation gates **arm B only**: arms A/C, W2, W3 and W4 run without
traces, and generation can be bundled into any pod session (it is minutes on
the teacher), so it does not have to be the first thing through the gate.

## Non-goals

- No imatrix / corpus calibration of the container.
- No changes to the frozen v1 recipe defaults; every new flag is off by default.
- No routing treatment is inherited from these papers: none of them report
  routing metrics, and the E1 placement rule stays the reference.
