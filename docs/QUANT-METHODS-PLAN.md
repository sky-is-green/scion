# Quant-methods plan — CAT-Q / AYOT / SignRoundV2 into the MoE line

**Status:** CPU side landed 2026-09-28 (tests green); Phase 1 (prefix arms A/C,
gate, `lmonly` ablation) complete on `scion-test`. The W1 gate fails on the
correction objective — the loss rebalance and the residual-mass term are now
measured (`RESEARCH-HANDOFF.md` §3/§7.8), and the third chain-rule piece
(TAD's D_KL2, sampled tail tokens) is built and queued. DeepSeek-V4.1 pulls are
folded in below, including the pre-rental expected-improvement gate.
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
| `moe/build_ayot_prompts.py` | builds the 512-row prompt set (128 math / 128 coding / 256 fineweb-edu, 50/50) | n/a |
| `moe/kld_eval.py` | **W1 gate instrument**: full-vocab KLD (mean/p99/p99.9/max) + top-1 agreement, FP prefix teacher vs corrected student | n/a |
| `moe/steer_probe.py` | **W3 runner**: hooks every `nn.Linear` input, scores it with `autogrid_ext.steer.deltaloss_linear` under the deployed quantizer, writes a ranked JSON | n/a |
| `moe/qwen35_moe_proxy.py` | flags `--quant catq`, `--catq-*`, `--kd-filter-frac`, `--corpus-file`, `--agentic-frac`, `--top-logits`; `build_student` shared by train/eval/`kld_eval` | frozen v1 |
| `autogrid_ext/steer.py` | DeltaLoss for a linear weight given calibration inputs | library only |
| tests | `moe/tests/` (36), `autogrid/tests/test_steer.py` (3) | green on CPU |

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

### Phase 1 result (2026-09-28): the gate FAILS, and top-k is not the reason

4-layer prefix, 4096 windows, wikitext seed 999, 4088 eval tokens:

| run | mean KLD | p99 | max | s-entropy | s-peak | sharper? |
|---|---|---|---|---|---|---|
| body (no branches) | 0.1452 | 0.6788 | 2.4232 | 10.83 | 0.0213 | no |
| armA (top-50) | 1.6391 | 5.5072 | 6.8429 | 7.42 | 0.1475 | yes |
| armC (top-512) | 1.3272 | 4.7196 | 5.8458 | 7.87 | 0.1475 | yes |
| lmonly (kd 0) | 4.5440 | 11.0062 | 15.3861 | 5.70 | 0.2024 | yes |

teacher: entropy 10.77 nats, top-1 mass 0.0230.

Top-k expansion does help — armC beats armA by 19% on mean, 14% on p99 and max.
But both trained arms are ~10x worse than the *uncorrected body*, so the gate
fails, and the diagnosis is that **the correction branches break the
distribution, not the top-k budget**. They are trained on LM + top-k KD only,
so they sharpen without constraint: entropy 10.77 → 7.42 nats, top-1 mass 6x
higher. PPL improves (83989 → 512) while full-vocab KLD worsens, which is the
canary failure mode from the tail plan, now quantified.

**The `lmonly` ablation identifies which term does it.** `--kd-weight 0` gives
mean KLD 4.5440 (+242% vs armC), p99 +133%, max +163%, and sharpens *harder*
(entropy 5.70, peak 0.202). So the **LM term is the primary cause**: dropping KD
makes everything worse, meaning KD was already the only thing resisting the
collapse. A tail constraint bolted onto the KD term is the wrong lever.

**And the KD term is blind for a measurable reason.** `kld_eval.py
--measure-topk 512`, from the full-vocab log-softmax, gives mean captured mass
**0.1704** (p01 0.0455, min 0.0366) — the cache holds 17% of teacher mass, and
the 512-support distribution the KD term optimises renormalises over the other
83% as if it did not exist. Consistent with 50 → 512 buying only 19%: coverage
rises slowly, not across a regime boundary. (Measured on wikitext eval windows
while the caches are fineweb-edu, so order-of-magnitude.)

Consequences for the order below:

1. **Rebalance the loss** — raise `--kd-weight`. One flag, and the ablation says
   the gradient exists. A trade, not a fix: the PPL gains are the point of the
   method, so read PPL and KLD together. v1 defaults stay frozen.
2. **Residual-mass term** (tail plan 2a) — the better second build now that 83%
   of the mass is out there to match, and since the failure is not specifically
   near-tie flips.
3. Not now: W4 (filters the largest KD losses, i.e. the term that is not the
   problem), W2a (body quantizer, cannot close a loss-function gap), arm B.

`kld-*.json` and `kld-cov512.json` in `$MOE_ARTIFACTS/qwen35/`. `kld_eval.py`
reports `sharper_than_teacher` and `topk_coverage` so this failure mode cannot be
mistaken for a quantizer regression. p99.9 was unresolved in all runs (rank 5 of
4088) — the comparison was argued on `max` and the mean. `top1_agreement` is
teacher/student *logit* argmax, **not** router agreement; the routing number
remains the in-run `router_agree`.

**Read the gate number with its sample size.**  `kld_eval.py` reports
`p999_rank`, the number of tokens at or above the p99.9 position, because that
percentile sits n/1000-th from the top by construction: at the default 8 eval
windows (4088 tokens) it is the 5th-worst token, so p99.9 and max are nearly the
same measurement and `p999_resolved` is false.  Lean on `max` and
`worst_tokens` (which record window and position, so a spike is inspectable
rather than merely reported) until the instrument runs at >= 1e5 tokens.
Resolving p99.9 properly needs more eval windows than host memory allows at a
248k vocab — the parked teacher log-probs are 4 GB for 8 windows — so a real
p99.9 wants teacher and student co-resident and streaming.  Follow-up, not this
run.

Prefix PPL is also not comparable to the 35B numbers: a 4-layer prefix with a
real `lm_head` on truncated hidden states is not a language model, so its PPL
(~10^2-10^3) is only meaningful arm-to-arm at fixed depth.

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
- `moe/steer_probe.py` is the runner (hook each linear's input, iterate modules,
  write ranked JSON).  The metric is CPU; the model wants a card.  Scope: the
  `nn.Linear` set the recipe leaves in FP — attention q/k/v/o, the GDN
  projections, `mlp.shared_expert`, and `mlp.shared_expert_gate`.  The fused
  expert banks and the router `mlp.gate` are the ternary *target*, not STEER
  candidates, and are excluded.
- **First run (4-layer prefix, 2 windows, lloyd g128, 35 linears):** by mean
  normalised DeltaLoss, `gdn` 0.365 > `attn` 0.294 >> `shared_expert` 0.043.
  Top tensors are `mlp.shared_expert_gate` (layers 0-1), then
  `linear_attn.in_proj_b`/`in_proj_a`, then `self_attn.q_proj`.  Read as a
  diagnostic only: 4 layers and 2 calibration windows is a very small sample,
  DeltaLoss scales with tensor size, and nothing here is a routing result.  The
  ordering does put the shared-expert gate and the GDN in-projections at the top,
  which is worth a look against the E1 placement rule.

### W4 — loss filtering (free)

- `--kd-filter-frac 0.001` on the winning arm; one variable, one decision.

## Order and gates

Revised twice: after the Phase 1 result, then again after the `lmonly` ablation
and the `--measure-topk` coverage result. The binding constraint is the **balance
between the two loss terms**, not the size of k and not a missing tail term.

1. **Rebalance: `--kd-weight` sweep** (`phase1-w1.sh train-kdw <w>`). The
   ablation says the LM term causes the collapse and KD is the only thing
   resisting it, so push the other way. Read PPL and KLD together — the PPL
   gains are the point of the method and this is a trade, not a fix. v1 defaults
   stay at 1.0; nothing ships off this without a case for it.
2. **Residual-mass term** (tail plan 2a) if rebalancing is not enough. Now the
   preferred tail term over rank/margin: the top-512 cache holds 17% of teacher
   mass, so 83% is available to match, and the failure is not specifically
   near-tie flips.
   **BUILT (2026-09-28, session 2), unmeasured, and kdw 2.0 is the argument for
   it:** kdw 2.0 moved the mean −25.3% and left p99/max flat (−1%), which is what
   a term that can only see 17% of the mass predicts. Built as the marginal KL of
   the two-way support/complement split — the piece of the gate's own metric that
   the top-k renorm discards — so it is exact and per-token, not the sampled
   estimator the tail plan first sketched. `--kd-tail-weight` (default 0.0),
   `phase1-w1.sh train-tail <w>` with `--kd-weight` pinned at 1.0. Needs one
   `phase1-w1.sh cache c` re-run first: the existing caches have no `w` field
   and the term refuses to substitute a constant.
   **MEASURED (2026-09-28/29).** The two levers compose on different axes:
   kdw 2.0 mean −25.3% with p99/max flat; tailw 1.0 **p99 −26.0% / max −19.6%**
   for +6.6% PPL; combo2t2 (kd 2.0 + tail 2.0) mean −38.6%, p99 −39.7%,
   max −30.8% at +57% PPL — the current recipe candidate. The gate's chain-rule
   decomposition then showed the remaining 81% of the absolute headroom is the
   *tail-conditional* piece, so the third build (TAD's D_KL2, estimated from
   sampled tail tokens — Sparse Logit Sampling) is **built and queued**:
   `--kd-tailcond-weight` (default 0.0), `phase1-w1.sh cache-tail` +
   `train-tailcond <w>`. Numbers: `RESEARCH-HANDOFF.md` §3/§7.8.

3. W1 re-run on the winning configuration: A vs C, and B once traces exist.
   Top-512 alone bought 19% and is worth keeping, but it is not the fix.
4. W4 (`--kd-filter-frac 0.001`) as a separate variable on the winning arm. It
   filters the *largest* per-token KD losses, so it acts on the opposite end of
   the problem from the tail — not a substitute for steps 1-2.
5. W2a on the winning setup; W2b only if warranted. CAT-Q changes the body
   quantizer, so it cannot fix a loss-function gap; do not expect it to.
6. W3 anytime (diagnostic).

Trace generation gates **arm B only**: arms A/C, W2, W3 and W4 run without
traces, and generation can be bundled into any pod session (it is minutes on
the teacher), so it does not have to be the first thing through the gate.

## Folded in: DeepSeek-V4.1-Flash pulls (2026-09-28)

The V4.1 technical report is a pretraining/serving report (no ternary or PTQ
content), but four mechanisms transfer. The full item-by-item mapping lives in
local design notes, not in this repo. What enters *this* plan, in the current
order:

### P0 — routing: bias-based load balancing (every router run)

- Mechanism: select by `logits + bias_e`, weight by the raw softmax over the
  selected experts; after each step `bias_e -= delta * sign(load_e - mean)`
  (delta ~1e-3; **sign corrected 2026-09-28 against arXiv 2408.15664 — overloaded
  experts move *down*, the earlier draft had `+=`, which would diverge**); keep
  a tiny sequence-level balance loss (~1e-4) as a safety net. V4.1 additionally
  keeps separate bias banks per modality.
- Why here: the MoTE proxy measured the Switch aux loss actively pushing the
  router toward redundant experts. Bias-based balancing removes that gradient
  from the correction objective and costs nothing at inference (per-expert
  scalar).
- Executable step: `--balance bias` in the router path (OLMoE first, then the
  qwen35 prefix), CPU unit tests for the update rule; A/B against the current
  balancing on the prefix, reading load entropy, `router_agree`, PPL and KLD.
  Orthogonal to the loss rebalance, so it runs as its own arm.

### P0 — supervision shape: full-vocabulary, on-policy

- V4.1's final post-training stage is full-vocabulary on-policy distillation
  (40+ heterogeneous teachers). Our KD optimises a renormalised 512-support
  distribution carrying 17% of teacher mass; the measured ~10x KLD regression
  is what that blind spot predicts.
- This is supporting evidence for the current Step 2 choices, not a new build:
  (1) the `--kd-weight` rebalance, already the top experiment, and (2) the
  sampled residual-mass term (tail plan 2a) — now the better second build.
- On-policy is the third axis (AYOT arm B; traces already built). Keep it
  deferred until the loss balance settles, per the current reading.

### P1 — serving track (separate from the correction recipe)

- **FP4 KV cache** (E2M1 + one E4M3 scale per 16 channels, no global scale;
  quantise after RoPE; dequantise before attention; FP8 for SWA KV): update
  `moe_tier`'s KV byte model and measure KV quantisation on Scion in the fork
  (`q8_0`/`q4_0` first).
- **DSpark drafter** (3-block SWA-128 drafter, frozen backbone,
  confidence-scheduled verification): measure off-the-shelf draft acceptance
  first; train a DSpark-shaped head only if the acceptance numbers justify it.
- **SWA bounded replay / CSA2 / CED / mHC / Engram** are trained-in
  architecture, not retrofittable to Qwen weights: design vocabulary for a
  future consumer MoE, notes only.

### Added 2026-09-28b — MoE-tricks sweep (K3 / GLM-5.3 / Hy4 / MiMo V2.6)

- **P0 routing, second candidate: Quantile Balancing (Kimi K3).** Same slot as the
  DS bias rule: auxiliary-loss-free, bias added only for Top-k *selection* and
  omitted from the mixture weights. Difference is the update: instead of a
  sign/step rule, each expert's bias is set from the `(1 − k/n)`-quantile of its
  per-token margins `s_i,j + b_j − α_i` (α_i = the token's Top-k cutoff), so each
  expert's expected load is `k/n`; the quantile is read from a histogram across
  ranks (few hundred bins, one all-reduce) rather than gathering margins.
  → run as a one-variable arm against the DS bias arm and the current balancing;
  read load entropy, `router_agree`, PPL and KLD. Add a router z-loss arm
  (OLMoE) as the cheap third point.
- **Tier 1 serving (planner math, not training arms): cross-layer index/KV reuse
  is now consensus** — GLM IndexShare (one indexer / 4 sparse layers, 2.9× FLOPs),
  Tencent Hy4 IndexCache, DS CSA2. Add the reuse pattern to the planner's KV
  model; watch MiMo V3's **HySparse2** (oracle full-attention layers select
  tokens; sparse layers share their KV; 11:1 validated at 80B, 5× prefill /
  4.5× KV at 1M) — unshipped, do not build on it yet.
- **Tier 2 (after the objective is fixed): correction-branch read/write gates.**
  Five current frontier models redesign the residual stream itself (Qwen Gated
  Residual, DS and GLM mHC, Hy4 iHC with 4 streams, K3 Attention Residuals). Our
  correction branches are a primitive of this surface; the foldable experiment is
  an element-wise read gate + scalar write gate on each branch, tested on the
  prefix against the plain additive branch.
- **Tier 2: AAR-style features into router training** — attention-derived
  statistics (sink/entropy over the last window) as extra router inputs, base
  frozen, routing only. Two-line hook in the prefix harness; queue behind the
  objective fix.
- **Future build only:** LatentMoE (K3: experts in a 0.5×-hidden space, 896
  experts top-16) and the no-shared-expert variant (MiMo V2.6 ships 384×8 with
  none; K3 ships 2) — inputs to any upcycle/architecture decision, not
  retrofittable to Scion.
- **Not foldable:** KDA/Gated MLA/QSA/MSA/SiTU-GLU/sparse attention (trained-in);
  Muon optimisers (frozen-body training does not need them); MiMo's 7k open RL
  environments (agent-eval work for hivebench, not the compression recipe).
- **Watchlist:** Qwen4 (in training), MiMo V3 (HySparse2), GLM-5.3-Pro,
  MiniMax M3.5/M4 weights, Meta Avocado (2027, closed).

### Parked — ToMoE expert masks (not a V4.1 pull)

`moe/olmoe_masks.py` + its test are untracked and stay parked: masks are a
*structure* change tested under an objective we now know is broken. Run them
only after the loss shape is fixed, and with bias-based routing as the balancer.

### Rental gate — expect the improvement before booking a pod

No rented GPU run until a short expected-improvement memo exists, written from
local evidence:

- measured local effect sizes for each accepted recipe change (prefix KLD/PPL
  deltas, per arm, with the coverage caveat);
- the projected full-model effect, with assumptions and error bars stated;
- a go/no-go threshold written down *before* the quote (e.g. "mean KLD <= body +
  x, max <= y, PPL not worse than z");
- cost per expected point of KLD, and what would falsify the projection.

The pod only runs the memo; AYOT trace generation can be bundled once the memo
says go.

## Non-goals

- No imatrix / corpus calibration of the container.
- No changes to the frozen v1 recipe defaults; every new flag is off by default.
- No routing treatment is inherited from these papers: none of them report
  routing metrics, and the E1 placement rule stays the reference.
