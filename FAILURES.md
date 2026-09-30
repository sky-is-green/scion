# Scion (MoE track) — negative register

> Companion to the dense-forensics register
> ([bonsai2-ternary-forensics/docs/FAILURES.md](https://github.com/sky-is-green/bonsai2-ternary-forensics/blob/main/docs/FAILURES.md)).
> Same schema: **id · claim/hypothesis · method · outcome · verdict · evidence ·
> status · what it rules out · cost to revisit.** Created 2026-09-27, at the
> repository split; entries before the split are the ones the MoE track
> produced.

| id | one line | verdict | status |
|---|---|---|---|
| [D1](#d1--rotation-is-not-the-missing-lever-for-moe) | Rotation before RTN does not help MoE | Falsified | closed |
| [D2](#d2--in-place-qat-at-local-budgets-does-not-recover-routing-damage) | In-place ternary QAT alone recovers ~1.9×, routing unchanged | Negative control | closed |
| [D3](#d3--per-expert-correction-branches-fail-the-placement-test) | Corrections inside experts are 25× worse than on the residual stream | Falsified | closed |
| [D4](#d4--router-kd-is-a-verified-no-op) | Router-KD term matched a zero-weight control within ~1.5% | No-op — dropped | closed |
| [D5](#d5--post-hoc-ternarisation-of-the-correction-sidecar-is-not-viable) | Post-hoc ternarised branches are 14–25× worse than STE-trained | Falsified | closed |
| [D6](#d6--mote-style-up-cycling-does-not-transfer-to-qwen36-35b-a3b) | The frozen-shared-expert trick needs a shared expert worth having | Not applicable | closed |
| [D7](#d7--hot-expert-gpu-cache-no-throughput-gain) | Static hot/cold expert split is performance-neutral by construction | Rejected (as a perf feature) | closed |
| [D8](#d8--tail-conditioned-kd-does-not-transfer-to-the-full-body) | Tail-conditioned KD does not transfer to the full body | Falsified at full scale | closed |

---

## D1 — Rotation is not the missing lever for MoE

- **Claim/hypothesis:** rotating expert banks into Prism's basis before
  ternarisation recovers the dense-forensics behaviour on MoE.
- **Method:** `moe/olmoe_rotate_rtn.py`; rotated vs raw absmean RTN on OLMoE
  experts, 8-window protocol.
- **Outcome:** weight rel err **0.5156 (raw) vs 0.5109 (rotated)**, zero share
  0.309 both; in-place RTN PPL 11.02 → **40,121 (3,640×) raw**, **56,611
  (5,135×) rotated**. The dense Gate-2 result (Prism's exact basis + RTN →
  1,270×) predicted this: the basis buys the container, the trained placement
  buys the model.
- **Verdict:** Falsified for MoE.
- **Evidence:** `docs/MOE-EXTENSION.md` §2.2; `moe/olmoe_rotate_rtn.py`;
  `moe/results/olmoe/`.
- **Status:** closed.
- **What it rules out:** basis rotation as the MoE fix; quantizer-side searches.
- **Cost to revisit:** low, but the direction is settled.

## D2 — In-place QAT at local budgets does not recover routing damage

- **Claim/hypothesis:** full-model ternary QAT at local budgets can retrain the
  routing back (the F5 route, on MoE).
- **Method:** `moe/olmoe_proxy.py` — in-place ternary QAT on OLMoE, 6.4B
  trainable, 0.5M tokens; plus an fp32-master 4-layer subset control.
- **Outcome:** bf16 masters: **1.9× recovery, routing unchanged (0.463 →
  0.464)**; fp32 masters on the subset removed the precision question but
  training still did nothing (**23.94 → 23.79**). Bulk update RMS ~5e-6 vs
  bf16 ULP ~6e-5 — precision is not the binding constraint.
- **Verdict:** Negative control; in-place QAT alone is not a route.
- **Evidence:** `docs/MOE-EXTENSION.md` §2.3; `moe/olmoe_proxy.py`;
  `moe/results/olmoe/`.
- **Status:** closed.
- **What it rules out:** "just train the whole thing locally" as the cheap MoE
  path; the correction branches are doing the work, not the body updates.
- **Cost to revisit:** only with a full-precision-master QAT budget (the F5
  priced run).

## D3 — Per-expert correction branches fail the placement test

- **Claim/hypothesis:** corrections placed inside the experts (per-expert
  branches) can fix the MoE quality loss.
- **Method:** `moe/olmoe_experts.py`, rank-8 per-expert branches (52.4M
  trainable) vs residual-stream per-layer branches (rank 64, 6.29M), same loss
  and steps.
- **Outcome:** per-expert **PPL 6,551.74, router agreement 0.425** — *below*
  the 0.463 untrained baseline, with 8× the parameters; residual-stream
  **253.85 / 0.506**. Mechanism: routing is decided from the residual stream
  entering the block; a correction inside an expert acts after the decision.
- **Verdict:** Falsified; **placement rule: corrections live on the residual
  stream, not inside the experts** (TAARDIS's per-matmul placement was derived
  on a dense model and does not transfer).
- **Evidence:** `docs/MOE-EXTENSION.md` §2.4; `moe/olmoe_experts.py`;
  `moe/results/olmoe/`.
- **Status:** closed.
- **What it rules out:** per-expert sidecars as the placement; parameter count
  as the lever for routing damage.
- **Cost to revisit:** low; not recommended.

## D4 — Router-KD is a verified no-op

- **Claim/hypothesis:** distilling the teacher's router decisions (`router-KD`)
  is required for routing-agreement recovery.
- **Method:** `--router-weight 0` control vs KD-on, same recipe, every
  checkpoint compared.
- **Outcome:** matched within **~1.5%** at every checkpoint, including the same
  late turnover. Routing-agreement recovery comes from repairing the state the
  router reads, not from distilling its decisions.
- **Verdict:** No-op — the term is dropped from the recipe.
- **Evidence:** `docs/MOE-EXTENSION.md` §2.4a; `moe/qwen35_moe_proxy.py`.
- **Status:** closed.
- **What it rules out:** router-loss tuning as a lever.
- **Cost to revisit:** none.

## D5 — Post-hoc ternarisation of the correction sidecar is not viable

- **Claim/hypothesis:** fp32-trained branches can be ternarised after training
  for the ~2 bpw total claim.
- **Method:** same rank-512 recipe, branches exported as fp32 vs g128/post-rank
  ternary, post-hoc vs trained-with-STE, matched steps.
- **Outcome:** at the 512-window point — fp32 **71.14**; g128 post-hoc
  **1,014.34**; per-rank post-hoc **1,759.11**; g128 **STE-trained 93.47**;
  per-rank STE **107.18**. Post-hoc costs 14–25×; training in the deployed
  format is required. (Mixed fp16 layers are the size/quality dial: 2/16 fp16
  = 1.37×, 4/16 = 1.28× at 24.6/40.2 MB.)
- **Verdict:** Falsified; train the sidecar in the deployed format.
- **Evidence:** `docs/MOE-EXTENSION.md` §2.4c; `moe/olmoe_corrections.py`;
  `moe/branch_sensitivity.py`.
- **Status:** closed.
- **What it rules out:** post-hoc compression of the corrections.
- **Cost to revisit:** none.

## D6 — MoTE-style up-cycling does not transfer to Qwen3.6-35B-A3B

- **Claim/hypothesis:** MoTE-style up-cycling (frozen BF16 shared expert +
  ternary routed experts) applies to the target architecture.
- **Method:** `moe/moe_proxy.py` proxy on Qwen3-1.7B; census of the target's
  shared-expert share.
- **Outcome:** the proxy works (1.048× with **zero training**) but the frozen
  FP component carries the function — training the experts made it worse
  (1.122×). For Qwen3.6-35B-A3B the shared expert is **~0.4% of weights**, so
  there is nothing to carry it: this implies re-architecture (route B), not
  in-place work.
- **Verdict:** Not applicable to the target as an in-place route.
- **Evidence:** `docs/MOE-EXTENSION.md` §2.5, §4 (routes B/C);
  `moe/moe_proxy.py`.
- **Status:** closed.
- **What it rules out:** MoTE up-cycling as the cheap in-place MoE route.
- **Cost to revisit:** only as a re-architecture decision.

## D7 — Hot-expert GPU cache: no throughput gain

- **Claim/hypothesis:** pinning the hot expert fraction on the GPU (sidecar +
  graph split) speeds up MoE decode under CPU expert offload.
- **Method:** sidecar + graph split in the TAARDIS fork; 16/32/64 of 64 experts
  pinned, `-ncmoe 16 -ngl 99 -t 8`; equivalence checks (hot64 PPL-identical;
  hot16 within 0.03% and greedy token-identical); server sweep.
- **Outcome:** baseline **93.3 t/s**, hot16 **89.4**, hot32 **76.9**, hot64
  **89.4** — no gain, by construction: `mul_mat_id` computes k experts per
  token regardless of weight, so the cold pass still does all 8 evaluations
  while the GPU adds the hot ones.
- **Verdict:** Rejected as a performance feature at this size. The real fix is
  **sparse per-token dispatch** (variable k), which is kernel work.
- **Evidence:** `serving/placement-sweep-20260927/THROUGHPUT.md`,
  `.../EQUIVALENCE.md`, `.../SUMMARY.md`; `serving/HANDOFF-EXPERT-CACHE.md`.
- **Status:** closed.
- **What it rules out:** static hot/cold residency as a throughput lever on
  OLMoE-class models; layer-level `-ncmoe` placement and single-GPU sizing stay
  the effective levers.
- **Cost to revisit:** the sparse-dispatch engine project (not scheduled).

## D8 — Tail-conditioned KD does not transfer to the full body

- **Claim/hypothesis:** output-KD on the teacher's top-50 logits is the binding
  constraint on full-vocabulary fidelity, so a tail-conditional KD term
  (D_KL1 marginal + D_KL2 conditional, TAD's chain rule) closes the KLD tail at
  full scale.
- **Method:** the local 4-layer prefix selected the recipe (`D_KL2 3.0` + router
  bias + hard-window curriculum 0.05, tag `cur05`; fallback `D_KL2 2.0 + bias`,
  `pred2.0`), then one 35B pod window (1×H100, `$12.60`) trained both arms from
  one cache with the signed recipe and gated them on the community protocol.
  Later re-measured on a second protocol (ARC / MMLU / TruthfulQA + PPL) and
  paired with the k=1 drafter.
- **Outcome — full 35B ship gate (community protocol: wikitext-2 PPL c512
  580 chunks; KLD 50 chunks vs BF16; HellaSwag/Winogrande 400):**

  | model | PPL | KLD mean | KLD med | KLD p99.9 | KLD max | HellaSwag | Winogrande |
  |---|---|---|---|---|---|---|---|
  | v1 release | 8.3539 | 0.2694 | 0.1346 | 4.7579 | 7.2432 | 79.00 | 76.25 |
  | v2 `cur05`-soup | 8.5478 | 0.2666 | 0.1256 | 5.4674 | 7.8929 | 78.00 | 75.25 |
  | v2 `cur05`-final | 8.6228 | 0.2715 | 0.1286 | 5.3851 | 7.8410 | — | — |
  | v2 `pred2.0`-soup | 8.5452 | 0.2675 | 0.1254 | 5.3657 | 7.9904 | 77.75 | 75.25 |

  On the prefix the same family improved the tail (first sub-2.0 prefix max).
  At full depth the result inverts on the axis that mattered: **mean/median KLD
  marginally better (−1.0% / −6.7%), but p99.9/max worse (+14.9% / +9.0%)**,
  PPL **+2.3%**, tasks a tie (all within CI). A second protocol reproduced the
  tie (v1/v2: ARC-C 57.19/57.19, ARC-E 80.00/79.47, MMLU 40.15/39.60,
  TruthfulQA 32.56/31.82; fresh PPL 8.2731/8.4517). Paired k=1 drafter: v1
  1.368×, v2 1.302× — the drafter transfers, the body does not improve.
  (Full artifacts: `hivebench/artifacts/ternary/moe/qwen35/rental-v2/gate/`,
  `.../community-v1-v2/`.)
- **Mechanism:** the full-model teacher puts **96–99%** of its mass on the
  top-512 cache, versus **17–20%** on the 4-layer prefix. The tail-conditioned
  terms act on the complement, so their effective strength drops ~20–80×
  (measured ~1/60 on the Flash-Next cache) — nearly inert by construction. The
  binding term at full depth is the top-512 support fit the v1 recipe already
  had; the 4-layer prefix is **not a valid calibration proxy for tail-weighted
  objectives**.
- **Verdict:** Falsified at full scale. Tail-conditioned KD does not close the
  KLD tail on a deep body, and prefix tail gains do not transfer.
- **Evidence:** `RESEARCH-HANDOFF.md` §3c/§3d/§9; `docs/SCION-RECIPE.md` §5;
  artifacts above.
- **Status:** closed (negative). The DSpark multi-token drafter is a separate
  closed negative in the drafter lane (handoff §3c).
- **What it rules out:** tail-conditioned KD (D_KL1/D_KL2) as a lever at full
  depth for this class; prefix-tail gains as a predictor of full-body tail; the
  shallow prefix as a proxy for any objective acting outside the top-512.
- **Cost to revisit:** a target/proxy whose body actually carries large tail
  mass, or a different tail mechanism (topological, not a loss term).
