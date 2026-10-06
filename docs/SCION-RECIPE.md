# The Scion training recipe — ternary MoE compression (model-agnostic)

**Purpose.** Everything needed to re-run the Scion recipe on a new MoE target —
in particular the forthcoming **Qwen 4.0** (`qwen4_exp`) — without re-deriving
it. Frozen, measured, signed for the 35B v1/v2 runs; the numbers below are the
evidence. Companion docs: [`MOE-EXTENSION.md`](MOE-EXTENSION.md) (the MoE
write-up), [`RELEASE-35B-MODEL-CARD.md`](RELEASE-35B-MODEL-CARD.md),
[`QUANT-RETENTION-35B.md`](QUANT-RETENTION-35B.md) (the gate protocol).

## 1. The method in one paragraph

Take a pretrained MoE (Qwen3.x/Llama-4-class): quantize the **expert banks** to
a ternary/2-bit container (`PQ2_0` = ggml type 142, g128 groups, Lloyd codes)
and train a small set of **low-rank correction branches (rank 512) on the MoE
output and the attention output**, plus per-expert router balancing, by
**distilling the frozen full-precision teacher's logits** (top-512 + a sampled
tail) on **self-generated / corpus text**. Calibration-free: teacher logits and
in-domain text only — **no imatrix, no external calibration corpus**. The
deployed artifact is the ternary body + an adapter (`.lora_a/.lora_b` +
router deltas) merged into a single GGUF.

## 2. The objective (what each term buys)

Loss per step (all on the ternary student's full-vocab logits):

```
loss = lm
     + kd_weight      * KD_top512(student, teacher ; temp)
     + kd_tail_weight * residual_mass_KL(student_mass, teacher_mass)     # D_KL1 marginal
     + kd_tailcond_weight * tail_conditional_piece(...)                  # D_KL2, TAD
     [+ balance term when --balance is on]
```

- **`lm`** — ordinary cross-entropy on the corpus (keeps the body honest).
- **`kd_weight` / temp** — top-512 KD. The original v1 lever; it moves the
  *mean* KL but leaves the tail (measured: combo 2.0+2.0 → mean 0.815, p99
  2.85, **max 4.05**).
- **`kd_tail_weight`** — a marginal KL on the two-way support/complement split;
  fixes the mass outside the cache's top-512.
- **`kd_tailcond_weight` (D_KL2)** — the **tail-conditional** term (TAD): the
  KL of the distribution *given* it is in the tail, estimated from
  `--tail-logits` sampled teacher-tail tokens. This is the single biggest lever
  on the worst token; at weight 3.0 + bias + curriculum it gave the program's
  first sub-2.0 max on the prefix. **80% of the v1 gap was tail-conditional**
  (decomposition: marginal 7%, support 13%, tail 80%).
- **`balance`** — router balancing. `bias` (a per-expert bias trained by an
  auxiliary load-balance update) was the keeper (best max, small mean cost);
  `quantile`/`cb`/`zloss` were neutral-to-worse. `--log-entropy` logs `H`,
  `loadH`, `seqvar`, `tcond` diagnostics.

**Best measured point on the prefix (local 4-layer, the candidate set):**
`D_KL2 3.0 + bias + curriculum 0.05` → deployed **0.3099 / 0.8265 / 1.9335**
(mean / p99 / max) — a strict win on every deployed axis vs the v1 candidate.
At **full 35B** the recipe did **not** beat v1 (§5): the prefix is far more
damaged than a 40-layer body, so prefix deltas are an upper bound, not a
forecast.

## 3. The exact recipe as run (35B v2, `box-run-v2.sh`)

Common flags for both arms (the **one variable** between them is
`--kd-tailcond-weight`):

```
--quant lloyd --branch-quant g128 --branch-target both --rank 512 \
--kd-weight 2.0 --kd-tail-weight 2.0 --temp 2.0 \
--balance bias --log-entropy \
--corpus-file $MIX --agentic-frac 0.05 \
--windows 4096 --corpus-chars 50000000 --seq 512 --seed 0 \
--epochs 1 --steps 4096 --eval-every 1000 --ckpt-every 1000 --log-every 100
```

| arm | `--kd-tailcond-weight` | tag | role |
|---|---|---|---|
| `cur05` (primary) | **3.0** | `cur05` | best deployed KLD; first sub-2.0 max |
| `pred2.0` (fallback) | **2.0** | `pred2.0` | PPL-conservative hedge (same cache) |

- `both` = corrections on MoE output **and** attention output.
- Do **not** add `--alloc-file` (the RCO/sensitivity allocations were measured
  and neither is a strict win over all-ternary).
- **`--resume` is unsafe for the main recipe** (the trainer pairs the cache
  iterator from 0 with a step-offset data index → misaligned windows). Restart
  the arm; the cache survives.

### Data / cache
- **Mix file**: `curric-combo.jsonl` (fineweb-edu + hard-window curriculum),
  `--agentic-frac 0.05`. The cache **refuses** to build without it — the cache
  corpus must equal the trained corpus.
- **Cache**: `--windows 4096 --corpus-chars 50000000 --seq 512 --seed 0
  --top-logits 512 --tail-logits 64`. Fields: `idx/val/w/router/tidx/tlp`
  (`w` = per-token teacher support mass, required by the tail terms; `tidx/tlp`
  = sampled tail tokens). ~10 GB at 40 layers (the router dict scales with
  depth); a 4-layer prefix is ~7.5 GB.

## 4. Stages and gates (`moe/box-run-v2.sh`)

`setup → smoke → cache → ref → train → train-fallback → eval → export → gate`

| stage | gate |
|---|---|
| setup | GPU visible; `causal_conv1d`/`fla` installed (**triton==3.8.0** pin matters on a near-full card); teacher shards complete (verify `model.safetensors.index.json`); mix file present |
| smoke | hidden drift 0.3–0.45, router top-8 agreement ~0.83, no fast-path warning |
| cache | the six fields present, tail fields present, size sane |
| train | **shape on the full model: `lm ~2.1`, `H ~2.2`, `kd ~0.3–0.7`, in-run PPL ~8.2–8.4** (see §6); abort on NaN or gross deviation |
| eval / export | PPL+router JSON written; adapters + soups produced |
| gate | community protocol on the merged GGUF (§5) |

## 5. Measured results

**Prefix (local 4-layer, arm-to-arm only; PPL not predictive in level):** the
D_KL2 sweep 0.5/1.0/2.0/3.0 → mean 0.584/0.462/0.357/0.333; `cur05` deployed
0.3099/0.8265/1.9335.

**Full 35B (community protocol: wikitext-2 PPL c512 580 chunks; KLD 50 chunks
vs BF16; HellaSwag/Winogrande 400):**

| model | PPL | KLD mean | KLD med | KLD p99.9 | KLD max | HS | WG |
|---|---|---|---|---|---|---|---|
| v1 release | 8.3539 | 0.2694 | 0.1346 | 4.7579 | 7.2432 | 79.00 | 76.25 |
| v2 `cur05` | 8.5478 | 0.2666 | 0.1256 | 5.4674 | 7.8929 | 78.00 | 75.25 |
| v2 `pred2.0` | 8.5452 | 0.2675 | 0.1254 | 5.3657 | 7.9904 | 77.75 | 75.25 |

**Read: v2 is a near-tie with v1** (mean/median KLD marginally better, tail
worse, PPL +2.3%, tasks within CI). The recipe transfers *mechanically* but the
full-body gain is small — budget accordingly and gate on the full model, never
the prefix.

## 6. Failure modes and lessons (all hit once)

1. **OOM at train (~2m in), FLA `chunk_bwd_dqkwg` autotune.** The v2 tail terms
   push the near-full card (68 GB fp16 latents + KD graph) past what Triton's
   autotune benchmark can allocate. Fix: `FLA_CACHE_MODE=default` +
   `FLA_CONFIG_DIR` with a pinned `default_config` (`num_warps 4, num_stages 2`)
   — a kernel-launch config, no numeric effect. `PYTORCH_ALLOC_CONF=
   expandable_segments:True` alone was not enough.
2. **`triton==3.8.0`** is required (3.4–3.7 refuse the Hopper GDN backward).
   The pip resolver warns it conflicts with torch's expected version — expected.
3. **Prefix vs full shape.** The runbook's train gate (`lm 7–9`, `H 9.8–10.4`,
   in-run PPL 2800–3000) is the **prefix** shape. The full 35B is healthy at
   `lm ~2.1`, `H ~2.2`, PPL 8.2–8.4. Do not abort on the prefix numbers.
4. **Partial HF download** → re-run and verify the shard index against files.
5. **Cache without the mix file** → refuses by design; do not work around.
6. **`eval` omits `--balance bias`** (the deployable number). The trained-with-
   bias model evaluated without the router patch differs materially
   (35B in-run 8.28 vs deployed eval 10.5) — report both, and decide the bias's
   serving story before shipping.
7. **`--resume`** — see §3.

## 7. The drafter lane (speculative decoding)

The release ships a frozen-body **drafter** trained on the release's own
post-norm hidden + the next token's embedding; the final projection reuses the
target's `output_norm`/`output` (no second vocab projection).

1. **Taps** — the fork's `test-mtp-probe <model.gguf> <tokens.bin> N seq outdir`
   (branch `moe-corr-runtime` in the runtime fork) dumps per-position
   fp32 post-norm hidden + the greedy argmax. Build `tokens.bin` with the
   target tokenizer over the training corpus (mix, 3088 windows × 512).
2. **k=1 sidecar** — `moe/mtp_release_train.py` (fc1/gelu/fc2, self-distilled
   to the release's greedy). Export `moe/mtp_sidecar_export.py` → `mtp.*` GGUF.
   Runtime: `draft-mtp-sidecar` (auto-detected). Measured 35B v2: wikitext
   0.359 / fineweb 0.487 teacher-forced; generation **1.678 tok/fwd**;
   llama-server **1.135×**, CLI **1.17×**.
3. **DSpark-shaped multi-token head** — `moe/mtp_dspark.py` (a GRU **state per
   drafted position** + a direct residual branch, k=3), export
   `moe/mtp_dspark_export.py` → extended **`mtp2`** GGUF. Runtime:
   `llama-mtp-sidecar.cpp` unrolled k-step graph + `llama_mtp_sidecar_draft_multi`,
   now wired into `common/speculative.cpp` (`mtp2` auto-detected; proposes
   `min(--spec-draft-n-max, k)`; `need_n_rs_seq` peeks `mtp2.k`; `accept()`
   rebases the draft seed to the last accepted verify row). **Measured and
   rejected:** the generation ideal holds (**1.875 tok/fwd** vs the k=1's
   1.708) but the net wall-clock loses to the k=1 sidecar — harness mtp2
   90.7 tok/s vs k=1 114.2 (**0.79×**); llama-server 1.04× vs 1.28× (prompt 1)
   and 0.81× vs 1.08× (prompt 2). Each extra verify row costs ~2.3–3.1 ms on
   this MoE (every verified token activates its own experts), so multi-token
   does not pay here. **Ship the k=1 sidecar; only revisit a multi-token head
   for a target with a cheaper verify (dense attention / shared experts).**
4. `moe/mtp_eval.py --chain K` measures the naive stale-hidden chain (a head
   without per-position state does **not** pay).

## 8. Porting to Qwen 4.0 (`qwen4_exp`) — checklist

Same recipe, new runtime work first:

1. **Runtime pin.** `qwen4_exp` is in transformers `main` (pin a commit) or
   vLLM/SGLang; shadow-install and smoke a tiny forward.
2. **Correction/KD port.** The harness needs the arch's forward + router hook
   and the expert-bank patch (the fused `gate_up_proj [E,2ff,h]` / `down_proj
   [E,h,ff]` layout is near-identical to `qwen35moe`). Watch the **hyper-
   connection residual topology** around `mlp_hyper_connection` — it is not a
   plain additive residual.
3. **Teacher.** Official FP8 (`Qwen3.8-Flash-Next-FP8`, 173 GiB) needs the
   `kernels` package and cc ≥ 8.9; verify `from_pretrained` keeps fp8 (the
   cache stage must load through `from_pretrained`, not a manual loader).
   2×H200 for the cache at the 180B scale.
4. **PLE precision** (`ple_layer_ids`, ~28.8 B n-gram table) is **not covered
   by the recipe** — decide ≥4-bit and measure.
5. **Drafter taps.** `set_capture_layers` covers `qwen35`/`step35`/`dspark`
   today, **not** `qwen35moe`/`qwen4_exp`; the k=1 sidecar only needs the final
   hidden via `embeddings_nextn` (works), the DSpark-proper head needs
   intermediate-layer capture (port it).
6. **Caps.** Flask the cost: cache is the expensive stage; the trainer reads the
   cached logits, so the teacher is resident only for the cache.

## 9. What "done" looks like

- One cache; two arms (`cur05` primary, `pred2.0` fallback) trained from it; a
  merged single-file GGUF per arm; adapters + eval JSONs; the community gate
  table; a k=1 sidecar (and, if the numbers hold, the DSpark multi-token head).
  **No release before the community gate and, for the multi-token drafter,
  before the real (batched-verify) speedup.**
