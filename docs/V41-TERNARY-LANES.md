# V4.1 ternary + DSpark lanes

**Status:** plan, 2026-10-25. The FP32 tap-projection gate failed, so do not
export or integrate that head. This plan starts from the valid capture and the
released DSpark stack, not from the failed checkpoint.

## What “ternary” means here

Keep these separate; they have different risk and different claims:

1. **Ternary expert overlay** — routed MoE experts are ternary; attention,
   router, shared experts, embeddings, norms, Engram and DSpark stay in the
   base precision. This is what the public `meshapplied` V4.1 pack does.
2. **All-ternary base** — every eligible matrix, including dense/attention, is
   ternary. This is a much larger quality and training project.
3. **Ternary DSpark sidecar** — the three-block draft stack itself is
   ternary-packed, while the target stays as-is.
4. **Ternary small head** — our 498M tap-projection head; only viable if it
   first passes the FP held-out gate.

PTQ1_0 and PQ2_0 are **packings of the same ternary codes**, not two quality
levels. The Bonsai parity result says the packings agree; it does not say
ternary equals the BF16/FP4 source.

## Public prior art (checked 2026-10-25)

| model | what is ternary | serving | notable result |
|---|---|---|---|
| `meshapplied/DeepSeek-V4.1-Flash-Ternary-Latest` | 40-layer routed-expert overlay, `w123`, g128, Hadamard GPTQ; attention/router/shared/Engram/DSpark stay base | Goethe + SGLang, 4× RTX PRO 6000 96 GB, TP=EP=4 | DSpark block 5; accept 3.78 @1, 4.08 @4; 194/309 tok/s agentic; held-out ΔNLL +0.202 (general +0.415) |
| `meshapplied/Qwen3.8-Flash-Next-Ternary-Latest` | 48-layer routed-expert overlay; base attention/router/shared/MTP stay | Goethe + SGLang, 96 GB | NEXTN 3/4; accept 2.99; 168 tok/s |
| `Jakevin/Qwen3.8-Flash-Next-ternary-mlx` | **mixed** MLX: 117/288 units ternary, remainder 2/3/4-bit affine; all-ternary retains ~57% of FP8 MC score | Apple MLX | not an all-ternary model |
| `prism-ml/Ternary-Bonsai-2-27B` | Bonsai 2 27B, not Flash-Next | Prism runtime | relevant packing/recipe evidence, different target |

Conclusion: we are **not** the only ternary-based Flash-Next. We may still be
the only **local, consumer-AMD, all-ternary/GSQ-RCO** lane, but that is a
narrower claim and needs its own measurement.

## Tooling we already have

- `bonsai_forensics/quant.py` — absmean g128, PQ2_0/PTQ1_0 research quantizer.
- `bonsai_forensics/pack_gguf.py`, `run_quant.py` — packed artifacts and
  round-trip checks.
- `bonsai_forensics/scripts/pilot/export_dspark_hf.py`,
  `dspark_export_check.py`, `pack_blockwise.py` — DSpark-shaped export and
  runtime inventory checks.
- `scion-v41-dspark/moe/mtp_dspark_export.py` — custom DSpark head GGUF export;
  `mtp_sidecar_export.py` — sidecar export.
- `strata-ds41/tools/v41/engine/ds41_dspark.cpp` — V4.1 DSpark loader and
  executor; currently consumes the released MXFP4/Q8/Q2_K sidecar.
- `strata-ds41/tools/v41/dspark_accept.py` and
  `build-hip-pre/test_ds41_dspark` — offline acceptance and runtime checks.

Missing pieces:

- a ternary type exporter/loader path for the DSpark sidecar;
- ternary matvec kernels for the relevant DSpark shapes;
- a DSpark-specific tensor manifest (sidecar vs target);
- QAT/lossy-quantization recovery for the draft stack;
- a wall-clock A/B on the same prompt/depth.

## Lane A — ternary small tap head (blocked)

**Precondition:** FP32/BF16 head passes the held-out gate against the released
DSpark at the same depth. It currently does not (ideal 1.036; no-spec is 1.0).

If it passes:

1. Freeze the FP checkpoint and record its exact prompt split.
2. QAT/STE the trainable dense matrices with ternary forward; do not start with
   post-hoc PTQ.
3. Export the same ternary codes in PTQ1_0 and PQ2_0.
4. Add an engine loader for the small-head tensor types.
5. Gate: acceptance ≥ DSpark at depth 3, decode tok/s > DSpark, token identity
   on the fixed prompt, VRAM within budget.
6. If it fails, keep the artifact as negative evidence; do not ship.

## Lane B — ternary full DSpark sidecar (first real lane)

Target: the released three-block DSpark sidecar, not the target body. The
target's attention/router/shared/Engram remain unchanged.

1. **Inventory** every `mtp.0–2.*` tensor: shape, current type, whether it is an
   expert matrix, attention projection, norm, markov head, or confidence head.
   Produce `sidecar-tensor-manifest.json`.
2. **Choose quantizable set.** Start with routed expert matrices only. Keep
   norms, embedding/markov/confidence and anything shape-sensitive in higher
   precision until measured.
3. **Quantize** with the existing absmean g128 codec; compare Hadamard-rotated
   and unrotated bases. Record codes, scales and dequant error.
4. **Pack** both PTQ1_0 and PQ2_0. Verify byte-level round trip and exact code
   equality between packings.
5. **Loader**: add the ternary tensor types to `ds41_dspark.cpp`; keep the
   existing Q2_K path as fallback. Test missing/wrong-type rejection.
6. **Acceptance gate** on the same fixed prompts and depth:
   - compare against current Q2_K DSpark;
   - report accepted tokens/pass, per-position acceptance, full-block rate;
   - require no regression beyond an explicitly recorded tolerance.
7. **Wall-clock gate**: decode tok/s with draft/verify overhead, not just
   acceptance. A smaller file that accepts less is a loss.
8. **Quality gate**: held-out NLL/KLD and the fixed greedy prompt; record token
   identity where verification allows it.
9. **Memory gate**: VRAM/GTT/host peak on the two-card box.
10. **Rollback**: keep the Q2_K sidecar selectable by env/setting.

The first goal is **size and speed parity**, not an all-ternary claim. If
ternary experts lose too much acceptance, stop; do not quantize the dense half
to chase a number.

## Lane C — full V4.1 base / all-ternary (separate, larger)

Only if Lane B proves the ternary kernel path and the user wants the larger
claim.

- Current GSQ is a mixed quant (experts 42.8% PTQ1_0 / 39.2% Q2_0 / 18.0%
  Q3_K; dense Q8_0), not all-ternary.
- All-ternary base needs QAT/corrections, not a naïve PTQ; the public
  expert-overlay result already shows a general-domain NLL cost.
- This lane needs a rental/large-memory trainer or a staged dense/expert split;
  it is not a two-card local job.
- Do not change the DSpark target or sidecar until the base lane has its own
  quality gate.

## Shared gates

| gate | question |
|---|---|
| **G0 format** | do packed and dequantized tensors match the intended code/scale? |
| **G1 load** | does the engine load the sidecar with all blocks and no fallback errors? |
| **G2 acceptance** | is accepted length ≥ the selected DSpark baseline at the same depth? |
| **G3 speed** | is decode tok/s better than the baseline, including draft cost? |
| **G4 quality** | are held-out NLL/KLD and fixed-prompt tokens acceptable? |
| **G5 memory** | does it fit with guard margin on the target box? |
| **G6 negative control** | does a wrong target/head/profile lose performance as expected? |

A smaller file, higher acceptance, or lower NLL alone is not a pass. The lane
ships only when G2–G5 all clear.

## Data for Lane A

The valid 250k capture has 2,048 windows but only **639 unique token windows**
and 2,033 unique taps; four prompts are not enough. The next capture must:

- use multiple distinct prompts per bucket, not eight seeds of one prompt;
- hold out several prompts, not one repeated systems prompt;
- deduplicate windows before splitting;
- record prompt/session provenance with every window;
- keep the same target and tap layout so Lane B remains comparable.

## Immediate order

1. Keep released DSpark as the default; no export of the failed head.
2. Build Lane B's tensor manifest and ternary sidecar loader test with synthetic
   tensors first.
3. Run one expert-only ternary repack; measure acceptance and tok/s.
4. If Lane B passes, revisit Lane A only after a better held-out corpus.
5. Lane C remains a separate decision and budget.

## Claim discipline

Use precise descriptions:

- “ternary routed experts, base attention/router/DSpark preserved” for the
  public overlay class;
- “all-ternary” only when dense and experts are ternary;
- “mixed 2/3/4-bit” for the MLX pack;
- “consumer two-card AMD” only for our local measured stack.

Do not claim “the only ternary Flash-Next.” The public record now contradicts it.
