# V4.1 ternary + DSpark lanes

**Status:** plan, 2026-10-25. The FP32 tap-projection gate failed, so do not
export or integrate that head. This plan starts from the valid capture and the
released DSpark stack, not from the failed checkpoint.

| lane | scope | status |
|---|---|---|
| **A** | ternary small tap head | blocked by FP held-out gate |
| **B** | ternary full DSpark sidecar | ready to start |
| **C** | full all-ternary V4.1 base | later / rental-scale |

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

## Actual sidecar inventory

Measured on `models-p310/V41-GSQ/V41-MTP-Vision.Q2_K.gguf`:

```text
file            5,661,492,320 bytes (5.66 GB)
tensors         347
types           F32 178, Q8_0 160, Q2_K 9
ternary set     9 tensors, 4,459,069,440 bytes
```

Only these nine tensors are ternary candidates:

```text
mtp.{0,1,2}.ffn.experts.w1.weight   [5120, 2304, 128]  Q2_K
mtp.{0,1,2}.ffn.experts.w2.weight   [2304, 5120, 128]  Q2_K
mtp.{0,1,2}.ffn.experts.w3.weight   [5120, 2304, 128]  Q2_K
```

Everything else stays as the base sidecar: attention (152 tensors),
shared experts (9), router (9), taps/main projection, norms, hyper-connections,
Markov and confidence heads. The 13.59B expert parameters would be about
**2.97 GB at PTQ1_0** or **3.61 GB at PQ2_0**, versus **4.46 GB at Q2_K**;
the expected saving is only **0.85–1.49 GB**. That is a real but modest win, so
the acceptance and wall-clock gates matter more than the file size.

Regenerate this manifest with:

```bash
.venv-rocm/bin/python moe/v41_sidecar_manifest.py \
  --gguf models-p310/V41-GSQ/V41-MTP-Vision.Q2_K.gguf \
  --gguf-py ~/llama.cpp/gguf-py \
  --out docs/V41-DSPARK-SIDECAR-MANIFEST.json
```

The checked-in [manifest](V41-DSPARK-SIDECAR-MANIFEST.json) is the inventory
for this exact file.

## Tooling we already have

- `bonsai_forensics/quant.py` — absmean g128 research quantizer.
- `bonsai_forensics/gate1_forensics.py` — **Prism PQ2_0 (type 142)** decoder and
  group-128 helpers.
- `bonsai_forensics/gate2_rtn_artifact.py`, `export_rmd.py`,
  `materialize.py`, `pack_blockwise.py` — PQ2_0 materialize/pack/patch paths.
- `bonsai_forensics/pack_gguf.py` — **TQ2_0 (type 35)**, 256-value blocks. This
  is *not* Prism PQ2_0 (type 142) and not PTQ1_0 (type 143). Do not mix the
  codecs in a report.
- `bonsai_forensics/scripts/pilot/export_dspark_hf.py`,
  `dspark_export_check.py` — DSpark-shaped export and runtime inventory checks.
- `scion-v41-dspark/moe/mtp_dspark_export.py` — custom DSpark head GGUF export;
  `mtp_sidecar_export.py` — sidecar export.
- `scion-v41-dspark/moe/v41_sidecar_manifest.py` — reads GGUF metadata and
  emits the tensor-role manifest; no model load.
- `strata-ds41/tools/v41/engine/ds41_dspark.cpp` — V4.1 DSpark loader and
  executor; currently consumes the released MXFP4/Q8/Q2_K sidecar.
- `strata-ds41/tools/v41/dspark_accept.py` and
  `build-hip-pre/test_ds41_dspark` — offline acceptance and runtime checks.

Missing pieces:

- a Prism PQ2_0/PTQ1_0 **writer** for an arbitrary sidecar tensor set (the
  existing PQ2_0 code is mostly reader/patch-oriented);
- a ternary type exporter/loader path for the DSpark sidecar;
- ternary matvec kernels for the relevant DSpark shapes;
- QAT/lossy-quantization recovery for the draft stack;
- a wall-clock A/B on the same prompt/depth;
- a **baseline report** for the current Q2_K sidecar to compare against.

## Handoff contract

A lane starts with these inputs and produces these outputs; nothing else is
required to reproduce the first result.

**Inputs**

- sidecar: `models-p310/V41-GSQ/V41-MTP-Vision.Q2_K.gguf`
- target: `models-p310/V41-GSQ/V41-GSQ.gguf`
- manifest: `docs/V41-DSPARK-SIDECAR-MANIFEST.json`
- engine: `strata-ds41/build-hip-pre/ds41_generate`
- tests: `build-hip-pre/test_ds41_dspark`
- worktree: any `v41-dspark` checkout; use `HIP_VISIBLE_DEVICES=1` (the free
  card). Never run on device 0/display during the capture/training lane.

**Outputs**

```text
sidecar-tensors.json                 # full manifest, regenerated if source changes
sidecar-ternary-<packing>.gguf       # PTQ1_0 or PQ2_0 artifact
sidecar-ternary-<packing>.sha256
bench-q2k.json                       # baseline, same prompts/depth
bench-ternary-<packing>.json         # candidate
loader.patch                         # only if the engine path changes
```

Do not commit multi-GB GGUF files to git. Keep them under `HIVE_HOME/models`
or a scratch path, with SHA-256 in the report.

**First hour**

1. Read the manifest summary; confirm 9 expert tensors and 338 keep tensors.
2. Run the current Q2_K baseline on the fixed prompt at depth 5; write
   `bench-q2k.json`.
3. Build a synthetic sidecar containing only the 9 expert tensors at tiny
   shapes; make the loader round-trip it.
4. Only then quantize a real expert tensor and compare dequantized error.
5. Stop if the loader/dequant path is not exact; do not start a full pack.

**Definition of done**

- manifest regenerated and checked in;
- one packing loads in the engine;
- acceptance and decode tok/s measured against Q2_K on the same prompt/depth;
- loader rejects wrong/missing ternary types cleanly;
- baseline and candidate reports committed under `docs/` (small JSON only);
- no target/base change and no DSpark default change without the gates.

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

## Packing decision for the first experiment

Start with **Prism PQ2_0 (type 142)** because the local tooling already has a
byte-exact reader and the Bonsai packings are understood. PTQ1_0 (type 143) is
the second packing, not a quality tier. Do **not** use `pack_gguf.py`'s TQ2_0
(type 35) for this lane unless the engine path is changed deliberately; it is a
different codec with a different block size.

For each of the nine expert tensors:

1. Dequantize the Q2_K payload to FP32/BF16 masters.
2. Quantize with absmean g128 at the **same group axis** as the intended
   packing; record codes/scales and dequant error.
3. Try both unrotated and the Hadamard-rotated basis. The public Prism pack uses
   the rotated basis; parity matters more than a nominal error metric.
4. Pack the same codes as PQ2_0 and PTQ1_0; assert code/scale equality between
   the two packings.
5. Only then add the engine tensor type/loader path. A synthetic 9-tensor
   sidecar must round-trip before the real 5.66 GB file is written.

The current `STRATA_DSPARK_EXP_TYPE` loader requantizes experts on load using
standard ggml quantizers; that is **not** automatic support for Prism PQ2_0 /
PTQ1_0. The ternary path needs an explicit loader and kernel test.

## Measurement contract

Every baseline/candidate pair uses the same:

- target GGUF and sidecar depth (default 5, block 5);
- tokenizer and prompt IDs;
- `--temp 0`, same context and prefill flags;
- `HIP_VISIBLE_DEVICES=1`, same guard/arena settings;
- three runs per prompt, report median and all runs.

Report at least:

```json
{
  "sidecar": "...",
  "packing": "q2_k|pq2_0|ptq1_0|fp16",
  "depth": 5,
  "prompts": ["..."],
  "accept_per_position": [0.0, 0.0, 0.0, 0.0, 0.0],
  "accepted_per_pass": 0.0,
  "tokens_per_pass": 0.0,
  "decode_tok_s": 0.0,
  "ttft_ms": 0.0,
  "vram_peak_gib": 0.0,
  "nll": null,
  "kld": null,
  "artifacts_sha256": {}
}
```

Do not compare a candidate on new prompts against a baseline on old prompts.

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
