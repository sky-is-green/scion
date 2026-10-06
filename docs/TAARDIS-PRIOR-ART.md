# External prior art: TAARDIS (CodeMasterCody3D) — notes for the dense Clef route

Surveyed 2026-10-03.  TAARDIS is the
direct ancestor of our fork (`taardis-lora`, `Q1_0_g128`, `blk.N.ssm_readout`
all come from it), so this is not just related work — it is the route's
upstream.  Everything below is public; none of it was downloaded into the
workspace.

## What TAARDIS is

**Ternary Adaptive Alignment & Rotation for Dense Integer Stacking** — Cody
Dixon's post-training pipeline + llama.cpp fork for full-ternary integer dense
models.  Result on Qwen3.8-27B:

- 1.75 bpw (base-3 five-trit `Q1_T_g128`; lossless repack of 2.125 bpw
  `Q1_0_g128`), 5.90 GB weights; embeddings, head, norms and group scales on
  the integer grid too (k8/k6 digit stacks).
- Wikitext c512: 13.61 PPL uncorrected -> **11.83 with "The Doctors"**
  (496 low-rank ternary correction branches, 0.32 GB).
- Rotation is load-bearing: `LLAMA_FORGE_ROT_DISABLE=1` -> PPL ~1.26M.
- Caveat from the card: post-training conversion + trained corrections; still
  behind PrismML's Ternary-Bonsai (11.01) which trains ternary weights.

Links:
- Model: <https://huggingface.co/CodeMasterCody3D/taardis-27b-full-ternary>
- Fork: <https://github.com/CodeMasterCody3D/taardis-llama.cpp> (branch
  `q1_0_g128-port`; TAARDIS.md is the feature list)
- AUTOGRID: <https://github.com/CodeMasterCody3D/autogrid> (noise-floor scanner,
  self-contained `autogrid.py`)

## Artifacts that matter to us

| artifact | what it contains | why we care |
|---|---|---|
| `taardis-dn-cache-reason` (dataset) | 48 x `readout_NN.safetensors` (~409 MB each, one per GDN layer of the 27B) + `manifest.pt` | the **training cache for per-head recurrent-readout branches** — the tap our route does not train |
| fork commit `fbcfe3712` (`dn-readout` branch) | `blk.N.ssm_readout` virtual LoRA target: `lora_a=B [head_dim,rank,heads]`, `lora_b=A [rank,head_dim,heads]`, applied per head on the recurrent readout before the gated norm | runtime side; **our fork already has this** (ported). The training/export side is missing in `scion/dense/` |
| `taardis-27b-teacher-bundle` (dataset) | `corpus/calib.txt` + `calib.lanes.json` + `mix.json`; per-block Hessian/H-map artifacts; `ruler/t27/calib.txt` | corpus recipe: 88.4% reasoning rails (9 lanes) + 35 sparse general lanes; chunk-level train/eval splits (`union.json`) |
| `taardis-reason-rails` (dataset) | 6 MB rail corpus, `union.json` chunk-level holdout, `pack.json` (2bit/g128) | the actual calibration/holdout pattern for a longer run |
| `taardis-27b-hess27carry` (dataset) | 64 per-block input-activation Hessians + `hessians.json` | per-linear placement; note `carry`: Hessians computed with the recurrent cache carried across chunks |
| `taardis-27b-damage-map-tsd11-0928` (dataset) | `heal_sweep_tsd11.json`: 60 broken probes, 374 linears measured, 142 shared culprits | per-tensor damage -> the principled input to rank allocation |
| `taardis-27b-gaterig` (dataset) | per-GDN-layer signal streams (`dn*`, `edge*`, `gvec*`, `kv*`, `vv*`) + rotation-basis / k5-lift analyses | instrumentation template for deciding where correction capacity goes |
| `CodeMasterCody3D/side-qwen35-08b-*` + `qwen35-08b-reasoning-ultramega-hmaps-*` | **Qwen3.5-0.8B** (same hybrid arch, `full_attention_interval=4`, 24 layers) ternarised to k2 and k1 with GPTQ + rotation + per-layer K5 self-distillation; published per-block reconstruction/metric JSONs + Hessians | the same-architecture validation of the V2 pipeline; mine the metrics for which Qwen3.5 tensors tolerate k2/k1 |

## Findings that should change our plans

1. **Per-head GDN readout is their main correction target on the hybrid
   layers.**  Our route corrects `linear_attn.out_proj` only.  Their 48-file
   readout cache and the `dn-readout` runtime hook say the recurrent readout
   (before the gated norm) is where they spend capacity.  The fork already
   exposes it; a third tap (`--target` extension + export layout
   `[head_dim,rank,heads]`) is the concrete next training-side change.
2. **"Cross-layer, jointly-trained low-rank branches ... measured 3.3x more
   effective than per-layer correction on held-out data."**  Our branches are
   joint end-to-end but uniform per-layer rank 512.  Their Doctors allocate
   rank 8..256 **per matmul by measured benefit** and are described as
   cross-layer.  Before a longer run, read the damage-map/heal-sweep JSONs and
   re-allocate rank by measured damage; consider whether any tap should read a
   different layer's activation.
3. **Rotation is their whole quality story.**  Clef's deployed PQ2_0 has no
   rotation (byte-verified container, no metadata).  So a "Clef V2" re-quant
   from the f16 body — per-linear block-Hadamard + Hessian-weighted GPTQ +
   flip-polish + Doctors — is likely a stronger lever than further repairing
   Cloudflare's unrotated Lloyd body.  Their Qwen3.5-0.8B run is the same-arch
   template, and our local fork already implements the rotated-basis runtime
   (`adapter.taardis.rotated_basis`, `forge.rotation.*`) and the taardis-lora
   container.  Keep as the fallback/parallel track to the current sidecar.
4. **AUTOGRID answers our mixed-precision open question** (which tensors can
   leave bf16 "free"; where `in_proj_a/b` and the GDN tensors fall).  It scans
   safetensors; we can dump the reverse-loaded f16 weights (or the f16 GGUF
   tensors) to HF layout and run `autogrid.py --scan --max-k 2` to get a
   Clef-specific FREE/TERNARY/STEER classification.  Cheap, no training.
5. **Containers:** `Q1_T_g128` (1.75 bpw) is lossless vs Q1_0_g128 but slower
   CPU unpack; for the CPU validator keep 2.125 bpw (`Q1_0_g128`).  Ternary KV
   cache is irrelevant at our 1k contexts.
6. **Their runtime discipline matches ours:** corrections ride as a LoRA
   sidecar (a low-rank fix cannot be folded into a ternary base without going
   off-grid), toggled at scale 1.0, with loud refusal on wrong basis.  Our
   packaging is the same shape; the `adapter.embedded` single-file path is a
   fork addition here (needs the explicit `llama_set_adapters_lora` call in
   raw-API consumers).

## AmelieSchreiber — surveyed, not applicable

Profile: <https://huggingface.co/AmelieSchreiber>.  The public record is
protein/small-molecule ML, persistent-homology diagnostics of attention, and
LoRA/QLoRA fine-tuning of ESM-2 (2023), plus the current ToricBLM/ToricGT
geometric small-model program.  No ternary, quantization, QAT, distillation or
GGUF artifacts; nothing we can lift for the Clef correction route.  The only
tangential asset is the 2023 ESM-2 LoRA/QLoRA code — our branch/trainer design
is already equivalent or beyond it.

## Not used as data

The 27B teacher caches, Hessians and `manifest.pt` are large and specific to
Qwen3.8-27B; useful as format references, not as data for Clef.
