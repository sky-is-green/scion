# Porting the MoE recipe to qwen3_5_moe (35B-A3B)

Status: prep, 2026-09-25.  Target inventory validated config-only; no 35B
training has run yet.

## Target

The `qwen3_5_moe` architecture of the 35B-A3B distill censused in this
harness:

| property | value |
|---|---|
| layers | 40 (30 Gated-DeltaNet, 10 full attention) |
| hidden size | 2048 |
| experts | 256 routed, top-8, intermediate 512 |
| shared expert | intermediate 512 |
| vocab / head | 248,320 tokens, untied |
| routed experts | fused parameters `experts.gate_up_proj` [256, 1024, 2048] and `experts.down_proj` [256, 512, 2048] |
| precision-exempt | router `mlp.gate`, `shared_expert_gate`, GDN `in_proj_a`/`in_proj_b`, norms |

## Validated so far (local)

- **Target profile:** `bonsai_forensics.targets` now has a `qwen3_5_moe`
  profile; `scripts/pilot/inspect_model_targets.py` (meta device, no weights)
  reports **251/251 selected linears**, `target_count_ok: true`, 1.91B linear
  params (99.79% of language-tower linear params), rotation width 2048.
  The 100 unselected language-tower linears are exactly the intended FP set
  (30+30 GDN `in_proj_a/b`, 40 `shared_expert_gate`); the router is not an
  `nn.Linear` and the visual tower / MTP head are excluded by name.
- **Harness smoke passed:** `qwen35_moe_proxy.py smoke` on embedding + layers
  0-3 (partial checkpoint) runs the fused-bank STE patch end to end: hidden
  drift 0.307 and router top-8 agreement 0.825 against the FP pass (in the E1
  probe's range), 10.49M correction-branch + router parameters trainable, and the LM loss
  falls 8.74 -> 5.38 over 10 steps.
- **Role map:** `role_map.py` projects the ternary build at **8.82 GiB /
  2.186 bpw** (per-tensor roles in `results/role-map-*.json`; router and the
  absorb-exempt shared expert stay FP16).
- **Routing drift:** `probe_router.py` on embedding + layers 0-3, router exact,
  experts rotate+RTN g128: top-8 agreement 1.00 -> 0.81 -> 0.76 -> 0.65,
  hidden drift 0.20 -> 0.37.

## What the port needs

1. **Harness variant** (`qwen35_moe_proxy.py`, adapted from `olmoe_proxy.py`):
   implemented and smoke-tested locally (see above).  The full-model
   `cache`/`train`/`eval` stages are written but only executable once the
   complete checkpoint can be resident:
   - patch `Qwen3_5MoeExperts.forward` to ternarise the fused banks (STE) and
     rebind instance forwards (`device_map="auto"` shadowing trap);
   - hook the router (`layer.mlp.gate`) for the cache and the agreement metric;
   - 40 layers in the cache/KD loops; the shared expert and attention
     projections stay FP for the first cut (role-map default).
2. **Teacher cache** for N windows (N as large as the budget allows; the OLMoE
   data slope was still positive at 4,096): top-50 output logits plus per-layer
   router top-8. Estimate ~1.5 MB/window -> ~6 GB at 4,096 windows. This step
   and the build are the only ones that need the FP 35B resident (~70 GB bf16).
3. **Ternary build:** `save_ternary_olmoe.py` as the template - RTN g128 on the
   fused expert banks, materialise a HF dir for AUTOGRID and eval.
4. **Corrections:** per-layer residual-stream branches (rank 512 to start) plus
   trainable routers, loss = LM + output KD (the router-KD term is dropped; the
   OLMoE control showed it is a no-op).  Train with ``--quant lloyd`` so the
   frozen body and the branches see the deployment quantizer (Q1_0_g128), and
   ship the sidecar at ``--branch-quant g128``.  At hidden 2048 and 40 layers,
   rank 512 is ~84M branch parameters, ~22 MB deployed at ~2.1 bpw.
5. **Compute estimate** (rough, from the OLMoE measurements): teacher cache on
   one 80 GB card, order 1-2 h; correction training 8-10k steps on the same
   card, order 8-12 h.  No fp32-master QAT of the body is needed.

## Harness status (2026-09-27)

The OLMoE recipe is now ported into `qwen35_moe_proxy.py`:

- `--quant {absmean,lloyd}` (default `lloyd`, the deployable rule);
- `--branch-target {moe_out,attn_out,both}` (default `both`); `attn_out`
  wraps `linear_attn.out_proj` on GDN layers (`ssm_out` in GGUF) and
  `self_attn.o_proj` on full-attention layers (`attn_output`);
- the validated schedule: `--kd-weight 1.0`, `--temp 2.0`,
  `--lr-half-every 500 --lr-decay-start 2000`, Adafactor;
- `ref` stage + `--ref-file` (teacher-free training: on an 80 GB card the FP
  teacher and the 35B student cannot be resident together);
- `--cache-file`, `--corpus-chars`, `--eval-every` with router agreement;
- `export_branches_lora.py` maps the qwen35moe keys (`ssm_out` /
  `attn_output` / `ffn_moe_out`) and writes the compact q1_0_g128 adapter.

Box run order: `cache` (FP teacher resident, ~1-2 h) -> `ref` (same
session) -> free teacher -> `train` (student only) -> `eval` -> export.

## Smoke-test plan

- `scripts/pilot/inspect_model_targets.py` on the real config — **green**.
- Prefix smoke of the patched forward + correction steps — **green** (drift
  0.307, agreement 0.825, loss 8.74 -> 5.38).
- Materialise the prefix's expert banks to a ternary HF dir and scan with
  AUTOGRID — next; needs a short single-card run.
- Full-model `cache`/`train`/`eval` dry run stays for a larger GPU (the
  full 35B does not fit here).

## Open decisions

- shared expert at FP16 (role-map default) vs ternary + STE: ~0.4 GiB smaller,
  unknown quality cost;
- embeddings/head ternary (`ternary_embed` per role map) vs FP16;
- attention/GDN projections ternary vs FP in the first cut (role map counts
  them ternary; the OLMoE result only measured expert banks).
