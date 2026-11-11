# Export audit - PLAN section 13.1 (gate)

Date: 2026-11-11.  Status: **the shipped v1 artifact is not code-identical to
the Python training rule; the deployed artifact is internally valid and the v2
pipeline is fixed by seeding from it.**  Do not train v2 corrections from the
raw BF16 master via the unmodified Python path.

## Artifacts

| role | file | sha256 |
|---|---|---|
| deployed | `models-p310/scion-strata/Scion-35B-A3B-PQ2_0-corr.gguf` | `4e9e3292...cffa040` |
| source | `models-p310/bf16/Qwen3.8-35B-A3B-Distill-BF16.gguf` | `ecbe9e21...666660` |
| training rule | `moe/moe_proxy.py:ternary_lloyd` (+ `_lloyd_scale`) | commit `494cfb8` |
| deploy rule | `quantize_row_pq2_0_lloyd_ref` in `prism-moe-corr/ggml/src/ggml-quants.c` | build `prism-qwen4exp-shrink` |

The source is the 69.4 GB APEX BF16 GGUF whose sampled tensors were verified
equal to the empero BF16 reference used for the KLD grid.  Its whole-file
sha256 matches the Hub LFS hash.

## Method

1. Decode every deployed expert tensor (type 142, 34-byte blocks: fp16 scale +
   128 two-bit codes) to trits and fp16 scales.
2. Read the matching BF16 tensor from the source GGUF (same dims, same
   `[in, out, expert]` flat order) and run the exact Python training rule on
   each 128-weight group: Lloyd fixed point from four starts, fp16 scale
   round-trip, `torch.round`.
3. Compare codes and fp16 scale bits per group and per tensor.
4. Reproduce the C++ rule literally (`v2/scripts/cpp_lloyd_check.py`:
   sequential float32 sums, `roundf`, fp16 scale) on layer 0 to identify the
   deployed quantizer.
5. Re-quantise the deployed dequantised values with the Python rule to prove
   the remediation (exact float32 and the bf16 round-trip the HF model stores).

Zero-scale groups are reported separately: the source BF16 contains whole
groups of denormals (`~7.8e-38`), which both quantizers map to scale 0 and all
zero trits.  They are not evidence of a codec problem.

## Results

### Embedded branch factors (type 43 == Q2_0_g128 layout)

| check | result |
|---|---|
| tensors / weights | 160 / 209,715,200 |
| fourth code state (`11`, value +2) | **0** |
| groups whose scale is not the group max | **0** |
| negative / non-finite scales | **0** |

The shipped factors use only `{-1,0,+1}`, and every group scale equals the
maximum absolute deployed value.  `export_branches_lora.py` calls
`deploy_weights()` **before** `add_factor()`/`pack_q1_0_g128()`, so the packer
is fed deployed ternary factors, not the float master.  The router deltas are
dense F16 SVD pairs and never enter `pack_q1_0_g128`.

Caveat: the v1 training checkpoint is not on this box (only the merged GGUF),
so branch-factor fidelity is proven at the packer/invariant level, not against
the original checkpoint tensors.  Keep checkpoints in v2 and add this diff to
the release checklist.

### Expert banks (120 tensors, 32,212,254,720 weights)

| metric | value |
|---|---|
| code mismatches | **117,926,147** |
| mismatches per million weights | **3,660.9** (3,688.6 on non-zero-scale weights) |
| fp16 scale mismatch, groups | 48,430,861 / 251,658,240 = **19.24%** |
| code mismatch where fp16 scale agrees | 241,143 weights = **9.27 ppm** |
| code mismatch where fp16 scale differs | 117,685,004 weights = **18,984 ppm** |
| zero-scale state disagreements | **0** |
| mean / max absolute dequant error | 0.00366 / 0.5047 |

Median tensor mismatch is 3,715 ppm; layer 0 is the best (1,752 ppm), layer 33
the worst (3,841 ppm).  Per-tensor rows: [audit.csv](audit.csv); raw JSON:
[audit.json](audit.json).

The root cause is the **fp16 scale**, not the codec.  When Python and C++ land
on the same fp16 scale, the codes agree to 9 ppm (float ties).  In 19.2% of
groups the Python fixed point rounds to the adjacent fp16 scale; ~1.9% of the
weights in those groups cross the `scale/2` boundary and flip one trit.  This
is the mechanism the prior-art entry 009 ("Lost in the bf16 Cast") warned
about, at the scale-rounding level.

### Which quantizer produced the file

`cpp_lloyd_check.py` on all three layer-0 banks (805,306,368 weights,
6,291,456 groups):

| tensor | code mismatch | ppm | scale mismatch |
|---|---|---|---|
| `blk.0.ffn_gate_exps` | 424 | 1.58 | 0 |
| `blk.0.ffn_up_exps` | 406 | 1.51 | 0 |
| `blk.0.ffn_down_exps` | 2,118 | 7.89 | 0 |
| **total** | **2,948** | **3.66** | **0** |

Scales match bit-for-bit and codes agree to 3.66 ppm.  The deployed body is
unambiguously the C++ `GGML_PQ2_0_LLOYD` output, as the pre-flight recorded.

### Remediation proof

Re-quantising the deployed dequantised banks with the Python training rule
(`--self-check`, all 120 tensors, 32,212,254,720 weights):

| precision | code mismatches | ppm | scale mismatch groups |
|---|---|---|---|
| float32 (exact deployed values) | **0** | **0.0** | 0 |
| bfloat16 (HF parameter dtype) | **0** | **0.0** | 217,140,810 |

The bf16 cast changes scale bits but never a code: the non-zero trits are a
full scale away from the decision boundary.  Therefore the fix is to make the
deployed GGUF the source of truth for the frozen body:
[`moe/seed_banks_from_gguf.py`](../../moe/seed_banks_from_gguf.py) copies the
deployed dequantised values into the HF expert banks before training.  Then
`ternary_lloyd` is idempotent and training sees exactly the deployed codes.
The body is not modified.

## Decision

- **v1 artifact:** strict equality with the Python training rule **fails**
  (3,661 ppm, 117.9M weights).  The artifact itself is valid C++ Lloyd output.
- **v2 gate:** pass with the mandatory pipeline change.  Correction training
  must be seeded from the deployed GGUF; re-run this audit after the first v2
  export and require 0 ppm against the v2 training seed.
- The v1 corrections were trained against 99.63% of the deployed codes.  The
  mismatch is a plausible contributor to the KLD tail, but it is too small to
  explain the hard-90 failure; the long-horizon data gap remains the primary
  cause.
- **No body re-pack.**  Body frozen; seed the training copy instead.

## Reproduce

```bash
# source (once; 69.4 GB, hash ecbe9e21...666660)
curl -L -o models-p310/bf16/Qwen3.8-35B-A3B-Distill-BF16.gguf \
  https://huggingface.co/MrFuzzihead/Qwen3.8-35B-A3B-Distill-APEX-GGUF/resolve/main/Qwen3.8-35B-A3B-Distill-BF16.gguf

# full bank + branch audit (~15 min on one 7900 XT)
python v2/scripts/export_audit.py \
  --deployed models-p310/scion-strata/Scion-35B-A3B-PQ2_0-corr.gguf \
  --source models-p310/bf16/Qwen3.8-35B-A3B-Distill-BF16.gguf \
  --device cuda:1 --out v2/export-audit/audit.json

# remediation proof
python v2/scripts/export_audit.py \
  --deployed models-p310/scion-strata/Scion-35B-A3B-PQ2_0-corr.gguf \
  --branches-only --self-check --device cuda:1 \
  --out v2/export-audit/selfcheck.json

# C++ provenance on layer 0
python v2/scripts/cpp_lloyd_check.py \
  models-p310/scion-strata/Scion-35B-A3B-PQ2_0-corr.gguf \
  models-p310/bf16/Qwen3.8-35B-A3B-Distill-BF16.gguf \
  --layers 0 --out v2/export-audit/cpp-parity.json
```

Files: [audit.json](audit.json), [audit.csv](audit.csv),
[selfcheck.json](selfcheck.json), [cpp-parity.json](cpp-parity.json).
