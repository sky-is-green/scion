# Scion runtime contract

Scion is a model and training project. It does **not** depend on Strata, Hive, or any other
serving engine. This page is the boundary an engine has to meet to run a Scion release.

## What the model is

A Scion release is one GGUF file:

| Part | 176B FlashNext | 35B A3B |
|---|---|---|
| Routed experts | `TQ1_0_g128` (type 143, formerly `PTQ1_0`), 128 values / 28 B | `Q2_0_g128` (type 142, formerly `PQ2_0`), 128 values / 34 B |
| Rest of the body | Q6_K (the direct release) or the exported mix | Q8_0 |
| Corrections | 96 F16 LoRA pairs, embedded | `Q1_0_g128` (type 43) + F16 router pairs, embedded |
| PLE / n-gram table | Q4_0 rows, 160 values / 90 B | none |
| Tokenizer | in the pack / GGUF | in the pack / GGUF |

The exact block layouts and aliases are in [`docs/CONTAINER-NAMES.md`](docs/CONTAINER-NAMES.md).
Numeric type IDs and block geometry are part of the artifact; a display rename never changes them.

## What an engine must implement

1. **The expert container.** Decode the ternary blocks and dot them against a Q8 activation
   (Q8_K for the 176B's `TQ1_0_g128`; the 35B uses its own Q8 path).
2. **The embedded corrections.** At the three block outputs:
   - `blk.N.ffn_moe_out.weight.lora_{a,b}` — the MoE block output, from the normalized FFN input;
   - `blk.N.ssm_out.weight.lora_{a,b}` — the GDN output projection, from its input;
   - `blk.N.attn_output.weight.lora_{a,b}` — the attention output, from the gated attention.
   `out += B (A x)`, A `[rank, n_in]`, B `[n_out, rank]`, both raw F16, scale 1.0.
   Decode, chunk-major prefill and verify must apply the same branch in the same place.
3. **The PLE table.** Q4_0 rows of 160 values (90 B), read row by row; the direct reader is
   preferred, and a mapped reader is acceptable if it honours the same row stride.
4. **Qwen4exp / FlashNext architecture support.** The 176B is `qwen4exp`; the 35B is
   `qwen35moe`. The engine owns the architecture, not Scion.

## Loaders

- **Reference loader:** the Prism `qwen4exp` llama.cpp fork. This is what the release was
  exported against and what the codec/parity tests in `moe/` target.
- **Optional loader:** Strata, through the `integrations/strata/` layer. Strata is one runtime,
  not a dependency of the model.

## Testing without a runtime

The core tests cover the codecs, the export and the container checks, and run with only Python
and PyTorch:

```sh
../hivebench/.venv/bin/python -m pytest -q moe/tests
```

They do not import Strata, do not need a Strata checkout, and do not need a GPU. Tests that do
need a served engine live under `integrations/strata/` and are optional.
