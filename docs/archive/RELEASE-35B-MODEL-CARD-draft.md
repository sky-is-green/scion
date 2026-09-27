# Qwen3.8-35B-A3B-Distill — ternary PQ2_0 + trained corrections (single-file GGUF)

Draft model card for the downloadable artifact.  Base: `empero-ai/Qwen3.8-35B-A3B-Distill`
(Apache-2.0), a Qwen3.8 distill into the Qwen3.6-35B-A3B MoE architecture
(`qwen35moe`, 40 layers, 256 experts, 8 routed/token, ~3B active, 262k context).
Confirm the upload name before publishing.

## Files

| file | size | sha256 | what |
|---|---|---|---|
| `qwen35-release.gguf` | 10.558 GiB (11.337 GB) | `545d83a93c6ad1474a97201a687f9a344df3e2d5f89ae7cc324aa3680be58d9c` | **the release**: ternary expert banks + Q8_0 rest + trained corrections, one file, no `--lora` |

Two-file variant (same weights, if you want to swap corrections at runtime):

| file | size | sha256 | what |
|---|---|---|---|
| `qwen35-body-pq2_0-q8rest.gguf` | 10.46 GiB | `6cede33aa00968dd78a4422fa49f55cf5f940e4fc2b95806a1bed686ff02551b` | uncorrected body (PQ2_0 experts, Q8_0 rest) |
| `qwen35-adapter.lora-soup.gguf` | 98 MiB | `8ee37faa567373c2e2bed24e8c2c92837f6c54a34602a08d564ee33d8ee1c665` | the shipped corrections (5-way same-run soup) |
| `qwen35-adapter.lora-final.gguf` | 98 MiB | `97548355e5bf37d4a4115c725bdb34b913781c89bf8c9bd6218b0a4a3d7d3145` | final-checkpoint corrections (slightly worse) |

Full checksums: `RELEASE-35B-SHA256SUMS.txt`.

## Method

1. Expert banks quantized to ternary **PQ2_0** (Lloyd–Max g128, `{−s,0,+s}`),
   every other quantizable weight to **Q8_0**, norms/routers/output f32 —
   2.61 bpw effective for the text path, ~2× smaller than Q4_K_M.
2. Trained rank-512 low-rank **corrections** on the attention output and the MoE
   block output, plus router deltas (Lloyd g128 banks, KD to the FP teacher's
   top-50 logits, kd 1.0 / temp 2.0, 4096 steps, 1 epoch, no LR schedule).
   Same-run checkpoint soup (SWA) shipped.
3. The corrections are **merged into the body GGUF** (`adapter.embedded=true`);
   the fork attaches them at load, so the release is a single file that runs
   with **no `--lora`**.

Toolchain: fork `sky-is-green/prism-ml-llama.cpp` branch `moe-corr-runtime`
@ `aa96b8c12`; harness `sky-is-green/bonsai2-ternary-forensics` @ `6a77019`
(`moe/merge_adapter_into_body.py`, `moe/box-run.sh`).

## Numbers (all measured)

Runtime, `wiki.test.raw`, `-c 512`, 580 chunks (local RX 7900 XT):

| build | PPL |
|---|---|
| uncorrected body | 11.6006 |
| + final corrections | 8.4113 |
| **+ soup corrections (shipped)** | **8.3539** |
| BF16 reference | 7.1595 |

Task retention (400 tasks each; BF16 reference in brackets):
**HellaSwag 79.00%** (BF16 81.25, Q4_K_M 80.00) ·
**Winogrande 76.25%** (BF16 76.00, Q4_K_M 76.00).

Distributional fidelity vs BF16 (KLD, 50 chunks): mean **0.2694**, median
0.1346, 99.9% 4.7579, max 7.2432.  Q4_K_M: 0.0314 / 1.141.

Full grid vs every quant of this model: `QUANT-RETENTION-35B.md`
(IQ2_M … Q8_0, BF16).  Summary: **task-level Q4-class at ~half the size, best
PPL of the 2-bit class, KLD tail is the known gap.**

## Usage

Needs the fork build (PQ2_0 + embedded-adapter support; stock llama.cpp cannot
load it):

```bash
git clone -b moe-corr-runtime https://github.com/sky-is-green/prism-ml-llama.cpp
cd prism-ml-llama.cpp && cmake -B build -DGGML_HIP=ON && cmake --build build -j --target llama-cli llama-server

# chat (reasoning model: answers open with a  thinking block)
./build/bin/llama-cli -m qwen35-release.gguf -ngl 99 --temp 0.6 --top-p 0.95 --top-k 20 -n 16384

# server
./build/bin/llama-server -m qwen35-release.gguf -ngl 99 --port 8080
```

Sampling per the base model card: `temperature=0.6, top_p=0.95, top_k=20`.
Text-only file (the vision tower is not included); it fits a 12 GB card for the
weights, ~16 GB comfortable.

## Caveats

- **KLD tail**: on distributional fidelity the release is a strong 2-bit-class
  model, not Q4 — see the grid.  The corrections improve mean likelihood more
  than the tail; tail-aware training is the planned v2 lever.
- **Text-only**: no vision tensors in the file (base card's vision path not
  evaluated).
- **Reasoning distill**: long ` thinking` spans; allow generous `max_new_tokens`.
- Corrections were trained on 50M chars of fineweb; the KLD reference is
  wikitext-2, so the comparison flatters the imatrix-calibrated peers slightly
  (Unsloth's own calibration caveat applies).

## Provenance

- Base weights: `empero-ai/Qwen3.8-35B-A3B-Distill` (Apache-2.0).
- BF16 GGUF used to build the body: `MrFuzzihead/Qwen3.8-35B-A3B-Distill-APEX-GGUF`,
  sha256 `ecbe9e21…666660` (verified equal, tensor-for-tensor, to the empero BF16
  reference used for KLD).
- Rental evidence and costs: `RENTAL-RESULTS-35B.md` ($7.26, H100 SXM, EUR-IS-3).
- License: **Apache-2.0**, inherited from the base.
