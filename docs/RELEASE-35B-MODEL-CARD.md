> **Snapshot note.** This is the published card for **Scion-35B-A3B** on
> Hugging Face (<https://huggingface.co/SkyIsNotGreen/Scion-35B-A3B>), kept
> here for provenance. Its "Our toolchain" line credits the harness as
> `sky-is-green/bonsai2-ternary-forensics`; the harness now lives in this
> repository (`moe/`, docs below) — the dense forensics stayed there.

---
library_name: llama.cpp
model_name: Scion-35B-A3B
base_model: empero-ai/Qwen3.8-35B-A3B-Distill
base_model_relation: quantized
license: apache-2.0
language:
- en
pipeline_tag: text-generation
tags:
- gguf
- llama.cpp
- moe
- qwen3.8
- ternary
- pq2_0
- quantized
- scion
---

# Scion-35B-A3B — ternary experts + trained corrections (single-file GGUF)

**Scion** is the release name: a scion is the shoot you graft onto a rootstock —
here the trained corrections are grafted onto a 2.125 bpw ternary body.

- 35B-A3B MoE distill — base: [empero-ai/Qwen3.8-35B-A3B-Distill](https://huggingface.co/empero-ai/Qwen3.8-35B-A3B-Distill) (Apache-2.0, verified), a Qwen3.8-line distill built on [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (`qwen3_5_moe`; llama.cpp arch `qwen35moe`; 40 layers, 256 experts, 8 routed/token, ~3B active, 262k context).
- Expert banks quantized to ternary **PQ2_0** (Lloyd–Max g128, {−s, 0, +s}); every other quantizable weight **Q8_0**; norms/routers/output f32 → **2.61 bpw** effective, ~2× smaller than Q4_K_M.
- Rank-512 low-rank **corrections** on the attention output and the MoE block output, plus router deltas — trained by output-KD against the BF16 teacher **with the deployed quantizer in the loop** (no post-hoc calibration); same-run checkpoint soup, 1 epoch.
- Corrections are **merged into the file** (`adapter.embedded=true`): one file, **no `--lora`**, no sidecar.
- **10.558 GiB / 11.337 GB** — fits a 12 GB card for the weights, ~16 GB comfortable.

![Scion-35B-A3B vs every community quant of the same model + BF16](./retention-grid.png)

*Retention grid: file size (left = smaller) vs quality (**up = better** — the PPL and KLD axes are inverted so the best end is up). Same protocol for every row: wikitext-2 PPL (`c512`, 580 chunks), KLD vs BF16 (50 chunks), HellaSwag + Winogrande 400 tasks.*

## Files

| file | size | sha256 |
|---|---|---|
| `Scion-35B-A3B-PQ2_0-corr.gguf` | 10.558 GiB (11.337 GB) | `545d83a93c6ad1474a97201a687f9a344df3e2d5f89ae7cc324aa3680be58d9c` |

## Numbers (all measured)

Runtime, `wiki.test.raw`, `-c 512`, 580 chunks (local RX 7900 XT):

| build | PPL |
|---|---|
| uncorrected ternary body | 11.6006 |
| + final corrections | 8.4113 |
| **+ soup corrections (shipped)** | **8.3539** |
| BF16 reference | 7.1595 |

Task retention (400 tasks each; BF16 in brackets): **HellaSwag 79.00%** (BF16 81.25, Q4_K_M 80.00) · **Winogrande 76.25%** (BF16 76.00, Q4_K_M 76.00).

Distributional fidelity vs BF16 (KLD, 50 chunks): mean **0.2694**, median 0.1346, 99.9% 4.7579, max 7.2432 (Q4_K_M: 0.0314 / 1.141).

### vs every quant of this model (same protocol)

| model | GB | bpw | PPL | KLD mean | HellaSwag | Winogrande |
|---|---|---|---|---|---|---|
| **Scion-35B-A3B (this)** | **11.34** | **2.61** | **8.354** | **0.269** | **79.00** | **76.25** |
| IQ2_M | 12.56 | 2.90 | 8.413 | 0.164 | 79.00 | 75.75 |
| Q2_K | 13.84 | 3.19 | 8.473 | 0.149 | 76.50 | 73.25 |
| IQ3_M | 16.34 | 3.77 | 7.520 | 0.057 | 80.00 | 74.25 |
| Q3_K_M | 17.66 | 4.07 | 7.432 | 0.058 | 79.75 | 76.00 |
| IQ4_XS | 19.63 | 4.53 | 7.264 | 0.022 | 80.75 | 75.75 |
| Q4_K_M | 21.71 | 5.01 | 7.235 | 0.031 | 80.00 | 76.00 |
| Q5_K_M | 25.35 | 5.84 | 7.273 | 0.015 | 80.50 | 76.25 |
| Q6_K | 29.21 | 6.73 | 7.153 | 0.008 | 80.75 | 74.75 |
| Q8_0 | 37.80 | 8.72 | 7.160 | 0.004 | 80.25 | 75.50 |
| BF16 (reference) | 71.07 | 16.38 | 7.160 | — | 81.25 | 76.00 |

Summary: **task-level Q4-class at ~half the size, best PPL of the 2-bit class; the KLD tail is the known gap.**

## Usage

Requires a llama.cpp build with the PQ2_0 container and embedded adapters (stock llama.cpp cannot load this file):

```bash
git clone -b moe-corr-runtime https://github.com/sky-is-green/prism-ml-llama.cpp
cd prism-ml-llama.cpp && cmake -B build -DGGML_CUDA=ON && cmake --build build -j --target llama-cli llama-server
# (use -DGGML_HIP=ON for ROCm)

# chat — reasoning model: answers open with a  thinking block
./build/bin/llama-cli -m Scion-35B-A3B-PQ2_0-corr.gguf -ngl 99 --temp 0.6 --top-p 0.95 --top-k 20 -n 16384

# server
./build/bin/llama-server -m Scion-35B-A3B-PQ2_0-corr.gguf -ngl 99 --port 8080
```

Verified on CPU and ROCm (gfx1100). CUDA builds are expected to work — reports welcome. Sampling per the base model card (`temperature=0.6, top_p=0.95, top_k=20`); allow generous `max_new_tokens` given the long reasoning spans.

## Caveats

- **KLD tail**: on distributional fidelity this is a strong 2-bit-class model, not Q4 — see the grid. The corrections improve mean likelihood more than the tail; tail-aware training is the next lever.
- Corrections were trained on 50M chars of fineweb; the KLD reference is wikitext-2, which slightly flatters the imatrix-calibrated peers (they are calibrated on Wikipedia-like data).
- **Text-only**: no vision tensors in the file.
- **Reasoning distill**: long ` thinking` spans.
- Task numbers are 400-task zero-shot runs (~±2% CI); treat sub-1% differences as ties.

## Provenance & credits

- Base weights: [empero-ai/Qwen3.8-35B-A3B-Distill](https://huggingface.co/empero-ai/Qwen3.8-35B-A3B-Distill) — **Apache-2.0 verified** (Hub cardData + tag, 2026-09-27); its declared base is [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (Apache-2.0), distilled from the Qwen3.8 line.
- BF16 GGUF used to build the body: [MrFuzzihead/Qwen3.8-35B-A3B-Distill-APEX-GGUF](https://huggingface.co/MrFuzzihead/Qwen3.8-35B-A3B-Distill-APEX-GGUF) (verified tensor-for-tensor against the empero BF16 reference used for KLD).
- Container + kernels: [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp) (`prism` branch, MIT) — PQ2_0 comes from there; the ternary-expert approach was validated publicly by Bonsai 2. We used the engine work, not Bonsai weights.
- Legacy container + adapter conventions: TAARDIS fork (`CodeMasterCody3D/prism-ml-llama.cpp`, MIT).
- Our toolchain: [sky-is-green/prism-ml-llama.cpp](https://github.com/sky-is-green/prism-ml-llama.cpp) branch `moe-corr-runtime` @ `aa96b8c12`; training harness `sky-is-green/bonsai2-ternary-forensics` (Apache-2.0).

## License & attribution

Apache-2.0, inherited from the base model — see `LICENSE` and `NOTICE`. Not affiliated with, endorsed by, or supported by Prism ML, empero-ai, or Alibaba Cloud.
