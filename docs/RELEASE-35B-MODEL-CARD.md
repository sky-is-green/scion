> **Snapshot note.** This is the model card as published on
> [Hugging Face](https://huggingface.co/SkyIsNotGreen/Scion-35B-A3B), kept here
> for provenance. The harness it references lives in this repository (`moe/`);
> the dense forensics stayed in
> [`bonsai2-ternary-forensics`](https://github.com/sky-is-green/bonsai2-ternary-forensics).
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
- ternary
- 2-bit
- pq2_0
- gguf
- llama-cpp
- moe
- qwen3.8
- quantized
- scion
---

<p align="center">
  <a href="https://github.com/sky-is-green/scion"><b>GitHub: harness and docs</b></a> &nbsp;|&nbsp;
  <a href="https://github.com/sky-is-green/bonsai2-ternary-forensics"><b>Forensics study</b></a> &nbsp;|&nbsp;
  <a href="https://github.com/sky-is-green/prism-ml-llama.cpp/tree/moe-corr-runtime"><b>Runtime fork</b></a> &nbsp;|&nbsp;
  <a href="https://huggingface.co/SkyIsNotGreen/Scion-35B-A3B/discussions"><b>Discussions</b></a>
</p>

# Scion-35B-A3B: ternary MoE experts + trained corrections

A full 35B-A3B MoE in one 11.3 GB GGUF for llama.cpp. Ternary expert banks plus
small trained corrections, with no full-precision masters and no full-model QAT.

> **2.61 bpw** | **11.34 GB** (6.3x smaller than the BF16 reference) | **best PPL of the 2-bit class** | **task retention at Q4-class level, at about half of Q4_K_M's size**

The name comes from grafting. A scion is the shoot grafted onto a rootstock, and
here the trained corrections are grafted onto a 2.125 bpw ternary body.

## Highlights

- **One 11.34 GB file at 2.61 bpw** (expert banks at 2.125 bpw). The BF16 reference of the same model is 71.07 GB; Q4_K_M is 21.71 GB and IQ2_M is 12.56 GB. It is the smallest published build of this model I have seen at this quality level.
- **Task retention inside the Q4 and BF16 noise band**: HellaSwag 400 gives **79.00%** (BF16 81.25, Q4_K_M 80.00) and Winogrande gives **76.25%** (BF16 76.00, Q4_K_M 76.00). It has the joint-best Winogrande row in the table and the **best PPL of the 2-bit class** (8.354 against IQ2_M 8.413 and Q2_K 8.473).
- **Trained, not calibrated**: rank-512 correction branches on the attention output and the MoE block output, plus router deltas, trained by output-KD against the BF16 teacher with the deployed quantizer in the loop (ternary Lloyd g128). No imatrix and no calibration corpus, which is what separates this build from the imatrix-calibrated quants on the chart.
- **One file, no `--lora`**: the corrections are embedded (`adapter.embedded=true`) and attached at load. There is no sidecar and no adapter plumbing.
- **The gap is stated, not hidden**: full-vocabulary KLD against BF16 is 0.269 mean, a strong 2-bit-class result but still behind Q4_K_M at 0.031. Closing the distributional tail is the active work ([`TAIL-EXPERIMENT-PLAN.md`](https://github.com/sky-is-green/scion/blob/main/docs/TAIL-EXPERIMENT-PLAN.md)).

## Resources

- **[GitHub `sky-is-green/scion`](https://github.com/sky-is-green/scion)**: the source of truth for this work. It carries the harness that produced the file, the full MoE write-up, the port decisions and the negative register.
- **[Bonsai 2 ternary forensics](https://github.com/sky-is-green/bonsai2-ternary-forensics)**: the dense-model study this method grew out of (format recovery, the trained-weight residual, the calibration-artifact result).
- **Runtime**: [`sky-is-green/prism-ml-llama.cpp`](https://github.com/sky-is-green/prism-ml-llama.cpp/tree/moe-corr-runtime), branch `moe-corr-runtime`, a fork of Prism ML's llama.cpp with the PQ2_0 container, the `ffn_moe_out` virtual target and embedded-adapter support.
- **[Retention grid](https://github.com/sky-is-green/scion/blob/main/retention-grid.png)**: this release against every community quant of the same model, same protocol.
- **[Discussions](https://huggingface.co/SkyIsNotGreen/Scion-35B-A3B/discussions)**: questions, test reports and failures are all welcome.

## Model Overview

| Item | Specification |
| :--- | :--- |
| Base model | [`empero-ai/Qwen3.8-35B-A3B-Distill`](https://huggingface.co/empero-ai/Qwen3.8-35B-A3B-Distill), a Qwen3.8-line reasoning distill built on [`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (Apache-2.0) |
| Parameters | 34.9B total, about 3B active per token (256 experts, top-8 plus shared) |
| Architecture | `qwen3_5_moe` (`qwen35moe` in llama.cpp): 40 layers, hybrid linear and full attention, MoE feed-forward |
| Context length | 262,144 tokens (inherited from the base model) |
| Weight format | Ternary **PQ2_0** g128 expert banks (codes in {-1, 0, +1} plus one fp16 group scale per 128 weights), **Q8_0** for the rest, embedded corrections in the legacy `q1_0_g128` container (rank-512) |
| Low-bit coverage | Expert banks only; attention, embeddings and the head stay Q8_0, and norms, routers and the output stay F32 |
| Deployed size | **10.558 GiB / 11.337 GB** (text only, single file) |
| Backends | llama.cpp fork; verified on CPU and ROCm/gfx1100, CUDA build expected to work but untested |
| License | Apache-2.0 (inherited from the base) |

## Weight Representation: ternary PQ2_0 + trained corrections

Each expert weight takes a value from {-1, 0, +1} with one shared FP16 scale per
group of 128 weights: 2-bit slots at 2.125 bits per weight. The rest of the
model is Q8_0 (attention, embeddings, LM head) and F32 (norms, routers, output).
The corrections are rank-512 low-rank branches on the attention output and the
MoE block output, plus exact router deltas. They ship in the compact legacy
`q1_0_g128` container (2-bit codes plus one fp16 group scale per 128) and are
merged into the file. Effective overall: **2.61 bpw**.

### Memory Requirement

| Format | bpw | Size | vs BF16 |
| :--- | ---: | ---: | ---: |
| BF16 (reference) | 16.38 | 71.07 GB | 1.0x |
| Q8_0 | 8.72 | 37.80 GB | 1.9x |
| Q4_K_M | 5.01 | 21.71 GB | 3.3x |
| IQ2_M | 2.90 | 12.56 GB | 5.7x |
| **Scion-35B-A3B** | **2.61** | **11.34 GB** | **6.3x** |

Sizes are the published on-disk files of the same model, measured under one
protocol (wikitext-2 PPL, KLD against BF16, HellaSwag and Winogrande 400; see
Benchmarks).

### Shipped Components

| Component | Pack | Size | Residency |
| :--- | :--- | ---: | :--- |
| Language model (this repo) | PQ2_0 experts, Q8_0 rest, embedded corrections | 11.34 GB | resident; the whole model |
| Uncorrected body (not uploaded) | PQ2_0 experts and Q8_0 rest | 10.46 GiB | for swap tests |
| Corrections (not uploaded) | rank-512 branches and router deltas (`q1_0_g128`) | 98 MiB | for swap tests |

Only the single file is published. The two-file variant (body plus separate
adapter) exists for reproducing the merge and swapping corrections at runtime.
Ask in Discussions if you want it.

> **About the Hub's quant chip.** The Hub parses file names and labels this file
> `Q2_0`; the same happens on Prism ML's own `PQ2_0` releases. The container is
> Prism's **`PQ2_0`** (legacy name `Q1_0_g128`, type 142/43, identical byte
> layout): 2-bit codes with one fp16 group scale per 128 weights. It is not
> upstream llama.cpp's g64 `Q2_0` (type 42). No single quant name fits the file
> anyway, because it is a mix: `PQ2_0` expert banks, the embedded corrections in
> the legacy `q1_0_g128` container, and `Q8_0` for the rest (norms and routers
> in F32).

## Best Practices

### Generation Parameters

Recommended values, from the base model card:

> - `temperature=0.6`, `top_p=0.95`, `top_k=20`

This is a reasoning distill, and answers open with a long thinking segment.
Allow generous `max_new_tokens` (for example `-n 16384`); a small cap ends
generation mid-thought, before any answer.

### System Prompt

A simple prompt works, for example `You are a helpful assistant`. The base is a
reasoning SFT distill and does not require a special system prompt.

### Choosing Context and Offload

- The weights fit a 12 GB card; about 16 GB is comfortable with context.
- `-ngl 99` on a single card. When VRAM is tight, offload experts to CPU with
  `-ncmoe`, which is the VRAM-budget dial; the ternary container roughly halves
  the CPU-tail penalty compared to an f16 expert bank.
- **Threads should equal physical cores** (`-t 8` on an 8C/16T CPU; SMT siblings
  collapse CPU expert throughput).
- Do not layer-split across two cards when one card fits; the proxy
  measurements showed a 40% generation loss.

## Quickstart

> The runtime is the fork, and the fork is the source of truth for running these
> files: [`sky-is-green/prism-ml-llama.cpp`](https://github.com/sky-is-green/prism-ml-llama.cpp/tree/moe-corr-runtime),
> branch `moe-corr-runtime`.

### These files need the fork build

The PQ2_0 container, the legacy `Q1_0_g128` import, the `ffn_moe_out` virtual
target and embedded adapters all live in the fork. **Stock llama.cpp will not
run this file**: it treats `PQ2_0` and `Q1_0_g128` as unknown tensor types.
Upstream's own `Q2_0` (type 42, g64) is a different container and is not a
substitute.

```bash
# build the fork
git clone -b moe-corr-runtime https://github.com/sky-is-green/prism-ml-llama.cpp
cd prism-ml-llama.cpp
cmake -B build -DGGML_CUDA=ON && cmake --build build -j --target llama-cli llama-server
# ROCm: -DGGML_HIP=ON     CPU-only: no flag
```

```bash
# fetch the weights
hf download SkyIsNotGreen/Scion-35B-A3B Scion-35B-A3B-PQ2_0-corr.gguf --local-dir .
```

```bash
# chat; the model thinks by default, so leave room for the trace
./build/bin/llama-cli -m Scion-35B-A3B-PQ2_0-corr.gguf \
    -ngl 99 -c 4096 -t 8 \
    --temp 0.6 --top-p 0.95 --top-k 20 \
    -p "Explain quantum computing in simple terms." -n 16384

# server
./build/bin/llama-server -m Scion-35B-A3B-PQ2_0-corr.gguf -ngl 99 -c 4096 -t 8 --port 8080
```

`-ngl 99` offloads every layer (`0` is CPU-only), `-c` sets context up to
262144, and `-t` should be the physical core count. Verified on CPU and ROCm
(gfx1100, RX 7900 XT, where the release was built and measured). A CUDA build is
expected to work; reports are welcome in Discussions.

## Benchmarks

Community protocol, the same for every row: wikitext-2 PPL (`c512`, 580 chunks);
KLD against BF16 logits over 50 chunks (25.5k tokens); HellaSwag 400 and
Winogrande 400 zero-shot (about ±2% CI). BF16 and all quants were measured in one
H100 session; this release was measured on the local card and cross-checked
against the pod (IQ2_M KLD local vs pod: 0.2%).

### Task Retention (400 tasks each)

| Variant | Size | bpw | HellaSwag | Winogrande |
| :--- | ---: | ---: | ---: | ---: |
| BF16 (reference) | 71.07 GB | 16.38 | 81.25 | 76.00 |
| Q4_K_M | 21.71 GB | 5.01 | 80.00 | 76.00 |
| IQ2_M | 12.56 GB | 2.90 | 79.00 | 75.75 |
| Q2_K | 13.84 GB | 3.19 | 76.50 | 73.25 |
| **Scion-35B-A3B** | **11.34 GB** | **2.61** | **79.00** | **76.25** |

Within ±2% noise of Q4_K_M and BF16; tied with IQ2_M on HellaSwag and ahead of
it on Winogrande at 1.2 GB less. Treat sub-1% differences as ties.

### Distributional Fidelity: the known gap

| Variant | PPL (lower better) | KLD mean vs BF16 (lower better) | KLD 99.9% |
| :--- | ---: | ---: | ---: |
| BF16 | 7.1595 | — | — |
| Q4_K_M | 7.2354 | 0.0314 | 1.141 |
| IQ2_M | 8.4133 | 0.1636 | 3.963 |
| Q2_K | 8.4729 | 0.1493 | 3.276 |
| **Scion-35B-A3B** | **8.3539** | **0.2694** | **4.758** |

The corrections improved mean token likelihood (PPL 11.60 uncorrected, 8.35
after) more than they improved the full-distribution tail. PPL ranks this build
first of the 2-bit class and KLD ranks it last, and that disagreement is the
research result: the training matches the teacher's top-50 logits, and the rest
of the distribution is unconstrained. Tail-aware training is the next step, not
a claim.

### Full Grid

![Scion-35B-A3B against every community quant of the same model, plus BF16](./retention-grid.png)

All ten community quants plus BF16; up is better, left is smaller. Full table
and method notes:
[`QUANT-RETENTION-35B.md`](https://github.com/sky-is-green/scion/blob/main/docs/QUANT-RETENTION-35B.md).

<details>
<summary>Full per-variant table</summary>

| model | size GB | bpw | PPL | KLD mean | KLD 99.9% | HellaSwag | Winogrande |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Scion-35B-A3B** | **11.34** | **2.61** | **8.354** | **0.269** | **4.758** | **79.00** | **76.25** |
| IQ2_M | 12.56 | 2.90 | 8.413 | 0.164 | 3.963 | 79.00 | 75.75 |
| Q2_K | 13.84 | 3.19 | 8.473 | 0.149 | 3.276 | 76.50 | 73.25 |
| IQ3_M | 16.34 | 3.77 | 7.520 | 0.057 | 1.225 | 80.00 | 74.25 |
| Q3_K_M | 17.66 | 4.07 | 7.432 | 0.058 | 1.611 | 79.75 | 76.00 |
| IQ4_XS | 19.63 | 4.53 | 7.264 | 0.022 | 0.636 | 80.75 | 75.75 |
| Q4_K_M | 21.71 | 5.01 | 7.235 | 0.031 | 1.141 | 80.00 | 76.00 |
| Q5_K_M | 25.35 | 5.84 | 7.273 | 0.015 | 0.717 | 80.50 | 76.25 |
| Q6_K | 29.21 | 6.73 | 7.153 | 0.008 | 0.360 | 80.75 | 74.75 |
| Q8_0 | 37.80 | 8.72 | 7.160 | 0.004 | 0.209 | 80.25 | 75.50 |
| BF16 | 71.07 | 16.38 | 7.160 | — | — | 81.25 | 76.00 |

</details>

## Use Cases

- **A 35B-A3B on one consumer GPU**: 11.3 GB of weights fit a 20 GB card with
  room for context, and can be tiered further with CPU expert offload.
- **Low-bit research and testing**: a reference point for "ternary experts plus
  trained corrections, no full-precision masters, no full-model QAT". The
  harness and every negative result are public.
- **Local-first serving**: the model was built for an offline assistant stack,
  so a single file with embedded corrections keeps deployment to one download
  and one set of flags.

## Limitations

- **KLD tail** (stated above): distributional fidelity is strong-2-bit, not
  Q4-class. PPL, HellaSwag and Winogrande look Q4-class; KLD is where the gap
  lives.
- **Text only**: the base model has a vision tower, and this file carries no
  vision tensors (Q8_0 language path only).
- **Reasoning distill**: long thinking traces; budget `max_new_tokens`
  accordingly.
- **Protocol caveats**: the KLD figure is 50 chunks; task numbers are 400-task
  runs (about ±2% CI); the 2-bit and 3-bit competitors are imatrix-calibrated on
  Wikipedia-like data, which flatters their wikitext KLD, and the corrections
  were trained on fineweb.
- **Platform coverage**: CPU and ROCm (gfx1100) verified. CUDA is expected to
  work but has not been tested, and Metal is untested.
- **The Hub chip says `Q2_0`**: it is filename-derived; see the note above, and
  use the fork.
- **Not affiliated** with Prism ML, empero-ai, or Alibaba Cloud. It builds on
  Prism ML's engine work (fork and PQ2_0 container) and community GGUF
  conversions.

## Citation

```bibtex
@misc{scion35b2026,
    title  = {Scion-35B-A3B: ternary MoE experts with trained corrections},
    author = {SkyIsNotGreen},
    year   = {2026},
    month  = {September},
    url    = {https://huggingface.co/SkyIsNotGreen/Scion-35B-A3B}
}
```

## License and Attribution

Apache-2.0, inherited from the base model; see [`LICENSE`](./LICENSE) and
[`NOTICE`](./NOTICE).
Base weights: [empero-ai](https://huggingface.co/empero-ai/Qwen3.8-35B-A3B-Distill) and
[Qwen / Alibaba Cloud](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (Apache-2.0);
BF16 conversion: [MrFuzzihead](https://huggingface.co/MrFuzzihead/Qwen3.8-35B-A3B-Distill-APEX-GGUF);
container and kernels: [Prism ML](https://github.com/PrismML-Eng/llama.cpp) (MIT) with
[TAARDIS](https://github.com/CodeMasterCody3D/prism-ml-llama.cpp) conventions (MIT);
engine: [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT).
