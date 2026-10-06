# Scion: ternary quantization with trained corrections

Scion is the project behind the released
[**Scion-35B-A3B**](https://huggingface.co/SkyIsNotGreen/Scion-35B-A3B) and
[**Scion-FlashNext-176B-A6B**](https://huggingface.co/SkyIsNotGreen/Scion-FlashNext-176B-A6B),
split out of
[bonsai2-ternary-forensics](https://github.com/sky-is-green/bonsai2-ternary-forensics)
so the harness, the port decisions and the release docs can be cloned and run on
their own. The dense-model forensics (Bonsai 2 27B) stayed in that repository.

**The idea.** Ternarise the expert banks of a pretrained MoE in place, then
recover the routing damage with small trained corrections on the residual
stream. No full-precision masters and no full-model QAT. The frozen body ships
at 2.125 bpw (`Q1_0_g128` / `PQ2_0` ternary experts, Q8_0 for the rest); the
corrections are rank-512 branches on the attention output and the MoE block
output, plus router deltas, trained by output-KD **with the deployed quantizer
in the loop**.

## Where the project is (2026-10)

| lane | status |
|---|---|
| **Scion-35B-A3B** | **released**: 11.34 GB / 2.61 bpw ternary experts + embedded corrections, plus an optional k=1 MTP drafter. Runtime `v0.4.1`. CPU and ROCm verified; CUDA/Windows binaries published but runtime-unverified (see [platform scope](docs/PLATFORM-SCOPE.md)). |
| **Scion-FlashNext-176B-A6B** | **released** (2026-10-05): the 125B MoE (6B activated) + the 51.2B-parameter n-gram table in one 60.17 GB file at 2.72 bpw; PTQ1_0 ternary experts + embedded corrections. CUDA, ROCm and CPU verified; runtime branch `qwen4exp-proto`. |
| **Dense (Clef-Flash)** | **in progress, not released**: the rotated-basis + GPTQ + QAT ladder reaches 23.25 PPL / 5.52 GiB, but free generation still degrades and a plain Q2_K (13.06 PPL, 3.56 GiB) beats it. The 2026-10-06 derisk run (0.8B, 1500 steps, mixed corpus + logits KD) plateaus at 42.4 PPL with generation still broken, so scaling the same recipe is not justified and the lane is on hold. See [`dense/README.md`](dense/README.md). |

Negative results are kept in [`FAILURES.md`](FAILURES.md); runtime support
boundaries and what a backend report must carry are in
[`docs/PLATFORM-SCOPE.md`](docs/PLATFORM-SCOPE.md).

## Reference release: Scion-35B-A3B

`empero-ai/Qwen3.8-35B-A3B-Distill` (a Qwen3.8-line distill on
`Qwen/Qwen3.6-35B-A3B`), ternary experts, Q8_0 rest and embedded corrections:
**11.34 GB / 2.61 bpw** in one file, no `--lora` step.

| metric | **Scion-35B-A3B** | Q4_K_M | BF16 |
|---|---|---|---|
| PPL, wikitext-2, c512, 580 chunks | **8.354** | 7.235 | 7.160 |
| KLD mean vs BF16, 50 chunks | 0.269 | 0.031 | — |
| HellaSwag 400 | 79.00 | 80.00 | 81.25 |
| Winogrande 400 | **76.25** | 76.00 | 76.00 |

Task accuracy is inside the ±2% noise band, PPL is the best of the 2-bit class,
and the KLD tail is the known gap and the current work item
(`docs/TAIL-EXPERIMENT-PLAN.md`). The full table against every community quant
of the same model is in `docs/QUANT-RETENTION-35B.md`.

![retention grid](retention-grid.png)

The grid shows file size (left is smaller) against quality, with the PPL and KLD
axes inverted so the better end is up. Every row uses the same protocol.

Card and weights:
[`SkyIsNotGreen/Scion-35B-A3B`](https://huggingface.co/SkyIsNotGreen/Scion-35B-A3B).
A snapshot of the card is kept at `docs/RELEASE-35B-MODEL-CARD.md`.

Optional speedup: a 50 MB k=1 **MTP drafter** ships at
[`SkyIsNotGreen/Scion-35B-A3B-mtp-drafter`](https://huggingface.co/SkyIsNotGreen/Scion-35B-A3B-mtp-drafter).
It drafts the next token from the model's own hidden state and gives about
1.2x generation (up to 1.37x on the bench prompt) with unchanged outputs.

## Release: Scion-FlashNext-176B-A6B

`Qwen/Qwen3.8-Flash-Next-FP8` quantized to **2.72 bpw**: the 125B language model
(6B activated per token, 512 experts top-10) plus the 51.2B-parameter n-gram
table in a single file. Ternary **PTQ1_0** expert banks with embedded
corrections; body Q6_K+Q8_0 (default) or F16; 60.17 GB and 65.80 GB variants.

| metric (one harness) | **Scion-FlashNext** | ISTA GSQ-RCO Q2_0 |
|---|---|---|
| HellaSwag 400 | **82.00** | 81.50 |
| Winogrande 400 | **77.50** | 74.75 |
| wikitext-2 PPL | 5.457 | **5.240** |

Card and weights:
[`SkyIsNotGreen/Scion-FlashNext-176B-A6B`](https://huggingface.co/SkyIsNotGreen/Scion-FlashNext-176B-A6B);
snapshot at `docs/RELEASE-FLASHNEXT-176B-A6B-MODEL-CARD.md`. It needs the
runtime fork's `qwen4exp-proto` branch, not stock llama.cpp.

## What the experiments established

- routing drift compounds and is the dominant failure mode (`docs/MOE-EXTENSION.md` §2.1);
- rotation is not the missing lever for MoE (§2.2);
- in-place ternary QAT on its own does not recover it at local budgets (§2.3);
- corrections must live on the **residual stream**, not inside experts; that is the placement rule (§2.4);
- router-KD is a verified no-op, and the correction repairs the state the router reads instead (§2.4a);
- train the sidecar in the deployed format, because post-hoc ternarisation is 14 to 25 times worse (§2.4c);
- the deployable Lloyd quantizer beats one-shot absmean (§2.4d);
- `attn_out` ships as a standard llama.cpp LoRA, and the better `moe_out` placement needs the ~90-line fork extension (§2.4e).

An earlier proof on the frozen proxy `allenai/OLMoE-1B-7B-0924`: the uncorrected
mixed GGUF measured **566.7 PPL, and 14.48** with both placements and routers
(2.12 GB body, 22.2 MB adapter), with CPU and HIP builds agreeing.

## Dense route (2026-10-03)

The recipe above repairs MoE routing damage. A dense variant is now open, with
Cloudflare's **Clef-Flash** (Qwen3.5-9B + joint schema head) as the test bed:
ternary body via `GGML_PQ2_0_LLOYD=1`, residual-stream corrections re-derived
for a dense/hybrid stack, and a typed-decision metric (accept/reject parity
against bf16) that the MoE PPL/KLD work did not have. Measured so far: absmax
is broken (hidden-state cosine 0.007), Lloyd alone gives 0.417, and uncorrected
decisions shift down ~0.43 - corrections are load-bearing here too. Plan:
[`docs/DENSE-TERNARY-QAT.md`](docs/DENSE-TERNARY-QAT.md). The work is a
postable ternary Clef community quant; the ladder so far (f16 = 12.59 PPL):
deployed 8684 -> signed RTN 476 -> GPTQ+act-order 141.6 -> 128-w Hessians 120.6
-> +`ffn_down` F16 75.2 -> **QAT 23.25** (5.52 GiB). The 2026-10-06 community
gate found a plain Q2_K at 13.06 PPL / 3.56 GiB beats the ternary artifact, so
nothing is released; the follow-up derisk run (0.8B, mixed corpus + logits KD,
1500 steps) confirmed free generation does not recover with scale on this
recipe (PPL plateau ~42.4; code/math still degenerate), so the lane is on hold.
Status: [`dense/README.md`](dense/README.md).

## Quickstart

**Run a release.** Both releases need the runtime fork,
[`sky-is-green/prism-ml-llama.cpp`](https://github.com/sky-is-green/prism-ml-llama.cpp).
The 35B loads on the default `moe-corr-runtime` branch; Flash-Next needs
`qwen4exp-proto`. Build once and run:

```bash
git clone https://github.com/sky-is-green/prism-ml-llama.cpp
cd prism-ml-llama.cpp
./verify-container-support.sh          # prints RESULT: OK
cmake -B build -DGGML_CUDA=ON          # or -DGGML_HIP=ON, or no flag for CPU
cmake --build build -j --target llama-cli llama-server

# Scion-35B-A3B with the drafter (branch moe-corr-runtime)
./build/bin/llama-server -m Scion-35B-A3B-PQ2_0-corr.gguf \
    -md Scion-35B-A3B-mtp-drafter.gguf -ngl 99 -c 4096 -t 8 --port 8080
```

Recommended sampling and context notes are on each model card. The fork's
`master` is an untouched upstream mirror and cannot load these files; a load
failure saying `invalid ggml type 142` means the wrong branch was built.

**Reproduce the recipe, locally and free: the OLMoE proxy.** Python 3.10 or
newer, torch (ROCm or CUDA wheels), `transformers`, `datasets`, `numpy` and
`pyyaml`. The run sequence is in [`moe/README.md`](moe/README.md). The in-place
ternary QAT negative control is `moe/olmoe_proxy.py` and the correction trainer
is `moe/olmoe_corrections.py`.

**Paths.** The harness resolves its external paths through `moe/scion_paths.py`
and `dense/clef_paths.py`; every default is an environment variable
(`SCION_WORKSPACE`, `SCION_MODELS`, `LLAMA_BIN`, `GGUF_PY`, ...). Set
`SCION_WORKSPACE` if models and sibling checkouts live outside the repo's
parent directory.

**The 35B path: one 80 GB GPU.** Quantize the deployment body
locally first (`llama-quantize` with the experts mapped to `pq2_0` and
`GGML_PQ2_0_LLOYD=1`), run the box stages, then merge the exported adapter into
the body with `moe/merge_adapter_into_body.py`:

```bash
bash moe/box-run.sh setup && bash moe/box-run.sh smoke && bash moe/box-run.sh cache
bash moe/box-run.sh ref   && bash moe/box-run.sh train && bash moe/box-run.sh eval && bash moe/box-run.sh export
```

Every stage is resumable, and checkpoints and the teacher cache land on the
volume. Port notes: `moe/PORT-QWEN35.md`; the architecture decision record:
`docs/QWEN35-PORT-DECISION.md`.

## Layout

| path | what |
|---|---|
| `moe/` | the harness: proxies, correction trainers, port, box-run scripts, export and bench tooling, result JSONs |
| `dense/` | the dense/hybrid route (Clef-Flash): reverse loader, correction trainer, rotated-basis quantizer, QAT |
| `docs/` | write-ups: the MoE extension, release tables, port decision, tail plan, platform scope, registers |
| `serving/` | serving experiments: placement sweep (ternary vs f16 offload, threads, split modes) and the expert-cache negative result |
| `scion_moe/` | vendored `rotation` and RTN quantizer used by the harness |
| `FAILURES.md` | the negative register for this track (D1-D8) |
| `retention-grid.png` | the release figure (regenerate with `moe/plot_bench.py`, which writes into `moe/results/qwen35-retention/`) |

## Serving notes

- the ternary container roughly **halves the CPU-expert-offload penalty** compared to f16 (2.07 GiB OLMoE: 34% and 50% versus 86% and 75% for prefill and decode);
- **threads should equal physical cores** (`-t 16` collapses generation from 95 to 35 t/s on an 8C/16T CPU);
- do not layer-split when one card fits (40% generation loss);
- `-ncmoe` is the VRAM-budget dial, and placement does not change the math;
- the optional MTP drafter gives about **1.2x generation** on the 35B (up to 1.37x on the bench prompt) with unchanged outputs;
- for the KV cache, `--cache-type-k/v q8_0` is lossless in practice; `q4_0` costs about +0.3% PPL at 4k context on the 35B;
- a hot-expert GPU cache gave **no throughput gain**, because `mul_mat_id` computes k experts per token regardless of the weights. It would need sparse per-token dispatch rather than a static split (`serving/placement-sweep-20260927/THROUGHPUT.md`).

## Provenance

Split from
[`sky-is-green/bonsai2-ternary-forensics`](https://github.com/sky-is-green/bonsai2-ternary-forensics)
at `b01f7e2` (2026-09-27). The dense forensics, the replication ladder and their
failure register stayed there; this track's register is [`FAILURES.md`](FAILURES.md).

## License and attribution

Apache-2.0; see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). Base models:
**empero-ai** (Apache-2.0) and **Qwen / Alibaba Cloud** (Apache-2.0); container
and kernels: **Prism ML**'s llama.cpp fork (MIT) with **TAARDIS** conventions
(MIT); engine: **llama.cpp** (MIT). Not affiliated with, endorsed by, or
supported by Prism ML, empero-ai, or Alibaba Cloud.
