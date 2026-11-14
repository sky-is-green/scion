---
library_name: llama.cpp
base_model: SkyIsNotGreen/Scion-FlashNext-176B-A6B
license: apache-2.0
language:
- en
tags:
- speculative-decoding
- draft-model
- mtp
- gguf
- llama.cpp
- scion
---

# Scion-FlashNext-176B-A6B — native MTP drafter (Strata)

A ~0.89 GB speculative-decoding head for
[**Scion-FlashNext-176B-A6B**](https://huggingface.co/SkyIsNotGreen/Scion-FlashNext-176B-A6B),
the 2-bit ternary release of Qwen3.8-Flash-Next. **It is not a trained drafter:
it is the model's own native MTP block.** Qwen3.8 ships a body-aligned MTP
layer in its BF16 checkpoint, and llama.cpp's converter drops it
(`supports_mtp_export = false`), so it never made it into the released GGUFs.
This repo packs it back for [Strata](https://github.com/sky-is-green/strata),
the project's own engine.

Because the head was trained by Qwen against this same body, no fitting pass
was run — the ternarization preserved the hidden geometry well enough that the
native head transfers at full strength:

| Measurement | Result |
| :--- | :--- |
| Draft acceptance (weighted, 10 mixed greedy prompts, 256 tok each) | **0.741** (1310/1769), per-request 0.640–0.810 |
| Draft acceptance (30 real long-form generations, T 0.6) | 0.62–0.72 per request |
| Decode during those generations (2x RX 7900 XT, Q6_K, `--spec 4`) | 52–78 tok/s (smoke test 21–58 tok/s by prompt/length) |
| Prefill, ~450-token prompts (same setup) | 116–135 tok/s |

Like all speculative decoding it is a **speed** option: the verifier checks
every draft against the target's own distribution, so outputs are unchanged
apart from rare near-tie flips on batched verification (same caveat as the
[35B drafter card](https://huggingface.co/SkyIsNotGreen/Scion-35B-A3B-mtp-drafter)).

## Requirement: the Strata engine

> **This is not a llama.cpp drafter.** The file carries the native full MTP
> layer (attention, indexer, hyper-connections, 512 routed experts) in
> Strata's layout. The fork's `draft-mtp-sidecar` lane takes a different,
> fc-head form and does not load this file.

```console
hf download SkyIsNotGreen/Scion-FlashNext-176B-A6B Scion-FlashNext-176B-A6B-Q6_K.gguf --local-dir .
hf download SkyIsNotGreen/Scion-FlashNext-176B-A6B-mtp-drafter Scion-FlashNext-176B-A6B-mtp-drafter.gguf --local-dir .

# one-time: convert the GGUF to the engine runtime
python tools/mtp_rt.py --gguf Scion-FlashNext-176B-A6B-mtp-drafter.gguf --out mtp/rt

strata --serve --pack <pack-q6k> \
    --native Scion-FlashNext-176B-A6B-Q6_K.gguf \
    --ple-gguf Scion-FlashNext-176B-A6B-Q6_K.gguf \
    --embd-gguf scion-embd-bf16.gguf \
    --expert-profile data/expert-profile.bin --expert-cache auto \
    --prefill 4096 --max-context 8192 --ple-io direct --mmap-experts \
    --vram-reserve-mib 3072 --pool-workers 15 --pcie-frac 0 \
    --pipeline-windows 2 --layer-split auto \
    --mtp mtp/rt --spec 4 --spec-min-p 0.5
```

The native-IQ packs require the MTP lane (`--spec` >= 2), so on Strata this
head is both the speed option and the enabler for serving the ternary 176B on
consumer cards.

## Provenance

The 31 `mtp.*` tensors were fetched from
`Qwen/Qwen3.8-Flash-Next` @ revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540`
by HTTP range request (5.21 GB of the 360 GB checkpoint, 131 shards — only the
named byte ranges), and every tensor was verified against SHA256. The manifest
with per-tensor shard, byte range and hash ships in this repo
(`mtp-manifest.json`). The pack is round-to-nearest Q2_0 for the routed
experts (the same grid format as the body's experts, so Strata's CPU VNNI and
GPU hit kernels serve it unchanged) and BF16 for the dense tensors; the packer's
per-tensor reconstruction report ships as `mtp-q2_0.gguf.report.json`. No
model runs, no distillation, no training: provenance is the checkpoint itself.

This mirrors — in reverse — the
[Scion-35B drafter](https://huggingface.co/SkyIsNotGreen/Scion-35B-A3B-mtp-drafter):
there the base distill ships no MTP block, so the head had to be trained
self-distilled (3,088 windows) to 67–83% acceptance; here Qwen's own head
accepts 74% weighted with zero training.

## File

| file | size | sha256 |
| :--- | ---: | :--- |
| `Scion-FlashNext-176B-A6B-mtp-drafter.gguf` | 0.889 GB | _see SHA256SUMS.txt_ |

License Apache-2.0, inherited from the base model. The base model is
Qwen/Qwen3.8-Flash-Next (see the
[model card](https://huggingface.co/SkyIsNotGreen/Scion-FlashNext-176B-A6B)
for the quantization chain and the NOTICE).
