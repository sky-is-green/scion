# Reddit copy-paste blocks: Scion-35B-A3B

Three fenced blocks, so a copy from here is exactly what you paste.

## 1. Body

````markdown
**TL;DR.** Scion-35B-A3B is a 35B-A3B distill at 2.61 bpw: 11.34 GB in one file, and it fits a 20 GB card. Ternary expert banks plus small trained corrections. Task accuracy sits in the Q4_K_M/BF16 noise band (HellaSwag 79.0, Winogrande 76.25) and PPL is the best of the 2-bit class; the KLD tail is still 2-bit, and that is the gap I am working on next. I would like people to run it and tell me where it breaks.

**What it is.** A scion is the shoot grafted onto a rootstock, and here the corrections are grafted onto the ternary body.

- expert banks in a ternary container (2-bit codes, one fp16 scale per 128 weights), everything else Q8_0
- rank-512 corrections on the attention output and the MoE block output, plus router deltas, trained by output-KD against BF16 with the deployed quantizer in the loop
- corrections embedded in the file, so there is no `--lora` step at load time

**Numbers.** wikitext-2 PPL, KLD against BF16, HellaSwag and Winogrande at 400 tasks each; the chart is attached (up is better).

```
model            GB     PPL    KLD     HS     WG
Scion-35B-A3B  11.34   8.354  0.269  79.00  76.25
IQ2_M          12.56   8.413  0.164  79.00  75.75
Q4_K_M         21.71   7.235  0.031  80.00  76.00
BF16           71.07   7.160     --   81.25  76.00
```

Best PPL of the 2-bit class at about half of Q4_K_M's size. The KLD number is the honest part: the competing quants are imatrix-calibrated, this one is not, and closing the distribution tail is the next step.

**Try it.** On Hugging Face: `SkyIsNotGreen/Scion-35B-A3B`, file `Scion-35B-A3B-PQ2_0-corr.gguf` (11.34 GB, sha256 `545d83a9...`). It needs my llama.cpp fork, `sky-is-green/prism-ml-llama.cpp`, whose default branch is the right one; run `./verify-container-support.sh` in the checkout before building and it tells you in a second. A load failure saying `invalid ggml type 142` means the wrong branch was built. CPU and ROCm are verified, CUDA is not.

```
hf download SkyIsNotGreen/Scion-35B-A3B Scion-35B-A3B-PQ2_0-corr.gguf
llama-cli -m Scion-35B-A3B-PQ2_0-corr.gguf -ngl 99 -c 4096 -t <physical cores>
```

Then tell me: where it derails over long chats, how it compares to your usual quant on your own prompts, speed (GPU, context, pp/tg t/s), and anything that fails to load.

Build scripts, MoE write-up and the negative register are at GitHub `sky-is-green/scion`; the dense forensics that started this is `sky-is-green/bonsai2-ternary-forensics`. Not affiliated with Prism ML or empero-ai; the model card carries the lineage and license notes.
````

## 2. First comment

```markdown
The attached chart is the retention grid: ten community quants plus BF16 under one protocol, and the full table with method notes is on the model card. If the link filter eats the thread, search the names in the post text and DM me. Happy to answer anything about the training recipe or the fork.
```

## 3. Detail comment

```markdown
**Protocol.** PPL is wikitext-2 at context 512 over 580 chunks; KLD is full-vocabulary against BF16 logits over 50 chunks (25.5k tokens); HellaSwag and Winogrande are 400 zero-shot tasks each, so treat sub-1% differences as ties. BF16 and the quants were measured in one H100 session; this file was measured on a local RX 7900 XT and cross-checked (IQ2_M KLD, local vs pod: 0.2%).

**Serving notes**, from two 7900 XT cards and a 7800X3D: threads should equal physical cores (SMT siblings halve CPU expert throughput); do not layer-split when one card fits (40% generation loss in the proxy runs); `-ncmoe` is the VRAM dial; the ternary container roughly halves the CPU-offload penalty versus an f16 expert bank. Negative result: a hot-expert GPU cache gave no throughput gain, because `mul_mat_id` computes k experts per token regardless of the weights, so it would need sparse per-token dispatch to be worth building.

**The Hub chip.** The file shows a `Q2_0` chip; that is the Hub's filename parser, not upstream's g64 `Q2_0`. The container is Prism's `PQ2_0`, and the file is a mix of container types. The card explains it.
```
