# Community post: empero-ai

Paste the block below into the Community tab of
`empero-ai/Qwen3.8-35B-A3B-Distill`. It is a code block so the markdown survives
copying.

```markdown
Thanks for the distill. I quantized `Qwen3.8-35B-A3B-Distill` down to **2.61 bpw** and wanted to show you the result.

**Scion-35B-A3B**: the expert banks in a ternary container (2-bit codes plus one fp16 group scale per 128 weights), everything else Q8_0, plus small **trained** rank-512 corrections (attention output, MoE block output and router deltas), trained by output-KD against the BF16 teacher with the deployed quantizer in the loop. No imatrix, no calibration corpus, no full-model QAT. The corrections are the only trained part, and the run cost about $7 of rented H100 time.

- **11.34 GB / 2.61 bpw, single file** (corrections embedded, no `--lora`), 6.3x smaller than the BF16 reference.
- Task retention (400 tasks each): HellaSwag **79.00** and Winogrande **76.25** against BF16's 81.25 and 76.00, inside the ±2% noise band; best PPL of the 2-bit class (8.354 versus IQ2_M's 8.413).
- The gap: full-vocabulary KLD against BF16 is still 2-bit-class (0.269 mean versus Q4_K_M's 0.031). Tail-aware training is my next step.

Apache-2.0, inherited, with attribution to empero-ai and Qwen (and the community BF16 GGUF conversion). It needs my llama.cpp fork (`PQ2_0` plus embedded adapters, default branch `moe-corr-runtime`; stock llama.cpp cannot load the tensor types, and the fork ships a one-second check for that). I built it to serve a local assistant stack on a single 20 GB card.

Card and weights: https://huggingface.co/SkyIsNotGreen/Scion-35B-A3B
Harness, write-up and failed routes: https://github.com/sky-is-green/scion

If anyone wants to try it or poke holes in the numbers, the model repo's discussions are open, and I am happy to share any part of the recipe in more detail.
```
