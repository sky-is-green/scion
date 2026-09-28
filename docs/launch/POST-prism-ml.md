# Community post: Prism ML

Paste the block below into the Community tab of
`prism-ml/Ternary-Bonsai-2-27B-gguf`. It is a code block so the markdown
survives copying. Follow-up to discussion #62.

```markdown
Update to my previous post ([#62](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/discussions/62)): the MoE side is finished.

**Scion-35B-A3B** is [empero-ai/Qwen3.8-35B-A3B-Distill](https://huggingface.co/empero-ai/Qwen3.8-35B-A3B-Distill) quantized to 2.61 bpw: 11.34 GB, one file, and it fits a single 20 GB card with room for context.

The recipe: the expert banks go into a ternary `PQ2_0` container (2-bit codes, one fp16 scale per 128 weights), and rank-512 low-rank corrections are trained on the residual stream (attention output, MoE block output, plus router deltas) by output-KD against the BF16 teacher, with the deployed quantizer in the loop. The corrections are embedded in the GGUF, so there is no `--lora` step at load time. Training took about 4,000 steps on rented H100 time ($7.26).

Numbers, all under one protocol (wikitext-2 PPL, KLD against BF16 logits, HellaSwag and Winogrande at 400 tasks):

- HellaSwag 79.00 and Winogrande 76.25, against 80.00 / 76.00 for Q4_K_M and 81.25 / 76.00 for BF16, at about half of Q4_K_M's file size.
- PPL 8.354, the best of the 2-bit class (IQ2_M 8.413, Q2_K 8.473).
- KLD 0.269 mean, still 2-bit-class against Q4_K_M's 0.031. That gap is the open item and the card says so up front.

Three findings from the build worth writing down:

- Rotation is not the MoE lever. Rotated and raw RTN on OLMoE experts came out even (relative error 0.511 vs 0.516). The loss is routing drift, and the corrections have to sit on the residual stream: per-expert branches were 25x worse PPL at 8x the parameters.
- The deployable Lloyd quantizer beats one-shot absmean on MoE experts (0.442 vs 0.517), and the GGUF expert codes reproduce the torch rule to 0.001 relative error.
- Router-KD is a no-op: it matched a zero-weight control within about 1.5%. What recovers router agreement is repairing the state the router reads.

Credits: the container, kernels and the branch this builds on come from [Prism ML's llama.cpp](https://github.com/PrismML-Eng/llama.cpp) and the [TAARDIS fork conventions](https://github.com/CodeMasterCody3D/prism-ml-llama.cpp), engine only, no Bonsai weights. Base weights from [empero-ai](https://huggingface.co/empero-ai/Qwen3.8-35B-A3B-Distill) and [Qwen](https://huggingface.co/Qwen/Qwen3.6-35B-A3B); BF16 GGUF conversion from [MrFuzzihead](https://huggingface.co/MrFuzzihead/Qwen3.8-35B-A3B-Distill-APEX-GGUF). Not affiliated with or endorsed by any of them. The runtime is my own fork of llama.cpp, [sky-is-green/prism-ml-llama.cpp](https://github.com/sky-is-green/prism-ml-llama.cpp), default branch `moe-corr-runtime` (embedded adapters, about 90 lines). Stock llama.cpp and the fork's `master` branch cannot load the container types; the branch ships `verify-container-support.sh`, which checks that in a second before you build. I am happy to upstream the loader bits if that is useful.

Model card: https://huggingface.co/SkyIsNotGreen/Scion-35B-A3B
Harness and MoE write-up: https://github.com/sky-is-green/scion
Earlier forensics work: https://github.com/sky-is-green/bonsai2-ternary-forensics
```
