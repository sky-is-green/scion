# 011 - Ternary Nemotron 3.5 Lightning 30B-A3B (artifact)

- Base: nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 (30B total,
  3B active, hybrid Mamba-2 + MoE: 52 blocks, 128 routed experts top-6 +
  shared, 6 attention layers, 1M context, OpenMDW-1.1).
- Ternary source: nota-ai/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Nota-
  Ternary-MLX-1014 (1.58-bit, MLX 2-bit affine group 64, values {-a,0,+a}).
- GGUF repack: boro64/Ternary-Nemotron-3.5-Lightning-30B-A3B-GGUF
  (8.82 GB Q2_0, bit-exact MLX->Q2_0).
- Status: analysed, direct consumer-tier competitor and possible baseline

## Numbers

- Nota AI: 60 GB BF16 -> 9.6 GB ternary (6.25x); the release is an
  intermediate checkpoint, "not fully recovered", recovering about 89% of the
  teacher's generative benchmark average (83.82 vs 93.80 across seven tasks).
- GGUF: 8.82 GB, all 162 linear tensors at Q2_0 (2.25 bpw), embeddings and
  lm_head Q8_0, norms/router f32. Fits a 12 GB card at up to 368K context
  (262K native, verified 341K on a needle test). ~172 tok/s decode,
  ~7,000 tok/s prefill; stock llama.cpp and the Prism fork.
- NVIDIA's own release quantizes to NVFP4 with quantization-aware
  distillation and reports a median 99.24% of BF16 accuracy at 22 GB; the
  MTP module is deliberately not quantized.

## Why it matters to us

1. **This is the bar for the consumer ternary tier.** 30B-A3B at 8.82 GB,
   long context, 172 tok/s, ~89% of its teacher on generative benchmarks,
   no trained corrections. Our 35B-A3B is 11.3 GB and slower per byte.
2. **The conversion is simple and bit-exact.** MLX 2-bit affine group 64 is
   bit-compatible with GGUF Q2_0 block 64; the repack copies codes and scales
   verbatim. That is the same "write codes directly" lesson as entry 009.
3. **The recovery is openly partial.** They ship an intermediate checkpoint
   at 89%, not a fully recovered model. If the hard-90 holds up, that sets an
   honest expectation for what a ternary release can claim.
4. **Architecture note:** hybrid Mamba-2 + only 6 attention layers gives
   262K-1M context with modest KV. If our long-horizon problem is partly
   attention over long contexts, this is the direction a future Scion
   architecture could take.

## What does not transfer

- Generative benchmark average is not the hard 90; long-horizon reasoning is
  untested in their card.
- Their quantization is affine 2-bit (values ternary), not our PTQ/Q2_0
  g128 with corrections; the bit accounting is 2.25 bpw for linears.
- The base is Mamba-2 hybrid, not Qwen attention; correction placement and
  sensitivity differ.

## Baseline we could run

- Download the 8.82 GB GGUF and run the hard 90 through llama.cpp on this
  box (the Prism fork already runs Q2_0). This gives the cleanest external
  answer to "is 14.4% a Scion problem or a ternary-tier problem".
- Also worth running the 30-item easy set to compare with our 76.7%.

## Open questions

- Does the 89% recovery hold on the hard 90, or does it collapse like Scion?
- What does 368K context do to long-horizon coherence (the needle passed, but
  needle retrieval is not reasoning)?
- Would our corrections on top of this base beat their 89%?
