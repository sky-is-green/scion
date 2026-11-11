# 008 - Embeddings Are Overpriced? A Frozen 16 MB Vocabulary for Small and Ternary Language Models

- arXiv: 2610.22428, 2026-10-15
- Link: https://arxiv.org/abs/2610.22428
- Status: analysed, size lever for the consumer tier

## Claim

In a small model the vocabulary is the single largest part, and because
quantization leaves the embedding in 16-bit it is most of the file once the
rest is compressed. They replace the embedding table with one frozen 16 MB
table of 512-bit codes (principal-component projection of a pretrained
embedding, then sign binarization), with no trainable parameters, and train
the same Gemma-shaped body from scratch with and without it.

## Numbers

- The learned table buys 0.20 nats of validation loss for 2.6x the trainable
  parameters.
- On LAMBADA, HellaSwag and PIQA the frozen-table model is better or tied;
  only ARC-Easy favors the learned table.
- With a ternary BitNet b1.58 body the whole model is a 46 MB file, ties the
  float model on LAMBADA and PIQA, and on a laptop CPU generates 615-740
  tok/s in 62 MB of memory, five to six times the 8-bit GGUF in llama.cpp at
  a seventh of the memory.
- The frozen table wins by 1.7-2.3 nats on rare words, accepts 10,000 new
  words after training with a 3% cost where the learned table collapses by
  43%.

## What transfers to Scion v2

1. **Our embedding and head are a real fraction of the 11.3 GB.** The 35B has
   a 248,320-token vocabulary; at Q8_0 the input embedding and the output
   head together are roughly 1.5-2 GB. That is the cheapest size target left
   in the tier.
2. **A frozen codebook is a consumer-tier move.** Replacing the embedding with
   a frozen codebook (or sharing/tieing the head more aggressively) could cut
   ~1-1.5 GB with little accuracy cost, if done as a conversion with a short
   recovery.
3. **Packing + embedding work together.** 5-trit expert packing (~20% off the
   expert body) plus a frozen embedding could take the tier from 11.3 GB to
   roughly 8-9 GB without changing the expert weights.

## What does not transfer directly

- They train the body from scratch with the frozen table; our body is frozen.
  A post-hoc replacement needs a recovery stage and risks the same
  distribution shift the paper avoids by training.
- The experiments are small dense models; the 35B's 248k vocabulary is much
  larger and its rare-word behavior is part of the task mix.
- The 512-bit code is 64 bytes per token; at 248k tokens that is already
  ~16 MB, but the codebook lookup and the head still cost storage.

## Experiments this suggests

1. Measure the exact embedding + head bytes in the current GGUF, at Q8_0 and
   at lower precision, and decide whether a frozen/compressed table is worth a
   v3 conversion.
2. If attempted: derive the frozen table from the current embedding (PCA +
   sign), retrain only the input norm/scale path for a short recovery, and
   eval hard-90 plus rare-word behavior.

## Open questions

- Does the frozen table survive a 248k multilingual vocabulary, or does it
  need a larger code?
- Can the output head stay separate and be quantized more aggressively than
  the input embedding?
