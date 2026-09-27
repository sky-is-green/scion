# Hot-cache equivalence (corrected OLMoE, 2026-09-27)

Model: `olmoe-q1exp-q8rest.gguf` + `branches-r1024-attnoutlloyd-step8000-q1g128-routers.lora.gguf`
(healthy operating point). Sidecars rebuilt from the same base model.

| run | PPL (32 chunks) | greedy 24 tokens |
|---|---|---|
| correction adapter only | 16.2089 ± 0.4910 | `Paris. The country is divided into four regions: ...` |
| + hot16 sidecar (25% experts) | 16.2035 ± 0.4907 | **identical** |
| + hot64 sidecar (all experts; control) | 16.2089 ± 0.49099 | **identical** |

Conclusions:

- **hot64 control is PPL-identical**: the split mechanism is exact (permuted
  bank order + extra masked pass add no measurable difference).
- **hot16 is within 0.03%** (inside the ±0.49 noise band) and **greedy
  token-identical**: the split is functionally equivalent.
- The earlier +0.05% PPL / greedy mismatch was the *degenerate* RTN model
  (PPL 43,000) exponentiating ~1e-4 logprob noise; at a healthy operating
  point (PPL 16) the same split is indistinguishable.

Reproduce:

```bash
tools/make_hot_sidecar.py --model olmoe-q1exp-q8rest.gguf \
  --profile ../placement-sweep-20260927/expert-hits.json --hot 16 --out hot16.sidecar.gguf
llama-perplexity -m olmoe-q1exp-q8rest.gguf \
  --lora branches-r1024-attnoutlloyd-step8000-q1g128-routers.lora.gguf \
  --lora hot16.sidecar.gguf -f wiki.test.raw -c 512 --chunks 32 -ngl 99
```
