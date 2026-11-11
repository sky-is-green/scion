# Evaluation state (2026-11-11)

## Runner

- `v2/scripts/hard90_scion.py` - copy of
  `hivebench/experiments/hard90_scion.py`, patched to record `seed`,
  `temperature` and `arm` in every row (the two runs below predate the patch).
- Dependency: `heldoutlab/bonsai2-27b-test` (harness and `items_r2`) cloned at
  `/tmp/opencode/bonsai2-27b-test`. The 12:08 reboot wiped it; re-clone and run
  `fetch.py` before the next run.
- Server config: `v2/configs/scion35b-strata.json` (engine
  `build-hip/strata-q36`, 32k context, chunk 512, GPU 0, port 8082,
  `reasoning_loop_recovery: recover`). Note: recovery only runs in the
  streaming path and the heldout harness sends `stream: false`, so it was
  inert for these runs.

## Results

| file | items | T | seed | correct | families | truncated | median wall | mean decode |
|---|---|---|---|---|---|---|---|---|
| `v2/eval/hard90-scion-32k/results.jsonl` | 90 | 0.6 | 1001 | 13 (14.4%) | R 4/30, C 1/30, T 8/30 | 17 | 287 s | 76.9 tok/s |
| `v2/eval/hard90-scion-32k-t01/results.jsonl` | 38 | 0.1 | 1001 | 10 (26.3%) | R 2/12, C 0/12, T 8/14 | 8 | - | - |

- The T=0.1 file is the partial archive from the run interrupted at 38 items.
- Sampler: top_p 0.95, top_k 20, min_p 0.05; 32k budget; thinking cap ~24.4k;
  one server at a time.
- Hashes:
  - 32k: `fb579b5d31c8fdd6f1b486bef6710279a19e2130b65ec9770e89037f22816f81`
  - 32k-t01: `8c9475cf1fd7bbbeb89fa070672d39b1d0c607c3d577db89bbb500051dd7144d`

## Protocol caveats

- 12/30 coding items truncated at 32k; 9 of those were lost to an unclosed
  ```python fence (the grader saw no complete code block).
- Loop onset measured at roughly 5-10k generated tokens; long items end at
  0.99 coverage.
- Needle test through the same engine: perfect recall at 2k-26.5k prompt
  tokens, so the runtime is not the failure.
- The model-default temperature is 0.6 (GGUF `general.sampling.*`); the 0.1
  file is the earlier card-default run.

## Reproduce

```bash
git clone https://github.com/heldoutlab/bonsai2-27b-test /tmp/opencode/bonsai2-27b-test
# fetch arms A and B per that repo's README, then start the server:
#   cd work/strata && .venv/bin/python -m serve.server --engine strata \
#     --config /path/to/v2/configs/scion35b-strata.json --port 8082
cd work/hivebench && python3.12 experiments/hard90_scion.py --temperature 0.6
```
