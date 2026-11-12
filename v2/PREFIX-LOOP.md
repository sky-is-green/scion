# Prefix loop (PLAN 13a step D) + the recovery findings

Date: 2026-11-12.  Status: **the full local loop runs end to end: seed ->
train -> export -> merge -> audit PASS -> serve -> harness eval; plus three
protocol findings that change how the v2 evals must run.**  No pod; one 7900
XT.

## The loop (all steps real, real cache, real deployed body)

1. **Train** (4-layer prefix, compact banks, real cache with tail tokens,
   rank 512 both, bias, 2 steps): `qwen35-corr-r512-g128-step2-prefixloop.pt`.
2. **Export** (`moe/export_branches_lora.py`, PYTHONPATH =
   `prism-qwen4exp-shrink/gguf-py` - the only local gguf-py with Q1_0_g128
   type 43): 8 branch tensors + 4 router deltas, 10.3 MB adapter GGUF.
3. **Merge** (`moe/merge_adapter_into_body.py --replace`): the deployed body
   already carries the v1 corrections (240 lora tensors), so a v2 adapter
   must SUPERSEDE them at the same sites - v2 branches train from zero and
   the v1 factors would apply twice.  New `--replace` flag (default off;
   v1 strict-refusal behaviour kept): 24 superseded body tensors dropped,
   973 tensors, 10.558 GiB, `adapter.embedded=true`.  The body's own
   `adapter.*` metadata is skipped so the file carries the v2 adapter's
   provenance.
4. **Audit** (new `v2/scripts/audit_merged_adapter.py`): (a) merge integrity
   - 949 body + 24 adapter tensors byte-identical, 0 mismatches; (b) factor
   fidelity - 20,971,520 packed Q1_0_g128 weights decode bit-exactly to
   `deploy_weights(checkpoint)` (0 mismatches); (c) router fidelity - the
   f16 SVD pairs reconstruct trained-base exactly at tolerance.  Plus
   `export_audit.py --branches-only` on the merged file: 160 tensors,
   209,715,200 weights, 0 fourth-state codes, 0 non-max scales, 0 bad
   scales.  **AUDIT PASS.**
5. **Serve + eval smoke**: strata server on the merged file
   (`v2/configs/scion35b-prefixloop.json`), harness re-cloned to
   /tmp/opencode/bonsai2-27b-test, 2 subset tool items through
   `v2/scripts/hard_subset.py`: ran end to end (43 tok/s decode, median
   84 s, 0/2 correct - expected: 2-step near-zero branches replaced the
   trained v1 corrections in layers 0-3; the point is the path).

The `balance_bias` finding: the proven sibling drops `balance_bias` at
deploy ("training-time shaping only", `qwen4exp_export.py`) - the served
model runs without it; recorded, applied to v2 by the same convention.

## The recovery findings (protocol-level, they change the eval design)

Driven by the question "does loop recovery actually run for the heldout
harness?" - three findings, each verified live:

1. **EVAL.md's caveat is wrong about the mechanism.** Recovery is NOT
   streaming-only: the non-streaming path consumes the same generator
   (`openai_collect` over the same `svc.run`), and the loop DETECTION fires
   live on non-streaming requests - proven with a "stop"-mode canary: the
   shipped v1 model on hard-90 item R07 (the harness protocol) hit
   `coverage=0.643 at 2560 tokens` and the reply ended per the stop contract
   (24.3 s, 2,560 tokens, deterministic reproduction of the saved hard-90
   row's 32,548-token loop otherwise intact).
2. **"recover" mode is silently inert for the 35B pack.**
   `focused_recovery_prompt` needs the chat template's xhigh effort sentence
   inside the first system message to splice the recovery prompt
   (`serve/server.py:266`); the pack's `chat_template.jinja` contains no
   effort sentence at all, and neither the heldout harness's nor a plain
   system prompt adds one - so `recover_prompt` is None, the detection
   (which DOES fire, coverage >= 0.25) silently does nothing, and no log
   line marks it.  Detection logs only exist in "stop" mode.
3. **The 18 `serve.test_reasoning_loop_recovery` regression tests pass** -
   the mechanism is sound; the live gap is the prompt dependency above.

Options for the v2 eval protocol (decision belongs in the experiment
freeze; recorded in `v2/EXPERIMENT-1.md`):
  (a) run recovery as "stop" (works today; changes grading semantics - the
      loop is cut, answers end as "length");
  (b) append the exact xhigh effort sentence to the harness system prompt
      for ALL runs (v2 arms AND the reference baseline, so the comparison
      stays like-for-like) - then the splice works as designed;
  (c) a strata-side fallback in `focused_recovery_prompt` (server change;
      also affects the 176B FlashNext runs - needs its own decision).

## The control that mattered (false positive avoided)

While probing recovery, the merged prefix model completed PR-001/PR-002
(the "looping" rollouts) cleanly - which looked like the loop was fixed.
The control run on the SHIPPED v1 model with the same protocol reproduced
PR-001 byte-identically, also clean - and recomputing the metric exposed
that the rollouts' recorded "coverage 1.0" was a broken analysis value (the
stored texts are clean solutions).  Corrected in `v2/PROGRESS.md` section 4,
`v2/PLAN.md` 13a and `v2/STATE.md` (commit `fb26b6e`); the hard-90 loop
evidence re-verified independently (all 30 items past 25k tokens at 0.65-0.97
duplication) and the experiment's primary gate stands on it.

## Operational notes

- A GPU **hang** (node-1, "HW Exception ... GPU Hang") killed the engine
  once during a 6.6k-token non-streaming generation; the server restarted it
  automatically (~1 min) and the request itself had already completed.  The
  same class of error appeared in the v1 hard-90 run (4 server errors on the
  Nemotron baseline).  One server at a time stays the rule.
- Server artifacts cleaned (memprobe/prefixloop checkpoints deleted); the
  merged loop GGUF and the adapter remain under /tmp/opencode/qwen35-realcache/
  for inspection.
- Servers started for these probes: all stopped (8083/8084/8085).

## Reproduce

```bash
# train (2 steps, real cache, compact path)
MOE_ARTIFACTS=../models-p310/empero-prefix SCION_GGUF=... \
  python moe/qwen35_moe_proxy.py train --prefix-layers 4 --device cuda:1 \
  --bank-gguf ... --cache-file /tmp/opencode/qwen35-realcache/cache-tail.pt \
  --compact-banks --grad-checkpoint --checkpoint-mode group --expert-group-size 64 \
  --quant lloyd --branch-quant g128 --branch-target both --rank 512 \
  --kd-weight 2.0 --kd-tail-weight 2.0 --kd-tailcond-weight 3.0 --temp 2.0 \
  --balance bias --steps 2 --tag prefixloop

# export + merge + audit
PYTHONPATH=../prism-qwen4exp-shrink/gguf-py python moe/export_branches_lora.py \
  --load .../qwen35-corr-r512-g128-step2-prefixloop.pt --arch qwen35moe \
  --target both --dtype q1_0_g128 --routers --base-model ../models-p310/empero-prefix/empero-hf \
  --out /tmp/adapter.gguf
PYTHONPATH=../prism-qwen4exp-shrink/gguf-py python moe/merge_adapter_into_body.py \
  --body .../Scion-35B-A3B-PQ2_0-corr.gguf --adapter /tmp/adapter.gguf \
  --out /tmp/merged.gguf --replace
PYTHONPATH=../prism-qwen4exp-shrink/gguf-py python v2/scripts/audit_merged_adapter.py \
  --merged /tmp/merged.gguf --body ... --adapter /tmp/adapter.gguf \
  --ckpt ... --base-model ../models-p310/empero-prefix/empero-hf
python v2/scripts/export_audit.py --deployed /tmp/merged.gguf --branches-only
```
