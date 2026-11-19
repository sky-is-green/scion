# Recipe review, 2026-11-19: what to change before the pod

Status: **review only; no pod created.**  The Runpod key is in place.  This
document records the last pre-flight audit of the v2 recipe, the trace-share
arithmetic, the loop-intervention question, and the wide art pass.  The
runbook fixes it refers to are in `moe/box-run-v2.sh` (commit `9b82e7d`).

## 1. The runbook had drifted from the frozen experiment

`EXPERIMENT-1.md` freezes the v2 stack: the deployed GGUF seeds the ternary
codes (`--bank-gguf`), the compact-bank/gradient-checkpoint path bounds
memory, windows are 4k, and selection is on the hard subset.  The full-run
runbook had regressed to the older v1 shape:

| item | frozen experiment | runbook before 2026-11-19 | now |
|---|---|---|---|
| ternary codes | `--bank-gguf` (0-ppm seed) | recomputed from BF16 | **fixed** |
| memory | `--compact-banks --grad-checkpoint --checkpoint-mode group --expert-group-size 64` | dense v1 path | **fixed** |
| vocab CE | `--chunked-ce` | full softmax | **fixed** |
| router | frozen (PLAN section 5) | trained by default | **fixed** (`--freeze-router`) |
| body on the pod | required for seed/merge/audit | not fetched | **fixed** (setup downloads it) |
| pilot | 200 steps is the cost gate | no pilot stage | **fixed** (`pilot`, `pilot-export`) |
| windows | 4k | 512 | **open** (see section 2) |
| selection | hard subset, not the last step | final + soup only | **open** |

The 512-token choice is not a bug by itself: it is the 176B FlashNext-proven
schedule, and it is what the 29 packed traces can support.  It is a deliberate
cost/data compromise, not the frozen 4k hypothesis.  The pilot must be read as
a test of the 512 schedule, not of the 4k one.

## 2. The mix is much more dilute than the name suggests

`build_curriculum.py` writes 205 hard fineweb windows (205 x 512 tokens) and
appends the 29 packed teacher traces (~34k words, ~44k tokens).  The mix is
therefore ~149k tokens, of which the traces are ~30%.

But `cache`/`train` pass `--agentic-frac 0.05`: `windows_from_texts()`
concatenates the mix and samples 4096 windows, then `mix_windows()` replaces
5% of the training rows (205 of 4096) with rows drawn from that pool.  The
trace share of the *training* windows is

```
0.05 x (44k / 149k) ~= 1.5%
```

i.e. roughly 60 of 4096 windows, or ~1.5% of the training tokens.  The hard
fineweb windows are ~3.5%.  This is a long way from "the teacher traces are
the second half of the mix" if the intent was a 5% trace signal.

Two separate causes, and they should be decided separately:

1. **Data volume.**  29 traces cannot fill a 5% share at 4k windows, and only
   ~1.5% at 512.  The 360-trace corpus is local now (`RUN-PLAN.md`); generating
   100-360 of them costs 2-10 h of local strata time, no pod.
2. **Mix semantics.**  `windows_from_texts()` treats the mix as one long text
   and resamples overlapping windows; it does not preserve the 205 curated
   rows.  A row-level mixer (chunk each text, sample whole windows) would make
   the curriculum mean what the docs say.  This is a code change; do not make
   it silently in the same run as a data change.

**Recommendation:** before the full run, expand the trace set and raise the
trace share.  At 512 tokens, 100 traces (~200k tokens) support a 10% trace
mix (410 of 4096) without starving the fineweb side.  This is the highest
expected-value change for the 5/30 -> 8-12/30 goal, because the 35B's gap to
the 176B is exactly the behavior the traces carry: tools 3/10 vs 9/10 and
reasoning 2/10 vs 3/10.  Coding is 0/10 for both; traces will not fix coding.

## 3. The loop intervention question: recovery already captured most of it

The shipped model under the current protocol:

| run | score | R | C | T | truncated | median wall |
|---|---:|---:|---:|---:|---:|---:|
| raw v1 (no recovery sentence) | 5/30 | 1/10 | 0/10 | 4/10 | 17 | 454 s |
| **v1 + recovery sentence (reference)** | **5/30** | **2/10** | **0/10** | **3/10** | **3** | **171 s** |

Recovery cuts truncations 17 -> 3 and the median wall time by 2.6x, but the
score does not move.  The 14 recovered items become wrong answers, not correct
ones.  The remaining 25 failures are capability failures (wrong reasoning,
wrong code, answer-room losses), not loop failures.

That changes the value of a better injected thought:

- It can plausibly turn the last 3 truncations into answers, and it can make
  the *raw* model (without the server recovery sentence) behave better.  Both
  are useful for the artifact and the model card.
- It cannot by itself move 5/30 much, because the recovered answers are wrong.
  The score lever is teacher data, not a cleverer cue.

### What the literature says about injecting a thought

The idea is feasible and our server is already 80% of the way there:
`focused_recovery_prompt()` is a token-level splice (prompt + generated ids,
with one system sentence swapped).  Appending a cue is a small extension.

The evidence is mixed, and it points to a specific design:

- **AID ("Well, Keep Thinking", 2503.10167)** injects a short phrase when EOS
  is likely and improves reasoning substantially.  Phrase choice matters:
  "Well" 50.6%, "Wait" 21.1%, "Keep reasoning" 12.2% on MultiArith.  A neutral
  continuation cue beats an explicit "you are looping" instruction.
- **Circular Reasoning (2601.05693)** shows loops are preceded by ~1500
  tokens of semantic circularity and a surge of high-entropy reflection
  tokens ("But", "Wait"), and are sustained by V-shaped attention to sinks and
  recent tokens.  By the time 12-gram coverage fires, the state is collapsed;
  a cue has to re-anchor attention on the task, not just say "stop".
- **ROM (2603.22016)** intervenes at a well-formed boundary: backtrace to a
  sentence/solution boundary, discard the redundant tail, append a
  final-answer cue.  Cut position and cue both matter.
- **EoT injection (2609.03633)** is the caution: injecting `</think>` mid-CoT
  often fails because later tokens do not attend to the injected token; the
  model regenerates the delimiter and keeps reasoning.  Text alone can be
  ignored.
- **One-neuron loop surgery (2606.13705)** is the second caution: suppressing
  verbatim loops at long budgets can shift the failure to "doom looping"
  (non-convergent self-correction) when the model lacks the fact.  Measure
  non-convergence, not just repeated n-grams.
- **STIR (EMNLP 2026)** builds state-conditioned correction vectors from the
  model's own successful/failed rollouts and validates them with a lookahead
  probe.  This is the activation-space version of the same idea; more work
  than a token splice, but it addresses the "cue is ignored" problem.

### The experiment worth running (if we run one)

Not "inject a thought" as a single arm.  Test the three things the literature
separates, on the 6 worst loopers (R09, R21, C03, C22, C05, C01), one card,
sequential:

| arm | intervention |
|---|---|
| A | current recover (effort swap + temp >= 1.0 + presence 1.5) |
| B | A + neutral cue appended after the generated text ("Well, let me re-read the task and take a different approach.") |
| C | rollback to the last sentence boundary before loop onset, then a task-anchored cue (first ~300 chars of the user request + "I am repeating myself; state what is missing and continue.") |
| D | stop at detection, no recovery (control) |

Each item is ~8 min at 32k on one 7900 XT; 4 arms x 6 items is ~3 h.  Primary
readout: completion (not truncated), loop recurrence, correct.  Secondary:
answer length and whether the model converges or doom-loops.  This is an
eval/serving experiment, not a training recipe change.  It should not block
the pod.

**Recommendation:** do not spend the pre-pod window on this.  The recovery
result says the score is data-bound.  Keep the cue experiment for the local
box while the pod trains, or for the model card after v2.

## 4. The stronger training-time loop fix (phase 2, not this run)

If the raw model still loops after v2, the literature's strongest targeted fix
is **Antidoom / FTPO** (Liquid AI, 2026): detect the exact loop-start token,
mark it rejected, choose a coherent alternative, and train a preference loss on
that single token.  Reported: 10.2% -> 1.4% doom loops on LFM2.5-2.6B and
22.9% -> 1.0% on Qwen3.5-4B, with benchmark scores improving.  A LoRA adapter
is enough; the rest of the model is untouched.

It fits our constraints better than a full preference run: the correction
branches are already a small trainable surface, and the loss touches one token
per loop.  It needs a DPO-family loss in `qwen35_moe_proxy.py` (not present)
and a set of loop boundaries from training-excluded prompts.  Cost is small;
implementation and validation are the work.  Keep it as the fallback if v2's
raw loop behavior is still bad, not as a change to the current pilot.

The on-policy alternative (Right On Q / SteerDistill) is the same story: the
training signal should come from the student's own failed states.  The current
mix has none.  The 12 captured rollouts are clean, and the only known loop
exemplars are the hard-90 items, which are training-excluded.  A loop-state
corpus therefore needs either (a) hard prompts from the 360-prompt corpus that
actually loop at 16k (a 30-prompt, ~2 h local check), or (b) synthetic loop
induction.  Do not assume the 360 prompts loop just because the hard-90 do.

## 5. Wide art pass (new since the 2026-11-11 survey)

New items that change or sharpen the plan:

| item | id | relevance |
|---|---|---|
| Circular Reasoning / LoopBench | 2601.05693 | Loop mechanism, semantic precursor, CUSUM early detection; supports early/boundary intervention. |
| ROM: Real-time Overthinking Mitigation | 2603.22016 | Boundary rollback + answer cue; the best template for a recovery intervention. |
| When Can One Neuron Fix Repetition Loops? | 2606.13705 | Sparse weight surgery; warns that loop suppression can become doom looping. |
| RADAR: Attention Dynamics | 2609.38817 | Attention realignment cuts loop rate ~10 pp; mechanistic, not needed now. |
| When Do Repetition Loops Lock? / DRY-C | 2610.24715 | Bounded penalties can stay locked; DRY's length term costs 2.5-14.2 accuracy points; a count term closes the blind spot. |
| </think> Doesn't Stop Reasoning | 2609.03633 | Injected tokens can be ignored; attention bias to the injection helps. Caution for cue design. |
| STIR: Self-Distilled Latent Tools | EMNLP 2026 | State-conditioned correction vectors from own rollouts; lookahead-gated intervention. |
| "Well, Keep Thinking" / AID | 2503.10167 | Short neutral injection phrase at the decision point; phrase choice matters. |
| Antidoom / FTPO | Liquid AI 2026 | Train the loop-start token with a preference loss; strongest targeted loop fix. |
| Long-Context Generation Is a Sampling Problem | 2611.22148 | At high temperature, top-p/top-k admit the tail and long output collapses; top-n-sigma stays tight. Our eval is T=0.6/top-k 20, so not the current failure, but recovery's temp >= 1.0 should keep the candidate set tight. |
| When to Intervene? | 2610.36471 | Agents often self-recover; intervention timing matters. Our loops are already locked at detection, so intervention is justified. |
| Echoes as Anchors | 2602.06600 | End-of-prompt attention refocusing; supports a task-anchored cue over a generic "stop". |

Already in the 2026-11-11 survey and still the core: Right On Q, ThinkBit,
SteerDistill, RoleQ, Align Then Correct, ProQuant, Lost in the bf16 Cast,
Ternary-Flint, Ternary Nemotron.  Nothing new invalidates the corrections-only
choice.  The new art mostly says: if you intervene, rollback to a boundary and
re-anchor the task; if you train, train the loop-start token or the student's
own failed states.

## 6. Decision list before the pod

Must be true before the pilot:

1. Runbook uses `--bank-gguf` + compact banks + frozen router + body fetch.
   **Done** (`9b82e7d`).
2. Pilot stage is 200 steps with a 100-step checkpoint, and exports the pilot
   adapter.  **Done.**
3. `setup` verifies the body sha256 (`4e9e3292...cffa040`) and the empero-hf
   config.  The runbook fetches the body but does not yet assert the sha.
4. The pilot command prints `memory_report` before the first checkpoint and
   the log gives s/step; stop if `s/step x 1024` cannot fit the approved cap.

Recommended but not blocking:

5. Expand the teacher traces to ~100 (2-3 h local) and raise the trace share
   to 5-10%; keep 512 windows for the pilot.  This is the score lever.
6. Keep the current recover protocol for all arms and the reference.  Do not
   change the cue and the training data in the same run.
7. Export 2-3 checkpoints and evaluate them on subset30; do not ship the last
   step by default.

Open questions:

- Does the 360-prompt corpus actually loop at 16k?  A 30-prompt/2 h local check
  answers it; if yes, on-policy loop-state data becomes available without
  touching hard-90.
- Does the 4k window buy enough over 512 to justify the extra step cost?  The
  pilot at 512 cannot answer this; a 2k/4k arm after the pilot can.
- Does freezing the router cost the tools family?  The 176B recipe trained
  routers; our plan says no router training.  If tools stall, this is the
  first one-variable retest.

## 7. Reproduce the numbers in this note

```bash
# trace share arithmetic (tokenizer-free estimate)
python - <<'EOF'
import json
rows=[json.loads(l) for l in open('v2/curriculum/teacher-traces.jsonl')]
words=sum(len(r['text'].split()) for r in rows)
trace_tokens=int(words*1.3); hard=205*512
print('trace share of mix', trace_tokens/(trace_tokens+hard))
print('trace share of 4096 training windows', 0.05*trace_tokens/(trace_tokens+hard))
EOF

# the recovery reference
cat v2/eval/hard-subset30-scion-recover/summary.json
```
