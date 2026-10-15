# Scion MoE speed lane, part 3 — the drafter's long-form round (handoff, 2026-10-15)

**Goal.** Continue the k=1 MTP drafter's quality work on Scion-35B-A3B.  The remaining weak regime is
**long-form text** (1500-2000-token generations), where acceptance averages ~51% against 68-96% on the
short protocols - the drafter is roughly break-even there, and the oracle bound for any control scheme on
that text is only +17%, so head quality is the lever.  Part 2's headline stands: the degenerate-loop worst
case is *trained away* (+33% at 96% with the shipped head, no flags).  This session is a coverage round for
the long-form styles, measured by the same A/B discipline as rounds 3-6.

**The box is shared and is being handed to the user's long-form tests.**  Run this lane's load tests
(serving, collection, benches) **only when the system is inactive** - see the rules at the bottom.

**Read first:** `SCION-MOE-SPEED.md` - "The k=1 MTP drafter in q36", the round 1-6 sections, "Round 6,
measured" (today's state), "The controller readout, measured" (the gate negative) and the Publication gate.
Then `NEXT-SESSION-SCION-MOE-SPEED-2.md` for the lane's conventions.

## State to trust (2026-10-15, end of part 2)

- **Engine:** `work/strata`, branch `scion-0.1.44`, HEAD **`2dd1d679`** (`5ee873c4` = `--mtp-period`,
  `2dd1d679` = the jev/control probe, `27d8c29f` = `--mtp-backoff`).  Binary `build-hip/strata-q36`.
- **Scion repo:** branch `main`, HEAD **`130ed6f`** (record + card snapshot), tree clean.
- **Shipped head:** `mtp-sidecar-op6-350.gguf` in `work/hivebench/artifacts/ternary/moe/qwen35/` - and the
  same bytes on HF as `SkyIsNotGreen/Scion-35B-A3B/Scion-35B-A3B-mtp-strata.gguf` (lfs sha256
  `a375a459`).  Served through `work/strata/strata-35b-a3b-op6-350.json`.
- **Live service:** `scion-flashnext` on 8081 (the Q6K 176B, reserve 3072) - keep it up except in
  serve-down windows, and restore it at the end of every window.  The 35B lane is down; it wants GPU 0,
  which the 176B holds (0+1).
- **Artifacts are safe:** the lane's heads (`head-op4/5/6.pt`), the merged dataset `probe9/` (237 windows),
  the raw dumps (`probe7.bin`, `probe8.bin`, `probe7/8-cli.bin`, `jev-served.bin`, `jev-cli.bin`), the
  scripts (`benchp.py`, `round5/6-measure.sh`), the prompt lists (`prompts*.txt`, `heldout*.txt`,
  `structured-*.txt`) and the train reports were copied to
  **`work/hivebench/artifacts/ternary/moe/qwen35/mtp-lane/`**; `/tmp/opencode/mtp-onpolicy/` still holds
  them for convenience but `/tmp` is tmpfs.
- **Published:** both cards carry the op6 rows (35B README `b44a3e1b`, drafter README `31c7b802`).

## The task (round 7): close the long-form gap

Acceptance on the two long-form collections (56 fresh prompts, 1500-2000 tokens) ranged **21-66%**, with
the lowest rows in dialogue, uncommon-language code (Ruby/Lua/Zig/Haskell), essays/reviews/emails and the
"begin now and do not stop early" factuals.

Recipe (this is rounds 4-6's pipeline, unchanged):

1. **Collect** ~50-60 *fresh* long-form prompts (1500-2500 tokens; the lowest styles above, plus formats)
   through the 35B serve with both probes: `STRATA_Q36_MTP_PROBE=<c> STRATA_Q36_JEV_PROBE=<j>
   serve/server.py --engine strata --config strata-35b-a3b-op6-350.json --port 8082`.  Keep 8-10 of the
   prompts out of training (the last requests) for the honest long-form measurement.
2. **Assemble** windows exactly as probe9 was built: cut records into requests by the `pos` deltas
   (`d != 1 and d != 2` is a boundary), one 512-record window per `floor(records/512)` with `h` shifted by
   one (`hh[:SEQ-2] = h[i+1:L-1]`), then order `train-windows + held-out-windows`.
3. **Merge with `probe9/`** and keep the replay rule: training data = all previous rounds' windows + the
   new ones, degenerate stretches oversampled ~x4.  Round 4 taught that skipping the replay loses prose.
4. **Train** from `head-op5.pt`-era conventions: `moe/mtp_release_train.py --probe-dir <merged> --n-train
   <T> --n-eval <E> --seq 512 --device cpu --init /tmp/opencode/mtp-onpolicy/head-op6.pt --out <new>
   --steps 400 --batch-windows 4 --lr 1e-4 --eval-every 50 --eval-train-n 8 --weight-decay 1e-4
   --report <json>` (~10 s/step on this box; the trainer keeps best-by-eval, and *its* export is the
   checkpoint to use).
5. **Measure** with `round6-measure.sh <head.pt> <tag>` (export -> serve -> the card protocol, the three
   structured sets, factual, technical, round-4 prose, and the long-form held-out at 1500 tokens), plus
   the long-form prompts from the *previous* collections where useful.  Ship only if the sets hold or rise;
   the standard sets' numbers are in the record's round-6 table for comparison.

Levers if coverage alone stalls: soft-target distillation (dump the top-8 logits in the probe, train
against the soft distribution) and the zero-padded width-1.5 head (`moe/mtp_pad_head.py`).  Both are
unexplored; the head saturates its training windows (~0.999) while held-out plateaus ~0.54, so the evidence
points at data first, capacity second.

## Gates, rules and the shared machine

- **Only run when the system is inactive.**  Before any load test: check ports 8081/8082 and the GPU are
  free of *other* work (`curl -s :8081/v1/models`, `rocm-smi` or the process list), never kill another
  session's processes, and wait out busy periods.  Stop the 176B only for a serve-down window and always
  `systemctl --user start scion-flashnext` afterwards, then verify with `curl :8081/v1/models`.  The CPU
  training (10 s/step) may run alongside whenever the CPU is otherwise idle.
- **Token identity:** every change leaves warmed greedy requests byte-identical - `serving/moe-exact.sh
  <tag> 8082 35b-base` (5 warmed bodies, 735 chars).  Known and documented: the *batched verify* can flip
  near-ties (both cards say so); the plain path must stay exact.
- **One step per commit, numbers in the message.  Never `git push`.**
- **Traps already paid for:** `pkill -f` patterns that appear later in the same command line kill your own
  shell (`serve/server[.]py` in a separate call); the CLI and the serve cannot share the GPU; `/tmp` is
  tmpfs; `--mtp-period` fires only on confirmed periods and `--mtp-backoff`'s pauses give up the rule's
  wins (relax or drop it when the rule is on); the engine's stats and probe lines go to stderr.
- **Do not re-derive the rejected list:** the abstain/verify gates (priced empty), a learned gate on raw h
  (cross-prompt AUC 0.63; within-prompt 0.67 - not yet worth a gate), k=2 drafting, the engine-side
  drafter levers (round 1), the fork's head in Strata (38%).

## Continuation prompt (copy-paste)

```
New session: Scion MoE speed lane part 3 - the drafter's long-form round.  Read
work/scion/docs/NEXT-SESSION-SCION-MOE-SPEED-3.md first (state, the task, the shared-machine rules), then
work/scion/docs/SCION-MOE-SPEED.md (the drafter rounds 1-6, the control results, the publication gate).

State to trust: engine work/strata branch scion-0.1.44 HEAD 2dd1d679 (binary build-hip/strata-q36); the
shipped head is mtp-sidecar-op6-350.gguf in work/hivebench/artifacts/ternary/moe/qwen35 (also the HF card's
file), served via work/strata/strata-35b-a3b-op6-350.json; the live service is the 176B on 8081 - keep it
up except in serve-down windows and restore it at the end of every one; the lane's heads, the merged
dataset probe9, the raw dumps, scripts and prompt lists were copied to
work/hivebench/artifacts/ternary/moe/qwen35/mtp-lane/ (and still sit in /tmp/opencode/mtp-onpolicy).

The box is shared: run our load tests (collection, benches, service windows) ONLY when the system is
inactive - check ports 8081/8082 and the GPU first, never kill another session's processes, wait out busy
periods; the CPU training may run whenever the CPU is otherwise idle.  Then do round 7: the last weak
regime is long-form text (~51% acceptance at 1500-2000 tokens against 68-96% short).  Collect ~50-60 fresh
long-form prompts (1500-2500 tokens, the lowest styles: dialogue, uncommon-language code, essays, reviews,
emails, "do not stop early" factuals) through the serve with STRATA_Q36_MTP_PROBE + STRATA_Q36_JEV_PROBE,
hold out 8-10 prompts, assemble windows as probe9 was (cut by pos deltas, h shifted by one), merge with
probe9 keeping the replay rule, train from head-op6.pt (lr 1e-4, 400 steps, best-by-eval), then run the
full A/B with round6-measure.sh plus the long-form held-out.  One step per commit with numbers; warmed
greedy requests stay byte-identical (serving/moe-exact.sh, 5 bodies); never git push; do not re-derive the
rejected list in the record.
```
