# Run plan (PLAN 13a step E draft) - v2 corrections on a rented pod

Status: **DRAFT 2026-11-13, awaiting approval (step F).  No pod has been
created.**  This document sizes the run from local measurements, states the
exact commands, resume points, artifact sync, cost cap and stop rules.
Everything marked MEASURED comes from this box; EXTRAPOLATED items are
re-measured on the pod at stage start before any long stage begins.

## Hardware and cost basis

### Local box VRAM policy (2026-11-14)

This box has 2x RX 7900 XT (20.5 GiB each).  **The displays are on card1**
(both DP connectors): it must keep ~2.5 GiB free for the desktop, i.e. the
engine's usage on it is capped at ~18 GiB.  Card0 drives nothing and may run
close to capacity (~1 GiB reserve).  In the strata serve configs this is
`"gpu": [1, 0]` (the FIRST split card gets `--vram-reserve-mib`), so the
display card carries `--vram-reserve-mib 3072` and the free card
`--vram-reserve-later-mib 1024`.  The gpu-list order is the reserve order -
do not "fix" it back to [0, 1].

**/tmp on this box is a 16 GiB tmpfs** (reboot-wiped, RAM-backed).  A 14.5
GiB logits dump written there hit ENOSPC mid-row; `std::ofstream` fails
silently afterwards, so the file stopped growing at ~15.3/17 chunks while the
run still printed its final perplexity.  Logit dumps (and any multi-GB
artifact) go to the work volume, never /tmp.

**The fork's HIP binaries can hang at backend init even with `-ngl 0`**
(observed on test-mtp-probe 2026-11-14: 0% CPU, model never touched, no
error).  For CPU-only jobs use the CPU-only build tree (`build/`, OpenMP ON)
- e.g. `build/bin/test-mtp-probe` - and keep the HIP tree (`build-hip/`) for
strata-style serving paths.

### Pod options

- Pod options (runpod, current stock unknown): 1x RTX 4090 24 GB (~$0.40/h),
  1x L40S / A6000 48 GB (~$0.8-1.6/h), 2x 48 GB (~$1.6-3.2/h), A100/H100 80
  GB (~$2-4/h).  The account spend limit is $80 and the balance ~$5.50:
  **the full 4096-step two-arm run at 4k windows does not fit the $80 cap**
  (see the runtime math below).  The plan is therefore: paid PILOT first
  (measures everything), then a decision with real numbers.

## UPDATE 2026-11-14: stages 0-3 are DONE locally, for $0

The teacher gate, the traces, the logit dump and the cache no longer need a
pod.  Measured on this box (2x 7900 XT, 30 GB RAM, NVMe streaming):

- **Serving the teacher: strata + the NATIVE MTP head.**  FlashNext IS the
  ternarized Qwen3.8-Flash-Next, and the Qwen3.8 BF16 checkpoint ships the
  body-aligned MTP head.  `strata/tools/mtp_fetch.py` pulled the 31 `mtp.*`
  tensors by HTTP range request (5.21 GB), `mtp_pack.py` packed them to
  `mtp-q2_0.gguf` (0.889 GB), `mtp_rt.py` wrote the engine runtime
  (`strata-data/mtp/rt`).  Serving config: `v2/configs/flashnext-q6k-176b.json`.
  Measured decode: **21-58 tok/s** with --spec 4 (the prism fork's
  llama-server, by contrast, is unusable on this box: 0.1 tok/s - its HIP
  build falls back on the qwen4exp custom ops; the 35B HIP perplexity was
  also slower than the known CPU baseline).
- **Teacher gate PASSED (free, 2026-11-14 ~00:50).**  30 traces (10 per
  family, 4096 cap, T 0.6 seed 1001 - the frozen teacher stage) via
  `capture_rollouts.py` against the local engine: median 10-gram dup 0.002,
  max 0.103, **zero traces in the degenerate zone** (>= 0.35).  The teacher
  does NOT loop on the loop-prone prompts.  Record:
  `v2/scripts/loop_signature.py --dir v2/traces/flashnext`.
- **Cache: BUILT and verified (2026-11-14).**  17/17 chunks survived the
  re-dump to the volume (the first dump died at 15.3/17 on the 16 GiB
  /tmp tmpfs).  `v2/traces/flashnext-cache.pt` (120 MB, 17 records, 2047
  rows/chunk, top-k 512, tail 64) verified: teacher mass 0.9986 (min
  0.854), row-0 top-1 matches the embedded next-token target, tidx/tlp
  present for the full recipe incl. D_KL2.  Two builder bugs were fixed on
  the way: the truncation path seeked the payload 32 KiB short (every row
  corrupted - "teacher mass 512.0") and non-multiple truncated dumps were
  rejected instead of floor-ed.
- **Consequence for the pod:** the pilot shrinks to the TRAIN stage only
  (~$8-12: setup + 200 steps + the 70 GiB load transition).  Stages 0-3
  (~$5-15 + transfer) are eliminated.  The full 360-trace corpus is also
  local now (~7-10 h of strata time) if the pilot's budget math wants it.

## What is measured locally (4-layer prefix, one 7900 XT 20 GB)

From `v2/MEMORY-PORT.md` (real recipe geometry, rank 512 both, tail terms on):

| config | 2k | 4k | 8k |
|---|---|---|---|
| dense v1 | OOM (18.3 GiB) | OOM | OOM |
| compact + ckpt g64 | 15.4 GiB OK | OOM (16.5) | OOM |
| compact + ckpt + chunked-CE | not run | **13.9 GiB OK** | OOM (18.6, +1.7 needed) |

- 4k windows on the FULL model extrapolate to ~20.5 GiB (activations x10
  assumed linear; vocab terms and the GDN transient are layer-independent):
  fits a 24 GB card tightly, a 48 GB card comfortably.  8k needs ~25-28 GiB
  -> 48 GB class.  **EXTRAPOLATED - re-measure at pod start with the real
  40-layer model before committing to a long stage.**
- Teacher dump (MEASURED on the 35B, 20 GB): 35.6 s per 4x512-chunk pass ->
  ~8.9 s per 512-token chunk.  The 176B Q6_K is ~5x the model and runs at
  4k context: budget 1-3 min per 4k chunk EXTRAPOLATED; 30 traces x ~4k
  tokens = ~30 chunks -> 1-1.5 h of pure dump + model load ~5-10 min.
- Cache build (MEASURED vectorized): minutes for 30 chunks (the scalar path
  was ~30 min per 60k rows).
- Eval (MEASURED): the harness on the v1 model, one item R07 = 452 s for a
  32.5k-token generation; the full subset30 baseline rerun is running now
  (~2.5 h expected on one 7900 XT).  On a pod card, ~2-4x faster.

## The runtime math that sets the shape of the plan (EXTRAPOLATED)

Local prefix at 4k with the full switch stack: ~13-15 s/step for 4 layers
(including layer recompute).  Layer work scales x10 for 40 layers ->
~2-2.5 min/step -> 4096 steps = 5.7-7 days PER ARM.  Even at the optimistic
end this is far beyond the $80 cap (2 arms x ~140 h x 2 cards x ~$1/h ~
$560).  Three consequences, all reflected below:

1. The PILOT (200-500 steps, ONE arm) is the only thing sized to the cap.
2. The pilot's measured s/step decides whether ANY full run is affordable,
   and at what window length (512-token windows are ~4-8x cheaper per step;
   the 176B FlashNext trained its 4096 steps at 512 on one H200).
3. Step count may trade against window length: the gate (loop onset > 16k)
   needs long-window DATA, not necessarily 4096 steps of it.  The freeze's
   falsification criterion stays; the budget may force a shorter schedule
   (e.g. 512-1024 steps at 4k) - decided with the pilot's numbers, not now.

**F0 (full-model 2-step probe on both local cards) is NOT feasible on this
box - corrected 2026-11-13.**  The load transition needs the whole BF16
host resident somewhere: 69.4 GiB of weights vs 40 GiB of VRAM + ~28 GiB of
usable RAM = ~68 GiB, and the compact conversion needs transient room on
top.  The pod does not have this problem (48 GB card + abundant RAM), and
the pilot's own startup `memory_report` is the measurement F0 would have
given: resident model, peak VRAM and s/step at the real 4k cache, printed
before the first checkpoint.  Consequence for sizing: the pod needs >= 70
GiB of VRAM+RAM combined for the load transition (a 48 GB card qualifies;
24 GB + 64 GB RAM also qualifies), which is now part of the pod
requirements below.

## Stage plan (all commands already exercised locally)

0. **Preflight (no GPU)**: upload the corpus
   (`v2/rollout-corpus-manifest.json` + prompts), the repo at the approved
   commit, and confirm model shas:
   - body `4e9e3292...cffa040` (11.3 GB)
   - FlashNext teacher Q6_K 60.2 GB
   - empero-hf (the BF16 training host)
1. **Setup** (`bash moe/runpod-pilot.sh setup`): pinned deps, the fork build
   (llama-server + llama-perplexity), models.  ~30-45 min.
2. **Teacher + score** (`teacher`, `score`): 30 teacher traces at 4k cap
   (budget ~$5-15), dump at `-c 4096`, cache via
   `build_trace_cache.py --top-k 512 --seq 4096 --tail-logits 64 --seed 0`
   (~210 MB).  The dump (~30 GB) is deleted after the cache build.
   **Stop check**: cache sanity printed by the builder (teacher mass ~0.9+,
   layout detected, ids present); abort if the teacher's own traces loop.
3. **Pilot train** (`train`, 200 steps, cur05 recipe, the full switch
   stack): MEASURED VRAM at start (the trainer's memory_report prints
   resident + peak), MEASURED s/step from the log.  Cost ~$2-6.
   **Stop check**: loss shape sane (lm finite, kd decreasing), no NaN, no
   eval-set contamination (the corpus manifest check), VRAM peak within the
   card.  If s/step x remaining budget cannot fund >= 1024 steps, STOP and
   report - the plan changes before more spend.
4. **Local eval of the pilot checkpoint** (FREE, this box): export ->
   merge --replace -> audit -> the 2-item smoke.  A 200-step pilot is a
   pipeline check, not a capability claim.
5. **Decision point (step F, with you)**: the pilot's measured cost/step,
   VRAM and cache quality decide whether a full (or reduced-schedule) run
   fits the cap.  Nothing longer runs without a new approval.

## Resume points and artifact sync

- Cache: resumable by construction (chunks); a crashed dump yields partial
  data (truncated-dump tolerance, loud warning).
- Training: `--ckpt-every 100` (pilot) / 500 (full) + `--resume` (step from
  the filename; optimizer restarts fresh - the documented convention).
- Sync BEFORE termination: `$Q/*.pt` checkpoints, the adapter + merged GGUF
  (11.3 GB - download last, verify sha256), all eval JSONs/summaries, the
  train log, and the cache (210 MB, reusable for the fallback arm).
- The volume keeps everything if the pod survives; sync is for the
  terminate-at-any-moment rule.

## Cost cap and stop rules (binding)

- Hard cap: **$80 account spend** (the existing limit).  The pilot is
  approved to run to its natural end within ~$15-25 total.
- Abort immediately on: NaN loss; the sharpening signature (entropy falling
  >2 nats with peak mass >0.5 - the W1 signature); eval-set contamination;
  an engine/card fault that recurs after one restart; or the cache
  sanity check failing.
- The teacher's own traces looping = a TEACHER problem (the freeze's
  "what would make us change this freeze" clause): stop, report, re-plan.

## Protocol notes carried from the local phase

- The eval runs with the recovery sentence appended to the harness system
  prompt (hard_subset.py does this by default now; --no-recovery-sentence
  opts out).  The baseline under the new protocol is being rerun on this
  box (v2/eval/hard-subset30-scion-recover/) - the v2 arms compare against
  THAT number, not the old 3/30.
- Recovery "recover" mode is live with the sentence (verified: detection
  fired at 4,608 tokens, coverage 0.277, the reply completed without
  truncation); "stop" mode is the fallback if the pod's server behaves
  differently.
- Every v2 export is re-audited to 0 ppm against the training seed
  (audit_merged_adapter.py + export_audit.py --branches-only) before any
  eval number is recorded.
