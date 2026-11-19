# 35B v2 pilot — 2026-11-19

Status: **complete; pod stopped.**  This is the measured cost/VRAM gate for the
full cur05 run.  No full run has started.

## Pod

- Runpod id `blqtl4901ye2p1`, Secure Cloud, **US-WA-1** (scheduler chose it;
  A100 SXM 80 GB was LOW in US-KS-2/US-MD-1/US-WA-1).
- 1x **NVIDIA A100-SXM4-80GB**, $2.19/h; 50 GB container disk + 150 GB volume
  at `/workspace`; 2 TB host RAM.
- Uptime 4,392 s (73 min); **cost $2.61**; balance after stop **$2.89**.
- The API key is scoped to pods/billing/catalog; it lacks `account:read`
  (`/account/limits` 403) and `pods/{id}/exec` 403.  Commands ran over the
  SSH proxy (PTY) and direct SSH for `scp`.

## What ran

1. `setup`: repo at `8549e96`, `transformers 5.23.0` (pulled in by
   `flash-linear-attention`; the 5.5.0 pin was overridden), `triton 3.8.0`
   (torch 2.8 wants 3.4.0; no observed failure), body 11 GB + empero 68 GB.
2. `smoke`: **PASS** — hidden drift **0.4395**, router top-8 **0.8251**, no
   fast-path warning.
3. `curriculum` (pilot-only 512 windows): worst CE 3.622, cut 2.294, best
   0.954; **205 hard fineweb windows + 346 clean teacher traces** (551 rows,
   2.1 MB).  The teacher corpus was loop-gated: 13/360 dropped at dup10 >=
   0.35 and 1 short.
4. `cache` (pilot-only 1024 windows): 2.51 GB, teacher support mass
   **0.9933 -> 0.9736** over the first 300, ~0.98 at the end.
5. `pilot` (200 steps): **6.10 s/step** (step 100 -> 200 wall time),
   router **frozen**, trainable **209.72M** branch params.

## Pilot losses

| step | lm | kd | tail | total | H | peak | mass s | mass t | tcond | loadH |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 2.3839 | 0.4662 | 0.0048 | 3.3404 | 2.161 | 0.5313 | 0.9746 | 0.9855 | 0.0048 | 4.403 |
| 200 | 2.6563 | 0.5134 | 0.0109 | 3.7273 | 2.581 | 0.4751 | 0.9581 | 0.9795 | 0.0074 | 4.575 |

The two rows are single batches on different windows, not a running mean; the
increase is not yet evidence of degradation.  No NaN, no sharpening collapse
(H rose, peak fell, loadH healthy).  The `lm` scale is the 35B BF16 host, not
the 176B cache's 7-9.

## Memory

- Compact-bank build: 7.50 GiB codes + 0.47 GiB scales (dense bf16 would be
  60.0 GiB), branches 0.781 GiB, routers 0.039 GiB, other 5.35 GiB ->
  **resident 14.14 GiB**, reserved 14.23 GiB.
- **Load peak 66.39 GiB**; step-1 peak 66.39 GiB.  Fits the 80 GB A100 with
  ~13 GiB headroom.  This is the number a 48 GB card cannot take without the
  compact-bank conversion being moved off the load path.

## Export / merge / audit

- Export (pod): 40 router deltas (rank 256, dense) + 160 branch tensors,
  **102.9 MB** per adapter.  `qwen35-adapter-pilot-soup.gguf` is complete;
  `qwen35-adapter-pilot-final.gguf` was truncated by the multi-file `scp`
  (96.6 MB local vs 102.9 MB header) and must not be used.  Pull after the
  export process has exited and check `stat` against the writer's size.
- Merge (local, `--replace`): **973 tensors, 10.558 GiB**; 240 v1 body tensors
  superseded; `adapter.embedded=true`.
- Branch audit (local, `export_audit.py --branches-only`):
  **160 tensors, 209,715,200 weights, 0 code3, 1,638,400 groups,
  0 scale_not_max, 0 bad_scale.**
- The audit initially reported 0 tensors because `validate_branches()`
  skipped type 142 (the v2 type).  Fixed in `v2/scripts/export_audit.py`;
  this was a real false-pass bug, not an adapter problem.

## Projection for the full run

At **6.10 s/step** and 4096 steps: **6.94 h/arm**.
At $2.19/h: **$15.20/arm**.

| item | hours | cost |
|---|---:|---:|
| cur05 arm (4096 steps) | 6.94 | $15.20 |
| full 4096 curriculum + cache + ref | ~1.4 | ~$3.1 |
| export + margin | ~0.3 | ~$0.7 |
| **cur05 total** | **~8.7** | **~$19** |
| pred2.0 fallback, if needed | +6.94 | +$15.2 |
| **both arms** | **~15.6** | **~$34** |

The $80 account limit is per hour; the balance is the total budget.  A top-up
to ~$40 covers both arms with margin; cur05 alone needs ~$22.

## Pilot-only deviations (recorded)

- **512-window curriculum, 1024-window cache**, not the full 4096/4096: the
  full curriculum was too slow for a $5.50 pilot.  The full run must rebuild
  both at 4096 and rerun `ref`.
- 10% curriculum mix (`--agentic-frac 0.10`) with 346 traces; this is the
  intended v2 mix, not a pilot shortcut.
- The pilot's `--eval-every 0`; no proxy eval.  The full run uses
  `--eval-every 1000` and needs the `ref` stage first.
- The pilot adapter is a pipeline artifact, not a capability claim.  The
  subset30 gate is a local strata run on the merged file.

## Files

- `qwen35-adapter-pilot-soup.gguf` (99 MiB, complete) — use this one.
- `qwen35-adapter-pilot-final.gguf` (93 MiB, truncated; do not use).
- `pilot.log`, `setup.log`, `curriculum.log`, `cache.log`,
  `pilot-export.log`, `curric-combo.jsonl`.

## Next

1. Top up to ~$40 (cur05 alone ~$22).
2. Start the stopped pod, or create a new A100 SXM 80 GB; reinstall the pip
   stack (the container disk is ephemeral) and verify the body sha.
3. Full `curriculum` (4096), `cache` (4096), `ref`, `train` (cur05).
4. Export, pull after the process exits, merge/audit locally, subset30 eval.
5. Run pred2.0 only if cur05 is ambiguous; select on subset30, not on the last
   step.
