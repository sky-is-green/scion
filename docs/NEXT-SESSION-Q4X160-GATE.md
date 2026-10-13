# Next session: Q4X160 gate (continuation prompt, 2026-10-12)

> **Executed 2026-10-12.** Gate passed: KLD 0.7819 -> 0.7818 (delta -0.0001),
> top1 0.6566 -> 0.6588, PPL 5.4561 -> 5.4784 (+0.41%), HS 82.00 (0.00), WG
> 77.75 -> 78.00; **Q4X160 chosen** (57.615 GB table-only, ~56.0 GB with the
> companions). Window $0.62. The paired code arms for the chosen file ran the
> same day with an adapted upload-path driver. Details:
> `docs/PLE-TABLE-TYPES.md` and `docs/SHRINK-PLAN.md`. This prompt is now
> history.

**Context.** Read `~/Desktop/SCION-LOCAL-ARCHITECTURE.md`, then
`docs/SHRINK-PLAN.md` (status at top), `docs/PLE-TABLE-TYPES.md` (container
specs, the measured ladder, gate results) and `docs/PLE-HEALING-PLAN.md`
(why the correction-re-fit path is parked). Expert pruning is closed; the
PLE table lane is the active one.

## State (2026-10-12)

- **Q3X32** (type 144, 70 B/row, 8-level codebook): built, parity-validated
  (fork CPU/HIP, Strata fixture), **gated** on the 48-layer release in one
  paired window: KLD 0.7819 -> 0.7927 (+0.011), PPL 5.4561 -> 5.5269
  (+1.30%), HS 82.00 -> 81.75, WG 77.75 -> 77.50. Table-only file 53.76 GB;
  ~52.2 GB with the body/corrections levers. Paired code arms NOT measured.
- **Q4X160** (type 146, 82 B/row, one fp16 scale per 160-element row,
  16-level codebook): built, parity-validated, exporter
  `--table-type q4x160`; table-only 57.60 GB (~56.0 GB with companions);
  expected cost ~+0.45-0.75% PPL (not yet gated).
- **Prebuild complete** (2026-10-12, 95.9 min): `work/tmp-export/
  q4x160-table.bin` = **26,240,125,952 bytes**, sha256
  `915bae37c28915d0378772fffd17329c9046a606555cabfcb93f3e5cd0803015`,
  produced by `tools/strata-bridge/table_only_transcode.py`.
- Fork patch `ple-table.patch` (recipe dir, staged in hivebench) now carries
  types 144/145/146 and is verified on a pristine `qwen4exp-proto` + the
  5-patch sequence.
- Budget **$10.40** (top-up 2026-10-12); no pods running; nothing pushed.
- Two Q4X160 gate attempts aborted cleanly on 2026-10-12 (host preflight,
  then the spend floor at the old $1.4 balance). Lessons in SHRINK-PLAN; the
  upload path removes host-speed risk.

## Goal

Run the Q4X160 gate (KLD + PPL 580 + HS 400 + WG 400) via the upload path,
then decide the shipped container (Q3X32 vs Q4X160). Optionally run the
paired code arms for the chosen file (~$1.5-3).

## Commands

1. Verify the prebuild (expect: unit inactive, file complete):
   ```sh
   cd ~/Desktop/work
   systemctl --user is-active p5-table-prebuild     # expect "inactive"
   ls -la tmp-export/q4x160-table.bin               # expect 26240125952 B
   sha256sum tmp-export/q4x160-table.bin            # expect 915bae37c28915d0...
   ```
2. Book + run the upload gate (A6000 preferred, L40S fallback). The driver
   uploads the table bytes in 5 GB chunks, the pod downloads the release and
   splices the bytes (`--table-bytes`), then runs KLD(release), KLD(table),
   PPL/HS/WG:
   ```sh
   cd work/hivebench/artifacts/ternary/moe/qwen4exp/pod/remote-w1w2/recipe
   systemd-run --user --unit=p5-q4x160-book3 --collect \
     --setenv="JOBDIR=/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/hivebench/artifacts/ternary/moe/qwen4exp/pod/remote-gate-q4x160-up" \
     --setenv="DRIVER=pod-table-gate-up-run.sh" \
     --setenv="UNIT=scion-tgu-driver" \
     bash "$PWD/pod-table-gate-book.sh"
   ```
   Monitor with
   `while systemctl --user is-active scion-tgu-driver; do sleep 60; done`;
   artifacts land in `pod/remote-gate-q4x160-up/bench/` (kld-release.json,
   kld-table.json, table-ppl.log, table-hellaswag.log, table-winogrande.log).
3. Read against the in-window release baselines: KLD mean 0.7819 / top1
   0.6566; PPL 5.4561; HS 82.00; WG 77.75. Acceptance: KLD delta < ~+0.02
   and PPL < ~+0.8% => ship Q4X160 (56.0 GB); anything worse than Q3X32's
   measured cost defeats its purpose.

## Fallbacks / notes

- Builder paths in `pod-build-table.sh`: uploaded bytes (`TABLE_BYTES`, the
  driver default), parallel pod encode (`TABLE_PARALLEL=1` via `PAR=1` on the
  driver), or the sequential host-preflight path (default).
  `parallel_table_transcode.py` is byte-identical but its one attempt did not
  finish on a pathologically slow host - add per-worker progress logging and
  cap workers (it passes `TABLE_PROCS`, default 32) before reuse.
- Code arms: the paired-code infrastructure (`pod-paired-code-run.sh`)
  works with any file; budget ~$1.5-3. The release-side baselines are
  gen 68.3/61.2 (all/test pass@1) and hard 50.0/49.0.
- Body/corrections companions (body Q6_K/Q8_0 -> Q4_K/Q5_K ~-1.3 GB,
  corrections mixed ~-0.3 GB) are unbuilt; one window could gate them.
- After a decision: update the model card (sizes + measured deltas) and these
  docs; nothing is pushed without asking.

## Constraints

- Never push git. Other agent lanes share the box: long jobs via
  `systemd-run --user --unit=...`; never kill their units.
- Ask before booking (balance $10.40 as of 2026-10-12).
