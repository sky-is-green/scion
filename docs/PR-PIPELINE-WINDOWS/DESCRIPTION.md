# PR draft — `--pipeline-windows 2` measured on two gfx11 cards

**Status: draft only. Nothing pushed, no PR opened.** This directory carries
everything needed to file it once approved:

- `pipeline-windows-gfx11-measured.patch` — the exact commit, based on upstream
  `main` @ `ed4773eb` (v0.1.44). One docs file, +16 lines.
- Local branch: `pipeline-windows-gfx11-measured` @ `efd6b857`, in the
  worktree `work/strata-pr-pw` (sibling of `work/strata`).

**Why a patch and a fresh fork are needed:** upstream re-imported its history
(v0.1.44's root is a fresh commit, `merge-base` with our fork is empty), so our
fork cannot open a PR against `Niko1221/Strata` directly. The steps at the
bottom create a fork whose history *is* theirs and apply this patch there.

---

## Title

`docs/MULTI_GPU: pipeline-windows on two gfx11 cards measured (+65% decode, draft layer stays on the last card)`

## Body

### What

`docs/MULTI_GPU.md` gains one measured bullet for `--pipeline-windows 2` on two
RX 7900 XT (gfx11), with the draft layer left on the last card.

### Why it may matter beyond our box

- The 0.1.44 two-stage auto rule opens only where `mtp_first_default_on()` is
  true (gfx12 WMMA parts) because "two stages pay off only with the drafter on
  the first card". On this gfx11 pair the opposite holds: the flag pays **+65%**
  with the drafter on the **last** card, and moving the drafter to the first
  card (`STRATA_MTP_STAGE0=1`) **loses 12-17 tok/s**. If that reproduces on
  other RDNA3 pairs, `pw_auto`'s min-stage rule could decouple the two.
- No quality trade observed: from the third request on, the full greedy text is
  byte-identical to the same engine without the flag (details below).

### Measurement

Scion-FlashNext 176B-A6B (release Q6K, embedded correction adapters, served
natively; engine 0.1.44 plus a local branch that ports the artifact support -
PTQ1_0 experts, P5 PLE table types, the correction adapters - and touches no
pipeline code). 2x RX 7900 XT 20 GB, Ryzen 7 7800X3D, 30 GB RAM, ROCm 7.2,
display on the second card. `--layer-split auto` (26/22), `--expert-cache auto`
(12007 + 10690 resident slots, 92% of the routed pairs), `--spec 4
--spec-min-p 0.5`, `--ple-io direct`, `--mmap-experts`, `--vram-reserve-mib
3072`, prefill 8192, expert profile.

Protocol: 4 requests x 128 tokens, temperature 0, one essay prompt, run 1 is
warm-up; the table is `gen tok/s` from the server timings.

| arm | runs 2-4 | steady |
| --- | --- | --- |
| off | 41.4 / 41.8 / 41.0 | ~41 |
| `--pipeline-windows 2` | 61.7 / 68.4 / 70.5 | **~70-71** (72.1 / 72.8 / 70.1 / 70.5; 70.7 / 70.9) |

Prompt reads 61.5 -> 98.4 tok/s. `STRATA_DECODE_TIMING` warmed window: verify
32.6-34.6 ms at 2.2-2.4 tokens/window, GPU-reach wait ~24-26 ms dominating.

Exactness: the full generated text (reasoning + content) is **byte-identical**
to the same engine without the flag from the third request on, and identical
across repeated warmed runs; the first two requests differ in both directions
because the adaptive tier is still swapping (`2.9 swaps/window` settling to
~0.3; run 1 is the documented warm-up). We did not use
`STRATA_IQ_MT_MIN=1 --pcie-frac 0 --adapt-every 0 / FORCE_MISS` for a
bit-exactness claim - the warmed-text comparison is what we ran.

Placement and split also measured: `STRATA_MTP_STAGE0=1` -> 52-59 tok/s
against 68-71 (keep the drafter on the last card); explicit `--layer-split 28`
-> 32.6 ms/window against 34.6 for auto, inside the run spread (auto kept).
Cost, from the log: "284 MiB of CUDA0 kept out of the expert cache (the second
verifier, the GDN snapshots)"; the start line reads "two verifiers per stage,
2 stages overlap across windows (decode too; the GDN snapshots beside the
window)".

### Reproduction

```
# engine: 0.1.44 + the artifact-support branch (no pipeline changes), --serve
... --pipeline-windows 2 --spec 4 --expert-cache auto --mmap-experts ...
# the fixed 4-request protocol (run 1 warm-up, 128 tokens, temperature 0)
bash serving/bench_strata.sh <label>
# warmed window timing
STRATA_DECODE_TIMING=1 in the engine env; the log prints ms/window per request
```

### Caveats

- One box, one model, greedy, essay-prompt decode; the vendor's 4x 7900 XT
  four-stage measurement (59.4 -> 67.6 tok/s at 4K, +27% at 32K) agrees in
  direction.
- The warmed-text identity check is weaker than `FORCE_MISS` bit-exactness; we
  state exactly what was run.

---

## Related PRs / issues (checked 2026-10-14; 504 open PRs in total)

Nothing duplicates this measurement.  The relevant threads, with one correction to
an earlier assumption in this draft:

- **#1656 (closed, ported into v0.1.42)** - the feature's home thread.  Niko1221's
  test there is where the "2 cards lose ~3%" comes from: **2x R9700, code -3.3%
  (0 of 4 pairs faster)**, which is why the flag stays opt-in; Cass67 replied that
  the flag's VRAM reserve can push a few experts to the CPU on a tight pair and
  "that alone could be the 2-card -3%".  Our +65% on a pair where nothing spills
  (92% resident, `--pcie-frac 0`) is evidence for that sizing reading.  **The speed
  result belongs in this thread - comment drafted in `COMMENT-1656.md`.**
- **#2245 (open)** - `--pipeline-windows 2` + MTP cross-queue stall (gfx1030; the
  same on Windows/gfx1100 per bcrawford-cpm), avoided on gfx1030 by
  `GPU_MAX_HW_QUEUES=2`.  Any proposal to default the flag on for gfx11 must
  reference this; it argues for an OS/driver-gated default, not a blanket one.
- **#2224 `pcie: PcieAuto default on gfx1100 too`** - **correction:** its diff adds
  a *separate* `pcie_auto_fork_arch_ok()` (gfx1100) consumed only by
  `pcie_frac_auto`; it does **not** touch `decode_defaults_arch_ok()`, so it has no
  drafter-first side effect.  No overlap with our ask.
- **#1120 `layer split: overlap the stages' hand-off` (opt-in, 2x RX 7900 XTX)** -
  the serial-window version of the same overlap; composes with ours.
- **#1813** exact speculative-window token parity gate (and **#1419** replay tools) -
  the rigorous form of our warmed-text identity check.
- **#1747** results-only pipeline-windows bench PR (4x GP100, +14.6-28.8%) - the
  format works upstream; **#1477** edits the same docs section (rebase watch).

---

## Filing steps, when approved

```sh
# 1. a fresh fork whose history is upstream's (the old fork cannot compare)
gh repo fork Niko1221/Strata --clone /path/to/strata-new
cd /path/to/strata-new
git checkout -b pipeline-windows-gfx11-measured main

# 2. apply the patch (it is based on main @ ed4773eb)
git am work/scion/docs/PR-PIPELINE-WINDOWS/pipeline-windows-gfx11-measured.patch

# 3. push to the fresh fork and open the PR
git push -u origin pipeline-windows-gfx11-measured
gh pr create --repo Niko1221/Strata --base main \
  --title "docs/MULTI_GPU: pipeline-windows on two gfx11 cards measured (+65% decode, draft layer stays on the last card)" \
  --body-file work/scion/docs/PR-PIPELINE-WINDOWS/DESCRIPTION.md   # body section above
```

The local branch and worktree (`work/strata-pr-pw`) stay in place until then.
