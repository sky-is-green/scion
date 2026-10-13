# Comment draft for #1656 (draft only - post on approval)

Target: https://github.com/Niko1221/Strata/pull/1656 (the ported pipeline-windows
thread, where the 2x R9700 "code -3.3%" opt-in decision lives).  gh is
authenticated as `sky-is-green`, so posting is one command when approved:

```sh
gh pr comment 1656 --repo Niko1221/Strata --body-file work/scion/docs/PR-PIPELINE-WINDOWS/COMMENT-1656.md
```

---8<--- text starts here ---8<---

Adding a two-card gfx11 data point, because the 2-card result in this thread
(Niko's 2x R9700, code -3.3%) is what keeps the flag opt-in.

On 2x RX 7900 XT (gfx1100, Linux, ROCm 7.2, display on the second card; engine
0.1.44 plus a local branch that only adds artifact support - PTQ1_0 experts, P5
PLE table types, the embedded correction adapters - and touches no pipeline
code), fixed 4-request 128-token protocol, temperature 0, one essay prompt, the
drafter on the last card:

| arm | runs 2-4 (tok/s) | steady |
| --- | --- | --- |
| off | 41.4 / 41.8 / 41.0 | ~41 |
| `--pipeline-windows 2` | 61.7 / 68.4 / 70.5 | **~70-71** |

Prompt reads 61.5 -> 98.4 tok/s; warmed verify window 32.6-34.6 ms at 2.2-2.4
tokens/window (GPU-reach wait ~24-26 ms).  Config: `--layer-split auto` (26/22),
`--expert-cache auto` resident 12007 + 10690 slots = 92% of the routed pairs,
`--spec 4 --spec-min-p 0.5`, `--ple-io direct --mmap-experts`, `--vram-reserve-mib
3072`, `--pcie-frac 0`.

Three things that may explain the difference from the 2-card R9700 number:

- **Expert residency.**  Nothing spills here - the windows are GPU-bound, the
  condition the docs say the flag pays in.  Cass67's sizing reading (the flag's
  cache reserve pushing a few experts to the CPU on a tight pair) fits: we lose
  ~284 MiB of CUDA0's cache to the second verifier and the GDN snapshots and
  still measure this.
- **Drafter placement.**  On this pair the drafter must stay on the **last**
  card: `STRATA_MTP_STAGE0=1` (the placement 0.1.44 defaults to on gfx12)
  measures **52-59 tok/s against 68-71** - the 26-layer card becomes the
  straggler when the chain moves onto it.  If the two-stage auto default is ever
  widened toward gfx11, it should not carry the drafter move with it.
- **Exactness.**  The warmed greedy text is byte-identical to the same engine
  without the flag (from the third request on; the first two differ in both
  directions while the adaptive tier swaps, 2.9 -> 0.3 swaps/window).  The
  forced-miss parity in CC-David-CC's #1813 is the rigorous form of that check.

One caveat for any default change: #2245 (open) reports a rare cross-queue stall
with `--pipeline-windows 2` on gfx1030 and on Windows/gfx1100, avoided there with
`GPU_MAX_HW_QUEUES=2`.  On this Linux/ROCm 7.2 pair, ~30 requests ran clean with
both drafter placements - a data point, not a claim that the stall is absent
(their rate is ~0.7 per 100 requests).
