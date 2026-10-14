# The EP HIP port — expert parallelism on two 7900 XT (working notes, 2026-10-14)

The active lane after `NEXT-SESSION-EP-HIP.md`. Upstream's `--expert-parallel-device` is SYCL-only (PRs
#1798/#2205): card 0 runs every layer, the peer holds the experts card 0 does not and both compute their own
experts of each window at once. This is the HIP port in `work/strata-ep` (branch `ep-hip`, rebased onto
`scion-0.1.44` = `e348c540` first).

## What HIP required that SYCL did not

**HIP IPC between the two cards fails** ("invalid device context"; `/tmp/opencode/ipc_test.cpp`), so there is no
P2P region to write the peer's rows into. Measured before porting (`/tmp/opencode/ep_xdev_test.cpp`,
`ep_flag_test.cpp`):

- one `cudaHostAlloc(Mapped|Portable)` region gets the **same VA on both cards** (HMM-mapped host memory); a
  kernel on card 0 writes it, a kernel on card 1 reads it, both directions, after stream syncs.
- the flag protocol works GPU-to-GPU with no host involvement: card 0's kernel stores + `__threadfence_system`
  then raises a mapped flag; card 1's kernel spins on it with volatile loads and then reads the data (and the
  reverse).  epoch flags (`(int32_t)(flag - want) < 0`, both counters bump once per window) work across 5
  windows with the two cards racing.
- two `cudaStreamBeginCapture(..., ThreadLocal)` on two devices' streams in one thread **work** on this ROCm
  (both graphs end clean), so one `record_window` pass records both cards' graphs.

The port therefore exchanges through host-mapped memory: one host allocation per region (inbox, in-flag lines,
peer rows, return-flag lines), with a device alias fetched for each card; `cudaHostAllocPortable` is enabled
for `--expert-parallel-device` (`set_peer_portable`).  The rows card 1 computes are written into the mapped
region and card 0 copies them into `parts_` with a volatile-read kernel (`ep_copy_peer_rows`) - the same
"inbound PCIe writes do not invalidate caches" reasoning as the engine's CPU-rows path.

## The files

| what | where |
| --- | --- |
| kernels | `include/strata/kernels/ep_kernels.hpp`, `src/kernels/cuda/ep_kernels.cu` |
| plan | `resident_plan_kernel` + `resident_plan_ep` in `src/kernels/cuda/verify_kernels.cu` |
| parity test | `src/kernels/ep_parity.cu` (`ep_parity`; `EP_PARITY_NO_B=1`: plans only) |
| verifier | `VerifyEp`, `ep_init/ep_free/ep_check` and the `record_window`/`capture`/`run` hooks in `verify.{hpp,cpp}` |
| placement/fill | `--expert-parallel-device` in `src/program/generate.cpp` (snake, peer cache, VerifyEp) |
| server | `layer_split_of` yields for `--expert-parallel-device` / `--expert-cache-device1..3` |

The protocol per window (one ring per layer-group, `ring = (l-lb)*G+grp+1`):

```
card 0 graph: ep_bump(ctr0) ... layer l: push(ids, q8_1 -> inbox[grp], inflag[ring] = ctr0)
                     resident_plan_ep(role 1)      plan only its own entries
                     grouped(kernel) -> parts_     its experts
                     ep_wait(ret[ring] >= ctr0); ep_copy_peer_rows -> parts_
card 1 graph: ep_bump(ctr1) ... layer l: ep_wait(inflag[ring] >= ctr1)
                     ep_copy_uncached(inbox -> in1); resident_plan_ep(role 2)
                     native_expert_grouped -> rows1; ep_send_rows(rows1 -> mapped rows, ret[ring] = ctr1)
```

An expert on neither card is a plan error (checked by `ep_check` after every window); every spin is bounded
(`STRATA_EP_SPIN_MAX`, default 2e6) and a bound writes the ring into mapped error words the host reads.

## Parity (green)

`./build-hip/ep_parity` on 2x7900XT: **151 plan cases, 0 failures** (both roles against a host replay; role 1
with every expert on card 0 word-for-word equal to `resident_plan`; an expert on neither card leaves an empty
plan and sets the error) and **5 plain + 5 captured-graph exchange windows, 0 failures** (checksum of the ids +
q8_1 words card 1 copied, window-dependent peer rows bit-exact in `parts_`, card 0's own rows untouched).
`resident_plan_parity` still passes (the kernel grew the EP parameters).

## Measured so far

**Phase A (the existing host-synced helper tier, the baseline the port must beat).** Server transient on 8081,
no layer split: `--expert-cache auto` + `--expert-cache-device1 auto` + `--remote-expert-opt`, same bench
protocol. Card 0: 10437 slots; CUDA1: 14139 additional experts (14.16 GiB).

| run (fixed protocol) | gen tok/s |
| --- | ---: |
| warm 1 | 68.6 |
| warm 2 | 68.9 |
| warm 3 | 68.9 |

- card busy during a 400-token decode: **drm card1 (HIP 0, primary) avg 44.4%**, **drm card0 (HIP 1, helper)
  avg 11.5%** (max 44%) - the helper idles between layers; its wait is 1705 ms of a 6497 ms decode (26%).
- so the host-synced hand-off loses to the pipelined config (74-78): **68.7 vs 74-78**.

**Phase B (the port), first served run.** `--expert-parallel-device 1`, no layer split:

- card 0 holds 217 experts/layer (10.43 GiB), card 1 295/layer (14.18 GiB, filled in 23.5 s, slot 0 verified)
  - **every expert is on one of the two cards** (217+295 = 512), so the zero-doorbell window runs with both
    tables.
- warm decode: **88.4 / 90.4 tok/s** on the fixed protocol (vs 74-78 pipelined).  A third warm run collapsed to
  5.7 tok/s because the other lane's 35B engine started loading on the box mid-run; not an EP number.
- the A/B window (bench both arms from the same build, card busy, output capture) is
  `work/scion/serving/ep-ab-window.sh`; it needs exclusive cards.  `gpu_busy_percent` on drm card0 reads a
  stuck ~100% since the standalone cross-card tests (~23:30), with the card empty (45 MiB) and 38 W idle; the
  A/B samples `power1_average` beside it as the independent busy signal (idle 38/70 W; decode expected
  200-300 W).

**Sizing note (why loans stay on).** With `--prefill 8192`, `no_prefill_borrow` reserves ~5.6 GiB on card 0
for the prompt path's own buffers, card 0's cache drops to ~103 slots/layer and the pair can no longer hold
every expert (the peer needs 19.66 GiB, 17.3 available).  Loans (the default with a profile) leave card 0
~217/layer and the peer 14.18 GiB.  Upstream's no-loan rule does not fit two 20 GB cards at this chunk; the
serial serve reads a prompt and returns its loan before decode windows run, and a window started while a loan
is out is refused by `capture()` (loudly), never computed wrong.

## Not done / next

- **Prefill EP** (upstream #2205): prompts read the peer's experts on card 0 from the GGUF (correct, slower);
  `set_ep_peer`'s prompt-chunk path over mapped regions is the follow-up.
- Batch windows refuse EP (the row layouts were not ported); `STRATA_EP_VERIFY`/`STRATA_EP_SELF` were left out
  (the parity test covers the kernel contract; a same-residency single-card oracle would be the A/B for the
  rounding-exact claim).
- The A/B window must run when the box is free (the 35B lane shares it).
- File the summary upstream either way (the vendor's engine; the SYCL PRs are the precedent).
