# Next session: the HIP expert-parallelism port (continuation prompt, 2026-10-14)

**Context.** Read `~/Desktop/SCION-LOCAL-ARCHITECTURE.md`, then
`work/scion/docs/EXPERT-PARALLEL-SCOPE.md` (why EP, the offline go/no-go), then
`NEXT-SESSION-MULTIGPU-PERF.md` (how the speed lane got here), then this doc. The
speed lane is finished and banked; the active lane is the **expert-parallelism port
to HIP** that the operator asked for after the upstream SYCL work was found.

## Where the lane stands (2026-10-14)

- **Speed lane, banked.** The release serves at **~74-78 tok/s warm** on both cards
  (`--pipeline-windows 2`, corrections reshaped, engine 0.1.44; spec sweep saturated
  at K=4 / `--spec-min-p 0.5` - `NEXT-SESSION-MULTIGPU-PERF.md`). The canonical
  tree `work/strata` (branch `scion-0.1.44`) has **moved on**: its latest lane
  commits add grouped-expert kernel work worth **+4% on the 176B** (`e348c540`) and
  the 35B work reaches ~100 tok/s. Re-bench with the fixed protocol before quoting
  a number; the service `scion-flashnext` is the user-facing one.
- **EP exists upstream, SYCL-first.** `--expert-parallel-device N` in open PRs
  **#1798** and **#2205** (recutita; #2468 is the container fix): *"card 0 runs
  every layer, card N holds the experts card 0 does not; both cards compute their
  own experts of each decode window at the same time."* Measured on 2x Arc Pro B70:
  decode **94.3 -> 107.9 tok/s (+14.4%)**; #2205 adds the prompt-chunk path. Needs
  #1589 (in main). **All of its code lives under `sycl/`; there is no HIP or CUDA
  EP, and nothing is merged.**
- **Our tree already has half of it.** `--expert-cache-device1..3` +
  `--remote-expert-opt` is a second-card expert tier: *"CUDA0 keeps all dense
  weights and state; a static, profile-filled expert tier on the other device
  computes its rows during decode"* (placement `stripe` by default,
  `--expert-cache-remote-placement stripe|layer`). Its hand-off is
  **host-synchronized** - `RemoteExperts::finish` is a `cudaStreamSynchronize`
  ("the layer waits for this GPU on the CPU pool's critical path") - and it turns
  pipeline windows and batching off. `--peer-device` is prompt-only ("decode
  unchanged"). So the port's missing piece is exactly the **capture-safe concurrent
  protocol**, not the cache or the placement.
- **Cross-card memory, measured.** `hipIpcOpenMemHandle` between the two 7900 XT
  fails ("invalid device context" - no P2P); **host-mapped memory works**
  (`/tmp/opencode/ipc_test.cpp`). The engine already has the graph-safe mapped
  pattern the port needs: a 64-byte mapped flag (`m_flag_`), `wait_flag_ge()`,
  `copy_rows_from_mapped_after()` - the same shape as upstream's EpRegion + epoch
  flags, through mapped RAM instead of IPC.
- **The worktree.** `work/strata-ep`, branch **`ep-hip`**, based on `scion-0.1.44`
  (`6ab2b5b2`, a few commits behind the canonical tree), configured for HIP/gfx1100
  with tests and **built** (`build-hip/strata`). `work/strata` is the engine lane's
  active tree; `work/strata-pr-pw` and `work/strata-bonsai2` are other lanes'
  worktrees - do not disturb any of them. Upstream diffs for study:
  `/tmp/opencode/ep-sycl/pr1798.diff`, `pr2205.diff`.

## Plan, in order

**Phase A - the existing helper path as a baseline (almost no code).**
Run the engine with **no layer split**: `--expert-cache auto` on card 0,
`--expert-cache-device1 auto` + `--remote-expert-opt` (stripe), measure decode vs
the current ~74-78 and card busy (`/sys/class/drm/cardX/device/gpu_busy_percent`).
The server auto-adds a layer split whenever two GPUs are listed; `layer_split_of`
in this branch's `serve/server.py` must also yield when `--expert-cache-device1..3`
is in the args. Run it as a transient service on 8081 (the other lane's
`bench-nocorr` pattern: `systemd-run --user --unit=... python serve/server.py
--config <cfg> --port 8081`), then restore `scion-flashnext`. Expect the
host-synced hand-off to lose to the pipelined config; this number is the baseline
the port must beat, and it validates the peer cache + placement on our hardware.

**Phase B - the port.** Follow the SYCL diff file by file:
1. `--expert-parallel-device N` + its constraints + the snake placement in
   `src/program/generate.cpp` (upstream's hunk ports directly; ours has the same
   `profile` vector, `ExpertCache::layer_slot_range`, `set_per_layer_admission`).
2. The peer's work captured as its **own graph beside card 0's window graph** in
   `record_window` (`src/core/verify.cpp`), with a mapped inbox / epoch flags /
   mapped return rows replacing `RemoteExperts::finish`'s host wait.
3. Kernels: send (ids + q8_1 rows into the mapped inbox, bump the epoch), the peer
   side (wait epoch, its own `resident_plan` + `native_expert_grouped`, rows into
   a mapped buffer, bump ret), card 0 (wait ret, `copy_rows_from_mapped` into
   `parts_` - the combine then stays single-card and unchanged).
4. A parity test modeled on `sycl/src/kernels/resident_plan_ep_parity.cpp`
   (placement + a row-exchange round trip, bit-exact vs single-card), then the
   served A/B. Success metric from the scope doc: **card busy ~50% -> >80%**; the
   decode delta is what it is (upstream measured +14% on their pair).
   Rebasing `ep-hip` onto the newest `scion-0.1.44` first is cheap and avoids
   porting against stale kernels.

**Either way, file the summary upstream** - it is the vendor's engine and they
handle AMD (the SYCL PRs are the precedent; the scoping numbers are in
`EXPERT-PARALLEL-SCOPE.md`).

## Key files

| what | where |
| --- | --- |
| upstream EP blueprint | `/tmp/opencode/ep-sycl/pr1798.diff`, `pr2205.diff` |
| existing peer tier | `src/core/remote_experts.{hpp,cpp}`, `src/core/remote_expert_opt.cu` |
| window/verify integration | `src/core/verify.cpp`, `include/strata/core/verify.hpp` (`Verifier`, `VerifyHits`) |
| mapped-flag machinery | `m_flag_`/`wait_flag_ge` (verify.cpp), `copy_rows_from_mapped*` (`src/kernels/cuda/elementwise.cu`) |
| options / placement / fill | `src/program/generate.cpp` (~2030, ~2418, ~3676, ~5452) |
| expert cache | `include/strata/core/expert_cache.hpp` |
| our scope + go/no-go | `work/scion/docs/EXPERT-PARALLEL-SCOPE.md` |

## Commands

```sh
# build the port tree
cd work/strata-ep && cmake --build build-hip -j10

# the four parity tests (4l fixture)
./build-hip/lora_corr_parity     /home/penis/models/scion-strata/4l-q2_0-q4table.gguf
./build-hip/ptq1_0_parity        /home/penis/models/scion-strata/4l-q2_0-q4table.gguf
./build-hip/ple_q4_parity        /home/penis/models/scion-strata/4l-q2_0-q4table.gguf

# fixed bench protocol (server on 8081)
bash work/scion/serving/bench_strata.sh <label>

# the EP metric during a decode
for c in 0 1; do cat /sys/class/drm/card$c/device/gpu_busy_percent; done
```

## Constraints

- Other lanes share the box; **never kill their units** (`work/strata` lane,
  `bench-*`, `scion-snapshot`); long jobs via `systemd-run --user --unit=...`.
- VRAM rule: display card keeps `max(2048 MiB, 10%)`, others `max(1024 MiB, 5%)`,
  reserve = target + 1024 (`work/scion/serving/vram_reserve.py`). In EP mode there
  is no layer split: card 0 must hold all dense weights plus its expert cache.
- EP is incompatible with `--layer-split`, `--pipeline-windows`, `--peer-device`
  and batching (upstream and ours agree). The A/B is **EP vs the current pipelined
  config**, not EP on top of it.
- Ask before pod spend; never push without asking.
- Doc style (strata AGENTS.md): plain words, measured numbers, no claims without
  a measurement.
