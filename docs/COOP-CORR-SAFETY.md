# Cooperative correction kernels — the safety case (2026-10-13)

**Status: shelved pending a safety case, not rejected.** The fused A+B idea (one
cooperative kernel per correction branch: phase 1 `t = A x`, `grid.sync()`, phase 2
`out += B t`) is sound and would remove the last launch pairing from the token path.
What is dead is the 2026-10-13 prototype, which wedged both GPUs and reset the
desktop. This note records the failure, the constraints that any revival must meet,
and the test protocol, so the next session does not repeat it.

## What happened (measured)

- `/tmp/opencode/fused_proto`, a scratch binary: `hipLaunchCooperativeKernel` with
  the full `hipOccupancyMaxActiveBlocksPerMultiprocessor` grid (336 blocks x 256
  threads), return code unchecked, no timeout, launched while the serving stack was
  resident on both cards.
- Journal, 22:51:12-22:51:18: `HW Exception ... reason: GPU Hang` (node-1); KWin
  `graphics reset`, `Pageflip timed out`; Firefox `DeviceReset::DRIVER_ERROR`;
  OpenCode desktop GPU process `exit_code=133`; the running measurement server
  aborted inside `libhsa-runtime64`. Both cards recovered without a reboot.
- Facts worth keeping: a 16-block cooperative kernel *ran correctly* on this box,
  and it *captured into a HIP graph and replayed* (`/tmp/opencode/coop_capture.cu`).
  The failure is in the full-grid attempt, cause not isolated.

## What NOT to do

1. Do not size a cooperative grid at the occupancy cap in one step.
2. Do not launch a cooperative kernel without checking `hipLaunchCooperativeKernel`'s
   return code (a non-fitting cooperative grid hangs; it does not always error).
3. Do not test new device-wide kernels while `scion-flashnext`, the expert cache, or
   anything else holds the cards.
4. Do not run any grid-synced kernel without a wall-clock timeout around the process.
5. Do not treat the P4 note "cooperative needs `-fgpu-rdc`" as a reason it cannot
   work: it is wrong for this build (HIP 7.2, gfx1100). It was neither the cause nor
   a blocker.

## What a revival needs (all of it)

- **Grid cap with margin**: hardcoded conservative grid, grown only by a verified
  sweep; a failed launch (or a query that errors) falls back to the existing
  two-kernel path. The cooperative path is always an optimization, never a
  correctness dependency.
- **Bounded duration**: per-branch corrections are tens of microseconds. Any design
  where a cooperative kernel can run for milliseconds is a design error.
- **A test protocol that cannot wedge a working box**:
  1. stop every GPU-using user unit (the release server first);
  2. confine to one card (`HIP_VISIBLE_DEVICES`);
  3. run under `timeout` with a trivial probe kernel before and after each step;
  4. sweep grids bottom-up (1, 2, 4, ... blocks) and record the first failure;
  5. only after a pass: repeat under desktop load, then with the server running, and
   compare decode tok/s.
- **The display card keeps working**: the user must be able to use the desktop with
  the model loaded. Cooperative work must be short, must not hold the device, and
  must be measured with the desktop active before it ships. The existing
  `--vram-reserve-mib` policy covers memory; this note adds the time/hang dimension.

## Where it fits

If revived, the natural shape is not a standalone fused branch kernel but folding
the phases into existing kernels: phase 1 of the ssm/attn output projection computes
`W y` and `A y` together (same y), phase 2 adds `B t`; the MoE combine kernel takes
`A x` in a phase 1 and `combine + B t` in phase 2. That is "fold into the layer's
GEMVs" with the grid sync paying for itself.

Until then, the local speed lane uses plain kernels: a `uint4`/`half2` A pass
(measured 2.2-4.5x faster than the split-4 form: ssm 69.6 -> 15.6 us, ffn 26.5 ->
9.5 us on rows=3, warm) plus the existing B kernel, with a small f16 rounding
pre-pass (~5 us, launch-bound). No grid sync, nothing that can deadlock a card.
