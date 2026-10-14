# Next session: multi-GPU & local speed path (continuation prompt, 2026-10-13)

**Context.** Read `~/Desktop/SCION-LOCAL-ARCHITECTURE.md`, then
`work/scion/docs/P4-STRATA-FIT.md` (the fit, the perf work, the VRAM policy,
the validation instrument and the Q4X160 local A/B - all measured), then this
doc. The shrink lane is decided (Q4X160); the active lane is **local serving
speed in Strata on the two 7900 XTs**.

## State (2026-10-13)

- **The release serves locally in Strata at ~30.6 tok/s with corrections**
  (99%+ expert-cache hit; first request after start needs ~40 s to load).
  Served by the user service **`scion-flashnext`** - web chat at
  `http://127.0.0.1:8081/`, OpenAI endpoint `/v1`. Unsloth Studio's custom
  provider ("Scion Flash-Next (Strata, local)") points at it.
  `scion-flashnext-q4x160` serves the shrunk file with
  `work/strata/strata-q4x160-176b.json` (stop one before starting the other).
- **The correction adapters are the measured bottleneck**: 30.6 tok/s with
  them, 46.5 without (measurement arm `pack-q6k-nocorr`), i.e. **~11 ms per
  token**. The kernels themselves are already 2.4-2.5x faster after the
  split-4 A-pass rewrite (`lora_corr_parity` stays bit-exact against the
  updated reference). What remains is structural: 96 two-pass branches per
  token, each a pair of small kernels on the per-layer critical path, captured
  into the token graphs (host call overhead is zero - measured with a counter).
- **Unsloth Studio cannot load the release with its bundled llama.cpp**
  (`invalid ggml type 143` for PTQ1_0; the 192 correction tensors are a second
  unknown). Studio-as-UI + Strata-as-engine via the custom provider is the
  working path. Studio's `UNSLOTH_LLAMA_CPP_PATH` / `LLAMA_SERVER_PATH` knobs
  can point at our fork build, but that runtime's GPU path on this box is
  fragile (HSA stalls at full offload; ~1 tok/s at safe partial offload).
- **Two-card parallelism research (done this session):**
  - llama.cpp upstream has **experimental tensor parallelism** (`-sm tensor`,
    weights+KV split via meta device); on ROCm it needs `-DGGML_HIP_RCCL=ON`
    (RCCL), which upstream keeps OFF by default ("not universally beneficial
    during testing"). Not applicable to our custom containers.
  - Vendor Strata is at **0.1.44 (released 2026-10-13)**; our tree is 0.1.37 +
    the Scion support commit. Their multi-GPU work is **still pipeline**
    (draft layer on the first card, 8K prompt chunks across the split,
    measured PCIe share); `--pipeline-windows` is opt-in for two cards and
    "loses about 3% on code" there. RDNA3 (gfx11) keeps 0.1.42 behaviour -
    "we have no such card to measure on".
  - **Expert-parallel MoE** (each card computes the experts it holds) is the
    design that fits this box (the two cards already hold disjoint expert
    caches). HarMoEny (arXiv 2506.12417) reports **+37-70%** under load
    imbalance. Needs engine work: dispatch is per-layer-owner today.
- **VRAM headroom policy** (user rule, applied): display card keeps
  `max(2048 MiB, 10%)` free, others `max(1024 MiB, 5%)`, reserve =
  target + 1024 (`--vram-reserve-mib 3072` here). Tool:
  `work/scion/serving/vram_reserve.py --apply <serve config>` / `--check`.
  Measured basis: at the fringe the display card sat at 428 MiB free and
  decode fell to ~25.8; with headroom 27-33.
- **Banked quality instruments**: PPL via `STRATA_LOGPOS` + `--short-read 600`
  + the assistant-role trick (the dump measures end-of-turn mass on user-role
  text - do not use that shape). Baseline **7.5728** Q6K / **7.5741** Q4X160,
  paired delta **+0.017% +/- 0.079%** (not significant). Results:
  `work/scion/moe/artifacts/strata-ppl/ppl-q6k.jsonl` and `ppl-q4x160.jsonl`.
  The number is chat-conditioned and NOT directly comparable to the fork's
  raw-LM 5.4561; an exact instrument needs per-position logits in the batched
  prefill path (the head machinery exists in `verify.cpp`).
- Balance **$6.45**, no pods, nothing pushed. Strata working tree = Scion
  commit `4f7fb6b5` + working changes (P5 table types, the corrections.cpp
  layer-map fix, split-4 kernels + reference).

## Goal

Close the local speed gap, **in this order**:

**Status 2026-10-13 (night): item 1 is DONE.**  The correction path was reshaped
rather than folded into the GEMVs: `lora_corr.cu` now reads uint4 rows and
converts f16 pairs with `__half22float2`, the A pass writes `t16` directly, the
B pass was rewritten to the same shape, and a small round pass is the only
f32 -> f16 RNE step.  Measured on the release with the fixed protocol:
**41.4-41.8 tok/s with corrections vs 46.3-46.5 without** (delta **2.6
ms/token**, was ~9-11); the verify stage table's correction-attributable stages
fell **7.66 -> 1.94 ms/window**.  `lora_corr_parity` is bit-exact on its updated
reference (8 pairs, both 4l fixtures); the change against the pre-rewrite order
is <= 1.2e-07 relative on the real factors (`corr_oldnew`).  Full record:
`P4-STRATA-FIT.md` item 5.  A cooperative fused A+B kernel was prototyped and is
**shelved**: a full-grid cooperative launch wedged both GPUs and reset the
desktop (2026-10-13); the safety case it must pass is in
`work/scion/docs/COOP-CORR-SAFETY.md`.

1. **Fold the corrections into the layer matmuls** (certain ~35%) - **DONE in
   the reshaped form above**; the remaining fold (B into the Q6_K mmvq/combine
   epilogues) was measured at only ~0.3-0.5 ms/window and not worth the pinned
   kernel surface.
   Today `corr_apply` runs `t = A x; out += B t` as two kernels per branch.
   Fold it llama.cpp-style into the layer's GEMVs (the A/B factors multiply
   the activation the layer already reads), or at minimum into one fused
   kernel per branch - the goal is *no separate kernels on the critical
   path*. Files: `src/kernels/cuda/lora_corr.cu` (kernels + launcher),
   call sites `src/core/layer.cpp` (attn/ssm ~L1234, ffn ~L1300),
   `src/core/verify.cpp` (windows, ~L546/668/804), `src/prefill/prefill.cpp`
   (~L1433); `src/kernels/lora_corr_parity.cpp` mirrors whatever order is
   used and must stay bit-exact. Declared done when: parity passes, the
   served with/without-corrections delta is a few ms, and the bench protocol
   shows ~40+ tok/s with corrections.
2. **Upgrade the engine to upstream 0.1.44** (we are 7 releases behind) - **DONE
   2026-10-13 (night)**.  The histories are unrelated (upstream re-imported the repo),
   so it was a 3-way patch port, not a merge: 29 hunks resolved, first build clean,
   branch `scion-0.1.44` (`319cf7f5` port, `42bef63a` PTQ1_0 split fix) - the only Scion branch
   since 2026-10-14, pushed (the old `scion-support` lineage is retired; its correction
   reshape lives here in evolved form as `src/kernels/f16_dot8.hpp`).  Parity:
   lora_corr 8 pairs 0 differ, ple_q4 PASS, ptq1_0 0 failures; ple_x32 needs its ad-hoc
   fixture (not in the repo).  Served: 0.1.44 alone 41-43.5 tok/s; with
   `--pipeline-windows 2` (now in both serve configs) **70-71 tok/s steady**, warmed
   output byte-identical to the non-pipelined engine.  Details in `P4-STRATA-FIT.md`.
   The two-card note itself - on gfx11 the split must be asked for (the auto rule
   follows `mtp_first`), the draft layer is better left on the **last** card
   (`STRATA_MTP_STAGE0=1` measured 52-59 against 68-71), cost 284 MiB of CUDA0 out of
   the expert cache - lives in `work/strata-pr-pw/docs/MULTI_GPU.md` (worktree branch
   `pipeline-windows-gfx11-measured`, commit `efd6b857`); no upstream PR filed yet.

3. **Scope expert-parallelism across the two cards** (the real 2x path) -
   **measured and scoped 2026-10-13 (night)**: during decode each card is only
   ~50% SM-busy (card0 48.9%, card1 54.8%; memory controllers 6-11%) because
   the verifier's stage chain is serial (26/48 and 22/48 layer shares), so the
   two arms' convergence is the shared ceiling, not the corrections.  The
   offline go/no-go over the 2.4M-token routing counts (`moe/ep_sim.py`) clears
   EP: resident routed mass stays 99.7% under expert-owner splits, and the load
   split goes 55.3/44.7 -> ~50/50.  Design and next steps:
   `docs/EXPERT-PARALLEL-SCOPE.md`; the per-token split tail needs a routing
   trace before the dispatcher is sized.  Or file the summary upstream - it is
   the vendor's engine and they handle AMD.

Side items if wanted: raw-prefill logits dump for an exact PPL instrument;
body/corrections companions (unbuilt, ~-1.6 GB).

## Spec tuning sweep (2026-10-14)

Cross-checked our decode knobs against the NInfer V100 engine's published lessons
(`github.com/geoffwatts/ninfer-v100`; its own real-text sweeps are 50-93 tok/s on a
single V100-32GB for a 27B dense model, with 219 tok/s only on a synthetic
continuation). Both transferable ideas are already in the engine: an
acceptance-adaptive window policy (`src/spec/draft_policy.cpp`) and a context-lookup
suffix drafter (`--suffix-draft`, on by default). The sweep found only noise:

| `--spec` cap | steady gen tok/s | draft acceptance |
| ---: | ---: | ---: |
| 2 | ~72 | 85% |
| 3 | ~71 | 76% |
| **4 (current)** | **~74-78** | 71% |
| 5 | ~64 | 63% |
| 6 | ~58 | 60% |

`--spec-min-p` at K=4: 0.3 -> ~66 (58% acceptance); 0.5 and 0.7 -> ~74-76 (within
noise). Suffix drafts fired about once per request on both the essay prompt and a
copy-heavy code prompt ("suffix drafts: 1 windows, 1 of 3 drafts accepted"), so the
lookup path is not a lever for these benchmarks. **No config change kept** (4 / 0.5
restored); the gap to engines like NInfer is structural - their model fits one card
and ours spans two over PCIe - not a missing technique. Numbers are 4-run windows
with the policy learning inside each process; treat +-2 tok/s as noise.

## Commands

```sh
# serve / chat (user services; stop one before starting the other)
systemctl --user start scion-flashnext      # release; web UI http://127.0.0.1:8081/
systemctl --user start scion-flashnext-q4x160
systemctl --user stop scion-flashnext

# fixed speed protocol (4 requests, 128 tokens; run 1 is warm-up)
bash work/scion/serving/bench_strata.sh <label>

# engine rebuild + the four parity tests (4l file)
cd work/strata && cmake --build build-hip -j16
./build-hip/ple_q4_parity  /home/penis/models/scion-strata/4l-q2_0-q4table.gguf
./build-hip/ptq1_0_parity  /home/penis/models/scion-strata/4l-q2_0-q4table.gguf
./build-hip/lora_corr_parity /home/penis/models/scion-strata/4l-q2_0-q4table.gguf
# (ple_x32_parity needs a transcode fixture; see docs/PLE-TABLE-TYPES.md)

# VRAM reserve for any serve config
python3 work/scion/serving/vram_reserve.py --apply work/strata/strata-q6k-176b.json
```

## Notes / fallbacks

- The fork's GPU path on this box (prism-ml-llama.cpp / prism-qwen4exp-shrink
  builds) stalls in HSA at high offload - do not spend more time there; Strata
  is the local engine.
- `pack-q6k-nocorr` (hardlinked copy with the 192 `.lora_` index lines
  removed) is a **measurement arm only** - not the shipped configuration.
- PPL runs: serve with `--short-read 600` + `STRATA_LOGPOS=<file>` (config
  `work/strata/strata-q6k-ppl.json` is the template); score with
  `.venv/bin/python work/scion/serving/strata_ppl.py --assistant`.
  User-role scoring is the wrong instrument (end-of-turn mass).
- Long jobs (`cmake`, PPL runs) via `systemd-run --user --unit=...`.

## Constraints

- Never push git. Other lanes share the box (e.g. `scion-snapshot`) - long
  jobs via `systemd-run --user --unit=...`; never kill their units.
- Ask before booking pods (none needed for these items; ~$0 cost).
- Doc style (strata AGENTS.md): plain words, measured numbers, no claims
  without a measurement.
