# Scion MoE speed lane — record (opened 2026-10-14)

Lane goal (from `NEXT-SESSION-SCION-MOE-SPEED.md`): apply the closed Bonsai-2 dense lane's speed methodology to
the two Scion MoE models in Strata — **Scion-FlashNext-176B-A6B** (live on the two cards) and
**Scion-35B-A3B** (no Strata lane yet).  Engine tree `work/strata`, branch `scion-0.1.44`; fixed protocol
`serving/bench_strata.sh`; `--vram-reserve-mib 3072`; never `git push`; model-load tests only in serve-down
windows.

## Step 1 — the verify-window instrument and the warmed budget (commit `58aa4712`)

### The instrument (opt-in; both unset = no kernel, node or stream change)

| env | what it adds |
| --- | --- |
| `STRATA_VERIFY_GRAPH_TIME=1` | per replay: the GPU span (events around the graph launch), per-window-size T averages, the launch-call host time, and the captured node count per T.  One line per stage per request with `STRATA_DECODE_TIMING=1`: `strata decode replay, stage N: replay ... \| T=.. \| nodes..` |
| `STRATA_VERIFY_ABLATE=1` | bench-only: the routed expert GEMMs are skipped, so the node-count delta and (with GRAPH_TIME) the replay split are measurable.  The output is garbage. |

The ablation's T collapses to 1 (garbage drafts end the spec chain), so its replay span is not a clean per-T
delta; the expert-GEMM share is read from the existing profile stamps instead (19->20 "VRAM hits" brackets the
grouped expert launches on the GPU clock).

### The warmed budget of the live serve (2026-10-14; Q6K + corrections, `--pipeline-windows 2`, reserve 3072)

Warmed 128-token greedy request, 2.33-2.37 tokens/window over 54-56 windows.  Fixed protocol gen: 68.3-77.0
tok/s (desktop-load spread; single runs up to 77.9).  Envs off except where said.

| item | ms/window | source |
| --- | ---: | --- |
| window wall | **30.0-30.9** (74-77 tok/s) | `STRATA_DECODE_TIMING` |
| GPU-reach wait | **22.1-22.7** | `STRATA_DECODE_TIMING` (the dominant item) |
| per-layer host | 1.1-2.3 (plan 0.09-0.13, actq 0.01, jobs ~0, CPU 0.6-0.8) | `STRATA_DECODE_TIMING` |
| host staging + launch call | 0.01 + 0.04-0.05 | GRAPH_TIME |
| unattributed pipeline host | ~6.3 | wall minus the parts above |
| replay, stage 0 (layers 0-25) | 16.0-17.0 over 40 windows/request | GRAPH_TIME |
| replay, stage 1 (layers 26-47) | 15.5-16.5 over 28 windows/request | GRAPH_TIME |
| nodes | stage0 T=1..6: 1307 / 1320 / 1334 / 1348 / 1362 / 1376; stage1: 1118 / 1129 / 1140 / 1151 / 1162 / 1173 | GRAPH_TIME |

Per-T replay spans (warmed averages; the served mix is mostly T=4 with T=1/2/3 mixed in):

| T | stage 0 | stage 1 |
| --: | ---: | ---: |
| 1 | 12.2 | 11.3 |
| 2 | 14.3 | 13.0 |
| 3 | 16.5 | 15.5 |
| 4 | 18.8 | 18.9 |

Profile arm (`STRATA_VERIFY_PROFILE=1`; fusions off, so ~2 ms/window pessimistic — wall 32.5-33.0, wait
24.2-24.5), GPU stage times per window, stage 0:

| item | ms/window |
| --- | ---: |
| expert GEMM ("VRAM hits", stamps 19->20) | 4.06 (GDN 3.15 + QSA 0.91) |
| hyper-connection hc0 + hc1 (norm/down/up) | ~3.0 |
| out-proj (corrections included) | 1.58 |
| q8+qkv / q-idx gemv | 1.08 |
| shared expert + quant | 1.16 |
| router + plan + doorbell | 0.93 |
| waits A/B/CPU | ~1.2 |
| copy+combine (corrections included) | 1.07 |
| total | 17.8 |

Stage 1 mirrors this and adds the head (1.21 ms/window).

Ablation node delta (clean): skipping the routed expert GEMMs removes **6 nodes per layer** (stage 0 T=1
1307 -> 1151 = 156 = 6 x 26 layers; stage 1 132 = 6 x 22).  In the profiled listing the expert path is
`native_gu_multi<143>`, `swiglu_q8_1_entries`, `native_down_multi<143>` plus the quantize/hit launches around
them.

**The wait decomposition:** a window's critical path is stage 0's graph (16.5) + hand-off + stage 1's graph
(16.2); consecutive windows overlap only ~3 ms, and the host's ring-to-ring wait (22.4) exceeds stage 0's replay
span by ~6 ms.  That excess is the hand-off/doorbell exposure, and it is step 2's first target.

## The identity finding: the shipped serve does not reproduce across restarts

Same binary, same config, same fixed protocol, cold restarts on 2026-10-14 produced **six distinct stable warmed
streams** (each stable over 7-16 consecutive requests inside its process; the bodies differ from token ~90 on, a
near-tie list flip):

| run | notes | warmed stream |
| --- | --- | --- |
| base / new2 / off3 | shipped config | C (565 chars) |
| off4 | shipped config | Y (572) |
| gt1 / gt2 / gt3 | instrumented | D (571) / A (571) / Y |
| instr-off | shipped config, no bench first | B (548) |

The startup expert-cache fill varies with desktop VRAM (auto: 11729 / 11733 / 11745 experts observed) and the
adaptive tier's warm-up swaps land on a timing-dependent schedule; both move experts between the VRAM cache and
the CPU pool, whose last bits differ.  The pre-change engine fails the cross-restart byte gate the same way, so
the gate as written is not defined on the shipped serve.

**Identity protocol for this lane:** the identity arm is the serve config + `--adapt-every 0` (adaptive tier
off; `work/strata/strata-q6k-176b-pin.json`), same fixed protocol, full-body compare of the warmed requests.  It
reproduces across restarts: three runs of the instrumented binary (auto fill 11745, and fixed 11000 slots) and
one run of the pre-change binary (exe-swap config `strata-q6k-176b-pin-ref.json` + `/tmp/opencode/scion-moe-ref`)
all produced state E (558 chars), all 8 bodies byte-identical.  Every subsequent change is checked there; the
shipped config is used for speed arms and reports its own (lottery-selected) warmed stream.

## Lane instruments

- `serving/moe-exact.sh <tag>` — the warmed full-body capture (8 requests of the fixed prompt; compares g4-g8 to
  the stored reference in `/tmp/opencode/scion-moe-exact`).
- `serving/moe-cycle.sh <tag>` — cold-restart + bench + exact; the standard shipped-config cycle.
- `serving/moe-pin-cycle.sh <tag>` — the identity arm in a serve-down window (pinned config on 8081; restores the
  service afterwards).

## Step 2 (next)

Attack the ~22.4 ms GPU-reach wait with the ISA-guided serialized-load lens: candidates by size are the expert
GEMM (4.06), the hyper-connection chain (~3.0), out-proj/corrections (1.6), and the hand-off structure itself
(the ~6 ms excess plus the per-layer `wait_flag_ge` doorbells).  Confirm each candidate with a GRAPH_TIME replay
delta on a real run; the micro-benchmark is a filter, never the result.  Then Scion-35B-A3B (step 3).
