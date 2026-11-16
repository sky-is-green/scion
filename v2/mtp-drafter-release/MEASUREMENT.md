# Scion MTP drafter - engine measurement

Date: 2026-11-15
Engine: `prism-ml-llama.cpp`, branch `engine/slotcache-sidecar`
(Ember expert slot cache + `mtp-sidecar` draft-mtp lane).
Target: `Scion-FlashNext-176B-A6B-Q6_K.gguf` (PTQ1_0 experts).
Drafter: `scion-mtp-sidecar-v1.gguf` (fc1/fc2/fc3, fp16, 108 MB).

## Test rig

| item | value |
|---|---|
| cards | 2x RX 7900 XT (20.5 GB each) |
| card policy | spare (sysfs card0) = primary 75% of layers + expert slot banks; display (card1) = 25%, less VRAM and less compute |
| host | 30 GB RAM, model file 60 GB, NVMe |
| launch | `HIP_VISIBLE_DEVICES=1,0 -ngl 99 -ncmoe 24 -msc 256 -sm layer -ts 0.75,0.25 -c 4096 -np 1 -fa auto` |
| slot cache | 24 host layers x 256 slots, 6.3 GiB, **90.0% hit rate** (36504 hits / 4056 fetches) |

## Correctness

Speculative output is **token-identical to the no-draft run** for the same prompt
and sampler (`The capital of France is` + 64 greedy tokens), on both the
checkpoint path and the recurrent-snapshot path.

The recurrent-snapshot rollback now works for qwen4exp (engine commit
`7e53b56be`). Two bugs kept it corrupting the state:

1. the GDN **conv state** (a short shift register) had no snapshots, so a
   rollback left the rejected tokens' history in the conv window;
2. the window was one slot short: a rollback can undo the sampled token too
   (`n_draft + 1`), which no snapshot plane held.

Both are fixed: the conv and ssm states now write per-token planes **and** the
pre-batch state (slot `n_tokens`), qwen4exp gets one extra window slot, and the
server only falls back to a checkpoint when the worst-case rollback exceeds the
window. The acceptance log shows **zero checkpoint restores** in the spec lane.

## Acceptance

Offline chain eval on the held-out probe windows (last 100 windows,
`moe/mtp_chain_eval.py`, depth 4, greedy):

| metric | value |
|---|---|
| P(accept draft 1) | **0.3065** |
| P(accept draft 2 \| 1 accepted) | 0.2552 |
| P(accept draft 3 \| 2 accepted) | 0.2471 |
| P(accept draft 4 \| 3 accepted) | 0.1953 |
| P(>=1 / >=2 / >=3 / >=4 accepted) | 0.3065 / 0.0782 / 0.0193 / 0.0038 |
| mean accepted drafts per target step | **0.4078** |
| ideal tokens per target step | **1.4078** |
| per-draft acceptance (all attempts) | 13.5% |

In-engine (llama.cpp server, `LLAMA_TRACE=1`, 64 greedy tokens, checkpoint
rollback path):

| metric | value |
|---|---|
| draft rounds | 29 |
| accepted drafts / drafted | 13 / 84 |
| per-draft acceptance | **15.5%** |
| accepted drafts per round | **0.45** |
| tokens per target step | **1.45** |

The engine number matches the offline chain within the metric definition: the
"30.5%" is P(first draft accepted), while the per-draft rate over all attempted
drafts is 13.5% offline vs 15.5% in-engine.


## v2 head - 360-trace corpus

Trained on the combined corpus (1,728 train windows: 1,000 old + 728 new
chat/reasoning; held out: 100 old + 100 new), 6,144 steps, warm start from v1.
The in-run blended held-out acceptance was **0.4668** (v1 on the same blend:
~0.343).

| held-out set | P(draft 1) | P(2 \| 1) | P(3 \| 2) | P(4 \| 3) | acc/round | tok/step |
|---|---:|---:|---:|---:|---:|---:|
| old (wiki/code) | 30.33% | 23.80% | 22.72% | 16.39% | 0.395 | 1.395 |
| new (chat/reasoning) | **63.06%** | 24.25% | 15.26% | 15.33% | **0.810** | **1.810** |

v2 is the release candidate: it roughly doubles first-draft acceptance on the
serving (chat/reasoning) distribution and keeps the old corpus at parity with
v1.  Artifacts: `mtp-release-head-v2.pt`, `scion-mtp-sidecar-v2.gguf`
(114.8 MB, round-trip verified).

An in-engine 64-token run with v2 reproduced the no-draft text exactly, but the
host is disk-bound (60 GB model / 30 GB RAM): per-step times swung from 0.05 to
0.3 t/s, so it gives no usable speed figure.  Speed has to be measured on a
resident host, or after the qwen4exp recurrent-snapshot rollback is fixed so the
checkpoint restores disappear.

## Speed

Same prompt, `n_predict = 64`, greedy, **warm page cache**, both arms warmed by
the same cold-first-request procedure:

| arm | predicted | tok/s |
|---|---:|---:|
| no draft | 2.81 s | **22.44** |
| draft-mtp v2 + RS rollback | 3.70 s | **17.21** |

The draft lane is still a net loss on this host even after removing every
checkpoint restore: the verify batch processes 4 token slots per step (1.51
tokens returned), and with the 176B's experts host-resident (60 GB model on
30 GB RAM) each extra token adds expert traffic. Cold-cache numbers are much
worse for both arms (0.08-0.23 tok/s) and are not meaningful.

The algorithmic result that transfers to a resident host is **1.51 tokens per
target step** on this prompt (1.81 on the chat/reasoning held-out set), i.e.
~1.5-1.8x fewer target steps, minus draft overhead.

In-engine acceptance on the wiki prompt (41 rounds): 21 accepted / 122 drafted,
mean accepted length 1.51; no-draft and draft outputs identical.
