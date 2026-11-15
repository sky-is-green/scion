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
and sampler (`The capital of France is` + 64 greedy tokens).

The recurrent-snapshot rollback path (`n_rs_seq`) **must stay off for qwen4exp**:
with it enabled the verify rollback corrupts the GDN state and the continuation
diverges after the first rejected round. The server therefore uses the memory
checkpoint path (`COMMON_CONTEXT_SEQ_RM_TYPE_FULL`) for this target.

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

## Speed

Same prompt, `n_predict = 64`, greedy, both arms on a warm server:

| arm | predicted | predicted ms | tok/s |
|---|---:|---:|---:|
| no draft | 64 | 184.7 s | **0.341** |
| draft-mtp | 64 | 207.2 s | **0.304** |

**The draft lane is currently ~11% slower than no draft on this rig.** The
acceptance gain (1.45 tokens per target step) is real, but the checkpoint
rollback path pays it back: every partial acceptance restores the full hybrid
memory state and replays the accepted prefix. The recurrent-snapshot rollback
would avoid that, but it corrupts qwen4exp's GDN state (see Correctness), so it
stays off.

On a host where the model fits in RAM (or with a cheaper rollback), the
algorithmic gain is bounded by the 1.45 tokens/step, i.e. up to ~1.45x fewer
target steps, minus the draft/verify overhead. Fixing the qwen4exp RS rollback
is the prerequisite for a net speed win here.

Both arms are paging expert rows on this box (30 GB RAM for a 60 GB model), so
the absolute tok/s is a floor, not the model's capability.
