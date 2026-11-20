# 35B v2 cur05 full run — 2026-11-20

Status: **trained, exported, merged and branch-audited. Subset30 eval pending.**
The full training is complete; this document records the run, the export
incident, and the artifact.

## Run

- Pod `f1f1pv7tbc5z57`, Secure Cloud, **EUR-IS-1**, 1x A100 SXM 80 GB,
  $2.19/h, 150 GB volume; 32 vCPU, ~1 TB host RAM.
- Setup 17:37–17:43 UTC; curriculum 17:47:51–18:17:01 (4096 windows scored);
  cache 18:17:01–19:00:32 (10.03 GB); ref 19:00:32–19:04:13; train
  19:04:13–02:53:28 (7 h 49 m, 4096 steps); first export completed
  **06:55:16** (see incident below).
- Peak memory: **14.14 GiB resident**, **66.39 GiB load peak**; router frozen;
  209.72M trainable branch params.
- Step time 6.37–6.88 s; 4096 steps = 7 h 49 m.
- Cost: ~$36 total (training + setup/cache/ref + the slow first export +
  the GPU re-export); balance after termination **$9.50**. The estimate was
  $15–35; the CPU export incident added the extra.

## Training diagnostics

| step | lm | kd | total | H | peak | loadH |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 2.3776 | 0.4629 | 3.3276 | 2.178 | 0.5277 | 4.407 |
| 1000 | 1.4508 | 0.3625 | 2.1879 | 1.371 | 0.6747 | 4.464 |
| 2000 | 2.1893 | 0.6021 | 3.4535 | 2.272 | 0.5682 | 4.873 |
| 3000 | 2.6054 | 0.4967 | 3.6468 | 2.346 | 0.5228 | 4.561 |
| 4000 | — | — | — | — | — | — |

Proxy eval (2 wikitext windows):

| step | ppl | router_agree | load_cv | peak | cold | dead (count/256) |
|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 8.19 | 0.6767 | 2.280 | 20.57 | 31.0% | 57.4 |
| 2000 | 8.35 | 0.6250 | 2.218 | 20.13 | 29.9% | 54.7 |
| 3000 | 8.31 | 0.6099 | 2.210 | 20.28 | 29.3% | 53.9 |
| 4000 | 8.28 | 0.5980 | 2.210 | 20.34 | 29.6% | 54.2 |

The router agreement is teacher-top-8 recall; it declined and then flattened
near 0.60 while PPL stayed flat. The v1 router deltas were dropped in the v2
merge (see below), so the deployed router is the BF16 router the branches were
trained with. This is a warning, not the gate.

## Export incident

The first export used the CPU Lloyd quantizer on the trained branch factors.
It ran ~4 h of wall time (the pod stayed alive to 06:55) and the balance was
exhausted at ~08:37 while the export was thought to be stuck. It had in fact
completed at **06:55:16** (`ALL-DONE`), writing the full 240-tensor soup
adapter; both adapters were on the volume. The pod stopped at $0; a $10 top-up
restarted it.

The fix: `export_branches_lora.py --device cuda` quantizes on the A100 and
moves the factors back to CPU before GGUF packing. The re-export of the final
and soup took **~1 minute each**. Branch-only adapters are 54 MB; the full
soup with zero router tensors is 99 MB.

## Artifacts

- `qwen35-adapter-cur05-soup.gguf` — **GPU, branch-only, 54 MB** (used).
- `qwen35-adapter-cur05-final.gguf` — GPU, branch-only, step 4096, 54 MB.
- `qwen35-adapter-cur05-soup-full.gguf` — first CPU export, full 240 tensors,
  99 MB (fallback; includes zero router deltas).
- `Scion-35B-A3B-v2-cur05.gguf` — local merge of the GPU soup branch adapter
  with `--replace`; 973 tensors, **10.558 GiB**, `adapter.embedded=true`; the
  80 v1 router tensors zeroed so the deployed router is the BF16 base.
- Branch audit: **160 tensors, 209,715,200 weights, 0 code3, 0
  scale_not_max, 0 bad_scale.**
- Logs: `full-run.log`, `train.log`, `export-final.log`, `export-soup.log`,
  `setup.log`, `curriculum.log`, `cache.log`, `ref.log`.

Hashes:

```
5e0f3a20cad0bdb10e5587efe62830f2e0d0bc65445373eee41fe5b434f95f21  qwen35-adapter-cur05-final.gguf
5d67443cf06dc2cf07a90b2653ca98375cf02f3d252985c92ca7c259002da510  qwen35-adapter-cur05-soup.gguf
c955a9e17916cde92916d3fdbd3c5e38826d80a4a14703cc5dcbf5203231482b  qwen35-adapter-cur05-soup-full.gguf
```

## Subset30

The protocol is the frozen one: `v2/scripts/hard_subset.py`, T=0.6, seed 1001,
top_p 0.95 / top_k 20 / min_p 0.05, 32k budget, recovery sentence appended.
Reference: **5/30** (v1 + recovery). Gate: **>= 8/30**.

Result: pending.

## Deviations and notes

- Full 4096-window curriculum/cache, 10% mix with 346 clean teacher traces.
- Router frozen (`--freeze-router`); the v1 router tensors were zeroed in the
  merged file because the v2 branches were trained against the BF16 router.
  This means the subset30 comparison also changes the router relative to v1;
  if the score is low, the next arm is router training or a router-preserving
  merge, not pred2.0.
- The export is CPU-bound in the original path; always use `--device cuda`
  for the branch factors.
- The pod/volume were terminated after the pull; the artifacts here are the
  only copies needed for the local eval.
