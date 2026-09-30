# RENTAL RUNBOOK — 35B v2 (+ Flash-Next prep)

**Status:** signed for execution 2026-09-30 (memo §5). The pod runs **only**
what `EXPECTED-IMPROVEMENT-MEMO.md` §5 says; nothing else is started on a
billed GPU without the human.

- **Budget:** $93.28 total (Runpod balance). Caps: 35B v2 **≤ $30**,
  Flash-Next **≤ $55**; at 80% of either cap, stop and report before the next
  stage.
- **Provider:** Runpod (EU, `eur-is-*`), API key from the human (created when
  we are ready). Pod creation/pre-flight/management via the API + SSH.
- **Rates (2026-09-30):** H100 80GB ~$2.4–3.2/hr; H200 141GB ~$3.6–4.4/hr.

## Booking shape (35B v2)

- 1× H100 80GB (H200 fine if the price is at parity), RunPod PyTorch image
  (torch 2.9.1+cu128 validated in the first run), **network volume** mounted
  at `/workspace` (≥ 200 GB for the teacher + cache + ckpts).
- The teacher (`empero-ai/Qwen3.8-35B-A3B-Distill`, ~67 GB BF16) is
  downloaded once into the volume; it survives pod termination if the volume
  persists — verify before booking.
- Code: session-8 tree via tarball (`$MOE_ARTIFACTS/scion-src.tar.gz`) or
  pushed `scion-test` at a pinned commit; **the corpus mix file**
  `$Q/curric-combo.jsonl` must be uploaded before `cache` (the stage refuses
  without it — the cache corpus must match the trained corpus exactly).
- Expected window: **6–9 h ≈ $15–28** (setup 0.5–1 h, cache 0.3–0.6 h,
  ref ~1 min, two trainings ≈ 2.5–3.5 h each, eval/export/gate ≈ 1–2 h).

## Stages (`scion/moe/box-run-v2.sh`)

| stage | expected | sanity gate before continuing |
|---|---|---|
| setup | 30–60 min | torch sees the GPU; `fla`/`causal_conv1d` installed (else fast-path WARN); teacher dir size ~67 GB; mix file present |
| smoke | ~3 min | hidden drift ~0.3–0.45; router top-8 agreement ~0.83; no "fast path not available" warning |
| cache | 20–35 min | cache ~7.5–8 GB with `idx/val/w/router/tidx/tlp`; teacher support mass ~0.17–0.20; tail fields present |
| ref | ~1 min | file written |
| train (cur05) | 2.5–3.5 h | at step 100–200: `lm` ~7–9, `kd` ~0.25–0.35, `tail` ~0.015–0.03, `tcond` ~0.13–0.25, `H` ~9.8–10.4, `loadH` ~5.1–5.4; in-run PPL step-1000 ~2800–3000, step-2000 ~2100–2200; abort on NaN or a >50% deviation from this shape |
| train-fallback (pred2.0) | 2.5–3.5 h | same shape, slightly lower total at the same step (kd-tailcond 2.0) |
| eval | ~5 min ×2 | PPL + router agreement written |
| export | ~5–10 min ×2 | adapters for final + soup, both arms |
| gate | 1–2 h | memo §3 clauses (community KLD, HellaSwag/Winogrande 400, canary) — see `docs/RELEASE-35B-MODEL-CARD.md` + `docs/QUANT-RETENTION-35B.md` |

## Management protocol (the money saver)

1. **15-minute cycles** (as `hive-ops/SLEEP-PROTOCOL.md`): status → diagnose →
   act. Read the stage log tail, the current step, and one checkpoint
   timestamp. Never let a stage run past a failed sanity gate.
2. **Kill criteria are pre-registered** (table above + memo §3). On a failed
   gate: kill the stage, diagnose, fix, restart — the cache is the expensive
   artifact and survives; do not "wait and see".
3. **Checkpoint discipline:** every 1000 steps to the volume; after each
   stage, verify the artifact exists and its size is sane before starting the
   next.
4. **Resource discipline:** terminate the pod (not stop) after the last
   artifact is fetched, unless the volume is needed for the Flash-Next run;
   fetch logs + JSONs first.
5. **Do not use `--resume` for main-recipe training:** the trainer pairs the
   cache iterator from 0 with a step-offset data index, so a resumed run
   trains against misaligned windows (session-8 finding; no test pins it).
   Restart the arm fresh — the cache is the expensive part and is kept.
6. One session per pod run; both write to the single handoff doc
   (`RESEARCH-HANDOFF.md`) in disjoint sections; the human relays any joint
   decision.

## Artifact checklist before terminating

- `$Q/*step*-cur05.pt`, `*step*-pred2.0.pt` (final + intermediates)
- `$Q/*-soup-*.pt` if soups were made
- adapters `$Q/qwen35-adapter-*.gguf` (both arms, final + soup)
- `$Q/qwen35-eval-*.json`, stage logs
- `$Q/prefix-top512-tail64-cur05.pt` (only if the Flash-Next/next window
  wants to skip the teacher pass; otherwise it can be rebuilt)

## Flash-Next (fill in after the port)

Blockers being removed locally (session 8): (1) runtime — `qwen4_exp` is not
in our transformers 5.5.0, support is on transformers `main` (pin a commit)
or vLLM/SGLang; (2) harness — scion has no `qwen4_exp` handling yet: the
cache stage needs a small port (forward + router hook), the correction/KD
training needs the architecture port, and PLE/n-gram precision is an open
local A/B. Planned pod shape once the port passes its local smoke: 2×H200 for
the FP8 teacher cache, release to 1×H200 for training/evals; total target
≤ $55.
