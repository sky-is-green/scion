# Solver plan — rotation-in-the-loop QAT of Clef (rental; needs approval)

Status: **draft for operator approval** (2026-10-05).  No spend until approved.

## Why

PTQ is exhausted (HANDOFF CURRENT THREAD): best artifacts are
`clef-flash-v2-signs-gptq-lloyd-ao-h128-nodown.gguf` (**75.2** PPL,
8.13 GiB, `ffn_down` F16) and `…-ao-h128.gguf` (**120.6**, 5.52 GiB,
all-ternary); f16 = 12.59.  Flip-polish was falsified.  The demonstrated
mechanism (forensics, TAARDIS) is training the ternary weights end-to-end
against the f16 teacher with the rotation in the loop.

## Derisk evidence (Qwen3.5-0.8B, same arch; `dense/qat_derisk.py`)

| run | post-hoc start | end | steps / time | vs f16 (27.2) |
|---|---|---|---|---|
| all-ternary | 34,938 | 57.1 | 300 / 21 min | 2.10x |
| all-ternary | 34,938 | 48.4 | 600 / 39 min | 1.78x |
| mixed (`down_proj` F16) | 10,020 | 48.8 | 300 / 18 min | 1.79x |

Peak VRAM 7.8 GB at 0.5B masters (f32), batch 4x512, hidden KD only; loop
stable, early recovery is 100x in 25 steps.  Both bases converge to ~48-49;
the mixed base gets there ~4x faster.

## Recipe (validated shape)

- Student: Clef f16 body reverse-loaded (`dense/clef_dense_load.py`,
  streamed, recurrent GDN) with the 176 attention/MLP linears folded
  `W' = W Rᵀ` (V2 basis, `--sign-seed 1337`); `ssm_out` unrotated-ternary;
  `in_proj_a/b`, norms, conv, embeds frozen.
- Ternary: deployed Lloyd group rule (g128) on the folded masters,
  straight-through gradients, scales detached per forward.
- Teacher: same f16 body (frozen), hidden-state KD = MSE + cosine.
- Optimizer: Adafactor lr 5e-5, warmup 10, cosine to 0.1x, clip 1.0,
  gradient checkpointing.

## Exact 9B budgets (from `clef-flash-f16.gguf`)

- ternary masters: **6.912B params** (176 rotated 6.510B + 24 unrotated
  0.403B) -> f32 **27.6 GB**, bf16 **13.8 GB**; grads equal again.
- frozen tensors: ~2.0B params (embed/head/norms) -> ~4 GB bf16.
- teacher f16 body: **16.7 GiB**.
- teacher hidden-state cache: 8 KB/token bf16 -> 1M tokens ~8 GB.

## Options

**A. 48 GB card (A40/A6000), student-only (recommended, cheapest).**
1. Local prep (~1-2 h GPU): harvest f16 teacher final-hidden states for
   1024-2048 wikitext-2 train windows with the recurrent GDN (the local card
   already runs this; hessian capture peak was 17.9 GB), and wrap/fold the
   9B masters locally to bf16 (~14 GB) + emit the init state.
2. Rental: student bf16 weights (masters 13.8 + frozen ~4) + bf16 grads
   13.8 + activations (checkpointed, batch 2-4) ~10-15 -> ~42-47 GB.
   ~600-1200 steps at ~15 s/step (A40 pilot pace) -> 2.5-5 h -> **~$1-4**.

**B. 80 GB card (A100/H100), online teacher (simplest).**
Student bf16 (18) + grads (14) + teacher bf16 (17) + activations -> ~65 GB.
~600-1200 steps, faster (~3-6 s/step) -> 1-2 h -> **~$5-12**.

Both upload the wrapped init (~14-18 GB) + code + corpus.  Cache path (A)
also uploads 4-8 GB of hidden states.

## Steps, acceptance, deliverables

- Base: start with the **mixed 75.2** artifact shape (converges faster),
  then optionally an all-ternary run for the 5.52 GiB community quant.
- Budget: 600-1200 steps at batch 4x512 (0.6-1.2M tokens) mirroring the
  derisk; extend if the eval curve is still falling.
- Acceptance: wikitext-2 PPL (c512, 100 chunks) materially below 75.2 —
  target **< 40**, stretch ~25; then re-run the frozen decision probe for
  the record (bf16 stays the decision validator).
- Deliverables: trained ternary GGUF (V2 container, prism metadata), PPL
  logs, doc updates, HF release prep per `docs/HF-RELEASE-NOTES.md`.

## Risks / mitigations

- 48 GB edge: if OOM, batch 1-2 or f32->bf16 frozen copies only.
- Local bf16 torch nondeterminism (observed on the 9B PQ2 body, ROCm) does
  not apply on CUDA; verify the first 10 steps' loss is monotonically sane.
- GDN backward: keep the recurrent patch; fall back to the chunked form if
  unstable (the derisk used the torch fallback throughout).
- Corrupted resumption: save master checkpoints every ~200 steps.

## Approval needed

Rental of option A (~$1-4) or B (~$5-12) plus local prep time.  After
approval: prep locally first, then create the pod, run, and report PPL and
the probe.

## Execution checklist (option A, approved 2026-10-05)

Local prep (free):
1. Teacher cache: `python dense/clef_cache.py --model models/clef-flash-ternary
   --out <out>/teacher-cache --generic 2048 --generic-only` (f16 body,
   recurrent GDN; ~8 GB fp16 hidden states + index.json).  **Blocked:** the
   non-display GPU is held by another lane's `llama-server` (Qwen4-exp);
   per policy, no torch beside a serving engine — waits for the card.
2. Upload set (rsync): `clef-flash-f16.gguf` (17.9 GB), `teacher-cache/`,
   `scion/dense/` (`qat_9b.py`, `clef_dense_load.py`, `qat_derisk.py`,
   `quant.py`, `clef_cache.py`), `bonsai2-ternary-forensics/bonsai_forensics/`
   (rotation utils), `llama.cpp/gguf-py`.
3. Pod env (mirror `RENTAL-RUNBOOK.md` §3): torch + transformers==5.5.0, no
   `fla`/`causal_conv1d`, `PYTHONPATH=<gguf-py>`,
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, version pin manifest.

Run (pod):
```
PYTHONPATH=<gguf-py> python dense/qat_9b.py --model ./clef-flash-ternary \
    --cache ./teacher-cache --out ./qat-run \
    --steps 800 --batch 2 --seq 512 --eval-every 50 --save-every 200
```
Eval / export (local, after rsync back): export the trained masters to the V2
GGUF (new `dense/clef_export_qat.py`: Lloyd g128 + `pack_q1_0_g128` + prism
metadata), then `llama-perplexity` (c512, 100 chunks) + the frozen probe.

Controls: rsync `masters-step*.pt` as they land; teardown via the provider API
from the local box; hard caps 4 GPU-h / $15 / 6 h wall + external watchdog.

## Outcome (2026-10-05/06)

Executed on one L40S 48 GB secure pod (US-TX-4, ~$3.9, pod deleted): QAT v1
(all-ternary, 1200 steps, hidden KD) reached **PPL 23.25** (1.85x f16), 5.52
GiB container.  Pre-post generation testing then found degenerate free
generation (code/math; f16 control clean); logits-KD (added, chunked +
checkpointed) and mixed-base derisks at 0.8B did not fix it at our training
scale, and the exported artifact is faithful (bridge hidden cos 0.838 vs f16
0.9999).  **Do not run this plan again as-is** until the community-baseline
gate (TQ1_0/PTQ1_0 Clef-Flash comparison, HANDOFF CURRENT THREAD item 4) says
the pipeline is worth a larger corpus + budget run.

## Proper run v2 — mixed corpus + top-k logits KD (draft for approval, 2026-10-06)

**Update (2026-10-06):** no rental yet.  Running the free, longer **0.8B
derisk first** (`qwen35-0.8b-qat-derisk/run_mixedcorpus`: mixed corpus
(`dense/clef_mixed_corpus.py`, 1024 prose / 410 math / 615 code windows) +
hidden+logits KD, 1500 steps, generation samples every 300 steps, same
generation suite) to see whether free generation cleans up with budget
before spending on the pod.  Pod decision is gated on it: if generation is
still degenerate at 4-8x the earlier derisk budget, re-scope or park.

**Draft for operator approval.  No spend until approved.**  The gate is in:
plain ternary codecs are unusable (1.9-2.1M PPL), the pipeline is the only
ternary route, but v1 fails free generation and a plain Q2_K beats it
(13.06 PPL, clean, 3.56 GiB).  v2 must fix generation first and land
PPL ≤ ~18 before any release.

### What changes vs v1

| | v1 (done, ~$3.9) | v2 (this plan) |
|---|---|---|
| KD target | f16 hidden states only | hidden + **top-k logits** (k=128, teacher, cached) |
| corpus | wikitext, 1.05M unique tokens (prose only) | **mixed** ~50% prose / 20% math / 30% code, ~2.5M unique tokens |
| token budget | 1.2M token-steps (~1.2 epochs) | **5-8M token-steps** (2-3 epochs) |
| acceptance | wikitext PPL | PPL + **free generation** + teacher-forced NLL + KL/top-1 vs f16 |
| base | all-ternary h128 GPTQ (120.62) | same (mixed `--skip-targets mlp.down_proj` documented as fallback) |

### Corpus (all local, no downloads on the pod)

- **prose** (50%): `Salesforce/wikitext` (103/2 train) from the local HF cache;
- **math** (20%): `EleutherAI/hendrycks_math` (7 configs) + `openai/gsm8k`
  train, both cached;
- **code** (30%): permissively-licensed local trees (`llama.cpp`, `scion`,
  `hivebench`, `FreeToken`, `llama-qwen4exp`, `prism-ml-llama.cpp`;
  `*.py,c,cc,cpp,h,hpp,cu,md`), build/output dirs excluded, deduplicated.
  No overlap with wikitext-2 test (PPL) or the cascade bench (decisions).
- New `dense/clef_mixed_corpus.py` -> `mixed-windows.npy` (int32
  `[n_windows, 512]`) + `mixed-corpus.json` (provenance, per-source token
  counts, SHA-256).  Target 2.5M tokens (4880 windows), cap 3M.

### Cache (built locally, uploaded)

`dense/clef_cache.py` gains `--topk 128 --windows-npy <file>`: per token
`hidden` (f16, ~8 KB) + `topk_idx` (int32) + `topk_val` (f16, ~0.75 KB at
k=128) -> **~22 GB** for 2.5M tokens on the external drive; rsync to the pod
(v1 uploaded 8.6 GB of cache + the 17.9 GB f16 GGUF the same way).

### Trainer deltas (`dense/qat_9b.py`)

- `HiddenCache` reads the optional top-k arrays; `--kd-logits W --kd-temp T`
  adds a top-k KD term (student logits gathered at the teacher's indices, KL
  against the teacher's renormalized top-k; tail mass documented).
- `--eval-nll`: holdout student NLL + teacher top-1 agreement — the
  generation-predictive metric (v1 only watched hidden cos 0.85).
- unchanged: rotation fold + Lloyd g128 STE, Adafactor lr 5e-5 cosine to
  0.1x, clip 1.0, checkpointing, `--kernel chunked` (matches the locally
  captured cache at cos 0.9999).
- pod smoke gate: 50 steps, sane loss trend, memory headroom logged; abort
  (cheaply) if not.

### Pod plan (offline teacher, 48 GB class)

Local prep first (free, ~1-2 h): corpus + cache build; the KL base-logits
file for acceptance can also be built locally later.

Pod: **1x 48 GB (L40S, as v1; v1-proven memory profile)** at ~$2.0-2.3/h
secure; hourly billing; hard caps **6 h wall / $15 / auto-teardown**, external
watchdog, `--max-steps-wall 21600` (`RENTAL-RUNBOOK.md` §3-§6).  If the smoke
shows batch 4 does not fit, step down 4 -> 3 -> 2 with the same wall cap
(batch 4 = ~7.4M tokens planned, batch 2 = ~3.7M; the wall/dollars are the
caps, the token count is the dial).  An 80 GB A100/H100 at ≤ $3/h is the
upgrade path if the operator prefers headroom/speed over the v1-proven shape.

```
PYTHONPATH=<gguf-py> python dense/qat_9b.py --model ./clef-flash-ternary \
    --cache ./teacher-cache-mixed --out ./qat-run2 \
    --steps 3600 --batch 4 --seq 512 --kd-logits 0.5 --kd-temp 1.0 \
    --eval-every 100 --eval-nll --save-every 300 --max-steps-wall 21600
```

Checkpoints + `qat.json` rsynced back as they land; nothing is left only on
the pod.

### Export + acceptance (local, free)

Export masters -> V2 all-ternary GGUF (`dense/clef_export_qat.py`, GDN
reorder), then:

1. `llama-perplexity` wikitext-2 c512 x100: **target ≤ 18** (1.43x f16;
   stretch ≤ 16) vs f16 12.59 / Q2_K 13.06 / v1 23.25.
2. Free generation, fixed-seed suite (same 3 probes + 3 extra code/math
   prompts): **no loops, correct math, valid code**; teacher-forced NLL on
   f16-clean text should approach f16 (v1: 2.02 vs 1.24).
3. Community norm on a ~250k-token mixed file: `llama-perplexity
   --save-all-logits` on f16, then each candidate with
   `--kl-divergence --kl-divergence-base` (reports KL, Δp RMS, same-top-p).
4. Only if 1-3 pass: release prep per `docs/HF-RELEASE-NOTES.md`.  Posting
   remains a separate decision; never automatic.

### Decision rules

- **Green** (release candidate): generation clean + PPL ≤ 18 + KL/top-1 in
  Q2_K-class distance -> prep the HF release and re-run the comparison table.
- **Amber**: generation clean but PPL 18-25 -> decide between a third run
  (longer / mixed base) and shipping the Q2_K class instead.
- **Red**: generation still degrades or PPL > 25 -> park the ternary lane
  (documented; no HF upload), bf16 Clef + Q2_K-class stay the offerings.

### Approval needed

1x 48 GB GPU for ≤ 6 h, hard cap **$15** (estimate $10-14; at the v1 pace
(5.2 s/step at batch 2) 3600 steps = 4-8 h depending on whether batch 4 fits,
so the 6 h wall cap is expected to bind first at **~5-7M token-steps**), plus
~1-2 h local prep and ~1 h local acceptance.  No downloads, no HF posting.
