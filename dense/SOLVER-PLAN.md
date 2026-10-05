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
