# Dense Clef corrections — rental runbook (9B, gated on local proof)

Status: **draft, not approved to run.**  The local box has twice OOM'd/restarted
on 9B torch training (15.9 GB body on a 21.5 GB card + 30 GB host).  This
runbook is the checklist to satisfy before spending money; the local prefix
smoke fills in the measured per-step numbers first.

## 1. GPU sizing

| config | VRAM | verdict |
|---|---|---|
| `both` taps, rank 512, seq 512, grad checkpointing | body 15.9 + branches/grads ~2.1 + activations ~2 + context ~1 = **~21 GB** | fails one 21.5 GB card (measured) |
| same, **24 GB** card | ~21 GB | workable but <3 GB headroom; risky |
| same, **48 GB** card (A6000 / L40S / A40) | ~21 GB | comfortable; **recommended** |
| **80 GB** (A100/H100) | ~21 GB, room for batch>1 or full-body QAT | overkill for the sidecar; needed only for the full-QAT fallback |

**Recommendation: one 48 GB card**, host RAM ≥ 64 GB, disk ≥ 50 GB.  The
reverse loader's worst host transient (Q4_K embedding dequant) plus the 2 GB
head/lm_head fit trivially at 64 GB; they do not at 30 GB.

## 2. Time and cost

- Per step (estimated): 9B fwd+bwd, seq ~512, batch 1, grad checkpointing →
  **~1–2 s/step** on a 48 GB card, plus a small CPU joint-head forward.
- Pilot: **300–500 steps** (~10–15 min) to get a parity read.
- Full sidecar run: **2000–4000 steps** (~1–2 h) + eval (~15 min).
- Teacher cache is built **locally** (already proven) and shipped, so the
  rental does no teacher forward.
- Budget: **< 4 GPU-hours, ~$3–8**.  Hard cap: **$15**, wall-clock **6 h**,
  auto-terminate on either.

## 3. What must be true before I say "positive" (failure routes)

Known-good locally (already verified):
- reverse loader vs the CPU bridge: **cos 0.99994**;
- joint head sidecar reproduces bf16 `p_correct` exactly (0.9219);
- teacher cache builds (158 sequences);
- cgroup `MemoryMax` guardrail works.

Must still be proven / controlled:

1. **The training loop itself** — pending the local 2-layer prefix smoke
   (`--prefix-layers 2`): validates branch STE, grad checkpointing, the
   double-grad hidden KD, the CPU decision-grad bridge, and the Adafactor step.
2. **Rental numerical parity** — before training, run the rental's f16 forward
   on the fixed 48-token probe and require **cos ≥ 0.999 vs the local CPU
   bridge** (`/tmp/opencode/f16_cpu.bin`).  If it fails, stop (hardware/version
   mismatch) before spending steps.
3. **Version pin** — `torch`, `transformers==5.5.0`, `numpy`, `safetensors`,
   `datasets`; record `pip freeze` + driver/CUDA in the run manifest.
4. **No `fla`/`causal_conv1d`** — the pure-torch recurrent GDN is deliberate;
   verify `is_fast_path_available is False` and that `chunk_gated_delta_rule`
   is patched.  Installing them would make the rental forward diverge from the
   local CPU deployment.
5. **OOM** — 3× the measured local budget; preflight `torch.cuda.mem_get_info`;
   body streaming; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
6. **VPS control** — SSH key (no browser-only consoles), non-interactive
   provider API, ability to kill the instance, **hourly billing**, and a
   `trap`/`EXIT` auto-teardown plus an external watchdog that stops the pod on
   wall-clock or heartbeat stall.
7. **Artifact persistence** — stream checkpoints + eval JSON back to the local
   box as they land (`rsync`), verify SHA-256; never leave the only copy on the
   pod.
8. **Research risk (cannot be eliminated, only bounded)** — the corrections may
   not reach parity (dense forensics F5/F11: dense QAT is hard).  Mitigation: a
   short pilot + parity eval before the full run, and the uncorrected ternary
   remains the shippable fallback.

Not yet ticked: **1** (running now), **2/6/7** (rental setup).  I will not ask
for spend until 1 is green and 2–8 have concrete, tested controls.

## 4. Upload set (no downloads on the rental)

- `clef-flash-PQ2_0.gguf` — 3.26 GB (the deployed body);
- `hf-head/` (joint head + tokenizer + configs) — ~0.27 GB;
- `lm_head.safetensors` — 2.03 GB;
- `dense/` code + the teacher cache (`cache-smoke`, 0.5 GB) + the 48-token
  parity probe + `f16_cpu.bin`.
- **Not** needed on the rental: the 17.9 GB f16 GGUF (the teacher cache is
  precomputed locally), unless we rebuild per-layer targets.

## 5. Teardown

On completion or abort: rsync artifacts, write `RUN-MANIFEST.json`, then stop
the instance via the provider API from the local box (not from the pod), and
confirm billing stopped.
