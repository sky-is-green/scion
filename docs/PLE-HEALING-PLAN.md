# PLE healing (QAT) plan (2026-10-12)

**Goal:** recover the measured table-quantization cost by training the model
to account for it. Current cost at Q3X32: **+0.011 KLD / +1.30% PPL** on the
48-layer release (paired gate, `docs/PLE-TABLE-TYPES.md`). If healing erases
it, the shipped file can be 52.2 GB at ~0 measured cost; if it also tames
Q2X32 (45.8 GB), even more.

**Status: the machinery already exists in `scion/moe/qwen4exp_proxy.py`; it
has not yet been run against the shipped containers.**

## What exists

- `apply_ple_qat(model, bits, group)` - STE hook: the student forward sees
  release-quantized PLE rows; gradients flow to the trainable **branches and
  gates** (the table is frozen). Used by `stage_train` when
  `--ple-qat-bits > 0` (`--ple-qat-group`, default 32).
- `stage_qat_screen` - 2-layer mirror, two arms (`qat0` full-precision rows,
  `qatN` N-bit STE), `--qat-steps` (default 30) each, then reads the N-bit
  damage `KLD(rows@Nbit || rows@full)` on held-out windows. A working hook
  shows both arms converging and the QAT arm with the smaller damage.
- `stage_ple_ab` - precision A/B (recorded on the pod, 2-layer): mean KLD vs
  the fp8 teacher 8-bit 6.3e-05, **4-bit 1.5e-03, 2-bit 5.7e-02** (top1 0.975
  / 0.869 / 0.388). The 2-bit region has a large signal to heal; 4-bit is
  nearly free.
- Training checkpoints: `hivebench/.../pod/remote-p2b/qwen4exp-corr-r512-
  g128-step{1000,2000,3000}-cur05.pt` (the release corrections are the
  step-3000 export).

## Gap before the heal targets the deployed format

The hook quantizes symmetric int per group (group 32), not the shipped
codebook+LS containers (Q3X32 = per-32 8-level codebook + least-squares
fp16 scale; Q4X160 = per-row 16-level). Add a container mode to
`apply_ple_qat` that applies the exact block quantizer with STE (~30 lines:
reuse the codebook constants and the per-block LS scale; apply on the
gathered rows). The screen then measures the damage of the *shipped* format.

## Plan, in order

1. **Local screen, existing int hook (free).** ROCm torch 2.14 + the local
   fp8 shards (`models/qwen38-flashnext-fp8/`) can run the 2-layer mirror on
   the 7900 XTs:
   ```
   python qwen4exp_proxy.py qat-screen --layers 2 --qat-steps 30 \
       --ple-qat-bits 2 --ple-qat-group 32 \
       --model-dir work/models/qwen38-flashnext-fp8 --device cuda:0
   ```
   Success: `qat2` damage < `qat0` damage and finite gradients in both arms.
2. **Container-mode hook** (q3x32 first), re-run the screen with
   `--ple-qat-bits 3 <container>`; success = the Q3X32 damage shrinks.
3. **Full heal (pod, 48 GB, ~4-8 h, ~$5-7):** resume the KD recipe
   (`train --resume <step3000 ckpt> --ple-qat-bits 3 <container> --steps N`),
   export the 192 branches (`export_branches_lora.py`), rebuild the file,
   **re-gate** (KLD + PPL/HS/WG; code arms if budget allows).
4. If the heal holds, consider the Q2X32 heal (45.8 GB) as the next tier.

## Screen results (2026-10-12, local, free)

Two runs on the 2-layer mirror (30 steps/arm, fineweb train / wikitext eval,
ROCm 7900 XT):

| arm | damage read | mean | vs control |
| :--- | :--- | ---: | ---: |
| qat0 (control, full-precision rows) | 2-bit | 0.0973 | - |
| qat2 (2-bit STE) | 2-bit | 0.0957 | **-1.6%** |
| qat0 (control) | 3-bit | 0.0121 | - |
| qat3 (3-bit STE) | 3-bit | 0.0121 | **~0%** |

Both arms converge with finite gradients, so the hook is mechanically sound.
But **branch-only adaptation does not measurably reduce the damage at 3 bits**
(the worst case, 2 bits, moved only -1.6% in 30 steps), and the 3-bit damage
itself is small (0.0121 vs 0.0957 at 2 bits, consistent with the recorded A/B:
4-bit nearly free).

**Recommendation: do not fund a correction re-fit pod (~$5-7) on this
evidence.** Revisit only with a longer run or table-QAT (training the table
itself); the branch adapter has too little leverage on quantization error.
The container choice (Q3X32 / Q4X160 / release) stands on the 48-layer gate
alone.

## Caveats

- The heal changes the corrections, so every gate must be re-run; the healed
  artifact is a new file with a new sha.
- The shipped corrections were trained against the Q4_0 table; healing any
  container (including the current release) realigns them properly.
- Screen numbers are on the 2-layer mirror; only the 48-layer gate decides.
- Pod balance was $1.77 on 2026-10-12; the full heal needs a top-up.
