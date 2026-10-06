# Scion-FlashNext-176B-A6B — publication plan (human actions; we never push)

State as of 2026-10-05 (W4 closed; release = step-3000 checkpoint).

- **Artifacts (local, drive W; backups on the 1.1T).  Publication renames:**
  - `qwen4exp-48l-ptq1_0-corr-step3000-ple.gguf` → publish as
    **`Scion-FlashNext-176B-A6B-F16body.gguf`** — 65.80 GB, sha
    `2f4bd083861b027ede8b81eacf5f8b808f04d3c72e612c42d54758a473aa22ba`
    (validated, generation-proven, cap_eval 14/20; backup on the 1.1T).
  - `qwen4exp-48l-ptq1_0-corr-step3000-ple-q6k.gguf` → publish as
    **`Scion-FlashNext-176B-A6B-Q6K.gguf`** — 60.17 GB,
    sha `236c6ef96f0a5c3bbf660101c51cb1fdcd6f63724d9341a3d5aedc568bd08cec`
    (containers byte-identical; body Q6_K/Q8_0; generation-proven;
    cap_eval 15/20).
  - Gate + cap_eval JSONs in `hivebench/artifacts/ternary/moe/qwen4exp/`;
    the step-4000 gate (W4 experiment) is in `pod/remote-gate4000/`
    (mean KLD 0.5880 — flat/worse; not shipped).
- **Runtime** lives only in the local worktree `work/llama-qwen4exp`, branch
  `qwen4exp-proto`, HEAD `0ec5be738` ("Merge commit '6c84c7d5d' into
  qwen4exp-proto"; fork of PrismML-Eng/llama.cpp with the Prism PQ2_0/PTQ1_0
  containers).  Uncommitted on top: `src/models/qwen4exp.cpp` (the
  `ffn_moe_out` correction hook, +12 lines) plus small spec text changes.
  **Nothing is pushed.**

## 1. Runtime repository (human)

1. Review the uncommitted diff; commit on `qwen4exp-proto`: qwen4exp arch +
   PTQ1_0/PQ2_0 containers + `ffn_moe_out` virtual target + embedded adapters
   + PLE table handling.
2. Push the branch to the fork `sky-is-green/prism-ml-llama.cpp`.
3. Tag a release (e.g. `qwen4exp-proto-2026-10`) after the HF files are up.
4. Optional: README section with the serving flags + memory budget (below).

## 2. HF model repository (human)

1. Create the model repo: **`SkyIsNotGreen/Scion-FlashNext-176B-A6B`**.
2. Upload both variants + `.sha256` files, `LICENSE`, `NOTICE` (**decision
   2026-10-05: upload both**; the **Q6_K is the headline/default** download -
   smaller and cap_eval 15/20 - and the F16-body file is the
   uncompressed-body fidelity variant, 14/20).  Staged for upload on the
   Desktop at `~/Desktop/UPLOAD-Scion-FlashNext-176B-A6B/` (sha-verified):
   ```
   pip install -U "huggingface_hub[cli]"
   hf auth login
   hf repo create SkyIsNotGreen/Scion-FlashNext-176B-A6B --repo-type model
   cd ~/Desktop/UPLOAD-Scion-FlashNext-176B-A6B
   hf upload-large-folder SkyIsNotGreen/Scion-FlashNext-176B-A6B . --repo-type model
   ```
   (`hf upload <repo> <file> <path>` per file also works; single files of
   60-66 GB are fine, and upload-large-folder resumes after a drop.  ~126 GB
   total, ~1-2 h at the home uplink.  License note: the base model is
   `license:other` (Qwen); the card follows ISTA's apache-2.0 treatment - the
   human owns this choice.)
3. Model card from `scion/docs/RELEASE-FLASHNEXT-176B-A6B-MODEL-CARD.md`
   (staged as the repo README; numbers final).
4. Note that the file **requires the fork runtime** (custom PTQ1_0 container);
   stock llama.cpp cannot load it.

## 3. Serving flags / memory budget

```
llama-server -m Scion-FlashNext-176B-A6B-*.gguf -c 512 -ngl 99 \
  --cpu-moe -ot per_layer_token_embd.weight=CPU -fa on --numa distribute
```

- Host needs >= 64 GB RAM (the 36 GB body mapping thrashes on a 30 GB host).
- `--cpu-moe` + table on CPU is the proven path; GPU offload of the body is
  slower on a 20 GB card.
- CPU-only llama-server (`-t 16`) is the working consumer/eval path.

## 4. Validation manifest (attach to the release)

| item | value |
|---|---|
| parse-back | 1416 tensors, 177.28 B params, 59 KV, 96 embedded adapter pairs |
| `ple_export_verify` | VERIFY OK (KV exact; table Q4_0 sample 0.0028; conv1d/norms exact; adapters finite) |
| `gguf_ref_sweep` | clean except PLE (no part-1 counterpart) + ref Q4_K noise on `blk.46.ssm_out.weight` |
| bytecheck (kq) | 192 adapters full + 144 experts + table sampled: 0 diffs vs primary |
| generation proof | `The capital of France is` -> ` Paris.`; `1,2,3,4,5,` -> ` 6..14,`; `def fibonacci(n):` -> valid Python (both variants) |
| KLD gate step3000 | mean 0.5800 / p99 4.4584 / max 13.0014 |
| KLD gate step4000 (W4, not shipped) | mean 0.5880 / p99 4.8389 / max 13.3137 |
| cap_eval primary | 14/20 = 0.700 |
| cap_eval kq | **15/20 = 0.750** (0 harness errors; math 5/5) |
| PPL / HellaSwag / Winogrande | **measured 2026-10-06** (one A100-SXM4-80GB session, same harness, vs ISTA GSQ-RCO Q2_0): PPL **5.4571 ± 0.0324** vs 5.2396 ± 0.0326; HellaSwag 400 **82.00** vs 81.50; Winogrande 400 **77.50 ± 2.09** vs 74.75 ± 2.18 (PPL reproduced on a second build; ISTA's reasoning suite not measured by us) |
