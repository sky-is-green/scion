# PTQ1_0 size track - 2026-11-11

PLAN section 13.6 / section 7 "Size track".  Status: **prototype passes; not
shipped.**  The container saves 1.51 GB at identical trits/scales, the HIP
kernels run it, and the engine's own 8-token generation matches, but the
cross-type logits are not bit-exact (KLD 0.0030).  A hard-subset eval is the
next gate before any release repack.

## What was done

1. `v2/scripts/transcode_experts_ptq1_0.py` decodes all 120 expert banks from
   the shipped Q2_0_g128 (type 142) bytes, repacks the same trits base-3 in the
   PTQ1_0 (type 143) layout, and copies every non-expert tensor and all
   metadata byte-for-byte.
2. Every transcoded tensor was decoded back and compared with the source:
   **120/120 tensors, 0 trit mismatches, 0 scale mismatches**.
3. The full file was loaded by `strata-q36` (the qwen35moe engine) and run
   against the Q2_0 file with the same 8-token prompt.
4. The existing `ptq1_0_parity` tool was rebuilt for the 35B geometry
   (`[2048, 512, E]`) and run on the real layer-0 blobs.

## Size

| file | bytes | GiB | notes |
|---|---:|---:|---|
| shipped Q2_0 | 11,336,908,288 | 10.56 | type 142 experts |
| PTQ1_0 transcode | 9,826,958,848 | 9.15 | type 143 experts, same everything else |
| **saving** | **1,509,949,440** | **1.41** | 34 -> 28 bytes per 128 weights |

Expert payload alone: 8,556,380,160 -> 7,046,430,720 bytes (1.51 GB).  Expert
rate 1.75 bpw (was 2.125).

## Kernel parity (real layer-0 blobs)

`ptq1_0_parity` (CPU reference + HIP kernels):

```
PTQ1_0 experts: gu max|err| 2.183e-11, down max|err| 9.313e-10 PASS
GPU: decode-once vs per-entry expert kernels: bitwise equal
GPU: swiglu h vs reference rel 9.38e-08; expert out vs reference rel 2.19e-08
GPU: iq_mmvq gate rows vs reference: rel 5.98e-08
GPU: dequant f32 rel 0.00e+00 (0 of 1048576 differ), f16 0 differ,
     gate/up fp16 rows 0 differ
PTQ1_0 parity: 0 failures
```

The tool was written for the 176B FlashNext shape `[2560, 640, E]`; the local
patch changes the geometry guard to `[2048, 512, E]` (the 35B).  The source
patch is three lines in `strata/src/kernels/ptq1_0_parity.cpp`; the file was
restored after the run, and the binary is
`work/strata/build-hip/ptq1_0_parity`.

## Engine-level check

Same prompt, same 8 tokens, `strata-q36 --tokens ... --n 8
--dump-full-logits`:

| metric | Q2_0 vs Q2_0 rerun | Q2_0 vs PTQ1_0 |
|---|---|---|
| generated tokens | identical | identical |
| top-1 agreement | 8/8 | 8/8 |
| max abs logit diff | 0 | 0.640 |
| mean abs logit diff | 0 | 0.0575 |
| relative L2 | 0 | 3.57% |
| KLD mean (per position) | 0 | **0.0030** |
| KLD max | 0 | 0.0158 |

WikiText-2 PPL (`llama-perplexity -c 512 --chunks 16`, same build, GPU 1):

| model | PPL |
|---|---:|
| Q2_0-corr | 9.0183 +/- 0.358 |
| PTQ1_0-corr | **9.0076 +/- 0.357** |

The PPL difference is within noise (the 16-chunk sample is deliberately
short; release-grid PPL uses 580 chunks).  The Q2_0 path is deterministic
(rerun bit-identical).  The PTQ1_0 path uses different kernels/accumulation
order, hence the small distribution shift.  This is much larger than the
per-kernel parity errors (1e-8) and is not a "free repack" in the strict
sense; it needs a hard-subset A/B before release.

Runtime: 4.42 GiB on the GPU for the dense part in both builds; routed experts
7.97 GiB (Q2_0) vs 6.56 GiB (PTQ1_0) resident.  Generation on the 8-token
probe was 96.4 tok/s (Q2_0) vs 126.0 tok/s (PTQ1_0), not a controlled
benchmark.

## Decision

- Keep the size track as a **prototype**; do not switch the v2 release
  container yet.
- The next gate is the 30-item hard subset with a merged PTQ1_0 body +
  corrections; if it ties the Q2_0 body, the 1.51 GB saving is real.
- The earlier "q36 HIP kernels are Q2_0-specific" note is outdated for this
  tree: type 143 is in `iq_kernels.cu` (`Fmt<143>`, `STRATA_GU_FMTS`,
  `STRATA_D_FMTS`) and the engine's `native_expert_supported(143,143,...)`
  path runs.

## Reproduce

```bash
# 1. full transcode (~10 min, 9.8 GB output)
PYTHONPATH=work/prism-qwen4exp-shrink/gguf-py \
  python v2/scripts/transcode_experts_ptq1_0.py \
    --in models-p310/scion-strata/Scion-35B-A3B-PQ2_0-corr.gguf \
    --out /tmp/opencode/Scion-35B-A3B-PTQ1_0-corr.gguf

# 2. engine probe (GPU 1)
cd work/strata
HIP_VISIBLE_DEVICES=1 ./build-hip/strata-q36 \
  --model /tmp/opencode/Scion-35B-A3B-PTQ1_0-corr.gguf \
  --max-context 512 --tokens "248045,9707,11,198,248046,248045,872,198" \
  --n 8 --dump-full-logits /tmp/opencode/ptq.logits

# 3. kernel parity (after the 3-line geometry patch + rebuild)
cmake --build build-hip --target ptq1_0_parity -j
./build-hip/ptq1_0_parity /tmp/opencode/ptq1_0-parity.gguf
```
