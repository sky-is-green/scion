# PLE table containers: Q3X32, Q2X32, Q4X160 (design, 2026-10-11; gates 2026-10-12)

**Status: Q3X32 and Q4X160 are gated; Q4X160 is the chosen container.**
Q3X32: +0.011 KLD / +1.30% PPL / -0.25 HS / -0.25 WG at 53.76 GB. Q4X160:
-0.0001 KLD / +0.41% PPL / 0.00 HS / +0.25 WG at 57.615 GB (both table-only).
Q2X32 is implemented but not gated. Fork types + readers + parity tool
(bit-exact on real rows, CPU and HIP), Strata reader + fixture parity
(bit-exact vs the fork's decoder), streamed exporter transcode (+ in-script
spot check), pod patch `ple-table.patch` verified to apply on a pristine
`qwen4exp-proto` + the 5-patch sequence. The codec is a learned codebook with
a per-block least-squares scale (8 levels per 32-element block; Q4X160: 16
levels per 160-element row).

These are the 32-element block types for the PLE n-gram table
(`per_layer_token_embd.weight`, 320,001,536 rows x 160 elements = 5 blocks).
They replace Q4_0 (90 B/row) to shrink the 28.80 GB table without touching the
experts:

| type | B/block | B/row | table size | saving |
| :--- | ---: | ---: | ---: | ---: |
| Q4_0 (release) | 18 | 90 | 28.80 GB | - |
| **Q3X32** (type 144) | 14 | 70 | **22.40 GB** | -6.40 GB |
| **Q2X32** (type 145) | 10 | 50 | **16.00 GB** | -12.80 GB |
| **Q4X160** (type 146) | 82 | 82 | **26.24 GB** | -2.56 GB |

The row geometry forces 32-element blocks: 160 = 5 x 32, and K-quants
(256-element blocks) do not divide the row. Q4X160 uses the whole row as one
block (one fp16 scale per 160 values), the smallest 16-level container the
row geometry allows.

### Measured container ladder (16384 real rows, test split)

Every container uses the same codebook + least-squares-scale pipeline.
Relative rmse against the Q4_0-dequantized values:

| container | bpw | B/row | table | rel rmse | est. PPL (48l)* |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Q3X32 (gated) | 3.5 | 70 | 22.40 | 16.5% | **+1.30% (measured)** |
| Q3X20 | 3.8 | 76 | 24.32 | 15.0% | ~+1.1% |
| Q3X16 | 4.0 | 80 | 25.60 | 14.3% | ~+1.0% |
| **Q4X160** (gated) | 4.1 | 82 | 26.24 | 9.7% | **+0.41% (measured)** |
| Q4X80 | 4.2 | 84 | 26.88 | 9.1% | ~+0.4-0.7% |
| Q4_0 (release) | 4.5 | 90 | 28.80 | 0% | 0 |

*anchored on the measured +1.30% at 16.5% error; brackets span error- and
error-squared scaling. 16-level containers cannot go below 4.1 bpw: the codes
alone fill 80 B of the 82 B row. 8-level containers at 3.9 bpw and below
(Q3X160 19.4%, Q3X20 15.0%) are dominated by Q3X16/Q4X160 at the same bytes.

## Quantizer rule (frozen): learned 8-level codebook + per-block least-squares scale

One fp16 scale per block; the 3-bit code indexes a fixed 8-level codebook.
The encoder picks the per-block scale by least squares for those levels:

    levels[8] = {-5.17308617, -3.14964318, -1.84705281, -0.830217063,
                  0.371574461, 1.65762842, 3.0041852, 5.05979347}
    amax = max|x| over the 32 elements;  s = amax / max(levels)
    repeat 3x:  c_j = argmin_l |x_j - s*levels[l]|;  s = sum(x*c_lv)/sum(c_lv^2)
    s16 = fp16(s);  c_j = argmin_l |x_j - s16*levels[l]|   (re-assigned on the stored scale)
    decode: x_j = levels[c_j] * s16

Q2X32 uses the mid-rise set `{-3, -1, 1, 3}` with the same least-squares scale
(decode `x = levels[c]*s`, i.e. `(2c-3)*s`). `amax = 0` gives `s = 0`.

Q4X160 uses a 16-level codebook with the same least-squares scale, one scale
per 160-element row (`levels[16]` in the fork's `kQ4x160_levels`):

    levels[16] = {-0.830319583, -0.56551832, -0.395098478, -0.278073937,
                  -0.207071558, -0.134492248, -0.105560891, -0.0803469718,
                  8.09813901e-06, 0.0911125243, 0.129564166, 0.206828877,
                  0.278073341, 0.395026445, 0.564443111, 0.827598035}

The codebook is trained on the release table itself (64k random rows,
Lloyd-Max on amax-normalized blocks) and is part of the type definition: the
constants live in the fork's `kQ3x32_levels`, the engine's `dequant.hpp`, the
parity tool and the exporter's `Q3X32_LEVELS`, and must match exactly. The
codebook was stable across 16k/64k samples; a forced-symmetric variant
measured worse (19.9 vs 19.4 percent relative rmse).

### Why this rule, measured on real rows

Same-size candidates on an 8192-row sample (4096 train / 4096 test), relative
rmse against the Q4_0-dequantized values:

| rule | rel rmse (test) | engine change |
| :--- | ---: | :--- |
| mid-rise, amax scale | 24.7% | none |
| mid-rise + least-squares scale | 22.5% | exporter only |
| **codebook + least-squares scale (chosen)** | **19.1%** | decoder constants |
| affine min/max (needs a second fp16, +2 B/block) | 21.9% | - |

The codebook beats even the more expensive affine option at the same 14/10 B
budgets. Per-block `rms/amax` is 0.43, i.e. blocks are filled, not spiky.

## Byte layout (LSB first, little endian)

    block_q3x32 (14 B):   d: fp16     (2 B)
                          qs[8]: low 2 bits of each code, 4 per byte
                          qh[4]: high 1 bit  of each code, 8 per byte
    block_q2x32 (10 B):   d: fp16     (2 B)
                          qs[8]: 2-bit code, 4 per byte
    block_q4x160 (82 B):  d: fp16     (2 B)
                          qs[80]: 4-bit code, 2 per byte; byte j holds
                          element j in the low nibble and element j+80 in
                          the high nibble (the Q4_0 convention)

Element `j` (0..31) of a block:

    Q3X32:  c = ((qs[j/4] >> ((j%4)*2)) & 3) | (((qh[j/8] >> (j%8)) & 1) << 2)
            x = levels[c] * s
    Q2X32:  c =  (qs[j/4] >> ((j%4)*2)) & 3
            x = (2c - 3) * s

The 3-bit code is split into byte-aligned planes on purpose: no bit-crossing,
the same extraction works scalar (CPU) and per-thread (CUDA/HIP), and all
implementations (fork C++, fork CUDA, Strata, parity tool, Python exporter)
stay bit-exact.

## Where the type is used

- Fork (`prism-ml-llama.cpp`, patch `ple-table.patch`): ggml type ids
  144/145/146, CPU `to_float`/`from_float_ref` + get_rows dispatch, CUDA/HIP
  `dequantize_q3x32/q2x32/q4x160` + get_rows cases, `tools/ple-table/`
  parity tool.
- Strata: `src/kernels/ngram.cpp` decode branch + type acceptance, mirroring
  the P2 Q4_0 precedent; `ple_x32_parity` bit-exact against the fork decoder
  fixture (`strata/tests/fixtures/`).
- Exporter (`hivebench/tools/strata-bridge/`): streamed Q4_0 -> Q3X32/Q2X32
  row transcode (never materialise the table), then the release repack.
- No new file type (ftype): the table type is per-tensor; the file keeps its
  original `general.file_type`.

## Parity chain

1. Fork tool vs an independent scalar decoder in the tool, on real 4l rows
   (bit-exact), CPU backend and HIP backend (same sources as CUDA).
2. Strata test (`ple_x32_parity`) vs a fixture written by the fork's decoder
   on the same rows (bit-exact), so the two engines cannot drift.
3. Pod: the KLD/code/general gates are the end-to-end check for the CUDA path.

## Validation (2026-10-11, local, zero pod spend)

Whole chain re-run after the codebook change (all bit-exact):

- Fork `tools/ple-table/ple-table-parity`:
  - real 4l artifact Q4_0 control, 320,001,536 rows: 0 of 5,920 differ;
  - 1,000,000 real rows transcoded by the exporter: Q3X32, Q2X32 and
    Q4X160 each 0 of 5,760 differ (CPU host-ptr path);
  - HIP/RX 7900 XT (`--device ROCm0`, the CUDA sources hipified): all four
    types 0 of 5,760 differ.
- Strata `ple_x32_parity` (fixture = the fork's `--dump` output):
  `Q3X32 PLE: 36 fixture rows, 0 of 5760 values differ, max_abs 0.000e+00
  PASS`, same for Q2X32 and Q4X160.
- Exporter round trip on real rows: Q4X160 rel rmse 0.0964 (matches the
  0.0966 codebook measurement); Q4X160 release dry run 60.164 -> 57.604 GB.
- Codebook constants verified bit-identical (float32 bit patterns) across the
  fork ggml table, the parity tool, Strata `dequant.hpp` and the exporter.
- Exporter quality on the first 1,000,000 real rows (relative rmse against the
  Q4_0-dequantized values): Q3X32 0.199 -> **0.162**, Q2X32 0.463 -> **0.354**
  after the codebook + least-squares encoder.
- Exporter `repack_experts_q2_0.py --table-type q3x32 --preserve-experts
  --dry-run` on the release: 1416 tensors, 60.164 GB -> 53.764 GB, table
  28.800 -> 22.400 GB, experts untouched.
- `ple-table.patch` (fork commit `c8beb3aeb` on local branch `p5-ple-table`)
  applies clean on a pristine `qwen4exp-proto` + patches 1-5 (re-verified
  after the codec change).
- Local model-level A/B (4l proof model, wikitext-2, same flags, paired):
  over the full 580-chunk corpus the Q4_0 table reads **PPL 357,826.90** and
  the Q3X32 table **356,831.09**, i.e. **-0.278%** (per-token
  +0.00279 +/- 0.00046 nats, t +6.1, 344/580 chunks positive). Repeat runs of
  each file are bit-identical, so the difference is the table alone. The table
  change is PPL-neutral-to-slightly-better on this proxy; no regression is
  detectable, and a regression beyond ~+0.2% is excluded (95% CI).

### Pod gate (2026-10-12, L40S, one window, paired)

Release vs the Q3X32 table file (60.164 -> 53.764 GB, experts untouched),
same build (patches 1-6), same instrument, same window:

| gate | release | Q3X32 table | delta |
| :--- | ---: | ---: | ---: |
| KLD mean (8x512 wikitext, parked teacher) | 0.7819 | 0.7927 | **+0.0108** |
| KLD top1 | 0.6566 | 0.6546 | -0.0020 |
| wikitext-2 PPL (580 x 512) | 5.4561 | 5.5269 | **+1.30%** |
| HellaSwag 400 | 82.00% | 81.75% | -0.25 |
| Winogrande 400 | 77.75% | 77.50% | -0.25 |

The release side reproduced the historical baselines (PPL 5.4561 vs 5.4571,
KLD 0.7819 vs 0.7819), so the comparison is same-instrument and clean. For
scale, the closed 25% prune cost +0.186 KLD / +11.6% PPL / -2.75 HS for
-6.64 GB; this table swap costs +0.011 KLD / +1.3% PPL / -0.25 HS / -0.25 WG
for -6.40 GB. The paired code arms were **not** measured (window budget; a
slow pod host burned ~2.3 h extra on the CPU-bound table export). Pod-side
transcode spot check: max|err| 6.8e-3, rel rmse 0.1622.

### Pod gate: Q4X160 (2026-10-12, L40S, upload path, window $0.62)

Release vs the Q4X160 table file (60.164 -> 57.615 GB, experts untouched),
same build (patches 1-6), same instrument, same window. The table stream was
prebuilt locally (`work/tmp-export/q4x160-table.bin`, 26,240,125,952 B,
sha256 `915bae37…3015`), uploaded in 5 GB chunks and spliced pod-side
(`--table-bytes`) - no pod-side CPU encode, host speed irrelevant (the two
earlier attempts died on exactly that).

| gate | release | Q4X160 table | delta |
| :--- | ---: | ---: | ---: |
| KLD mean (8x512 wikitext, parked teacher) | 0.7819 | 0.7818 | **-0.0001** |
| KLD top1 | 0.6566 | 0.6588 | +0.0022 |
| wikitext-2 PPL (580 x 512) | 5.4561 | 5.4784 | **+0.41%** |
| HellaSwag 400 | 82.00% | 82.00% | 0.00 |
| Winogrande 400 | 77.75% | 78.00% | +0.25 |

The release side reproduced history exactly again (KLD 0.7819 / 0.6566), so
the deltas are clean; PPL/HS/WG compare against the Q3X32-window release
numbers (5.4561 / 82.00 / 77.75), the Q4X160 side is from this window. The
acceptance bar (KLD delta < ~+0.02, PPL < ~+0.8%) passes with margin, and
the ladder's +0.45-0.75% estimate was high. Pod-side export spot check:
max|err| 6.5e-3, rel rmse 0.0965 (matches the local 0.0966). Same-pod paired
code arms (Q4X160 vs release, gen 664 + hard, cap 4096) ran 2026-10-12
19:00Z with the same upload path (`paired-code-table.sh` +
`pod-paired-code-table-run.sh`): gen pass@1 73.2 -> 68.3 all / 71.4 -> 61.2
test, hard 55.5 -> 53.0 all (cap 2048), hard 57.9 -> 56.1 all (cap 4096);
paired flips 22/14 (gen), 24/20, 23/20 - inside the McNemar noise band, no
significant code/thinking regression.

## Mixed-precision option: the row-hit map (2026-10-12)

Replaying the exact PLE hash (ngram 3, 8 heads per gram, 16 heads, EOS resets)
over wikitext + the web/math + coder calibration corpora (4.9M tokens, 78.5M
row reads) shows the table is extremely sparsely exercised:

- only 8.4% of the 320M rows are ever hit; 90% of all reads land in 6.0% of
  rows (19.1M), 95% in 7.2%, 99% in 8.2%;
- keeping the hot rows at the release Q4_0 bytes (bit-exact) and the cold rows
  at Q3X32 costs +0.38 GB (90% coverage) to +0.52 GB (99%) and should leave
  ~0.1-0.3% PPL cost instead of the measured +1.3% for all-rows Q3X32
  (first-order: cost scales with the degraded read share);
- needs two tensors (hot Q4_0 + compacted cold Q3X32) + a sorted hot-row list
  (~40-76 MB); the fork resolves PLE rows host-side already, so the hot/cold
  index remap and the select can live there; Strata routes in `PleTable`.

Not built; recorded as the design of record if the uniform +1.3% is rejected.

## Decision (2026-10-12)

- **Q4X160 ships**: KLD ~0, PPL +0.41%, HS 0.00, WG +0.25 at 57.615 GB
  table-only (~56.0 GB with the body/corrections levers). The acceptance bar
  (KLD delta < ~+0.02, PPL < ~+0.8%) passes with margin.
- Q3X32 (52.2 GB with levers, +1.30% PPL) stays as the small-size fallback.
- Q2X32 stays unbuilt/ungated: it was only worth gating if the Q3 gate held
  and the budget allowed; at Q4X160's measured +0.41% the 3-bit step is
  dominated.
- The mixed-precision hot/cold option (above) stays parked, superseded.
