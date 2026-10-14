# Scion MoE speed lane, part 2 — the 35B corrections + serve lane (handoff, 2026-10-14)

**Goal.** Fill the gaps left by part 1 on **Scion-35B-A3B**: apply the release's embedded corrections in the
q36 engine, stand up its serve lane, baseline it with the fixed protocol — then resume the speed work on this
model.  The 176B stays the live service throughout this session.

**Read first:** `SCION-MOE-SPEED.md` (part 1 record: the verify instruments, the warmed budget, the identity
protocol, the 35B bring-up), then `NEXT-SESSION-BONSAI2-SPEED.md` (the methodology that transfers) and
`P4-STRATA-FIT.md` (the fit, the 176B corrections precedent, the VRAM policy).

## State to trust

- **Engine:** `work/strata`, branch `scion-0.1.44`, HEAD **`3d34f904`** (the 35B port; `58aa4712` the verify
  instrument), binary `work/strata/build-hip/strata`.  Tree clean.
- **Live service:** `scion-flashnext` on 8081 — the Q6K 176B + corrections + `--pipeline-windows 2`, reserve
  3072 (config `work/strata/strata-q6k-176b.json`), **68-77 tok/s warmed**.  Keep it running except in
  serve-down windows; stop it before any load test on the same cards.
- **35B release:** `/home/penis/models/scion-strata/Scion-35B-A3B-PQ2_0-corr.gguf` (11.337 GB, sha
  `4e9e3292c2af320b436b9c0cc10d10aee86689b0f0ce9d390be9d8365cffa040`, fetched 2026-10-14 and verified) —
  `qwen35moe`, 40 layers, 256 experts top-8, hidden 2048.  Expert banks are **type 142** (supported since
  `3d34f904`); the embedded corrections are **type 43** + F16 router pairs (**present but not applied**); the
  released v1 has **no MTP block**.
- **Measured on the release (part 1):** loads (973 tensors, 4.32 GiB dense + 7.97 GiB experts on one card);
  **pp8 346 tok/s / tg8 102 tok/s** at `--max-context 256/512/1024`, uncorrected and without MTP;
  `tests/pq2_0_expert_parity` mean |gpu-ref|/mean |ref| 0.2%.
- **The q36 engine is single-device.**  There is no layer split / two-card arm in it today; "two-card" is new
  engine work, not a config.  Do not promise it in this session — scope it after corrections + serve.

## Gap 1 — the corrections (first move; this is the quality gate)

The release's quality is the *corrected* one (v1 card: 11.60 -> 8.35 PPL with corrections, 79.0 HellaSwag).
The q36 engine reads none of the `.lora_` tensors today.  Per layer (from the file census):

| tensor | shape | type | layers | placement |
| --- | --- | --- | --- | --- |
| `ffn_moe_out.weight.lora_a` | [2048, 512] | type 43 | all 40 | MoE output |
| `ffn_moe_out.weight.lora_b` | [512, 2048] | type 43 | all 40 | |
| `ssm_out.weight.lora_a` / `.b` | [4096, 512] / [512, 2048] | type 43 | 30 GDN | GDN out-proj |
| `attn_output.weight.lora_a` / `.b` | [4096, 512] / [512, 2048] | type 43 | 10 attn | attention out-proj |
| `ffn_gate_inp.weight.lora_a` / `.b` | [2048, 256] / [256, 256] | F16 | all 40 | router delta |

**First move: nail type 43's decode and the apply order from the sources, not from guesswork.**  Leads:

- the release's own pipeline: `work/scion/moe/export_branches_lora.py`, `merge_adapter_into_body.py` (what the
  container is and how the factors are encoded), `work/hivebench/artifacts/ternary/moe/qwen35/rental-v2/...`
  and the model cards under `work/scion/docs/archive/` and `work/hivebench/.../hf-release/README.md`;
- the fork that runs it: `work/prism-moe-corr` (sky-is-green/prism-ml-llama.cpp, `moe-corr-runtime`) —
  `src/llama-adapter.cpp` (embedded adapter attach), the ggml type table for the g128 container, the graph
  hook (`build_lora_branch` style) that adds the branch;
- the 176B precedent in this tree: `src/core/corrections.cpp`, `src/kernels/cuda/lora_corr.cu`, call sites
  `src/core/layer.cpp` / `verify.cpp` / `prefill.cpp`, and `lora_corr_parity` as the oracle style (here the
  factors are type 43/F16 instead of the 176B's f16 pairs, so the parity reference needs a type-43 row decode
  — port it from the fork's `ggml-cuda/vecdotq.cuh`).

Then implement in q36 (decode **and** prompt): `out += B * (A * x)` at the three placements off the same
normalised inputs the 176B uses, and the router delta into `route_topk`'s logits.  Oracle: the fork's outputs
on the same prompts (the lane's `tools/qwen35/parity.py` / `kl.py` and the `scion/moe/box-run*.sh` scripts),
or the stored results under `work/hivebench/...` if usable.  Acceptance: warmed greedy token streams match the
fork's on the fixed prompts; record the served PPL only if the harness is at hand.

## Gap 2 — the serve lane

- Config `work/strata/strata-35b-a3b.json` beside `strata-q6k-176b.json` (model = the 35B GGUF, `--gpu 0`,
  `--max-context` and `--chunk` chosen, no MTP), and a user unit `scion-35b` (port **8082**, so it can run
  beside nothing else but coexist in config; the two services cannot run at once on the two cards anyway —
  176B uses both).
- `serving/bench_strata.sh` hardcodes 8081: add a port argument (the `moe-pin-cycle.sh` pattern) or a sibling
  script; the fixed protocol and `moe-exact.sh` then work unchanged.
- VRAM policy: run `serving/vram_reserve.py --apply` for the 35B config; the release fits one card (7.97 GiB
  experts + ~4.3 GiB dense), so the display card needs only the policy reserve.
- The 35B identity stash: once the lane exists, capture its warmed full text the way the 176B stash works
  (`serving/moe-exact.sh`), and keep it in `/tmp/opencode/scion-moe-exact`.
- No drafter arm: v1 carries no `nextn` block; the published file is drafterless.

## Gap 3 — the speed budget on the 35B (after 1-2)

- The part-1 `STRATA_VERIFY_GRAPH_TIME`/`ABLATE` instruments live in the **Flash-Next verify window** and do
  not touch q36.  The first move here is a q36 decode budget: events around the decode graph launch + a
  kernel-class split, or the q36 timing prints extended.  Publish one table before touching kernels.
- Then part 1's step-2 candidates as they apply: the grouped expert GEMM's effective rate vs its footprint
  (the 35B reads ~13 MB/layer/window instead of the 176B's ~28 MB — the shapes differ), the hyper-connection
  chain, and the corrections' cost once applied (the 176B's is 2.6 ms/token after its reshape — use the same
  reshape as the starting point, not the naive branch).
- MMQ for type 142 (currently the prompt runs "the native grouped kernels") is a measurable prompt-speed
  candidate: implement `MMQ::supported(142)` + the quantize/apply path, or leave it if the numbers say the
  prompt is not the wall.

## Gates and rules (unchanged)

- **Token identity:** every change leaves warmed greedy requests byte-identical.  176B arm: config +
  `--adapt-every 0` (`strata-q6k-176b-pin.json`, `serving/moe-pin-cycle.sh`); the pre-change reference binary
  at `/tmp/opencode/scion-moe-ref` is tmpfs and may be gone — rebuild from `58aa4712`'s parent if needed.
- **One step per commit, numbers in the message.  Never `git push`.**  Model-load tests only in serve-down
  windows; stop one service before starting another on the same port; `--vram-reserve-mib 3072` for the 176B.
- **Do not re-derive the rejected list:** correction cooperative fusion, RCCL tensor parallel, MTP stage-0
  placement, the two-rows-per-warp GEMV ports, the doorbell-spin split wait.
- Report when a measurement changes something — the budget, the surprises and the reverted dead ends are the
  output, not the edit count.

## Continuation prompt (copy-paste)

```
New session: Scion MoE speed lane part 2 — fill the 35B gaps. Read work/scion/docs/NEXT-SESSION-SCION-MOE-SPEED-2.md
first (the handoff: state, the corrections task, the serve lane, the rules), then work/scion/docs/SCION-MOE-SPEED.md
(part 1: the instruments, the budget, the identity protocol, the 35B bring-up).

State to trust: the engine tree is work/strata, branch scion-0.1.44 (HEAD 3d34f904), binary build-hip/strata; the
live service scion-flashnext on 8081 serves the Q6K 176B + corrections + --pipeline-windows 2 at 68-77 tok/s
(reserve 3072; keep it running except in serve-down windows); the 35B release is
/home/penis/models/scion-strata/Scion-35B-A3B-PQ2_0-corr.gguf (11.337 GB, sha 4e9e3292..., type-142 experts run
at pp8 346 / tg8 102 tok/s on one card, uncorrected, no MTP).

Do this in order: (1) apply the release's embedded corrections in the q36 engine (type 43 + F16 router pairs;
nail the decode/apply order from work/prism-moe-corr and the scion/moe export scripts, oracle against the fork),
(2) stand up the 35B serve config + service and baseline it with the fixed protocol, (3) build the 35B's decode
budget and apply part 1's speed lens.  One step per commit with numbers; every change must leave warmed greedy
requests byte-identical (fixed protocol: serving/bench_strata.sh; keep --vram-reserve-mib 3072); model-load tests
only in serve-down windows; never git push; do not re-derive the rejected items in the part-1 record.  Do not stop
for approval; report when a measurement changes something.
```
