# The decision-head port — readout heads on our engine (working notes, 2026-10-15)

The lane after the EP port, and the reason for it: the decision-head study
(`huggingface.co/datasets/SkyIsNotGreen/decision-heads`, locally in
`work/scion/decision-heads`) shows a correctness verdict is already linearly readable from a
generation model's own last-token hidden state. A closed-form ridge + Platt head (about 66 KB for the
35B's 2048 dims) beats a prompted self-judge (test AUC 0.999 against 0.903) and beats purpose-trained
judges, and its two paying uses are selection, not steering: gated sequential sampling reaches
best-of-3 accuracy at about half the pool cost, and the veto (majority plus head override) adds 14
fixes / 2 breaks over 398 held-out tasks with no extra generation. Early cancellation is the studied
negative result: even a perfect predictor saves only 21-34 percent of tokens at 64-256 horizon, so it
is not worth building. The study's own next step says "package the veto/raw readout as an engine
feature"; this lane is that port onto our engine and the 176B.

The port has three stages. Stage 1 is done; the numbers are here.

## Stage 1 — the engine serves the hidden state (done)

`EMB path=<file> [norm=l2] <id,id,...>` on `strata-q36` (and later the main engine): runs the prompt
and writes N float32, the last processed position's hidden **after `output_norm`** (the unembedding
input, the same vector llama.cpp returns for `--pooling last --embd-normalize -1`; raw, no L2), to the
file, then answers `EMB <dim> <file>`.  Only `EMB` lines take this path, so `GEN`/`GENI` are untouched
(checked by the smoke below and by the replay).  The code is `src/qwen35/q36_engine.cu` (58 lines),
commit `2b62dee5` on branch `decision-head` in `work/strata-dh` (based on `scion-0.1.44` = `2b5d28d0`),
built with the AMD_HIP.md recipe (`build-hip/strata-q36`).

Validation is a token-exact replay of the study's judge bench on our engine:

- bench: `judge-bench.jsonl`, 378 items, frozen 262/116 train/test split (97 pos / 281 neg).
- prompts: the study's own prompt (system "You are a decision function…", the A/B body), rendered with
  the pack's template (`pack-35b/tokenizer/chat_template.jinja`, `enable_thinking` false) through
  `serve/frontend.py`'s `ChatTemplate` and encoded with `tools/strata_tokenizer.py`; ids 112 to 642
  tokens, mean 273, ending `<|im_start|>assistant\n<think>\n\n</think>\n\n`.
- engine: `strata-q36 --serve --model Scion-35B-A3B-PQ2_0-corr.gguf --max-context 4096 --chunk 512`,
  one 7900 XT (HIP device 1, the card the other lane was not using; its 35B stayed up on device 0).
  378 EMB passes in **75 s** (about 1,500 prompt tokens/s), load 3 s (page cache warm), warmup 0.4 s.
- result (test split, 116 items):

| readout | AUC | best BA | acc at 0.87 |
| --- | --- | --- | --- |
| the study's **published** head applied to our states | **0.9996** | 0.9943 | 0.9828 |
| the same head where it was fit (the study's number) | 0.9988 | 0.9943 | — |
| a refit on our states | 1.0000 | 1.0000 | 0.9914 |

Score correlation between the published and refit readouts on the test split: 0.97.  The frozen head
transfers across the two engines on the same body, which says the embedded prompt (template and
tokenizer) and the extracted state (final layer, last position, post-norm) match the study closely
enough that a head fit on one engine scores on the other.  It also says the extraction is right:
a wrong layer or a pre-norm state would tank the frozen head.

Artifacts (session): `work/scion/decision-heads-work/dh_replay.py` (tokens / probe / score),
`judge-prompt-ids.jsonl`, `dh-probe-states.npz`, `dh-replay.json`, `dh-probe.engine.log`.
The endpoint was smoke-tested for `EMB` then `GEN` on the same ids (state reuse works).

## Stage 2 — the 176B head (done, first pass)

The served 176B is `qwen4exp` (hidden 2560) through the main engine, so `emb` was added there too:
`GEN 1 emb=<file> <ids>` writes the last prompt token's head input (`head_mixed_`: after the output
mixer and norm, the unembedding input) and ends the request without emitting a token.  The first
decode window holds the last prompt token alone, so its row 0 is that vector; a pipelined request
falls back to the serial loop for it (the `pl_serial` reason), so the served `--pipeline-windows 2`
config works unchanged (commit `5b05d404`).  The 35B and the 176B share one tokenizer and template:
all 378 judge prompts are token-identical across the two bodies (checked item by item), so the two
were scored on the same inputs.

Measured 2026-10-15 on `Scion-FlashNext-176B-A6B-Q6K.gguf`, 2x7900XT, layer split auto with
`--pipeline-windows 2`, the CPU busy with the MTP lane's training job (the other lane's service was
stopped for the run and restarted after it):

| readout | AUC | best BA | acc at 0.87 |
| --- | --- | --- | --- |
| refit on the 176B's states (test, 116 items) | **0.9941** | 0.9828 @ 0.60 | 0.9655 |
| our 35B replay, the study's published head | 0.9996 | 0.9943 | 0.9828 |
| the same 35B head where it was fit (the study) | 0.9988 | 0.9943 | — |

378 judge passes took 1594 s (4.2 s each: the prompt read is about 3 s in this config, the hidden
window itself 41 ms).  The extracted head is
`work/scion/decision-heads-work/scion-176b-a6b-decision-head.json` (83 KB, dim 2560): the study's
export schema, the same frozen bench (sha `3289dcb8…`), n_train 262 / n_pos 68 / n_neg 194.  A 176B
value in the same 0.988-0.999 band as the study's bodies says the main-path state is the right one; a
wrong row, a wrong layer, or a pre-norm state would not read this well.

The caveat: 0.9941 is the judge number (reading the correctness of given candidates).  The paying
uses are selection policies over candidate pools (gating and the veto), and those numbers (the
study's veto 14 fixes / 2 breaks over 398 tasks; gating at about half the pool cost) still have to be
reproduced on the 176B once pools exist.  One selection preview is already measurable on this run:
the judge bench's own pools (100 tasks, 91 of them with 4 candidates) let the head pick the best
candidate, and the 176B's head picks the correct one in **97/100 tasks** (test split 29/30).  The
honest cost so far: a judge pass on the 176B is about 3 s in the served config, against 0.2 s on the
35B; the readout itself is a dot product.

## Stage 3 — the endpoints and the community arms (built; the overnight run is queued)

The readout's vector is served through the API (commit `69cb2515`): `POST /embedding` (llama.cpp's shape,
`embd_normalize` `-1` raw or `2` L2), `POST /v1/embeddings` (OpenAI's, unit-norm), and `POST /apply-template`
(renders the model's own template).  8 unit tests (`serve/test_embeddings.py`, no GPU or pack).  With those,
the study's own scripts run against Strata unchanged: `community_eval.py` and `veto_eval.py` are pointed at our
server, which is the closest thing to comparing like with like.  A five-minute end-to-end check through the
server (2026-10-15, 176B, the decision-head build): `/apply-template` ends
`<|im_start|>assistant\n<think>\n\n</think>\n\n` (thinking off), `/embedding` answers a 2560-dim raw vector,
`/v1/chat/completions` generated `"Hi"` with usage 18/2, and `scion-flashnext` came back `active`.

The overnight run (`decision-heads-work/dh_overnight.sh`, a transient unit, smoke-gated, resumes where it
stopped, and restarts the service either way):

1. stops `scion-flashnext`, serves the 176B through the decision-head build on 8897 with the study's sampling
   (temperature 0.7, top_p 0.95, top_k 40, min_p 0.05, thinking off) - the same defaults the study's llama.cpp
   had;
2. the study's community protocol on the 664-task set: 3 samples/task, then the judge-prompt states for each
   sample (answer and reasoning-trace variants) with the resumable embedder (`dh_embed.py`; the study's own
   embed phase restarts from zero on a crash);
3. the eval (arms A single, C gated, D best-of-N, majority, oracle; the J-Bench-frozen head as a chooser too)
   and both veto arms (in-domain head, frozen J-Bench head);
4. restores `scion-flashnext`.

The think-harder arm (B) is left out on purpose: it was dominated by gating in the study and its 2,048-token
thinking runs on the 176B would add about five hours for an arm already answered.

What the night should produce, for comparison against the study's Scion numbers (single 142/199, majority 154,
gated 152-154, best-of-3 149-156, veto 161, oracle 169): the 176B's own arm table, its in-domain and frozen
head AUCs on 1,992 samples, and the honest token costs.  Nothing about those numbers is known yet.

## Box notes

The other lane was serving the 35B (`strata-q36` + `serve/server.py`) on HIP device 0 throughout;
this lane ran on HIP device 1 only (verified with a `hipGetDeviceProperties` probe: HIP 0 = PCI
03:00.0, HIP 1 = PCI 07:00.0, the free card).  The replay used the free card at nice 10, 75 s of
prompt passes, no `--vram-reserve-mib` (the q36 engine ignores it: "not an option of this engine").
