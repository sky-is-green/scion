# Byte distillation, the two ways this touches us (working notes, 2026-10-15)

Paper: "Breaking the Token Ceiling: Distilling Smaller, Stronger Byte Models" (Marathe, Pagnoni et al.,
UW + Meta FAIR, arXiv:2609.12303).  A 1B-scale study of tokenization (Llama-3 tokens vs bytes vs bytes
with an `<eot>` symbol) crossed with the objective (cross-entropy vs distillation from Llama-3-8B).  Two
reusable pieces, one we can use now and one that is a direction, not a deployment.

## The two conversion methods, verified on our tokenizer

`Marginalize-It` (approximate) marginalizes the teacher's token distribution over the tokens matching the
observed byte prefix, renormalizing at each byte; `End-Of-Token` (exact) appends an `<eot>` symbol after
every token's bytes so the residual mass becomes "ends here" and nothing is lost.  Both are single-pass
over one teacher distribution (no extra teacher forwards).  Implementation and checks:
`byte_logits.py` (the paper's Appendix F.1, plus the table builder from a pack's `vocab.json`).

Measured 2026-10-15 on `pack-q6k/tokenizer` (248,320 tokens, widest 128 bytes), teacher = a Zipf-ish
distribution over the text's own tokens plus 4,000 random ones:

- the paper's Figure 2 example reproduces exactly (Marginalize-It 0.8/0.2; End-Of-Token 0.667/0.167/0.167);
- **End-Of-Token is exact on the real 248k vocabulary**: the product of the converted byte distributions
  along a token's path equals the teacher's probability of that exact byte string to 4.2e-17 (machine
  precision), over every token checked;
- **Marginalize-It's local error is real on 248k**: relative gap to the exact string probability median
  2.6%, p90 77%, 18% of tokens more than 10% off.  The paper's "works well in practice" is an end-task
  statement; per-token the approximation drops visible mass.  Use End-Of-Token.

## The dump economics (why this is the usable half)

`byte_table.py` exports the engine's table (`byte-table.bin`, 64.1 MB for the 248k EOT table) and reads
the dump back; the identity above is the dump's own check (`--verify`).  On our tokenizer, at 2.903
bytes/token measured on `tasks-community.json` (575,348 chars, 198,212 tokens):

| teacher logits | bytes per text token | per 1M tokens | note |
| --- | --- | --- | --- |
| full, float32 | 993,280 | 993 GB | what "no top-k" would cost |
| top-600 (the paper's controlled k) | 4,804 | 4.80 GB | 1.20x the EOT float32 |
| top-4096 | 32,772 | 32.8 GB | 8.2x |
| top-16384 | 131,076 | 131 GB | 32.7x |
| **End-Of-Token, float32** | **4,012** | **4.01 GB** | exact, no truncation |
| End-Of-Token, float16 | 2,006 | 2.01 GB | half, at the precision a distillation loss needs |

So at the paper's own k the byte dump is at parity with a top-600 token dump while being exact; against
anything anyone would actually use for serious distillation (k in the thousands) it is 8-33x smaller and
never drops the tail.  A 6.7e12-byte-unit EOT run (the paper's crossover point) is ~1.2e12 Llama-3-8B
tokens; at 4.01 GB per 1M tokens that is 4.8 TB of dump, against 39 TB for top-4096.

## The engine side (built, waiting for the cards)

`strata-q36` writes the dump itself (commit in `work/strata-dh`), so the conversion happens where the
logits already are and the full distribution never lands on disk:

- `STRATA_Q36_BYTE_TABLE=<byte-table.bin>` + `STRATA_Q36_LOGIT_DUMP=<file>`;
- during a prompt read, each row whose continuation is known gets the head computed (groups of 8 rows),
  softmaxed on the host, converted with the same math as `byte_logits.py`, and written as one record:
  gold token, its probability, the exact byte-string probability, and the EOT byte distributions
  (float16).  A table without the EOT slot gives Marginalize-It instead;
- the record carries `p_exact` precisely so the dump can check itself: `byte_table.py --verify` recomputes
  the product along the gold path and compares.  The format and that check are exercised now by
  `byte_table.py --selftest` (a dump written Python-side the engine's way, 21 records, worst |diff| 3.5e-6,
  passes).

Not run yet: the first real dump (one 35B prompt on a free card), the write overhead per token, and a
`--verify` of a real engine dump.  That is the next free-card step, together with the decision-head night.

## The direction we are not taking: a byte-level model line

The paper's headline (distilled End-Of-Token-1B beats distilled Token-1B asymptotically) is an
extrapolation; the crossover compute they compute is 6.3e22 FLOPs for EOT (2.0e23 for Marginalize-It),
which is far outside this box.  At the budgets they actually trained, the token model still wins
downstream.  And inference stays expensive: a 1.28B byte model spends ~5.5 forward steps per text token
(4.5 bytes + EOT), which is ~9.4 GFLOPs per text token, about the same FLOPs as our 176B A6B's ~12
GFLOPs.  Our box is bandwidth-bound, not FLOP-bound, so the honest comparison is bytes read per text
token: 5.5 steps x 2.56 GB (1.28B in f16) = ~14 GB per text token, against the 176B A6B's ~5-6 GB of
active weights per token.  A byte model would be ~2.5x slower and far weaker unless it is quantized to
4-bit (then ~3.5 GB per text token, comparable), and it would still be a small-model line, not a 176B
replacement.

If we ever want the tiny-scale version of their experiment (does the EOT/byte advantage show up at
CPU-trainable scale on our own data), the pieces are here: the converter, the table, the dump format.
That is a multi-hour CPU training pilot to run when the box is free; no GPU is needed for it.
