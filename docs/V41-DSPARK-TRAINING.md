# V4.1 DSpark training lane

**Status: first local training attempt complete; held-out gate failed.** See
[`V41-DSPARK-RESULTS.md`](V41-DSPARK-RESULTS.md) for the measured result. This
is the working plan for a custom-trained drafter for DeepSeek-V4.1-Flash on the
local AMD box. It does not add a serving path for another vendor.

## Scope

First target: a **small target-hidden DSpark-shaped head**, not a full fine-tune
of the released 3-block MoE DSpark.

The released V4.1 DSpark is ~5.1 GB of MTP tensors (three blocks, 128 experts
each). The first experiment trains a GRU-carried multi-token head that reuses
the target's own `output_norm`/`output` and consumes the target's hidden states
at layers 37–39. That keeps the first loop trainable on one card and answers the
real question — can self-distillation lift acceptance and wall-clock speed? —
before paying for a full DSpark fine-tune.

A full 3-block DSpark fine-tune remains a later branch. Do not start with it.

## Baseline

Before training, measure the existing engine with:

- no speculation;
- DSpark depth 1;
- DSpark depth 3/5 if it fits the same memory budget.

Record acceptance, tokens/pass, decode tok/s, token identity on a fixed prompt,
VRAM/GTT, and host RAM. The previous local result (depth 4, 19.07 vs 25.4 tok/s)
is the reason depth is a first-class gate, not an afterthought.

## Training signal

The target is the teacher. No API text is used for the distillation signal:
the API does not expose the hidden states and cannot load this drafter.

Capture, per engine step:

```text
int32 token, int32 pos, 3 × 5120 float32
```

This is the `STRATA_DUMP_MAIN` record documented in
`strata-ds41/tools/v41/dspark_accept.py`: the hyper-connection mean of the
stream entering layers 37, 38 and 39. The accompanying generated token stream
provides the target next-token labels.

Data contract for the first corpus:

- fixed prompt set, greedy decode;
- `tokens.i32`, `positions.i32`, `taps.f32` or sharded windows;
- `argmax.i32` = the target's own next token per position;
- a held-out split by prompt, never by random token;
- near-tie margins recorded so the loss can weight draft-target near-ties.

Capture is a GPU job and must run under the existing memory guard: one engine,
one card, no concurrent V4.1 process.

## Student

The first student is a V4.1 `DSparkHead`:

- input: the three 5120-wide taps, projected to the model width (the released
  DSpark's `main_proj`/`main_norm` role);
- GRU state across the drafted block;
- direct residual MLP branch (the k=1 form);
- target `output_norm` and `output` shared and frozen;
- confidence head trained on acceptance, used only for scheduling;
- start with k=3, then test k=1 and k=5.

The head must not duplicate the 129280-token vocabulary projection. It uses the
target's own output head, exactly like the Scion k=1/dspark heads.

## Loss and gates

Loss: cross-entropy on each draft slot plus a near-tie-aware term and a
confidence BCE. The target weights stay frozen.

A candidate is kept only if:

1. acceptance improves over the existing DSpark at the same depth;
2. **decode tok/s** improves over no-spec on the same prompt and context;
3. the greedy stream is token-identical on the fixed quality prompt;
4. VRAM/GTT and host RAM stay inside the capture/training budget.

Acceptance alone is not a pass. A drafter that accepts more but verifies slower
is a revert.

## Export and ternary packing

Train in BF16/FP32 first. Then export the same ternary weights in both Scion
packings:

- **PTQ1_0** — 1.75 bpw, smaller file, memory-bound cards;
- **PQ2_0** — 2.125 bpw, cheaper unpack where the kernel benefits.

Bonsai 2's PTQ1_0/PQ2_0 parity result is the model: the packing choice is a
runtime/file-size decision, not a quality decision. Measure both on the target
engine.

The V4.1 engine needs a loader for the new small-head sidecar. Keep that loader
in `strata-ds41`; keep the trainer/quantizer/exporter in Scion.

## Other-model compatibility

DSpark is target-conditioned. A trained head is not a universal drafter:

- same V4.1 family/quant variants: test zero-shot; hidden-state drift may lower
  acceptance;
- other model families: retrain on that model's taps, tokenizer and output
  head;
- the reusable artifact is the pipeline, not the weights.

A compatibility test should report acceptance with the wrong target as a
negative control, not just a same-model success.

## First CPU steps

1. Add a converter from `main.bin` to the window layout used by the Scion
   trainers.
2. Add synthetic tests for the record size, alignment and labels.
3. Add the V4.1 head wrapper with a frozen target head.
4. Only then start one guarded capture run.

## Memory/budget rules

- Never run the teacher and a second full engine at once.
- Capture on one card with the existing guard; stop before the next phase.
- Train the small head on one card if it fits; otherwise rent a temporary
  80 GB trainer. Do not develop an NVIDIA serving path for this.
- Export the trained weights back to the AMD engine.

## Later, if the small head pays

Train the full 3-block DSpark with adapters or a constrained fine-tune, compare
against the small head, and only then decide whether the extra capacity is worth
the memory and complexity.
