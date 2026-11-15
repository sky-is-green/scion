# Scion-35B-A3B decision head

> Mirrored from the [Hugging Face repo](https://huggingface.co/SkyIsNotGreen/Scion-35B-A3B/tree/main/decision-head); the head weights and JSON are included here.

A closed-form correctness readout for this body: a ridge on the standardised last-prompt-token
hidden state plus Platt calibration, 66 KB of weights.  It answers "is this candidate answer
correct?" for the model's own outputs, and it is what the veto and gating policies in the study
([`SkyIsNotGreen/decision-heads`](https://huggingface.co/datasets/SkyIsNotGreen/decision-heads))
are built on.

## Files

| file | what |
| --- | --- |
| `scion-35b-a3b-decision-head-q36.json` | the exchange form (the study's export schema: `weights.w/mu/sd/b/platt`, the prompt, the training record, the test metrics) |
| `scion-35b-a3b-decision-head-q36.bin` | the Strata engines' runtime form (magic `SJH1`: dim, then w, mu, sd, b, a, c as float32), converted from the JSON |

## The judge prompt (format-locked)

The head is fitted on this exact wording, with the body's own chat template and thinking disabled:

```
system: You are a decision function. Reply with the answer label only.
user:   Question: {task}
        Candidate answer: {candidate}
        Is the candidate answer correct?
        A. Yes
        B. No
assistant (generation prompt, thinking disabled)
```

The readout is `p_correct = sigmoid(a * (w . (h - mu) / sd + b) + c)` over `h`, the last prompt
token's hidden state (the unembedding input).

## Measured

- Frozen judge bench (378 items, 116 test): AUC 1.0000 for this q36-fitted head; the study's
  llama.cpp-fitted head applied to Strata's states scores 0.9996 where it was fit at 0.9988 —
  the readout transfers across engines on the same body.
- Strata q36 sidecar: engine verdicts vs the same head in Python, worst |diff| 5.0e-7 over 378
  judge passes; one pass takes 325 ms mean on one 7900 XT.
- Policies on the held-out community set (see the study's reports): veto over majority +7 tasks
  (9 fixes / 2 breaks), gating reaches best-of-3 accuracy at about half the pool cost.
- Refit in the serving format is the rule: heads do not transfer across bodies or prompt formats.

## Use

Strata q36 engine sidecar:

```
STRATA_Q36_JEV_HEAD=decision-head/scion-35b-a3b-decision-head-q36.bin \
  strata-q36 --serve --model /path/to/Scion-35B-A3B-PQ2_0-corr.gguf
# an EMB request then answers:   EMB 2048 <file> <p_correct>
```

Strata server route (any body; `"decision_head": "<path>.json"` in the run config):

```
POST /v1/verdict   {"task": "...", "candidate": "..."}
-> {"p_correct": 0.997, "threshold": 0.9, "accept": true, "tokens": 271, "ms": 318.4}
```

Full record, arms tables and the port's numbers: the dataset's `PORT-REPORT.md`.
