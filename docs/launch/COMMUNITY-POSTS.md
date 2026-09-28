# Community posts: Prism ML and empero-ai

Two short notes for the creators' Community tabs, posted around the Reddit
launch. Both credit the upstreams first, because the release sits on their work.

The post bodies live in their own files so they can be copied as raw markdown
(code blocks, no `>` prefixes, nothing to reformat by hand):

| where | file | tab |
|---|---|---|
| Prism ML | `POST-prism-ml.md` | `prism-ml/Ternary-Bonsai-2-27B-gguf` → Community |
| empero-ai | `POST-empero-ai.md` | `empero-ai/Qwen3.8-35B-A3B-Distill` → Community |

**Rules of the room:** one post per repo; the owner can lock, hide or delete
anything; no marketing tone; never write "Created using Bonsai by Prism ML"
(that string is for models derived from Bonsai weights, and this is not). If a
maintainer engages, reply quickly; the embedded-adapter branch is ready if they
want a PR.

## Posting notes

- **Timing:** post both notes on the day the Reddit thread goes live, or the
  evening before. They are short reads, not announcements meant to carry the
  launch.
- **Tone:** credit first, numbers second, no superlatives that are not in the
  card, and name the KLD gap in both.
- **Handling replies:** if Prism ML asks about the embedded-adapter diff, the
  branch is `sky-is-green/prism-ml-llama.cpp` → `moe-corr-runtime`, and I can
  open a PR. If empero-ai wants the recipe, point at
  `scion/docs/MOE-EXTENSION.md` and `moe/box-run.sh`.
- **Optional extras:** a one-liner in Prism ML's Discord (linked from their
  model card), or a post on HF Posts (`huggingface.co/posts`) if a broader
  write-up is ever wanted. Neither is required.
- **Do not** post the identical text as the Reddit thread; the Reddit body is
  link-free on purpose.

## Current state of each post

- **Prism ML:** final, links wired for every reference (#62, base model, Prism
  ML's engine, TAARDIS fork, empero-ai, Qwen, MrFuzzihead's conversion, the
  runtime fork, card, harness repo, forensics repo). Framed as an update to
  #62 rather than as a message to Prism.
- **empero-ai:** wording settled, links still bare URLs at the end. Wants the
  same inline-link treatment as the Prism note, and a register check: the
  opener is warmer ("Thanks for the distill") than the Prism one on purpose.
