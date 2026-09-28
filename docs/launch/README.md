# Launch copy

Public text for the release, versioned here so it does not live as loose
Desktop files. Every post body is a fenced code block: copy out of the block and
paste, and the markdown survives.

| file | what | where it goes |
| --- | --- | --- |
| `REDDIT-BODY.md` | post body, first comment, optional detail comment | r/LocalLLaMA, image attached natively |
| `POST-prism-ml.md` | community note | `prism-ml/Ternary-Bonsai-2-27B-gguf` → Community |
| `POST-empero-ai.md` | community note | `empero-ai/Qwen3.8-35B-A3B-Distill` → Community |
| `DISCUSSION-REPLY-01.md` | reply to the first tester report | the model's Discussions tab |
| `COMMUNITY-POSTS.md` | tab map, room rules, posting notes, per-post status | internal, not posted |

## Rules baked into the copy

- **Reddit: no links at all**, in the body or the comments. Subreddit filters and
  new-account limits can remove a submission that carries outbound links, so
  every name is spelled out (`SkyIsNotGreen/Scion-35B-A3B`,
  `sky-is-green/scion`, `sky-is-green/prism-ml-llama.cpp`) and the figure is
  attached natively. Re-grep for `http` after any edit.
- **Reddit does not render pipe tables**, so the numbers are a fenced code block.
- **Community notes may link**, and every reference is a real link rather than a
  bare name.
- **Credit first, numbers second**, and the KLD gap is named in every post.
- **Never** write "Created using Bonsai by Prism ML"; that string is for models
  derived from Bonsai weights and this is not one.

## The runtime trap, in one paragraph

The release needs the fork at `sky-is-green/prism-ml-llama.cpp`. That fork's
`master` branch is an untouched upstream mirror whose type table stops at 43, so
a binary built from it rejects the file with `invalid ggml type 142. should be
in [0, 43)`. The `moe-corr-runtime` branch is now the fork's **default** branch,
it defines `GGML_TYPE_PQ2_0 = 142` and `GGML_TYPE_COUNT = 144`, and it ships
`verify-container-support.sh` to check that in a second. Anyone who hits the
error should delete the checkout and the `build/` directory and start from a
fresh clone, because a stale CMake cache or an older `llama-cli` on `PATH`
produces the identical message.
