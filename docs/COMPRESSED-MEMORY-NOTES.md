# Compressed memory (CRAM) and fast byte search (memchr) — assessed, 2026-10-14

Pointers from the operator: `crates.io/crates/memchr`, and Meta's **CRAM** (talk:
*A Compressed RAM Service*, Gregory Price, LPC 2026,
`lpc.events/event/20/contributions/2424/`; coverage: Phoronix, Tom's Hardware).
Question: do they help the local-serving / shrink lanes? Short answer: neither is a
lever today; the measurements below are why, kept so the check is not repeated.

## CRAM — hardware compression, and the data decides it

What it is: kernel support for **hardware-offloaded** compressed memory exposed as a
normal memory tier — cacheline/byte access, page-table mapped, page-cache present, *no
software decompression step* (unlike zswap/zram). Near-DRAM reads on read-only
workloads (452x ZRAM in Tom's read-only microbenchmark; 5.4x in the worst tested write
case, 20% writes). A tested prototype (`mm/cram.c`); not upstream; needs
compression-capable memory hardware — this box has none.

The decisive part is the data. Measured here (zstd -3, 128 MiB samples):

| artifact | ratio |
| --- | ---: |
| `Scion-FlashNext-176B-A6B-Q2_0.gguf` | 1.032 |
| `Scion-FlashNext-176B-A6B-Q6K.gguf` | 1.062 |
| `Scion-FlashNext-176B-A6B-q4x160.gguf` (table region) | 1.043 |
| `pack-q6k/dense.bin` (2.1 GB body) | 1.406 |
| `scion-embd-bf16.gguf` | 1.277 |
| anonymous pages already in zram (live) | ~2.8x (3.7 GB -> 1.3 GB) |

The shipped weights are incompressible (1.03-1.06x): **no compressed RAM tier can
expand the model's effective footprint**. The compressible parts are small (2.1 GB
body at 1.41x, 1.2 GB bf16 bridge at 1.28x). The box already runs the *software* form
of the idea for non-model memory (zram+zstd, 31 GB, `vm.swappiness=150`); the engines'
pools stay resident under it (live check: engine Rss 17.3 GB, Swap 0 kB; the swapped
pages are desktop apps — Steam/VLC/electron ~300 MB each).

Where the concept could matter later: (a) compression-capable memory hardware, if it
ever ships — but our page-cache data is the same incompressible weights, so the
benefit would be for the *non-model* pages, which zram already covers; (b) the
bf16/fp32 experiment regimes (RCO DB builds, 35B/176B calibration) where models are
70-350 GB against 30 GB of RAM — a compressed tier there would change what runs
locally, but ~1.3x on bf16 is not enough; pods remain the answer.

## memchr — no hot byte search in this stack

`memchr` is SIMD-accelerated 1-3-byte and substring search (Rust; the foundation of
aho-corasick/regex/ripgrep). Checked our paths: the engine's suffix drafter matches
**token-id n-grams through a hash table** (`src/spec/suffix_drafter.cpp`), not bytes;
the tokenizer is HF's Rust stack in the Python server; the engine's `.find` uses are
map lookups and log-line parsing (`conversation_memory.cpp`, `expert_source.cpp`);
the corpus/table builds are Python. Nothing hot scans bytes, so the crate itself does
not plug into the engine, the server or the tooling. The transferable class — SIMD
multi-pattern scanning plus aho-corasick/FST dictionaries — is the right tool if a
corpus/table scan or a grammar/stop-string matcher ever becomes a bottleneck; none
does today.

## Verdict

- CRAM: hardware watch item; the attached number is *our weights are incompressible*,
  so the useful place for compressed memory on this box is exactly where zram already
  is (non-model pages).
- memchr: technique pointer for future Rust tooling (corpus/table building), not a
  current lever.
