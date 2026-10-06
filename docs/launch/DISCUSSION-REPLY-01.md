# Discussion reply 01: "invalid ggml type 142"

Reply to the first tester report (discussions/1). Code block so the markdown
survives copying: upstream `master` and the fork's old default branch both stop
at `GGML_TYPE_COUNT = 43`; `moe-corr-runtime` defines `GGML_TYPE_PQ2_0 = 142`
and `GGML_TYPE_COUNT = 144`.

````markdown
Thanks, and good catch on the error text.

Nothing to do with your OS or hardware. `142` is `GGML_TYPE_PQ2_0`, the container the expert banks use, and "should be in [0, 43)" is the giveaway: your binary reports 43 known types, which is exactly upstream `master`. The branch that reads this file defines `GGML_TYPE_PQ2_0 = 142` and `GGML_TYPE_COUNT = 144`, and it imports the legacy type 43 (the embedded corrections) as `PQ2_0` automatically, so nothing else is missing.

Cleanest path: throw the checkout away and start over. Re-running cmake in place tends to keep the old configuration, and an older `llama-cli` earlier on your `PATH` gives the same error, so a fresh tree is faster than debugging it.

```bash
rm -rf prism-ml-llama.cpp
git clone https://github.com/sky-is-green/prism-ml-llama.cpp
cd prism-ml-llama.cpp
./verify-container-support.sh      # prints RESULT: OK
cmake -B build -DGGML_CUDA=ON && cmake --build build -j --target llama-cli llama-server
./build/bin/llama-cli -m Scion-35B-A3B-PQ2_0-corr.gguf -ngl 99 -c 4096 -t <physical cores>
```

I have just made `moe-corr-runtime` the fork's default branch, so a plain clone is the right tree now, and the branch README carries the same explanation. If you would rather keep what you have, `git fetch origin && git checkout moe-corr-runtime && rm -rf build` also works; just start from a clean `build/` either way.

When it loads, please send `git log -1 --format=%h`, your GPU and VRAM, and pp512/tg128 t/s. CUDA is the one path I could not test myself, so a report from an NVIDIA box is exactly what I want. If anything else fails, paste the full log.
````

## Troubleshooting

Three checks that localise the problem in one round:

```bash
git -C <checkout> rev-parse --abbrev-ref HEAD; git -C <checkout> log -1 --format=%h
grep -n "GGML_TYPE_PQ2_0" <checkout>/ggml/include/ggml.h
which -a llama-cli; <checkout>/build/bin/llama-cli --version
```

| symptom | cause | fix |
| --- | --- | --- |
| no `GGML_TYPE_PQ2_0` line in `ggml.h` | built `master` or stock llama.cpp | clone the fork, default branch is the right one |
| line present, error persists | a different binary is running | use the absolute path to the freshly built `llama-cli` |
| loads, then wrong numbers | built with the wrong quantizer rule | rebuild with `GGML_PQ2_0_LLOYD=1` and compare against the card's PPL 8.354 |
