# Platform scope and reporting

Which runtime paths the release is verified on, and what a report needs to
carry.

## Scope

| path | status |
|---|---|
| CPU (all platforms) | verified end to end |
| Linux x64 + AMD ROCm (gfx1100/gfx11) | verified end to end; all release measurements use this path |
| Linux x64 + NVIDIA CUDA | prebuilt binaries published; runtime-unverified |
| Windows x64 (CPU, CUDA) | prebuilt binaries published; runtime-unverified |
| macOS Metal | kernels present; untested |
| Vulkan | unsupported: no PQ2_0 kernels |

"Verified" means the release files were run on that path.

## Reporting an issue

Include:

1. exact command line;
2. full console output (for the MTP sidecar, the decisive line is
   `failed to load sidecar '<path>': <reason>`);
3. release tag and artifact (backend, CUDA version, CPU vs GPU);
4. GPU model and driver version;
5. model and drafter file SHA256;
6. for speed claims, `llama-bench` numbers for both sides of the comparison.

For a published but unverified path, a fix needs a reproduction on that
platform to be validated.

## Known packaging note (AMD)

The prebuilt `linux-x64-hip.tar.gz` from `v0.4.1` links ROCm 6.x sonames
(`libhipblas.so.2`, `librocblas.so.4`, `libamdhip64.so.6`) and does not load
models on ROCm 7.x. Use the source build (`-DGGML_HIP=ON`) or rebuild the
artifact against the ROCm you run.
