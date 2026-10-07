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
   `speculative decoding enabled: draft-mtp-sidecar`; a `loading draft model`
   line for the drafter file means `--spec-type` selected the wrong path);
3. release tag and artifact (backend, CUDA version, CPU vs GPU);
4. GPU model and driver version;
5. model and drafter file SHA256;
6. for speed claims, `llama-bench` numbers for both sides of the comparison.

For a published but unverified path, a fix needs a reproduction on that
platform to be validated.

## Triage: "unknown model architecture: 'mtp'"

The follow-up log from the 2026-10-06 Windows CUDA report shows the drafter was
requested as a regular draft model:

```
common_speculative_init_result: loading draft model '<...mtp-drafter.gguf>'
error loading model: unknown model architecture: 'mtp'
```

The sidecar path logs `speculative decoding enabled: draft-mtp-sidecar`
instead. This is a type selection issue, not CUDA: `--spec-type draft-mtp`
asks for the MTP head inside a full model, while the Scion drafter is the
target-context `draft-mtp-sidecar`, auto-detected from the GGUF when no
`--spec-type` is given. Both Windows zips contain the sidecar code.

Fork fixes on `fix/scion-mtp-sidecar`:

- an explicit draft-model `--spec-type` is replaced with `draft-mtp-sidecar`
  (with a warning) when the `-md` file is a Scion sidecar;
- a GGUF that claims architecture `mtp` but has no head tensors no longer
  aborts the type detector;
- the model loader prints an actionable error for the `mtp` architecture.


## Known packaging note (AMD)

The prebuilt `linux-x64-hip.tar.gz` from `v0.4.1` links ROCm 6.x sonames
(`libhipblas.so.2`, `librocblas.so.4`, `libamdhip64.so.6`) and does not load
models on ROCm 7.x. Use the source build (`-DGGML_HIP=ON`) or rebuild the
artifact against the ROCm you run.
