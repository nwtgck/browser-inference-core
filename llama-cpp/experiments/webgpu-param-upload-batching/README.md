# Batched WebGPU parameter uploads

This source-bound experiment coalesces per-kernel uniform writes at existing graph command submission boundaries. Runtime integration is available through a separate OFF-by-default build option. See [integration](../../docs/webgpu-param-upload-batching.md); these files retain the standalone source checks.

## Scope

Repository patch base: browser-inference-core `4e04bb9b833b97d9367569fcdd225156039dfb9b`.

The upstream patch targets `ggml/src/ggml-webgpu/ggml-webgpu.cpp` at llama.cpp `d81235049384534c167caea52b85a694f6103d14`. Exact source and optional header identities are in `inputs.json`. It does not change shaders, model weights, tensor data uploads, public APIs, graph construction, or checkpoint formats.

The current backend allocates one aligned uniform slot per kernel and uploads every parameter vector separately. It already submits graph commands in batches with a 64-kernel threshold. This experiment adds a bounded CPU mirror of the existing parameter arena and copies each vector into its assigned slot. One `WriteBuffer` uploads the populated mirror immediately before each existing batch submission, including the final partial batch.

`GGML_WEBGPU_BATCH_PARAM_UPLOADS` must be defined explicitly when compiling the patched translation unit. With the macro undefined, the upstream source behavior is preserved. The optional bicore integration sets this guard only for an explicitly enabled build.

## Ordering, lifetime, and costs

- Keep the existing sequence: upload batch A, submit A, reset slots, upload B, submit B. Same-queue order protects A's readers before B overwrites reused slots. Do not move a later upload before the preceding submit.
- Copy the CPU parameter vectors while they are alive. `WriteBuffer` consumes the upload bytes before returning; replacing this with delayed references would invalidate CPU mirror reuse.
- Keep the 64-kernel threshold, multi-kernel operation overshoot, final partial submission, waits, readbacks, and reset sites unchanged. This does not permit concurrent use of a context or add thread safety.
- Zero the complete slot stride before copying parameters. Check parameter length, slot alignment, allocated slot ownership, and backing-storage bounds.
- The extra CPU vector is `74 * slot_stride` bytes per backend context. The stride is 128 bytes rounded up to the device's uniform-offset alignment. At 256-byte alignment it is 18,944 bytes. No GPU buffer is added.
- Upload bytes increase because padding is transferred. A 64-slot batch at stride 256 uploads 16,384 bytes; the original writes transfer only the actual parameter vector lengths, each at most 128 bytes.

Reducing API call count is not a measured speedup or a reduction in resident GPU/browser memory. Evaluate the extra copies and padding, browser staging behavior, and instrumentation overhead on the target device.

## Reproducible checks

Run from the bicore repository root with a pristine checkout of the pinned upstream:

```sh
python3 llama-cpp/experiments/webgpu-param-upload-batching/test_candidate.py /path/to/upstream-d812
```

The runner checks the input hash and applies the patch to a temporary source copy. It verifies disabled-mode equivalence and both graph flush locations. It extracts the actual parameter arena and graph-compute function into instrumented CPU tests. No upstream checkout is changed and no dependencies are downloaded.

Coverage includes zero-work and no-op graphs, full and final partial batches, multiple kernels per operation, graph/batch reuse, both compute-pass settings, alignments 128/256/512, padding, parameter-vector lifetime, delayed queue consumers, bounds rejections, and preservation of the set-rows check. Queue/API behavior and kernel encoding are mocked; the arena and graph loop are real source. The tested graph suite uses identical kernel and submission counts in both modes.

Add `--sanitize` for AddressSanitizer and UndefinedBehaviorSanitizer. If LeakSanitizer is unavailable under a debugger or sandbox, use `ASAN_OPTIONS=detect_leaks=0`; that disables leak checking only. `--prepare-only` checks source identity/application/equivalence without compiling. `--compiler` chooses the host C++ compiler.

### Real-header syntax checks

The optional check compiles the actual arena in both modes against real pinned ggml and Dawn headers. It reuses the Dawn package used by the tensor-copy experiment: `v20260908.214631`, revision `94c3c9cc0d5fb2e85aebb370fa8d37b71aa34655`.

Obtain the [pinned package](https://github.com/google/dawn/releases/download/v20260908.214631/emdawnwebgpu_pkg-v20260908.214631.zip), verify archive SHA-256 `9c36eb46ada070b9cc0de2bdfd04c1fa1a1fd852cc1ebb371e96e68e1efe6c3c`, and extract it. The runner checks required individual header hashes even with Python optimization enabled.

```sh
python3 llama-cpp/experiments/webgpu-param-upload-batching/test_candidate.py /path/to/upstream-d812 --dawn-package /path/to/emdawnwebgpu_pkg
```

This is host syntax checking with browser/wasm64 branches selected, not a complete translation-unit build, Emscripten build, ABI check, linking, or browser execution. A host GCC compiler cannot necessarily parse the pinned Emscripten Clang annotations in a full translation-unit check.

## Integration gates and maintenance

Before adopting this experiment for production:

1. Compile the complete patched target using its actual pinned Emscripten/Dawn toolchain.
2. Validate GPU outputs, slot reuse, submission order, partial batches, and supported profiling configurations on target browsers/devices.
3. Measure wall-clock performance and allocation behavior with and without diagnostic instrumentation; account for larger upload byte volume.
4. Recheck every arena consumer, reset, and graph submission site when updating upstream. Successful patch application alone does not establish semantic compatibility.
5. Keep experimental build integration OFF by default until the required validation is complete. Preserve source provenance and keep independent upstream patches separate.

The no-patch alternative is to retain upstream uploads or wait for upstream coalescing support. Remove this experiment when upstream provides suitable batching or it no longer offers a justified benefit. It does not change runtime defaults on its own.
