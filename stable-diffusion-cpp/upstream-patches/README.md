# Browser/WebGPU enablement patches

The user explicitly authorized the implementing agent to choose upstream patches
for this image-runtime experiment on 2026-09-24. The exception is local to
`stable-diffusion-cpp`; it does not relax llama-cpp's approval policy.

`series.json` is the ordered, exact-input/exact-output inventory. Apply only to
private build-tree copies, leaving both upstream checkouts clean. The independent
llama.cpp checkout is used only as a source for its `ggml/` subtree.

1. Model reader: avoid native thread spawning in single-threaded Emscripten.
2. Group normalization: decompose contiguous channel-divisible F32 shapes into
   existing normalization/reshape operations; retain other upstream paths.
3. WebGPU memory: report unknown device capacity, not a single-buffer ceiling.
   The **application** supplies a managed-memory budget. Allocation can still
   fail; this is not a measurement of free GPU memory or driver overhead.
4. Unsplit GGUF: metadata-only v2/v3 parsing and 64-bit file positions independent
   of Wasm address width. Avoid whole-model `size_t` accumulation, validate file
   ranges/type sizes/counts and keep per-tensor memory limits. Preserve supported
   higher-rank flattening with checked arithmetic. Synchronous random-access
   sources and the decision to disable mmap/prefetch belong to the caller.
5. GGUF stream cursor: keep the checked logical metadata position in 64 bits
   instead of calling `tellg()` before every field. Small metadata skips consume
   the existing stream buffer; large arrays still use a bounded 64-bit seek.
   This avoids libc++ buffer invalidation without loosening bounds or loading
   tensor payloads. Remove when upstream provides an equivalent buffered reader.

6. Repository-native file I/O: preserve 64-bit safetensors sizes/positions;
   bound and validate header/index JSON, tensor ranges, and local sibling paths.
   Resolve and validate complete standard GGUF shard groups without concatenating
   or rewriting the original files. Duplicate/missing tensors and inconsistent
   shard metadata fail closed. This patch does not add application model recipes.

7. Qwen timestep activation: use out-of-place SiLU under `SD_BROWSER_WEBGPU`.
   Qwen Image 2.1's BF16 first linear may run on CPU while the runner pins its
   supported SiLU to WebGPU. An in-place output retains a CPU `view_src` despite
   the scheduler copying `src[0]` to WebGPU; binding that output as a WebGPU buffer
   traps before shader execution. Keep the original activation formula, weight
   types and backend selection; only the result allocation changes. Other builds
   retain the upstream in-place path. See `tests/qwen-timestep-probe.cpp` for the
   actual Qwen block's alias/placement/numerical regression. Remove this patch
   only after a reviewed upstream equivalent makes the mixed-BF16 placement and
   numerical regression pass. If the upstream scheduler safely relocates views,
   adapt the probe's pre-allocation no-view assertion as part of that review.

8. Allocation placement diagnostics: snapshot at most 256 BF16 weight-matmul
   operands before scheduler allocation replaces their sources, then report
   assigned CPU/WebGPU/other node counts through the existing log callback.
   Seen versus inspected counts expose the cap. Bytes count operand uses, which
   may repeat a weight; WebGPU buffer/CPU assignment is a scheduled boundary,
   not actual readback, completed execution or physical memory residency.
   The hook performs two metadata-only node scans per successful allocation,
   at most 256 support checks and bounded stack storage, even when the caller
   discards debug logs. It adds no synchronization, evaluation callback, tensor
   readback, graph rewrite or public API. The synthetic timestep probe now uses
   the production workspace and its CPU/optional WebGPU smoke checks validate
   original weight placement as well as arithmetic. Remove when an upstream
   passive allocation summary provides equivalent coverage without names/data.

9. BF16 parameter storage: under `SD_BROWSER_WEBGPU`, register original BF16
   parameters for WebGPU compute as F32 before allocating buffers. The per-context
   `webgpu_bf16_type` field accepts F32 (default, exact widening) or F16 (explicit
   lossy conversion, including possible overflow/underflow). CPU compute and other
   parameter types, including quantized weights, retain their existing types.
   Source metadata and files stay unchanged; the existing loader converts into
   the destination storage using per-tensor temporary buffers. Converted weights
   remain in the selected residency backend until released or evicted.
   F32 doubles the converted parameters' raw bytes; F16 retains their byte count.
   Strides, registered bytes and allocation budgets use the destination type.
   Auto-fit includes widening conservatively before deciding placement, so CPU
   components can be overestimated. A fixed log reports unique converted bytes,
   not graph operand uses, alignment overhead or peak memory. The synthetic
   `bf16-weights-probe.cpp` covers both file formats, preserved quantization,
   values, accounting and optional WebGPU placement; it does not certify model
   quality, speed or bitwise parity with CPU arithmetic. Remove when upstream
   provides equivalent backend-aware BF16 loading with caller-selected precision.

The smoke fixture reports byte ranges and total bytes rather than guessing from
read-call counts. Its 4 KiB chunk / 256 KiB total budget applies only to the tiny
synthetic fixture; it is not a metadata or model size limit in the core. Runtime
input size and source chunking remain caller-owned.

Real GPU numerical parity, large dispatches, all architectures and quantizations
are not certified. There is no application resolution cap, sampler selection,
GPU budget or Qwen-specific sampling policy in these patches or the bridge.

Removal criteria: replace each patch with an upstream equivalent when its same
threading/normalization/memory/large-file tests pass. Never silently skip a failed
patch, relax input hashes, or substitute a different upstream revision.
