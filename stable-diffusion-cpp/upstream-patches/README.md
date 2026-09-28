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

10. Graph traversal: replace the external GGML parent's recursive postorder walk
    with an explicit heap-backed stack. Deep dependency chains no longer consume
    the browser's native/Wasm call stack. Preserve source ordering, leaf/parameter
    classification, automatic names, shared operand use counts and compute-flag
    propagation through already-visited nodes. Allocation growth is checked and
    the temporary stack is freed on completion or growth failure. Traversal depth
    is bounded by the existing visited capacity; malformed cycles cannot grow
    the heap indefinitely. Invalid graphs retain fatal errors. No stack-size
    increase, model-specific branch or public ABI change. The test-only graph
    probe covers both traversal orders,
    a shared DAG, repeated expansion, and a 32,768-node unselected chain followed
    by compute propagation. Native CI and browser smoke run the same probe; it
    does not evaluate weights or certify image quality. Remove when the pinned
    external GGML has an equivalent non-recursive traversal passing these cases.

11. 3D convolution bias: under `SD_BROWSER_WEBGPU`, use an out-of-place addition
    for the shared 3D convolution helper. Its direct convolution can fall back to
    CPU while the runner assigns the supported bias addition to WebGPU. The
    in-place output would retain the CPU convolution's `view_src`; copying the
    addition's input does not relocate that output. Keep convolution selection,
    arithmetic and parameter types unchanged. Non-browser builds retain the
    original in-place path. Each bias output is an additional F32 tensor of the
    convolution's output shape; allocator reuse determines the peak overhead.
    `tests/conv3d-bias-probe.cpp` checks direct, automatic and forced-F32 paths,
    with and without bias, against an independent small convolution reference.
    Native/test-variant CPU checks detect the alias before allocation. Optional
    WebGPU smoke additionally requires CPU convolution/im2col and WebGPU bias
    placement, compatible buffers and completed numerical evaluation. This is
    not trained-model VAE validation or proof of any particular browser crash's
    cause. Remove after a reviewed upstream equivalent passes the same mixed
    placement and arithmetic checks; revise the no-view assertion if an upstream
    scheduler safely relocates views instead.

12. Single-depth 3D convolution: when the automatic browser path cannot use
    `IM2COL_3D`, fold depth into channels and use supported direct `CONV_2D` for
    one sequence and one full-depth window. Require contiguous input/weights,
    equal input/kernel depth, temporal stride/dilation 1 and padding 0, F32
    input, F16/F32 weights, and backend support including buffer binding limits.
    Reshape the output back to the original 3D layout and retain patch 11's
    separate bias output. Unsupported cases, explicit direct requests and
    forced-F32 requests retain their previous paths. No new weight conversion,
    model-name branch, full im2col allocation or public API is introduced.
    Views add only tensor metadata; the direct output replaces the 3D output.
    CPU direct convolution rounds input patches to the weight type, whereas
    WebGPU direct convolution can retain F32 input, so parity is not bitwise.
    The existing convolution probe checks F16/F32 arithmetic, bias/no-bias and
    the excluded temporal, batch, backend and explicit-mode cases. CPU checks
    simulate unsupported IM2COL only during graph selection, then evaluate on
    the real CPU. Optional WebGPU checks require the real 2D operation on GPU.
    Neither CPU simulation nor small GPU graphs establish trained-model speed
    or memory use. Remove when the pinned upstream provides an equivalent
    type-preserving selection passing the same shape and fallback checks.

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

13. Passive context snapshots: expose versioned runtime and memory records plus
    borrowed context configuration. Getters use the existing execution mutex with
    a nonblocking lock and reject reentry; normal generation gains no new locks,
    timers, hooks, metadata walks, synchronization or readbacks. The context
    initializer also takes ownership of the previously omitted audio-encoder path
    with one string copy, preventing a borrowed snapshot from returning a dangling
    caller pointer. Memory collection
    walks existing manager bookkeeping only when requested. Logical tensor bytes,
    unique manager buffer-handle sizes (host/non-host), and runners' last published
    retained runtime bytes remain separate; none is physical/free/peak GPU memory
    or a complete process allocation total. Aggregation uses saturating uint64_t
    on both Wasm address widths. Configuration pointers are borrowed until the
    next mutation or context destruction; resolved thread/runner flags are in the
    runtime record instead. Existing BF16 loader probes check logical/allocated
    distinction, buffer deduplication/release and reports above 4 GiB; every
    browser smoke variant checks the public records and null-failure atomicity.
    These tests do not validate loaded-model snapshots, performance or real GPU
    memory accounting. Remove when upstream exposes equivalent documented idle
    snapshots with the same failure/ownership and passive-observation guarantees.
