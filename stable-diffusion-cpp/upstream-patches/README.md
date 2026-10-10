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

14. Opt-in graph-stage diagnostics: `sd_set_graph_diagnostics` enables bounded
    `graph-stage-v1` debug logs through the existing log callback. It is disabled
    by default. Each enable call opens a new window; each runner reports its
    first attempt after graph-cut plan resolution, including failure. Re-enable
    before another request or graph/resolution and disable after the request.
    Earlier failures retain their existing logs. Boundaries cover measurement,
    scheduler setup/reserve, weight preparation, workspace preparation and actual
    allocation, input copy, prefetch and execution. `reserve-including-sync`
    includes the scheduler's existing synchronization and size estimation; it
    does not isolate that synchronization. The consumer can timestamp callbacks;
    no native timers, added tensor walks, GPU synchronization or readbacks are
    introduced. Disabled calls only check the gate/null label, without formatting
    or callbacks. Generation behavior and arithmetic are unchanged. The existing
    callback smoke probe checks default-off, begin/end and failed-attempt gating,
    re-enabling and disabling. Its simulated attempt does not establish real
    runner timing or diagnose a trained-model stall. Remove when upstream offers
    equivalent explicitly enabled, bounded preparation/execution diagnostics.
15. Preserve external tensor address metadata in scheduler measurement clones.
    The former pointer value `1` was only a marker, but the pinned WebGPU
    `GET_ROWS` support check interprets it as a backend address. Relative to its
    `0x1000` base, the resulting odd offset cannot satisfy F16/F32 alignment;
    its backwards alignment loop can take an impractical number of iterations
    on memory64. Keep the source `data` and view offsets already copied with the
    tensor. The existing non-null, zero-size backend buffer is sufficient to
    classify an external weight, including one whose data is still null. No
    tensor data is read, allocated or converted by this correction, and backend
    selection policy is unchanged. The existing Qwen timestep test probe also
    checks real scheduler measurement with F16/F32 resident and unallocated
    weights, nonzero row views, unchanged `GET_ROWS` support, reservation sizes
    and source tensors. CPU checks exercise clone ownership; optional WebGPU
    checks exercise that backend's actual support predicate. This is not
    trained-model inference. Remove when upstream preserves valid address
    metadata when constructing measurement graphs.

16. Temporal pointwise convolution: when WebGPU cannot execute IM2COL_3D,
    a contiguous multi-frame 1x1x1 kernel with unit stride/dilation and no padding
    can be expressed as channel matrix multiplication plus contiguous layout
    changes. This covers H3's 24-channel post-quantization video VAE convolution
    without materializing a 3D im2col buffer. Every materialized node must pass the
    backend capability predicate; otherwise the existing fallback remains. The
    old single-frame/full-depth path, explicit direct-convolution choice and
    forced-precision branch are unchanged. Native tests exercise 40 synthetic
    type/batch/bias/fallback cases against an independent CPU numerical reference.
    These checks are not H3 inference or WebGPU performance/accuracy validation.
    The lowering becomes redundant if upstream supplies an equivalent supported
    temporal pointwise path or the backend supports the original operation.

18. Superseded by patch 0019 (retained in the ordered source history).
    A finite video queue completion budget: allow 180 seconds for submitted GPU
    work, while keeping buffer-map and event synchronization at 30 seconds.
    User H3 traces had completed steps around 22-28 seconds before the next
    30-second queue wait aborted; this does not establish a driver failure.
    The wait-status and callback-status checks remain unchanged. No repeated
    wait, queue resubmission, tensor change or precision fallback is added.
    Timeout messages include the actual selected budget. The extracted-source
    contract probe checks both call sites and synthetic completion/failure cases;
    it does not execute WebGPU or H3. All SD WebGPU builds receive this queue
    budget; the separately built llama runtime is unchanged. Replaced by an
    upstream configurable finite workload budget with equivalent failure checks.


19. Completion-based WebGPU waits: remove application wall-clock deadlines for
    queue completion, buffer mapping and backend events. Map/event waits can also
    depend on queued GPU work; keeping their 30-second deadlines would move the
    same workload-dependent failure to another synchronization point. UINT64_MAX
    selects the Emdawn completion-only path already used for adapter/device waits.
    This is a single wait on the original future, not polling or resubmission.
    Wait errors, unexpected wait statuses and failed/cancelled callbacks still
    abort. Backend events now retain the callback result per recording; a late
    callback cannot access a freed event or overwrite a later recording.
    The runtime announces completion-wait-policy-v1 at device initialization.
    Callers must retain explicit cancellation/Worker termination and device-loss
    handling. This patch does not disable browser/OS driver watchdogs or force
    an unresponsive GPU to recover. All SD WebGPU profiles receive the policy;
    the separately built llama runtime remains unchanged. The prepared-source
    probe covers long completions, errors and callback lifetimes; it does not
    execute GPU work. The patch is removable when the pinned upstream implements
    equivalent completion-based waits and failure/ownership checks.

20. Single-image causal convolution: when the input has one frame, no saved
    history and the temporal kernel is three with the standard causal zero pad,
    pack only the last temporal slice and use an existing direct 2D convolution.
    Check the selected backend's support before selecting the candidate. Preserve
    history/video, multiple batches, unsupported types, explicit 3D direct mode,
    temporal stride/dilation changes and both circular flags. Spatial padding
    continues through the existing helper, including its supported fallback.
    Keep a separate bias output; no weight precision or global copy policy changes.
    `image-causal-test` compares 62 synthetic CPU conditions, each with two inputs,
    against the legacy path. This is not trained-model/WebGPU speed validation.
    Remove when upstream has an equivalent guarded lowering passing these tests.

21. Anima conditioning: separate adapter/weighting/trimming from image-dependent
    computation. Preserve structural zero-padding mass with explicit optional
    F32 attention sinks, attached before capability selection; ordinary and flash
    paths retain the same mathematical denominator. Use RunnerCache only with
    bounded immutable condition IDs owned by one sampling call. Enabled extensions,
    arbitrary weight adapters and nondefault numeric scales do not get reuse.
    Cache hits materialize a graph-local copy so graph-cut metadata cannot rename
    or modify persistent input tensors. Allocation failure clears and disables
    caching for that sample, then retries uncached once; other failures are not
    retried. Independent model arguments can disable caching or compaction.
    The actual Anima layer and a small actual AnimaRunner are tested with synthetic
    weights, including ownership, metadata, condition isolation and injected cache
    allocation failure. No application package pin or generated asset is changed.
    Remove when upstream supplies equivalent sink semantics, explicit condition
    identity, memory accounting and lifecycle/fallback contracts with regressions.

22. Same-device WebGPU tensor copies: implement the buffer's synchronous copy
    callback and the backend's asynchronous copy callback using the existing
    device queue. Accept only equal-type/equal-layout ranges on the same device,
    different GPU buffers, four-byte-aligned offsets/sizes and checked bounds.
    Refuse before encoding otherwise, preserving GGML's generic fallback. Never
    round a partial F16 tail over a neighboring tensor. The synchronous callback
    waits for completion; the asynchronous callback does not. This is an SD-only
    prepared GGML change, not a change to the separately built llama runtime.
    Extracted-source API doubles check range, ownership and queue contracts; they
    are not a WebGPU compiler, device execution or speed measurement. Remove when
    upstream has equivalent callbacks passing these contracts and real GPU tests.

23. Cache capture batching: allocate all destinations before any asynchronous
    submission, retain all owners through completion, and publish only afterward.
    Contiguous same-WebGPU-device copies share one completion boundary per capture.
    Other devices and strided spans retain their previous copy paths. Standalone
    CachedTensor::copy stays synchronous; source tensor metadata is never mutated.
    This benefits enabled Qwen prefix/graph-cut caches as well as Anima conditioning,
    but does not enable an application-disabled cache. Allocation/submission failure
    tests verify no stranded destination lifetime. This does not merge submissions:
    eight tensors still make eight queue submissions, followed by one explicit wait.
    Remove when upstream capture has equivalent completion and ownership semantics.

24. Backend-admitted bounded materialization: when WebGPU rejects the full F16
    convolution expansion or manual attention score matrix, try at most 64 output
    row/query chunks. Admit the alternative only if every introduced compute node
    is supported. Convolution preserves the full input halo and baseline F16
    expansion/matmul; attention retains all keys, masks, sinks and F32 score
    accumulation. Chained SET writes own one final output without concatenation.
    CPU, small accepted graphs, explicit direct convolutions and unsupported
    candidates retain the original path. Flash Attention is not enabled silently.
    This is not a guarantee of peak memory or device residency: allocation,
    scheduler cuts and runtime limits still matter. Native tests replay the pinned
    WebGPU capability predicate and compare real CPU arithmetic; optional browser
    smoke exercises small bounded graphs on the requested real backend. Remove
    when upstream provides an equivalent backend-aware bounded implementation.

25. Opt-in compute placement: with the existing graph diagnostics enabled, emit
    browser-placement-ops-v1 alongside the unchanged browser-placement-v1 record.
    Exclude metadata-only nodes and count assigned CPU/WebGPU/other operations,
    CPU operation kinds, current CPU-node rejection by the preferred WebGPU backend,
    and the largest logical CPU operand. The check occurs after scheduler source
    rewriting; it is not a historical rejection cause, transferred bytes, a GPU
    execution trace or timing. No tensor names/data, new waits or reads are added.
    Disabled diagnostics add only the gate check. Remove when upstream provides an
    equivalent opt-in, name-free compute-placement summary with these semantics.


26. WebGPU parameter upload batching: the image backend stages per-kernel uniform
    bytes in the existing aligned slot arena and flushes once before each existing
    command submission. Submission, throttling, profile/compute-pass boundaries,
    parameter bindings and slot reuse are preserved. The 74-slot default mirror is
    18,944 bytes at 256-byte alignment; padding trades more small transfer bytes
    for fewer API calls. Prepared-source tests exercise the live arena, dispatch
    builder and graph loop with an ordered queue double, not GPU measurements.
    CMake explicitly enables the definition on the image-only WebGPU target.
    Remove when upstream provides equivalent bounded staging and queue ordering.

27. Empty cache capture: return without a queue completion callback when no new
    runner/graph-cut outputs need saving. Segment execution and cleanup retain
    their own waits; non-empty and partially submitted captures preserve their
    completion/ownership rules. Tests include outputs in other segments, already
    pending entries, no future cut consumer, normal saves and failures. Remove
    when upstream avoids equivalent empty waits with these lifetime guarantees.
