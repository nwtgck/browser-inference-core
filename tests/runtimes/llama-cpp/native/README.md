# Current native contracts

`native-validation.test.ts` prepares a complete source tree through the production
TypeScript recipe and configures this directory in Release mode. The ordinary
and diagnostic providers include the complete prepared `ggml-webgpu.cpp` and its
complete factory headers. Current acceptance does not depend on historical
composer stages, fixed source digests, extracted functions or AST-selected
branches. Checks remain active with `NDEBUG`.

The WebGPU fixture uses the pinned SDK C++ bindings and records the external C
API boundary in `support/webgpu-recorder.hpp`. Real GGML type/layout functions
come from `ggml-base`. An API function without an implementation fails to link;
unexpected arguments at implemented boundaries fail the fixture. This checks
host decisions, cache ownership, encoding and upload bytes, not GPU execution or
the numerical result of a shader.

| Fixture | Meaningful assertions | Explicit boundary |
| --- | --- | --- |
| `provider.cpp` | Q8/Q4/Q6 selection and fallback, factory/cache identity, emitted WGSL, actual matmul grid, binding cache keys and lifetime, 164 warm allocation/pass/order/upload/borrowed-input cases, profile timestamp pairs and 14 query-budget shapes in both pass modes | Recorded WebGPU C API; metadata-only storage bindings; query cases replace only node encoding through the production typed callback |
| `readback.cpp` | Actual public ggml/provider getters, 272 alignment cases per platform arm, blocking output lifetime, same-queue wait elimination, fallback waits, map/read errors, checkpoint batching and staging cap | Deferred C API queue; native pointer and Emscripten direct-copy branches compiled as native executables |
| `transfer.cpp` | Registry capability caching, real scheduler input snapshots and fallback waits, copied-input lifetime, prefetch before host storage reuse, cancellation and reuse after caller synchronization, event-prefix completion; Scale/SiLU and RMS/Scale fusion admission and warm reuse | Complete `ggml-backend.cpp` and provider, public scheduler/backend APIs, real GGML graph/gallocr; controlled external host backend and recorded WebGPU C API |
| `cases/expert.hpp` | Q5_1/IQ4_NL short and fallback factories, distinct cache entries and warm revisit | Four current shapes emit `{type}-{128,256}.wgsl` and `cases.json` to `webgpu-provider-shaders` for an optional external GPU experiment; the WG256 output is the current fallback, not a historical provider |
| `hc-admission.cpp` | Actual HC header/type admission, layout/shape/mode/device-limit rejection and shared-root conservatism | Normal-only target shares the prepared tree; checks remain active in Release without a second preparation job |
| `cases/q8-diagnostic.hpp` | Execute-before-prepare rejection, warm prepare/execute without pipeline creation, actual diagnostic reentry BUSY, undersized report without writes, copied bridge request and CPU/foreign registry rejection; SHA-256 known vectors and padding/block boundaries | Profile-only complete provider plus the ordinary production diagnostic bridge TU; controlled typed registry and recorded WebGPU C API, native pointer bounds rather than Wasm heap execution |
| `checkpoint-writer.cpp` | Host offsets, batch/fallback counts, repeated flush and exception lifetime | Complete wasm-selected `llama-context.cpp`; actual ggml types with controlled backend callbacks |
| `cases/copy.hpp` | CPU producer synchronization before caller-byte snapshot, storage lifetime after return, checked no-side-effect rejection, same-device copy and alias, generic fallback between devices | Complete provider and public ggml calls; deferred C API queue. `transfer.cpp` exercises the real scheduler integration |
| `cases/subgroup.hpp` | Current Q4/Q6/Q8 fixed/requested factories, optional validation rejection memo, shape-dependent canonical fallback, Q6 admission/broadcast/cache and actual grid, C8 grid7/8 | Shared C API observer; emitted normal/profile shader comparison is host/source evidence |
| `cases/matrix.hpp` | Current half-pairs/wide-M/canonical mode, tile/cache separation in both shape orders, Q6 parallelism threshold and overflow-safe independent oracle, real provider dtype admission and actual matmul grid | Complete factory/provider; invalid layout inputs exercise factory admission only. Shader arithmetic and historical output bytes remain separate guarantees |
| `cases/cache.hpp` | Real GGML 130-node graph through public backend compute, warm coefficient uploads and group reuse, changed binding keys, short/empty tail retirement, bounded slots, key-only SDK references, free epoch before buffer destruction, queued old-group ownership, diagnostic reentry rejection | Actual graph loop and node encoder; Scale shader execution remains a C API observation |
| `mtmd-ownership.cpp` | Real constructor partial failure, alias backend cleanup, real vector allocation faults, scheduler retired before backends | Typed external backend/scheduler API doubles |
| `mtmd-upload.cpp` | Upload backend initialization failures, device identity, drain before backend and weight retirement | Typed public ggml API doubles using actual backend declarations |
| `mtmd-borrowed.cpp` | Shared real CPU owner survives multiple borrowed contexts, graph failure, destruction and reuse; actual formatter bytes | Throwing graph factory |
| `mtmd-entry.cpp` | Real entry/context/loader reject unknown raw i32 before file loading, preserve old default, isolate nested policies, recover after load failure | Real small synthetic GGUF; unused text-model and graph entry points fail |
| `mtmd-loader.cpp` | Real loader preserves audio legacy storage, retains CPU vision policy, rejects requested unavailable GPU retention before weight loading | Real small synthetic GGUF and CPU buffer allocation |
| `mtmd-bf16.cpp`, `mtmd-retained.cpp` | Bit-pattern/chunk upload, backend identity, retained roles and graph planning | Deterministic CPU buffers and explicit backend fixtures |
| `mtmd-graphs.cpp` | Model-specific graph shapes and retained-type boundaries, including Gemma4UV dimensions | Metadata-only constructor doubles using real class declarations; complete graph/model source files |

The real upstream mtmd CMake target is configured separately for CPU and WebGPU
preparations. Its source list must remain inside the selected coherent tree,
retain the expected core translation units without duplicates, and use the same
public header list before and after runtime include wiring.

Linker section collection discards unexercised functions after the complete
translation units have compiled. Successful execution establishes the assertions
above; it does not establish linkage or GPU behavior for discarded functions.
Select retained tests by their useful feature, API, performance or ownership
guarantee and maintenance cost. Obsolete composer, source-text and historical
implementation checks do not require replacement machinery.

Queue completion and buffer mapping capture the submitted prefix when the API
request is made. Waiting for that prefix must leave later work pending. The
scheduler cases observe copied input bytes before a recorded kernel, so they can
detect reuse before snapshots or missing producer barriers without emulating a
shader. The host backend deliberately fails on an unexpected graph operation;
its explicit cancellation case requires caller synchronization before reuse.
The profiling path may collect and wait at graph completion, which is tested as
a separate intended ordering. Fusion cases use actual graph flags, consumer
counts and allocator buffers; partial overlap must reject without encoding or
consuming nodes. None of these cases establishes shader arithmetic or GPU speed.

The warm allocation bounds allow at most two batch scratch arrays and one
binding list per dispatch, with owned result labels only in profiling builds.
They scale with the actual SDK types and dispatch/binding count. The ordinary
path has no parameter-size or pipeline-name allocation charge. Eight warm
repetitions must reuse bind groups and stay within those bounds. Recorder C API
allocations are outside that host-code measurement window. The test deliberately
does not maintain a historical provider/SDK or claim a speedup over it.

The actual diagnostic graph loop and collector keep query counts within the
query set, preserve upload/dispatch order across arena resets, and collect only
when another maximum-sized arena needs headroom or at graph completion. Mapping
waits equal the number of collections; the unconstrained queue needs no separate
queue wait. The external query resolver returns deterministic timestamps, which
are not GPU timing evidence. The copied historical initializer assertion is
retired: these tests exercise the real 74-slot arena, its overflow rejection and
the actual query boundary, not the complete public device initializer.
