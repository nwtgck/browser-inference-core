# WebGPU image performance: bounded work and cache transport

This unit extends the Anima changes in `anima-image-performance.md` and applies
only to the image runtime. It uses patches 0022–0030 over the pinned SD and GGML
sources; vendor commits, the llama runtime and application presets are unchanged.

## Separate execution placement from transport

A supported WebGPU operation does not imply that its result is copied efficiently.
The pinned backend had neither a buffer copy nor an asynchronous backend copy
callback. GGML therefore used host readback/upload when persisting GPU cache
entries. Patch 0022 supplies compatible same-device copies; patch 0023 captures
all outputs before one completion boundary. It changes neither the cache key nor
its enabling policy. Non-contiguous graph-cut spans and unsupported transfers keep
the old path. Allocation finishes before submission, and every submitted destination
is retained until completion, including exceptional exits.

The original Anima compact-context sink is already accepted by this pinned
backend's normal and Flash Attention implementations. Also, the pre-0020 browser
causal convolution already had a WebGPU-aware 3D-to-2D lowering. Do not claim that
0020 eliminates an IM2COL_3D fallback by comparing it to a CPU-only reference graph.
Its remaining hypothesis is reduced temporal zero work, minus weight packing cost;
actual GPU throughput remains to be measured.

## Bound parameter uploads at the existing submission boundary

Patch 0026 enables `GGML_WEBGPU_BATCH_PARAM_UPLOADS` on the prepared image-only
WebGPU target. The existing parameter arena keeps one aligned host mirror, stages
kernel parameters into their original slots, then issues one `WriteBuffer` before
each existing full or tail command submission. The graph's command cadence,
queue throttling, binding offsets, compute passes and reset boundaries are unchanged.
Queue order stays upload A / submit A / upload B / submit B, even when slot storage
is reused before prior work completes. Unused slot bytes are zeroed.

At the default 64-kernel command budget, the 10-slot safety margin and 256-byte
alignment require 74 * 256 = 18,944 host bytes. Padding increases uploaded bytes;
this is a reduction in API calls, not a reduction in every transfer metric and not
a measured generation speedup. Non-parameter uploads and cache-copy submission
counts are unchanged. The separately built llama runtime already has a related
implementation; it is neither repinned nor modified here.

Patch 0027 skips cache completion only when capture has no new destinations.
Runner outputs absent from this segment, already-pending entries and graph cuts
without future consumers no longer cause an empty wait. Actual computation and
segment cleanup retain their own completion boundaries. Non-empty copies still
complete before ownership is published, including exceptions.

## Bound materialized intermediates, not the user's image

Patch 0024 changes graph construction only when the preferred WebGPU backend
rejects the full intermediate. It probes a decreasing chunk size, with a maximum
of 64 chunks, and checks every introduced compute operation. No CPU policy is
replaced, no precision is reduced, and neither VAE image tile limits nor Flash
Attention settings are changed.

- Convolution splits output rows. Original spatial halo, stride, dilation, batch,
  output layout and F16 input expansion are preserved. A single final output is
  filled with ordered SET nodes. Unsupported padding, layout, matrix multiplication
  or SET operations retain the original graph.
- Manual attention splits queries, never keys. Every query keeps the complete
  softmax denominator, including structural Anima sinks and its mask slice. The
  final output is filled without repeated full-output concatenation. Masks whose
  strided spans remain too large can still require the original fallback.

The native fixture extracts the exact capability predicate and pure helpers from
prepared GGML source. Its advertised device limits are synthetic. A zero rejection
count is graph admission evidence, not actual placement, completed GPU execution,
peak allocation, driver compilation or a trained-model speedup.

## Operation placement diagnostics

Use the existing public `sd_set_graph_diagnostics` control and log callback. No
new application ABI is introduced. `browser-placement-ops-v1` is emitted only while
that control is enabled, after workspace allocation:

| Field | Meaning |
| --- | --- |
| cpu / webgpu / other | Assigned compute nodes, excluding views/reshapes/transposes |
| cpu_rejected_now | CPU nodes currently rejected by preferred WebGPU support checks |
| largest_cpu_operand_bytes | Largest logical input span of an assigned CPU operation |
| cpu_ops | Histogram of CPU operation kinds, without tensor or prompt names |

The original BF16-oriented `browser-placement-v1` log is unchanged. These logs
record scheduled placement, not completed execution. A CPU operation may reflect
intentional residency, an explicit CPU component, unsupported operations or other
scheduler policy; `cpu_rejected_now` is not a complete attribution of its cause.

## Reproduction and verification boundary

Native CMake checks include `webgpu-bounded`, `webgpu-transfer`, the three Anima
regressions and `browser-boundary`. Source extraction intentionally fails if
upstream signatures/vtables drift. Transfer API doubles inspect encoded ranges,
copy/read/write counts, completion and failure lifetimes, not actual WGPU calls.

`../../tests/runtimes/stable_diffusion_cpp/webgpu-performance-probe.cpp` is a test-variant-only, model-free runtime
probe. It builds small convolution and manual-attention graphs, forces a small
construction budget, and then requires the requested real backend to support and
execute every selected operation. It compares against CPU references, captures
outputs, changes inputs, and verifies independent cache ownership after the
original graph and weights are destroyed. It also compares observable independent
scale/SET outputs across submission-size boundaries and repeated graph runs, so later normalization cannot hide stale
parameters. It has no time/speed assertion.

The existing browser smoke runner invokes it for CPU and, with
`SDCB_TEST_WEBGPU=1`, WebGPU. Use that runner's existing working directory,
Playwright setup and packaged runtime argument. The opt-in fails rather than
silently skipping an unavailable GPU. The probe and its suspending export are
absent from ordinary browser artifacts. Native tests execute its CPU path only;
CMake export-contract tests are not a Wasm build.

The `webgpu-backend-*` native contracts compile the complete prepared backend
with real SDK declarations and observe its external queue API. They check arena
bounds, batched/per-kernel uploads, alignment, command budgets, in-flight limits,
empty/tail batches, deferred uniform ownership and zero padding. The mixed-queue
cases use real VIEW, ARGSORT and SET_ROWS graph nodes to distinguish node counts
from kernel counts and check the optional SET_ROWS readback boundary. Native
observation does not execute shaders or replace Wasm and device integration.

## Browser smoke and publication must agree

The browser harness and typed saved-package validator use the same owned
`runtimes/stable-diffusion-cpp/config/browser-smoke-contract.json` through their
runtime-owned contract readers. The JSON, browser reader and executing typed
reader are included in the test-input identity checked around smoke execution.
Required arithmetic evidence includes timestep, BF16 loading, 3D convolution and
WebGPU performance probes, with exactly the requested backend list for each test
profile. Browser variants cannot claim test-only arithmetic. CPU-only execution
is never relabeled as GPU execution. The recorded browser evidence is separate
from the prose description of its scope.

`tools/validation/runtime.ts` invokes the external browser runner, checks its
result against the selected package and writes the typed receipt. Node tests
under `tests/tools/validation/` exercise captured browser results and input
identity changes without starting a browser. Package tests use synthetic payloads
and real npm packing where relevant. A missing or mismatched required result
prevents validation completion. Native checks, captured-result tests and actual
browser execution remain separate evidence.

Full-model WebGPU validation must still cover trained weights, device limits,
GPU memory pressure, graph cuts, cache-enabled/disabled runs, cancellation/device
loss, repeated generations and image differences. Do not extrapolate CPU timings
or synthetic admission results to end-to-end generation speed.


## Reuse each invocation's softmax cache (0028)

The pinned shader owns sixteen private floats **per invocation**, not sixteen
floats for the whole row. Index by the lane's loop iteration, rather than its
strided global column. All three passes use the same index. Explicit branches
replace `select(storage_read, cache_read, condition)`: selecting a result does
not guard evaluation of the other argument or its out-of-range cache index.

A row of at most `16 * WG_SIZE` columns now retains its logits and exponentials
in private values through the reductions. Larger rows spill only the remainder
through the existing intermediate output. The workgroup size, sixteen-float
budget, barriers, reductions, mask/sink rules and output layout are unchanged.
Private storage can still be spilled by a real shader compiler; host-model
access counts are not hardware counters or a measured speedup.

The test-only real-backend probe executes softmax around workgroup/cache
boundaries through 32,771 columns, with masks, sinks, in-place/out-of-place paths,
two scales and different input distributions. It checks an independent scalar
oracle, nonzero input/mask offsets, surrounding guards and repeated graph use.
Native tests execute CPU operators; explicit WebGPU smoke executes shaders.
The former WGSL-to-C++ lane model was removed: its logical storage counts did
not establish compiler register allocation, hardware traffic or GPU speed.

## Fuse unmasked score scaling without rescaling queries (0029)

For a supported WebGPU F32 unmasked manual-attention softmax, use its existing
scale parameter. The multiplication still occurs **after** F32 score
accumulation. Neither Q/K values nor matmul input precision are changed. Scaling
queries instead would alter rounding and potentially mixed-precision conversion;
that earlier experimental approach is not used.

Masked paths retain the original explicit scale and broadcast addition. A
backend rejection retains the old path. Sink-bearing softmax remains
out-of-place, and sink attachment precedes admission. Bounded attention uses the
same helper per chunk. This removes a separate score-buffer read/write pass;
it does not enable Flash Attention or approximate the denominator.

`webgpu-attention-softmax` compares the production builder to its unmodified CPU
path for F16/F32 inputs, head dimensions, batches, masks, sinks, bounded cases and
injected rejection. The argument named `kv_scale` is a Flash overflow control,
not a requested softmax temperature; tests inspect the actual `1/sqrt(head)`
softmax parameter and ensure that the input is still the accumulated score.

## Share the Qwen position-matrix preparation within a graph (0030)

Qwen Image 2.1 has separate Q and K rotation consumers in each transformer layer.
Previously every consumer built another `cont(permute(pe))`. The WebGPU model
builder now prepares its two matrix-row views once and explicitly passes that
representation to its layers. Other callers retain the original wrapper. A
32-layer monolithic graph has one preparation instead of sixty-four, with no
position values, tensor precision, rotary layout or model weights changed.

This is **graph-local shared computation**, not a persistent pointer-keyed value
cache. The position producer is rerun when graph inputs change. With graph cuts,
a segment may recompute it; do not claim one upload for an entire segmented run.
The existing planner's dependency and ownership rules remain responsible for
segment lifetime. Native regression tests check the original frozen rotation
against the shared builder bit-for-bit, change the same input storage between
executions, and construct valid 2/32-layer cut plans with prefix reuse both off
and on. The optional real-backend probe covers shared rotation arithmetic too.

Graph-local sharing can reduce repeated preparation and associated uploads, but
the actual byte count depends on graph cuts, input lifetimes and backend
scheduling. Measure queue uploads separately from GPU-internal traffic, live
residency and physical memory. Native graph-count and arithmetic checks do not
replace actual WebGPU validation or cold/warm full-model measurements.

## Qwen Image 2.1: remove repeated elementwise work (0031–0032)

Rotary matrix sharing leaves separate layer-local modulation work and
intermediate MLP activations to optimize. Qwen's four modulation matrices depend on the
current timestep, not on the layer. For an accepted WebGPU graph, prepare their
scale/tanh rows once and reuse those producers across layers. Prefix reuse needs
only target rows; an initial full-context graph also prepares the prefix rows.
Every new graph evaluates its current inputs. No pointer-identity or cross-run
modulation cache is introduced, and layer/graph-cut marker ownership is unchanged.

Modulation uses out-of-place MUL. Read-only contiguous target/prefix spans can
therefore be borrowed directly; noncontiguous spans retain explicit compaction.
This is a local alias argument, not permission to delete all CONT operators.

The MLP uses the pinned backend's existing F32 SwiGLU for compatible packed or
separate gate/up projections. Guard the GGML constructor's layout/type/shape
preconditions before building a candidate and ask the preferred backend to accept
it. Unsupported inputs keep separate SiLU and MUL. No shader/precision override,
weight conversion, quantization, image-size, step-count, or Flash setting changes.

For controlled comparisons, the context-creation model argument
`qwen_image_2_1_webgpu_work=false` preserves the previous three graph choices.
It does not disable earlier RoPE/softmax/transport patches. Creating a new context
is required to change this flag; never mutate it while a graph is active. A false
setting is compared with the frozen pre-work-sharing reference on CPU. Same-artifact
experiments must still account for run order and cannot infer thermal state from
timing alone.

`webgpu-qwen-work-test` uses real model methods with synthetic weights, frozen
pre-work-sharing references, repeated input/timestep values in the same storage, negative
admission/type/layout cases, and real graph-cut binding/cache execution on CPU.
The optional model-free WebGPU smoke adds requested-backend elementwise checks.
Policy replay and CPU execution are not GPU speed or trained-image validation.

When existing graph diagnostics are enabled, `browser-placement-ops-v1` additionally
includes `webgpu_ops`, `largest_webgpu_output_bytes`, `webgpu_silu`, `webgpu_tanh`,
and `webgpu_swiglu`. These are allocation metadata; backend fusion can mean fewer
actual dispatches. A sum of logical outputs across layers is neither peak VRAM nor
measured GPU memory traffic. The diagnostic does not change wait or submit policy.

## Keep large linear dispatches native (patch 0033)

A supported tensor shape can require more workgroups than one dispatch axis
allows. This is distinct from storage-buffer limits or total memory capacity.
Row normalization (including its multiply fusion), softmax, and GLU previously
sent their linear domain in X alone. Large image shapes consequently relied on
an application wrapper to transform shaders and split dispatch calls.

These four builders now use a tight two-dimensional grid, preserving a one-axis
shape for small domains. The row shaders compute `x + y * num_workgroups.x` and
return for padding rows before storage access or barriers. Workgroup ID, dispatch
size and the uniform row count are uniform within each workgroup. GLU additionally
checks the group before multiplying by its workgroup size, then checks the final
partial group per invocation. Host ceil-division uses uint64_t and caps each axis
at 65,535 so the flattened index cannot wrap u32. Other kernels and their existing
grid helpers are unchanged.

At an axis limit of 65,535, a logical 131,072-row operation can be encoded as
43,691 by 3 groups in a single dispatch instead of three application-side calls;
one padding group is inactive. A 65,536-row operation uses 32,768 by 2 groups.
This reduces dispatch splitting, not matrix-multiplication work, storage traffic
inside a kernel or shader arithmetic precision. The distribution across axes can
have device-dependent scheduling effects: end-to-end performance is unmeasured.
No Flash setting, tile policy, workgroup size, memory budget or wait is changed.

The `webgpu-backend-*` native contracts call the actual builders and grid helper
from the complete prepared backend. SDK observations check uniform records,
bindings, large integer domains and fused-operation overlap branches. These are
host packet checks; they do not translate shader guards or execute WGSL. The
requested-backend probe also runs NORM, RMS_NORM, RMS_NORM+MUL and masked/sink
softmax at 66,049 rows, with separate/in-place destinations and changed inputs.
CPU CI runs its CPU path. Only explicit WebGPU smoke exercises shaders.

## Stable registry/handle ownership (patch 0034)

Repeatedly retrieving a registry must not reset its discovered device count.
Retrieving a device must not request a different adapter or replace the shared
Device/Queue while tensors still own buffers from the original device. Discovery
and device publication are now separate once-only local-static initializations.
The admitted adapter is reused; the device is published after its queue and shared
buffers are ready. The existing deliberate registry lifetime through teardown is
preserved. A failed discovery remains unavailable and a lost device stays lost;
this is not an automatic browser reset, adapter retry or device-loss recovery.

Adapter and device request callbacks capture independently owned result records.
WaitAny failures do not inspect or publish potentially unfinished callback data.
A late callback may finish its own record but cannot reference a returned stack
or mutate the shared device. Callback status and non-null handles are checked
before successful publication. The existing completion-based wait policy remains.

The `webgpu-registry-*` and `webgpu-native-registry-*` native contracts compile
the complete backend against browser/native SDK declarations. Their external API
observer covers concurrent and reordered lookups, retained buffer/queue ownership,
creation failures, delayed callbacks, failed waits, lost devices and the native
implicit-synchronization requirement. It executes neither Dawn nor a browser.
The WebGPU-only runtime probe additionally performs repeated
registry/device lookups while real test buffers are live; the CPU probe does not
pretend to check GPU registry state.

The shared smoke description records the new large-row coverage and explicitly
qualifies live-registry checks as WebGPU-only. Publication negative tests require
the current source-bound description and verify that removing required scope text
actually changes the fixture; old probe evidence cannot validate a new artifact.
