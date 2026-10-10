# WebGPU image performance: bounded work and cache transport

This unit extends the Anima changes in `anima-image-performance.md` and applies
only to the image runtime. It uses patches 0022–0027 over the pinned SD and GGML
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

`tests/webgpu-performance-probe.cpp` is a test-variant-only, model-free runtime
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

`webgpu-param-upload-contract` extracts the actual prepared arena, dispatch
builder, submission helper and graph loop. An ordered queue double checks both
per-kernel and batched modes, multiple alignments, command budgets, in-flight
limits, empty/tail batches, staging lifetime, zero padding and bounds failures.
It does not replace Wasm compilation or device tests. The evaluated CMake contract
also checks that every browser/test profile enables the backend definition.

## Browser smoke and publication must agree

The producer and `scripts/validate_runtime_package.py` use the same source-bound
`tests/browser-smoke-contract.json`. Its JavaScript reader and JSON bytes are part
of the test-input identity checked before and after smoke. Required arithmetic
evidence includes timestep, BF16 loading, 3D convolution and WebGPU performance
probes, with exactly the requested backend list for every test profile. Browser
variants cannot claim test-only arithmetic. CPU-only execution is never relabeled
as GPU execution. Verified probe evidence is retained separately from the prose
scope in `validation.browserSmokeEvidence`.

The image Python suite runs the root snapshot boundary tests in an isolated process,
including the actual JavaScript scope producer, so producer/consumer drift is caught
by image-native CI instead of only after expensive package assembly. That job pins
Node 22.16.0, matching the package job. These host
checks use synthetic package bytes and mocked browser invocation; `npm pack` and
its payload integrity checks are real. Missing evidence prevents a completion
receipt or final pack. No publication, source-identity or payload check is bypassed.

Full-model WebGPU validation must still cover trained weights, device limits,
GPU memory pressure, graph cuts, cache-enabled/disabled runs, cancellation/device
loss, repeated generations and image differences. Do not extrapolate CPU timings
or synthetic admission results to end-to-end generation speed.
