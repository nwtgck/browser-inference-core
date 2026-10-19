# Short-input Q8_0 matvec workgroups

This source-bound selection is part of the owner's explicitly requested Qwen Next
performance work. It is not an alternative quantizer or model pipeline.

## Measured motivation and conservative scope

The owner's post-HC profile contains 97 matvecs per generated token with K=320,
M=10240 and another 48 with K=640, M=2560, in both weight packages. Upstream's
Q8_0 accumulator assigns four lanes to a 32-weight block (eight weights per lane).
Its fixed 256-lane workgroup therefore has only 40 or 80 lanes doing input work
on these shapes. HC fusion is already visible in that profile: 96 PRE and 96 POST
operations per token, and 3740 rather than 4508 total profiled passes.

The selector uses only the existing upstream `WG_SIZE` specialization. It does
not change WGSL, quantized bytes, block mapping, per-lane accumulation, output
rows per workgroup, bind groups, workgroup-grid construction or dispatch count.
No additional Wasm profile, telemetry, synchronization, loading policy or model
name branch is introduced.

Select a smaller power of two only for subgroup Q8_0 x F32, one input column,
K in [32, 1024] divisible by 32, and M >= 1024. Exclude packed-integer-dot and
source-overlap paths. Reject unknown subgroup limits. The floor is the greater
of 32 and the reported maximum subgroup size, avoiding partial subgroups even
on adapters exposing a range. Otherwise preserve the upstream 256-lane path.
The output-row threshold is deliberately conservative, not a universal tuning
claim. At K=320 the common 32-lane-subgroup device uses 64 lanes; at K=640 it uses
128. All useful lanes still visit exactly the same one block as before.

## Cache and profiling identity

The compiled workgroup size is part of both the pipeline key equality and hash.
A K=320 pipeline must never be reused as the K=640 or long-input pipeline.
The name adds `_wg32`, `_wg64` or `_wg128` only on a tuned path. Existing Naidan
GPU-profile collection can see these names without a consumer API change.

## Preparation and publication

`prepare_webgpu_q8_short.py` runs after the reviewed HC composition and checks
its exact header input, the transitive upstream shader include closure, the
preprocessor, the emitter, the patch and the resulting header. The canonical
upstream shader header remains authoritative; no additional embedded shader
copy is generated. CMake tracks all inputs for reconfiguration. The patch is
explicitly listed in the policy test and in classified publication provenance.
CPU builds report no compiled copy or enabled variants for this overlay.

## Validation boundaries

Host tests exercise the extracted actual selector, pipeline-key hash/equality
and pipeline factory with recording dependencies, plus fail-closed preparation,
CMake reconfiguration and provenance. The optional standalone GPU test uses the
real pinned emitter/preprocessor and upstream WGSL, comparing against its
256-lane version with exact output-bit and input/guard checks. Software-Vulkan
measurements are not Mac throughput or learned-model quality measurements.
Although useful lanes retain their accumulation order, a different number of
zero-only subgroups may affect compiler reduction decisions. Real-device model
validation remains required; no universal bit identity or speed claim is made.

No-patch alternative: keep the upstream 256-lane workgroup. No existing public
runtime setting selects this shape-specific workgroup. Maintenance is limited
to the internal pipeline key, factory and reviewed shader accumulation/reduction
contract. Remove this overlay, preparation and provenance together when upstream
provides suitable short-Q8 selection, or if target-device results reject it.

## Optional source-bound experiment

With local Node/Dawn, a C++ compiler and the pinned upstream already available:

```sh
python3 llama-cpp/tests/webgpu-q8-short/check_gpu.py \
  --llama-source /path/to/llama.cpp \
  --work /path/to/experiment-output \
  --webgpu-module /path/to/node_modules/webgpu/index.js \
  --node-command-json '["node"]' --force-gc --benchmarks
```

The output records upstream/patch/prepared-header and emitted-shader identities.
The 107-case matrix includes all 32-aligned short input widths, tail output rows,
nonzero byte offsets, padded strides, multidimensional batches/broadcasting and
actual 320-by-10240 and 640-by-2560 shapes. Outputs are checked bitwise against
the unchanged 256-lane specialization; inputs and output guards must remain
unchanged. Optional GPU timestamp comparisons alternate AB/BA on the same
buffers, discard four warm-up pairs and retain twelve measured pairs including
a long-input unchanged-path control. An injected output error must fail rather
than being hidden by a relaxed numerical tolerance.

The returned Dawn owner is rooted until process exit, including forced garbage
collection. A successful standalone comparison does not establish performance
or model quality on a different adapter, the browser, or the compiled runtime.
