# Short-input Q8_0 matvec workgroups

This optimization selects the existing upstream workgroup specialization for
short-input matvecs. It does not change quantization or the model pipeline.

## Lane utilization and conservative scope

Upstream's Q8_0 accumulator assigns four lanes to a 32-weight block, with eight
weights per lane. A 256-lane workgroup therefore has only 40 lanes doing input
work at K=320, or 80 at K=640. These shapes motivate the guarded smaller
workgroups below; lane utilization alone is not a measured speed result.

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
The name adds `_wg32`, `_wg64` or `_wg128` only on a tuned path. Application
GPU-profile collection can see these names without a consumer API change.

## Preparation and publication

The runtime's typed recipe applies the individual workgroup patch to the final
private shader-library source. The unchanged upstream emitter generates headers
from the selected WGSL inputs. No replacement implementation is hidden in a
second shader string. Preparation records the actual inputs, patch and final
outputs; actual compiler inputs establish selection. CPU builds do not claim a
compiled WebGPU copy.

## Validation boundaries

The current complete-source native provider exercises the selector, pipeline-key
hash/equality, factory and real dispatch geometry using recorded SDK calls.
Those observations do not execute shaders. An optional GPU comparison must use
the same emitted WGSL and compare the smaller specialization with the 256-lane
reference, including input/guard preservation. Software-adapter results cannot
establish model quality or physical-device throughput. A different number of
zero-only subgroups can affect compiler reduction decisions even when useful
lanes retain their arithmetic order.

No-patch alternative: keep the upstream 256-lane workgroup. No existing public
runtime setting selects this shape-specific workgroup. Maintenance is limited
to the internal pipeline key, factory and reviewed shader accumulation/reduction
contract. Remove this overlay, preparation and provenance together when upstream
provides suitable short-Q8 selection, or if target-device results reject it.

## Optional source-bound experiment

The standalone MJS runner consumes supplied WGSL, not a source checkout. Provide
`q8-32.wgsl`, `q8-64.wgsl`, `q8-128.wgsl` and `q8-256.wgsl` with entry point
`main` and the bindings/uniform layout expected by the harness. The current
native provider emits these names for its Q8 cases, but a filename alone does not
establish that an independently supplied file came from that factory. Record
and review the input shader bytes separately when making that claim.

With compatible shader inputs and a locally installed Dawn module, run from the
repository root:

```sh
LCB_TEST_WEBGPU_MODULE=/absolute/path/to/webgpu/index.js \
LCB_Q8_BENCHMARKS=1 \
node tests/runtimes/llama_cpp/webgpu-q8-short/check_gpu.mjs \
  /absolute/path/to/shaders /absolute/path/to/q8-report.json
```

Omit `LCB_Q8_BENCHMARKS` for numerical checks without timestamps. Benchmark mode
requires `timestamp-query`; all runs require subgroups and shader-f16. To stress
native instance ownership, set `LCB_Q8_FORCE_GC=1` and pass `--expose-gc` to Node.
Missing requested capabilities fail rather than silently disabling the check.

The report records the adapter, requested cases, errors and optional paired
timings. It does not verify upstream, patch or production-factory provenance.
The case matrix includes all 32-aligned short input widths, tail output rows,
nonzero byte offsets, padded strides, multidimensional batches/broadcasting and
actual 320-by-10240 and 640-by-2560 shapes. Outputs are checked bitwise against
the supplied 256-lane reference; inputs and output guards must remain
unchanged. Optional GPU timestamp comparisons alternate AB/BA on the same
buffers, discard four warm-up pairs and retain twelve measured pairs including
a long-input unchanged-path control. An injected output error must fail rather
than being hidden by a relaxed numerical tolerance.

The returned Dawn owner is rooted until process exit, including forced garbage
collection. A successful standalone comparison does not establish performance
or model quality on a different adapter, the browser, or the compiled runtime.
