# WebGPU four-stream HC PRE/POST

This narrow WebGPU fusion preserves the existing model graph and staged
fallback. Compilation and native recorded-call tests do not establish actual
GPU execution, target-device speed or full-model numerical equivalence.

## Scope and fallback

Support is limited to contiguous F32 tensors, exactly four hyper-connection
streams, gated PRE, identity-combination POST, positive shapes, distinct underlying tensor/view roots, and checked
storage and dispatch limits. Unsupported modes and shapes remain unsupported;
the existing upstream fusion probe selects the existing staged graph. The
model graph, CPU implementation, public API, profiles and variants do not change.
There is no model-name, filename, quantization-name or adapter-name switch.

For each eligible invocation the staged PRE has seven dispatches and the staged
POST has three; the fused path has one of each. Two pairs per layer over 48 layers
would eliminate 768 dispatches per decode. This is a
source-derived count, not a measured throughput improvement. The final HC head
is excluded by upstream's `il >= 0` fusion condition.

PRE retains the four products, sequential sums, final scaling and runtime +0
bias used at the existing SCALE boundary. POST retains product followed by the
residual addition. Storage bindings use read_write categories to allow disjoint
logical tensors in a shared allocator buffer. The kernels write only dst. The
upstream allocator does not admit these HC operations for in-place allocation.
These source properties do not by themselves prove compiler contraction,
signed-zero, exceptional-value or alias behavior on a physical device.

## Maintenance contract

The typed recipe applies the individual backend and shader-library patches to
one private coherent source tree. Operation definitions, allocator and fusion
contracts remain upstream dependencies. Dedicated WGSL inputs use the unchanged
upstream emitter. Preparation receipts record selected inputs and final outputs;
the checked-out upstream is never edited. No experimental indexer,
attention-mask or unrelated candidate is bundled into this exception.

The current native provider compiles the complete prepared backend and observes
admission, bindings, parameter bits, launch geometry and fallback using SDK C API
recorders. It does not execute the shader. Full-model/scheduler placement,
compiled-Wasm numerical checks and target-device throughput require their own
measurements. Keeping the staged graph is the no-patch option. Remove the patch,
preparation and associated checks when equivalent upstream support is adopted or
the requirement is withdrawn.

## Optional GPU regression

The standalone MJS runner reads supplied `hc_pre.wgsl` and `hc_post.wgsl`,
each exposing `main` with the harness's storage/uniform layout. It compares those
inputs with separately submitted staged GPU expressions. It does not identify
their upstream/patch provenance or prove agreement with the compiled production
factory. Record and review input identities separately.

With a locally installed Vulkan-capable Dawn module and compatible inputs, run
from the repository root:

```sh
LCB_TEST_WEBGPU_MODULE=/absolute/path/to/webgpu/index.js \
node tests/runtimes/llama_cpp/webgpu-hc/check-gpu.mjs \
  /absolute/path/to/shaders /absolute/path/to/hc-report.json
```

Optional trailing `CASE_START CASE_COUNT` integers select a contiguous range of
the 288 cases. The default runs all cases. This runner is numerical/lifetime
coverage, without a timestamp benchmark.

Coverage includes independent and pooled buffers, distinct binding residues,
1D/2D launches and padded groups, exact and tail dimensions, finite extremes,
cancellation, signed zeros, and positive/negative/zero PRE scaling. It checks
input/guard preservation and exact non-NaN output bits against separately
submitted GPU expressions preserving the staged operation order. Those
expressions are not the complete compiled bicore backend, and this is not a
speed benchmark. Per-case JSON Lines survive a native process crash.

A deliberately overlapping writable binding is required to fail WebGPU
validation. Host admission rejects shared tensor/view roots before placement;
it does not expand the backend with merged-binding kernel permutations.
The current native provider executes the complete prepared C++ dispatcher with
recording WGPU objects and checks parameter bits and rectangular launches. These
are host decisions rather than physical-GPU execution.

## Native instance lifetime and continuous regression

The Node/Dawn `create()` result owns its native instance. This standalone test
keeps an explicit global Symbol-keyed reference until all cleanup and report
writes finish, then releases it immediately before synchronous process exit.
A local variable used only to request an adapter is not a sufficient lifetime
anchor for the rest of the asynchronous program. Device and buffer references
alone do not establish ownership of the native instance.

To stress this contract, set `LCB_HC_FORCE_GC=1` and run `node --expose-gc`. The checker rejects that mode when `globalThis.gc` is
unavailable; it never silently skips the requested collection stress. Collection
runs between cases, and the report records `forcedGc` and the Node version.
This only changes the standalone test, not browser inference or profiling.

A lifetime test must distinguish dropping the native instance owner from keeping
it rooted, including forced garbage collection and delayed asynchronous work.
That is a harness ownership check; it does not diagnose browser stalls or measure
model speed. Numerical reports must identify their adapter and retain
all requested cases, failures and input/guard checks.
