# WebGPU four-stream HC PRE/POST

This is a narrow optimization for the explicitly requested Qwen3.8-Flash-Next
performance investigation. It has not established target-device speed or
full-model numerical equivalence. Do not describe compilation or host tests as
GPU execution. The independent input candidate is reduced to HC only and rebased
onto llama.cpp `620d2225cc436ab0b001e3200e590e3b286ef9ab`.

## Scope and fallback

Support is limited to contiguous F32 tensors, exactly four hyper-connection
streams, gated PRE, identity-combination POST, positive shapes, distinct underlying tensor/view roots, and checked
storage and dispatch limits. Unsupported modes and shapes remain unsupported;
the existing upstream fusion probe selects the existing staged graph. The
model graph, CPU implementation, public API, profiles and variants do not change.
There is no model-name, filename, quantization-name or adapter-name switch.

For each eligible invocation the staged PRE has seven dispatches and the staged
POST has three; the candidate has one of each. Two pairs per layer over 48 layers
would eliminate 768 dispatches per decode (17.0% of the observed 4,508). This is a
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

The preparation checks the previously composed backend and shader-library
identities, operation definitions, CPU reference, allocator, fusion probe and
Qwen graph. It never edits the checked-out upstream. New shaders live in a
dedicated two-symbol header, not a replacement canonical WGSL header. The patch,
helper and final compiled sources are separately recorded in provenance.
Unknown inputs, output mismatches and a shadowing canonical header fail closed.
No experimental indexer, attention-mask, bind-group-cache or other candidate is
bundled into this exception.

Local checks cover preparation, host admission, dispatcher parameters and
compile-time interfaces. The standalone Dawn/Mesa software-GPU checker covers
288 shader/bind-group cases, including pooled buffers and forced garbage
collection. Full-model/scheduler placement, compiled WebAssembly numerical
regression and paired Mac throughput measurements remain to be performed. Keeping the staged graph is the no-patch option.
Remove this exception and its preparation/provenance/tests when equivalent
upstream support is adopted or the candidate is rejected.

## Optional GPU regression

`tests/webgpu-hc/check_gpu.py` extracts the two shader strings from the actual
prepared, fingerprint-checked C++ embedding and executes them using an externally
installed Node/Dawn module. Set `LCB_TEST_WEBGPU_MODULE` to its `index.js` absolute
path, then invoke:

```sh
python3 llama-cpp/tests/webgpu-hc/check_gpu.py --source /path/to/llama.cpp --output /tmp/hc-gpu.json
```

Use `--node-command-json '["/path/to/node"]'` for a different interpreter, or
include a loader and its arguments in that array for an offline Linux bundle.
`--case-start` and `--case-count` select a contiguous range of the 288 cases.
The test does not add an npm dependency or run during ordinary inference.

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
Local graph-dispatch tests execute the actual source-extracted C++ dispatcher
with recording WGPU objects and verify parameter bits, source order and the
rectangular launch. They are not physical-GPU execution.

## Native instance lifetime and continuous regression

The Node/Dawn `create()` result owns its native instance. This standalone test
keeps an explicit global Symbol-keyed reference until all cleanup and report
writes finish, then releases it immediately before synchronous process exit.
A local variable used only to request an adapter is not a sufficient lifetime
anchor for the rest of the asynchronous program. Device and buffer references
alone did not prevent instance collection in the supplied Dawn package.

To stress this contract, set `LCB_HC_FORCE_GC=1` and include `--expose-gc` in
`--node-command-json`. The checker rejects that mode when `globalThis.gc` is
unavailable; it never silently skips the requested collection stress. Collection
runs between cases, and the report records `forcedGc` and the Node version.
This only changes the standalone test, not browser inference or profiling.

The previously observed continuous native crash was reproduced in
`InstanceBase::ProcessEvents`, entered by the Node `AsyncRunner` callback. In a
controlled test, removing the explicit instance root triggered finalization and
SIGSEGV; keeping it completed all 288 cases with exact non-NaN output bits and
intact input/guard regions. Five ordinary full runs and forced-collection full
runs passed in the supplied software-GPU environment. This resolves the local
checker failure; it neither diagnoses Mac stalls nor measures model speed.
