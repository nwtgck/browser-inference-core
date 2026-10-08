# Optional same-device WebGPU tensor copy

The pinned WebGPU backend leaves its optional destination-buffer `cpy_tensor`
hook null. Generic copies consequently use a host read and write. This opt-in
build-tree overlay implements that existing hook for compatible GPU buffers.
It adds no public API and changes no model, quantization, state format,
checkpoint selection or runtime capability advertisement.

## Activation

Default builds and direct CMake configuration leave the option OFF. For an
experimental build, run from `llama-cpp/` with the standard pinned toolchain:

```sh
python scripts/build.py --fresh --profile webgpu-wasm64-jspi --variant browser --webgpu-tensor-copy
```

The other configured WebGPU profiles and the test variant also support the
flag. CPU requests fail clearly. Every ordinary invocation passes explicit OFF,
including when reusing a previously enabled build directory. Direct CMake uses
`-DLCB_WEBGPU_TENSOR_COPY=ON` together with WebGPU and the pinned
`EMDAWNWEBGPU_DIR` package; unknown headers fail configuration.

The build workflow has a default-false `webgpu_tensor_copy` manual input. Enabled
manual runs build and validate the existing matrix and upload packages, but skip
publication. OFF and ON can produce different bytes from one source commit, so
an experimental run must not occupy the ordinary immutable publication identity.
Consumer integration requires an actual built artifact and a separate adoption
and publication decision. No artifact hash or performance result is implied.

## Contract and compatibility

- Reviewed upstream: `d81235049384534c167caea52b85a694f6103d14`.
- Reviewed Dawn: `v20260908.214631`. The enabled CMake hook verifies its actual
  four public C/C++ headers, not merely a version label.
- `prepare_webgpu_tensor_copy.py` binds the patch and output identities, plus
  only the backend, generic fallback, buffer/layout declarations and CMake/header
  lookup contracts. Unrelated model, shader and audio files are not hash-pinned
  by this optional overlay. Upstream revision changes still require review.
- Preparation never edits vendor sources. CMake replaces exactly one source of
  the existing `ggml-webgpu` target, retaining compile flags and dependencies.
  Its appended sibling include does not override the MoE shader overlay's
  BEFORE priority. BF16 and audio remain independent mtmd source overlays.
- Reconfiguration tracks all guarded source/header and patch inputs. Unknown
  source, patch, header or target layout fails closed; no stale source is chosen.
- Provenance inventories the patch even when disabled, records explicit ON/OFF
  profile variants, and reports a patched compiled-copy identity only when ON.
  Updater preflight reports optional incompatibility separately, allowing
  unchanged default configurations to continue.

## Copy behavior and constraints

The hook validates the source backend before casting, matches the owner context
(device and queue) and layout, resolves source views, and checks byte ranges
without overflowing additions. Empty and exact same-resource/same-offset copies
are no-ops. Underlying handles use `Buffer::Get()` comparisons. Other same-resource
copies, missing copy usages and non-four-byte-aligned intervals use the existing
generic fallback. No padding is copied and invalid tensor inputs are not made
safe by returning false.

Eligible copies encode and immediately submit on the existing common queue,
without staging memory or CPU waits. Serialized operations retain queue order;
return means submitted, not physically complete. Existing synchronization or
readback observes completion. Concurrent callers or unsubmitted foreign work
are outside this contract. Resources submitted before destruction must remain
valid through WebGPU's submitted-work lifetime semantics.

This does not make ON_DEVICE checkpoints a safe default: allocation failure,
peak GPU memory/OOM behavior, lifetime and checkpoint capability contracts remain
separate prerequisites. Retain host snapshots as the no-patch alternative, or
wait for upstream support. Remove this exception and its integration when
upstream supplies equivalent behavior or the optional requirement is dropped.

## Verification

```sh
LCB_TEST_LLAMA_SOURCE=/path/to/pristine/llama.cpp \
LCB_TEST_DAWN_PACKAGE=/path/to/emdawnwebgpu_pkg \
python -m unittest discover -s tests -p 'test_webgpu_tensor_copy_overlay.py' -v
```

Target tests configure the actual upstream WebGPU target, inspect its compile
commands, check ON/OFF/ON-cache transitions, and exercise simultaneous MoE,
BF16 and audio overlay preparation. An extracted actual prepared hook and generic
fallback run under a deterministic CPU mock, covering byte correctness, ordering,
views, ranges, fallback, submission and resource lifetime. Optional real-header
syntax checks validate the hook against actual ggml/Dawn declarations.
These checks are not full Wasm linking, browser conformance, GPU execution,
model equivalence or speed evidence. Complete the standard target CI build and
real-device copy/lifetime/fallback and checkpoint-equivalence checks before use.

When parameter upload batching is also enabled, `prepare_webgpu_source.py` composes both patches into one checked translation unit. The options remain independent and both default to OFF. See [parameter upload batching](webgpu-param-upload-batching.md).
