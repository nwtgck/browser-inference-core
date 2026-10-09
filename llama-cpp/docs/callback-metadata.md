# Synchronous metadata getters inside native evaluation callbacks

JSPI (JavaScript Promise Integration) makes the ordinary generated exports return
Promises even when their implementation does not suspend. They cannot be called
as synchronous numeric/pointer getters from an in-progress native evaluation
callback. Awaiting a getter there would violate the native callback contract.

The binding generator now emits an explicit, narrow, synchronous surface:
`lcb_callback_metadata_version()` returns `1`, and seven
`lcb_callback_<public-name>` exports forward only to the public metadata getters
listed in `CALLBACK_METADATA_GETTERS`. The existing pointer/size representation
and checked conversion are reused. Ordinary generated function exports and their
Promise contract are unchanged. The callback aliases are exported, but **not**
listed in `JSPI_EXPORTS`; the schema records their version and identity, so the
ordinary schema hash and distribution provenance include this addition.

## Scope and lifetime

The getters read operation descriptions, backend registry/device capability,
and destination-buffer metadata. They do not submit work, map GPU buffers, read
tensor values, create/free contexts, or implement model operations. Pointer
arguments must be live native objects belonging to the currently active module.
A tensor pointer is valid only while the evaluation callback owns it. Do not
retain native pointers or use these exports as a general reentrant API.

Buffer location and a device's `supports_op` result **do not identify the actual
scheduler-selected execution backend**, a CPU fallback reason, or GPU time.
Installing an evaluation callback can itself alter synchronization; isolate that
instrumented run from normal performance measurements. The aliases by themselves
install no callback and change no scheduling in ordinary inference.

## Compatibility and maintenance

This is a binding-layer addition using public upstream functions. No upstream
source, existing exception patch, compiler transformation, or private runtime
export is rewritten. No WebAssembly export-table bypass is needed in consumers.
The name set and wire signatures fail closed if public declarations disappear,
are deprecated, or change incompatibly. Inspect the implementations at every
upstream update to keep the synchronous, non-submitting property true; signature
checks alone cannot prove that property. Add only reviewed metadata getters,
never make this a wildcard exemption for all native functions.

A consumer can feature-detect `_lcb_callback_metadata_version`. Existing JSPI
artifacts without it must omit these fields or use validated tensor layout data;
they must not invoke ordinary Promise exports synchronously. Build fresh native
artifacts, keep their real source/schema identities, then use the normal consumer
pin-update procedure. This patch does not publish or invent artifacts or hashes.

## Validation

`python -m unittest discover -s llama-cpp/tests -p test_callback_metadata_bindings.py -v`
checks the export/JSPI split, schema hash, missing/deprecated/signature changes,
and compiles/runs the generated wrappers with a host C++ fixture. This is not a
browser Wasm build or a real WebGPU execution test. The two existing partial-AST
binding tests explicitly omit the new getter set because their fixtures do not
represent the full public headers; real generation never disables the set.

The development environment for this patch has no pinned upstream checkout or
Emscripten toolchain. Real rebuilt WebGPU/JSPI artifacts still require build and
browser verification before use; do not replace source-bound artifact checks with
fixture results.
