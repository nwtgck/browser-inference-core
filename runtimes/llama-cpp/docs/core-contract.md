# Low-level API and ownership

## Boundary

The build publishes llama.cpp, selected backends, generated C bindings, and Embind registrations for upstream C++ chat types. It does not own model catalogs, downloads, storage paths, chat sessions, generation loops, UI, or worker protocols. `host/` is always shipped as tested reference code for application host implementations. Its use is optional; changing application host code does not require rebuilding Wasm.

The generated ABI 3 facade retains upstream function names. Scalar signatures use the original C exports; ordinary `bicore_` bridges adapt records where required by the Wasm/JavaScript ABI. There is no application-specific JSON command dispatcher. Deprecated, variadic, and unsupported video declarations are listed with exclusion reasons in the C schema. GGML graph-building declarations accepted by the C binding generator are exposed with generated types and schema checks. The published schema is the inventory for that build; this is not a guarantee that every upstream helper is exported or that every backend implements every operation. See [native chat and multimodal bindings](chat-and-multimodal.md) for the separate Embind surface and generated `mtmd` APIs. The conventions below describe the C bindings and optional reference wrapper.

Upstream v1.0.0 removes `ggml_diag_mask_inf`, `ggml_diag_mask_inf_inplace`,
`ggml_diag_mask_zero`, and `ggml_diag_mask_zero_inplace` from the generated C
surface. It also removes `llama_context_params.defrag_thold`, the ADD1 and
DIAG_MASK operation constants, and the old `GGML_PREC_DEFAULT` alias. Read the
current schema and record metadata before using these APIs; do not reuse field
IDs or offsets from an older artifact. The newly public embedding-row and
context-state helpers are generated from the upstream public headers. Private
helpers outside that header collection remain outside the generated surface.

## Calling convention

| Native type | Host representation and native call |
| --- | --- |
| Object, buffer, or function pointer | Checked `bigint`; null is `0n`. Converted to the selected target's pointer width at the call boundary. |
| `size_t` | Checked `bigint`, converted according to the target's actual native width. |
| `uint64_t` or `int64_t` | `bigint`. |
| 32-bit integer, enum, or bool | `number`; bool is 0 or 1. |
| float or double | `number`. |
| Structure passed by value | Caller-owned record pointer; an ordinary bridge passes the record by value. |
| Structure returned by value | Explicit caller-owned return-storage pointer; the Promise resolves to `void`. |

For example, `llama_model_default_params()` becomes:

```js
const params = core.allocRecord('llama_model_params');
await core.api.llama_model_default_params(params);
// Use the parameters while their referenced resources remain valid.
core.free(params);
```

The native core reports structure sizes, alignments, and field offsets. Do not treat a C++ structure as a JavaScript object or reuse offsets from another build. Types and the target schema catalog are generated together; attachment selects the layout using the actual module ABI version and fingerprint.

Every reference `core.api` function returns a Promise, including CPU functions. This gives JSPI and Asyncify WebGPU calls the same interface; it does not move synchronous CPU work to another thread. Overlapping calls through that wrapper are rejected. Applications provide sequencing across all C and Embind calls; direct native calls bypass the reference wrapper's checks.

Both `webgpu-wasm32-jspi` and `webgpu-wasm64-jspi` expose JSPI Promise exports
through the default reference wrapper. The wasm32 variant needs no memory64 and
has a 4 GiB linear-memory ceiling. The public host API uses `bigint` for pointers and 64-bit arguments; its target
adapter performs the actual Wasm32/Wasm64 scalar conversion.

With `webgpu-wasm32-asyncify`, the reference loader uses Emscripten's
`ccall(name, returnType, argumentTypes, arguments, { async: true })`. Raw Asyncify
exports can return before a GPU operation finishes; merely awaiting that return
is insufficient. When attaching a module yourself, pass
`attachCore(module, schema, { suspension: 'asyncify', createApi })`, supplying the
generated API factory. Host pointer, record-address and 64-bit arguments remain
`bigint` even with 32-bit Wasm memory. Direct exports require their actual target
scalar types; the generated facade performs that conversion. Synchronous layout/allocation
helpers and Embind chat operations do not require suspension wrappers.

## Memory and lifetime

Memory returned by `alloc`, `allocRecord`, or `utf8` belongs to the caller and must be released with `core.free`. Models, contexts, and samplers require their corresponding upstream destructors, such as `llama_model_free`, `llama_free`, and `llama_sampler_free`. Do not pass borrowed vocabulary, logits, or string pointers to `core.free`.

A structure's allocation and its pointed-to resources have separate lifetimes. Keep callbacks, model overrides, sampler arrays, and other referenced data alive for as long as the native owner can use them. Consult the pinned upstream header for ownership transfers.

`bytes` returns a borrowed view of the current Wasm memory. Memory growth can invalidate earlier views. Reacquire the view after a native or asynchronous call that may grow memory. The helpers themselves reacquire `HEAPU8` on each access.

## Callbacks and raw exports

`module.addFunction`, `removeFunction`, and table growth are exposed. A callback must use the upstream C signature for the selected build's ABI, including the correct pointer width. Do not remove a function-table entry while native code can still reference it.

Callbacks do not have a generic asynchronous suspension contract. Treat logging, progress, and cancellation callbacks as synchronous. Raw exports are an escape hatch, not permission to ignore signatures, reentrancy, or lifetime rules.

## Files

`mountReadOnlyFile` registers an ordinary read-only file in Emscripten's JavaScript filesystem. The caller supplies a fixed file size and synchronous `read(destination, offset)` implementation. OPFS, an existing Blob, or another store can provide the bytes; storage layout is not part of the core contract.

Read requests are split at the configured chunk limit. Short reads advance by the actual returned length. Promises, negative or oversized results, and exceptions become I/O errors. File offsets use nonnegative safe integers without 32-bit bitwise truncation. The source must not change while any native owner can still read it.

Use `load_mode = LLAMA_LOAD_MODE_NONE` and, by default, `lazy_mode = LLAMA_LAZY_MODE_OFF`. The optional versioned [file-backed Qwen4Exp row extension](file-backed-rows.md) permits `LLAMA_LAZY_MODE_ON` on supported 64-bit builds; it requires retaining the selected shard through model destruction. The custom file rejects mmap rather than silently creating a file-wide emulated mapping. This does not bound every allocation made by the model loader, including GPU staging allocations.

The caller owns any OPFS access handle. Removing a mounted file is rejected while native code still has it open. Destroy contexts and models in the appropriate order, remove the file, and then close its storage handle.

## Errors and support

Bindings check argument types, pointer-width conversion, and explicit memory-access bounds. They do not prove that an arbitrary native pointer identifies a valid live object. After misuse of raw APIs, an assertion failure, or an unexpected native exception, recovery of the same instance is not guaranteed. Applications should be able to discard and recreate the owning worker and core instance.

Model support depends on the pinned llama.cpp implementation and selected backend. Exporting a function is not a claim that every model, operation, or browser is supported. Some upstream training and quantization functions are exposed, but require their own validation before being offered as browser application features.

## Experimental model-graph boundary

`model-graph-v1` is an optional, synchronous leaf extension. Its raw exports use
the `lcb_model_graph_` prefix and the same checked `bigint`/`uint64_t` pointer
convention. Each runtime variant ships `model-graph-v1.schema.json` and
`model-graph-v1.functions.d.ts`; the schema fingerprint is reported by the
module and recorded with the checked chat overlay in build provenance. Package
validation rejects absent, mixed, or inconsistent extension contracts. The base
C binding ABI is unchanged. A host must reject a missing/mismatched extension
before calling it, and serialize calls with the ordinary C/Embind boundary.

The extension owns vocabulary objects independently of a model, checks whether
an architecture name is registered, and applies existing CPU samplers to host
graph logits through reusable native candidate storage. The vocabulary owner
copies GGUF tokenizer and chat metadata; the metadata context can then be freed.
The borrowed vocabulary pointer and chat template objects must not outlive the
owner. Destroy chat/sampler objects before freeing the vocabulary. Free each
sampling workspace separately; its candidate storage is reset for every sample.
The chat constructor for an independent owner is a separate overload, not a
dummy model or context. Error strings are borrowed until the next extension
call on that thread. Null handles and count/width errors are rejected, but
arbitrary forged pointers are not sandboxed by this C boundary.

This is a deliberately unstable experimental contract. Breaking changes should
change the extension identifier/version/fingerprint and generated host types,
not add speculative compatibility shims. Older artifacts without the extension
remain usable for native models; they cannot run a host graph requiring it.
Neither graph definitions nor model scripts are evaluated by the core. Graph
interpretation, shape/resource checks, file policies, execution scheduling and
user consent stay in the host. New tokenizers, kernels and unsupported graph
operations still need an implementation update. Native model graph builders,
`llama_decode`, and existing kernels are unchanged by this extension.
