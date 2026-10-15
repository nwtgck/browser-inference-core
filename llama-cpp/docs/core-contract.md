# Low-level API and ownership

## Boundary

The build publishes llama.cpp, selected backends, generated C bindings, and Embind registrations for upstream C++ chat types. It does not own model catalogs, downloads, storage paths, chat sessions, generation loops, UI, or worker protocols. `examples/runtime/` is always shipped as tested reference code for application host implementations. Its use is optional; changing application host code does not require rebuilding Wasm.

The `lcb_` prefix identifies normalized C bindings while retaining upstream function names. There is no application-specific JSON command dispatcher. Deprecated, variadic, and unsupported video declarations are listed with exclusion reasons in the C schema. The complete GGML graph-building API remains outside the public surface. See [native chat and multimodal bindings](chat-and-multimodal.md) for the separate Embind surface and generated `mtmd` APIs. The conventions below describe the C bindings and optional reference wrapper.

## Calling convention

| Native type | Normalized binding | JavaScript type |
|---|---|---|
| Object, buffer, or function pointer | `uint64_t`, checked against the target pointer width | `bigint`; null is `0n` |
| `size_t` or `uint64_t` | `uint64_t`; conversion to `size_t` checks address width | `bigint` |
| `int64_t` | `int64_t` | `bigint` |
| 32-bit integer, enum, or bool | 32-bit integer | `number`; bool is 0 or 1 |
| float or double | Original floating-point type | `number` |
| Structure passed by value | Pointer to the structure; dereferenced by the binding | `bigint` |
| Structure returned by value | Additional first argument pointing to return storage | Promise resolving to `void` |

For example, `llama_model_default_params()` becomes:

```js
const params = core.allocRecord('llama_model_params');
await core.api.llama_model_default_params(params);
// Use the parameters while their referenced resources remain valid.
core.free(params);
```

The native core reports structure sizes, alignments, and field offsets. Do not treat a C++ structure as a JavaScript object or reuse offsets from another build. Types and schemas are generated together; a mismatched schema fingerprint is rejected when attaching the core.

Every reference `core.api` function returns a Promise, including CPU functions. This gives JSPI and Asyncify WebGPU calls the same interface; it does not move synchronous CPU work to another thread. Overlapping calls through that wrapper are rejected. Applications provide sequencing across all C and Embind calls; direct native calls bypass the reference wrapper's checks.

Both `webgpu-wasm32-jspi` and `webgpu-wasm64-jspi` expose JSPI Promise exports
through the default reference wrapper. The wasm32 variant needs no memory64 and
has a 4 GiB linear-memory ceiling. Its normalized pointer and 64-bit arguments
still use `bigint`; pointer width does not select a different binding ABI.

With `webgpu-wasm32-asyncify`, the reference loader uses Emscripten's
`ccall(name, returnType, argumentTypes, arguments, { async: true })`. Raw Asyncify
exports can return before a GPU operation finishes; merely awaiting that return
is insufficient. When attaching a module yourself, pass
`attachCore(module, schema, { suspension: 'asyncify' })`. Normalized `lcb_` pointer,
record, and 64-bit arguments remain `bigint` even with 32-bit Wasm memory. Raw
upstream exports instead retain their native ABI. Synchronous layout/allocation
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
