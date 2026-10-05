# Thin host bindings (ABI 2)

Import `attachCore`, `schema`, and `mountReadOnlyFile` from this directory.
Instantiate a chosen `profiles/<profile>/<variant>/core.mjs` factory yourself,
then call `attachCore(module, schema, { suspension: 'asyncify' })` for Asyncify
or `{ suspension: 'direct' }` for JSPI. Do not invoke suspending raw exports
without the appropriate Promise bridge.

The helpers have no model singleton, Worker, catalog, download, storage, thread,
sampler, memory-budget, or generation policy. Functions and record
layouts in the pinned public SD header are generated; the compiled schema fingerprint
must match. `api` calls normalize pointers and 64-bit integers to bigint, serialize
native access, and preserve upstream defaults and results. Record layouts are
queried from the compiled module, never guessed from JavaScript.

Allocate records, call their upstream initializer, set fields, and invoke upstream
functions. `getField`/`setField` handle native scalar widths; use `fieldAddress`
for nested records and pointer-width DataView access for pointer out-parameters.
Call `free_sd_images` for image arrays (including every channel buffer), `free_sd_ctx`
for contexts, and `free` only for caller allocations. Borrowed strings/preview
images must not be freed; copy preview data before returning from a callback.

`module.addFunction` / `removeFunction` expose upstream callbacks. Use the native
pointer signature `p` (not a fixed `i`) for pointer arguments, e.g. log `vipp`,
progress `viifp`, preview `viipip`, graph evaluation `ipip`. Callback registrations
are upstream module-global; separate instances do not share them. A callback must
be synchronous, catch its own errors, and must not start a second native operation
while one is pending. Clear native registrations before removing table entries.
A Worker receiving no events while native code runs cannot accept a cancellation
message; the caller can terminate that Worker, or explicitly design a shared
control mechanism. The core does not choose for the caller.

## Large files

`mountReadOnlyFile(core, path, { size, read(destination, offset) }, { maxChunkBytes })`
accepts safe-integer byte positions independently of Wasm memory size. It caps each
read, retries short reads, rejects async reads and mmap, and never allocates the
whole model. The caller owns file handles, locking, consistency, storage, and
unmount timing. Unmount only after every native context using the file is freed.
A FileReaderSync + Blob.slice adapter, OPFS sync handle, or another synchronous
range source can implement `read` without changing this package.

The browser image loader preserves 64-bit GGUF metadata offsets and does not use
ggml's aggregate-size_t parser. Files above 4 GiB need not be sharded. This is not
a promise that all large models fit: each tensor, allocation, working graph,
WebGPU binding, and device budget must still fit. wasm64 enlarges the linear
address space, not GPU limits. Models requiring independent components still need
one unsharded GGUF per component. Accepted types follow the selected upstream ggml;
FP8 and custom SD fork types are not supplied by this compatibility build.

## On-demand observation

The public SD functions include context image/video/ControlNet capabilities,
model version, default sampler/scheduler, enum names, system information and
`sd_list_devices`. Query them through `core.api` while the core is idle. The
system/device calls can initialize the backend registry on first use; they are
not a lightweight polling protocol. `core.busy` describes this host binding's
pending native call, and `module.HEAPU8.byteLength` is allocated Wasm linear-memory
capacity, not live heap use or GPU memory.

`sd_ctx_get_runtime_info`, `sd_ctx_get_memory_info` and `sd_ctx_get_params` take a
context, caller output record and its byte size. Allocate with `allocRecord` and
pass `BigInt(core.recordSize(recordName))`. They return 1 on success and 0 for
null, undersized or busy-context requests; failed requests do not touch output.
The two info records include `struct_size` and `version`. The schema fingerprint
and compiled record layout must match the runtime; do not hard-code offsets.
Never call these APIs from a generation callback or overlap a native operation.

Runtime information reports resolved thread count and existing engine flags.
Configuration is the engine's saved request (including later ControlNet changes),
not effective tensor placement. Its pointers, including strings and embeddings,
are borrowed and read-only until the next context mutation or destruction. Copy
needed values immediately; free only the caller's output record.

Memory information does work only when requested and does not synchronize GPUs,
read tensors back, refresh caches or enable continuous instrumentation. It reports
three distinct categories: unique registered tensor logical bytes; manager-held
buffer-handle sizes split by host/non-host; and runners' last published retained
workspace/cache reports split by CPU/non-CPU/unknown backend device. uint64 byte
sums remain bigint on wasm32; `saturated` marks overflow. Do not add these columns
into a total. Aliased handles can overlap, runtime reports may be stale, and bare
mappings, file buffers, other allocators and driver costs are excluded. These are
not physical VRAM, free GPU memory or peak usage. When no snapshot is requested,
the getters add no work to generation; the context initializer additionally owns
the previously omitted audio-encoder path with one string copy. This design property is
separate from an actual performance measurement.

## Opt-in graph-stage logs

Register a log callback, then call `await core.api.sd_set_graph_diagnostics(1)`
while the runtime is idle before the request to inspect. In a `finally` block,
after that native request has settled, call
`await core.api.sd_set_graph_diagnostics(0)`. These module-global diagnostics
start disabled; callbacks must not re-enter a native operation. Each enable
call opens a new observation window, including when already enabled. Each
runner logs only its first attempt after graph-cut plan resolution in that
window. Enable again before another request or a changed graph/resolution.
This bounds repeated denoiser-step logging; failures before plan resolution
remain outside these markers.

Debug messages use `graph-stage-v1 runner=... stage=... event=begin|end|failed`.
Receiver timestamps can locate a long measurement, weight preparation,
workspace allocation, input copy or execution interval. `workspace-reprepare`
covers a possible remeasurement and capacity checks; `workspace-allocate`
covers the actual workspace allocation call. `reserve-including-sync` includes
the scheduler's existing synchronization, graph splitting and allocator size
estimation, without distinguishing those internal operations. A last `begin`
without an `end` locates an unfinished interval, not proof of a deadlock.

No native timers, additional tensor walks, GPU synchronization or readbacks are
added. With diagnostics disabled, boundary helpers only check the gate/null
label and perform no formatting or callback work. Enabled logging has callback
overhead and is not a performance benchmark. This API requires an artifact built
with the graph-stage patch; older ABI 2 artifacts do not expose it.

## Public GGML queries

The generated schema also includes explicitly selected functions from pinned
`ggml.h`, `ggml-backend.h`, `ggml-cpu.h` and `gguf.h`: version/type/operation and
CPU capability queries, tensor/buffer metadata, backend/device registry and
capabilities, and GGUF construction/metadata reading. Tagged public records such
as device properties/caps, tensor/type traits and `gguf_init_params` use compiled
layouts too. Enumerated constants and selected numeric macros are exported.
`schema.ggmlCoverage` and the exclusion list identify unselected public declarations;
this is not a blanket export of every graph-building/backend mutation function.
These are raw source-bound APIs, not checked high-level parsers.

Query devices explicitly while idle. Registry loading and property queries may
initialize native backend state; device memory values retain their backend's
meaning and must not be advertised as accurate free GPU memory. They do not
start an automatic sampler or add generation instrumentation.

`gguf_init_from_file`, `gguf_init_from_buffer` and `gguf_init_from_callback` accept
a caller-allocated `gguf_init_params` record through the by-value record bridge.
The caller owns the source, any returned GGML context and synchronous callback
lifetime. Use `gguf_free` for the GGUF context; strings, arrays and tensor-shape
pointers returned by getters are borrowed until that context is freed. Before a
typed `gguf_get_val_*`/array getter, check the key's type and bounds: mismatches
can trigger native abort. The generic GGUF API has its upstream allocation and
address-width limits; it does not replace the large-file model-reader contract
above. `ggml_set_abort_callback` is module-global and returns the previous
callback pointer; manage JS callback-table lifetime yourself and never throw
across the native boundary.
