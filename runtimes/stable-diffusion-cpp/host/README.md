# Thin host bindings (ABI 2)

Import `attachCore`, `schema`, and `mountReadOnlyFile` from
`browser-inference-core/stable-diffusion-cpp/host`.
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

ABI 2 describes this wire convention, not compatibility between upstream API
revisions. A matching schema and Wasm can still require changes in the caller
when a function signature, record field, ownership rule, or field unit changes.

The image manifest's `capabilities.bf16WeightRetention` identifies builds exposing
`await core.api.sd_browser_bf16_retention_version()`, which returns `1` for this
browser policy. Check the source-bound capability and the generated API/version
before selecting `sd_ctx_params_t.webgpu_bf16_type = core.constant('SD_TYPE_BF16')`.
This explicit option retains original BF16 parameters for supported WebGPU weight
operations. The initializer still selects `SD_TYPE_F32` widening; `SD_TYPE_F16`
remains the caller's explicit narrowing choice. It does not change source files,
other parameter types, activation precision or the runtime profile. The version
reports compiled functionality, not successful execution on a device or proof
that a complete model fits. The separate model-IO capability bits remain `7`.
Package validation also requires the same ordered kernel, bounded upload,
bounded reader and explicit retention patch hashes in every source-bound profile
and variant; the version getter alone does not establish this capability.

Allocate records, call their upstream initializer, set fields, and invoke upstream
functions. `getField`/`setField` handle native scalar widths; use `fieldAddress`
for nested records and pointer-width DataView access for pointer out-parameters.
Re-read the memory view after an awaited native call; Wasm memory may have grown.

`generate_image(ctx, params, resultOut)` writes an owned
`sd_image_generation_result_t*` to caller-allocated pointer storage. Read `items`
and `count` from that result, then use `recordSize('sd_image_result_t')` to step
through the items and `fieldAddress(..., 'image')` for each embedded `sd_image_t`.
Each item owns its pixels and its UTF-8 JSON `generation_info`. Copy the data the
application needs before calling `free_sd_image_generation_result(result)`.
The result is independent of the context lifetime. Do not separately free its
items, pixels or metadata. A failed generation sets the output pointer to null.

`generate_video` and `upscale_video` return an owned
`sd_video_generation_result_t*`, released by `free_sd_video_generation_result`.
`upscale` and `adetail_image` use the image-result owner. `free_sd_images` still
exists for separately owned image arrays; it does not release a generation-result
owner. Call `free_sd_ctx` for contexts and `free` only for caller allocations.
Borrowed preview images must not be freed; copy their data before returning from
a callback.

`sd_tiling_params_t.tile_size_w` and `tile_size_h` are image-pixel dimensions for
both VAE encode and decode; zero selects the upstream default of 256 pixels.
The old `tile_size_x/y` names and latent-space units are no longer accepted.
`rel_size_w/h` values up to one are dimension fractions; larger positive values
specify target tile counts. The application owns migration of saved settings.

`module.addFunction` / `removeFunction` expose upstream callbacks. Use the native
pointer signature `p` (not a fixed `i`) for pointer arguments, e.g. log `vipp`,
progress `viifp`, generation progress `vpp`, preview `viipip`, graph evaluation
`ipip`. Log, legacy progress and preview registrations are module-global;
`sd_set_generation_progress_callback(ctx, callback, data)` is context-local.
Generation-progress records and their strings are borrowed only during the
callback; copy needed values synchronously. Its 64-bit counters use bigint, and
`total == 0` means an unknown total. Register and clear callbacks while the
context is idle, before removing their function-table entries. A callback must
be synchronous, catch its own errors, and must not start a second host-helper
operation while one is pending.

`sd_get_preview_info` returns a record by value in C, so the generated call is
`await core.api.sd_get_preview_info(preallocatedOutput)`. The helper's serialization
guard also applies to this call; do not invoke it through `core.api` from a pending
generation callback. Separate Wasm instances do not share callback state.
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

## Public GGML control

The generated schema includes supported public declarations from pinned `ggml.h`,
`ggml-backend.h`, `ggml-cpu.h`, `ggml-alloc.h` and `gguf.h`. The caller can create
contexts/tensors/graphs, choose a backend, allocate buffers, upload inputs, compute,
download results and release resources without adding runtime-specific probes.
Complete public records such as `ggml_init_params`, `ggml_cplan`, device properties,
tensor/type traits and `gguf_init_params` use compiled layouts. Enumerated constants
and selected numeric macros are exported. `schema.ggmlCoverage` and the exclusion
list describe the supported surface and explicit omissions.

Pointers, byte counts and dimension integers follow the generated schema: pointers
and `size_t`/`int64_t` use bigint on both Wasm widths. By-value records use caller-owned
storage; a returned record adds an output pointer as the first argument. All calls
remain serialized by the host helper. Native callbacks keep their original pinned
C signatures and must not re-enter a pending host-helper operation.

The caller owns context/graph metadata, backend and buffer lifetimes. Keep graph
inputs/outputs and callback table entries alive until computation settles; release
backend buffers before their backend/context owners. Registry loading and graph
mutation are explicit caller operations, not automatic generation instrumentation.

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

## Node ABI contract probe

An already compiled `test` variant can run the small Node-only ABI probe without
a browser or model files. From the runtime source directory, for example:

```sh
node --experimental-wasm-jspi ../../tests/runtimes/stable_diffusion_cpp/node-core-contract.mjs build/webgpu-wasm32-jspi/test direct 4
node ../../tests/runtimes/stable_diffusion_cpp/node-core-contract.mjs build/webgpu-wasm32-asyncify/test asyncify 4
```

Use a Node version with JSPI support for the first command. Pass `8` for a wasm64
build. The directory must contain `runtime/core.mjs`, `runtime/core.wasm` and its
matching `generated/schema.mjs`. The probe checks BF16 retention version/defaults,
record layouts, pointer width,
64-bit fields, by-value preview info, null-context generation results and snapshot
failure atomicity. It never queries devices, creates model contexts or computes
graphs, and it does not update browser-smoke provenance.
