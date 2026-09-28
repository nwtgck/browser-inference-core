# Thin host bindings (ABI 2)

Import `attachCore`, `schema`, and `mountReadOnlyFile` from this directory.
Instantiate a chosen `profiles/<profile>/<variant>/core.mjs` factory yourself,
then call `attachCore(module, schema, { suspension: 'asyncify' })` for Asyncify
or `{ suspension: 'direct' }` for JSPI. Do not invoke suspending raw exports
without the appropriate Promise bridge.

The helpers have no model singleton, Worker, catalog, download, storage, thread,
sampler, memory-budget, or generation policy. All 63 functions and 18 record
layouts in the pinned public header are generated; the compiled schema fingerprint
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
