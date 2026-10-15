# File-backed Qwen4Exp rows (extension version 1)

This opt-in extension replaces only Qwen4Exp's per-layer embedding (PLE) table
with positioned file reads. It is not a general model streaming mode or a
promise that the remaining weights fit in browser or device memory.

## Activation and lifetime

`LCB_FILE_BACKED_PLE=ON` is selected by `scripts/build.py` only for memory64
profiles. The build-tree patch is reviewed against the inputs in
`tests/file-backed-rows/inputs.json`; changes to those inputs stop preparation.
Native Linux checks may enable it explicitly. The upstream checkout stays clean.

The actual module exposes two synchronous metadata functions, never JSPI
(JavaScript Promise Integration) operations:

* `_lcb_file_backed_rows_version()`: 1 when compiled in, 0 when disabled; older
  artifacts may omit it entirely.
* `_lcb_model_file_backed_shard(modelPointer)`: -1 for a resident model, otherwise
  the zero-based index in the **exact split-path array used to load the model**.
  For single-file models this is 0. The argument must identify a live model and
  no other native operation may be in flight.

Callers opt in with `load_mode=LLAMA_LOAD_MODE_NONE` and
`lazy_mode=LLAMA_LAZY_MODE_ON`. Only Qwen4Exp uses the positioned reader.
Other architectures keep eager loading for that combination; this deliberately
prevents Gemma4's mmap-based lazy table from being enabled on the same mounts.
Explicit mmap modes and AUTO retain upstream policy. Old callers using OFF are
unchanged. Absence of the extension is not grounds to send ON to an old runtime.

After successful load, query the retained shard **before** removing mounts or
closing storage. Keep that shard's read source, mount, and cache alive until
all contexts and the model have been destroyed. Release unrelated shards after
load. During a failed or cancelled load, free any returned model first and then
remove all mounts and close sources. Never attach the initial request's aborted
signal to a source that outlives that request. Sources must remain immutable.

The reader owns a duplicated descriptor, not the loader's `llama_file`. The
JavaScript filesystem `stream_ops.dup` hook must increment open ownership:
Emscripten's `dupStream` does not call `open` on the duplicate. `close` releases
one owner; unlink/removal remains forbidden while any owner is live.

## Algorithm and limits

The existing n-gram hashing, predecessor/EOS handling, head order, graph
reshape, projections and remaining Qwen4Exp operations are unchanged.
Each microbatch gathers the requested rows, dequantizes with ggml's own type
traits, and supplies an F32 input tensor. Sorting changes only read order;
duplicate indices are read once per microbatch and restored to their original
positions. No thread pool, global cache, mutable seek cursor or full-table mmap
is used. Short reads and EINTR are handled; invalid indices, premature EOF,
unsupported row types, range overflow and descriptor failures throw. No backend
input is published until the entire gather succeeds.

The 64 MiB limit applies separately to packed rows and the F32 staging buffer,
not to total process memory. A gather has at most 1,048,576 indices, plus an
ordering vector and a backend input copy. Cached graphs and multiple contexts
can retain additional buffers. For the affected IQ4_NL table, a 160-element row
is 90 bytes and 16 rows produce 10,240 F32 bytes per token. Prefill scales with
the microbatch token count; it does not read all table rows.

`check_tensors=true` is rejected for this reader: partial validation must not
silently stand in for a full integrity scan. Model export is also rejected
before creating a destination, because the resident tensor list omits the table.
Quantizing the original GGUF files is a separate operation. Model size and
parameter queries still include the backing table, so they are not resident-memory
measurements. Resident tensor enumeration omits it. The source shard sizes and
GGUF metadata remain the file-size authority.

The only new native error line forwarded by the consumer design is a fixed
`qwen4exp_file_rows: gather failed` marker. Source paths, row indices, token IDs,
raw stderr, and arbitrary exception strings are not a telemetry contract.
Unexpected native exceptions may require destroying the worker; recovery of an
arbitrary failed context is not guaranteed.

## Build coherence and provenance

Changing a derived class in `models.h` requires every model translation unit
and `llama-model.cpp` to use the same definition. The preparer copies the source
tree, applies the patch there, and CMake selects those model units plus the
model-save entry point. Include precedence is explicit, including unity builds.
The independent WebGPU loader overlay remains selected rather than overwritten.
Provenance records changed compiled files and a compact digest of the coherent
source set. It does not publish a fabricated upstream commit or modify a gitlink.

## Local checks

From the repository's `llama-cpp` directory, with a separately available pinned
upstream checkout (no download of the large model is needed):

```sh
cmake -S . -B build/file-rows-native -DLCB_NATIVE_CHECK=ON \
  -DLCB_FILE_BACKED_PLE=ON -DLCB_LLAMA_SOURCE=/absolute/path/to/llama.cpp \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/file-rows-native --target core -j2
cmake -S tests/file-backed-rows -B build/file-rows-tests \
  -DLCB_LLAMA_SOURCE=/absolute/path/to/llama.cpp \
  -DLCB_NATIVE_BUILD="$PWD/build/file-rows-native" -DCMAKE_BUILD_TYPE=Release
cmake --build build/file-rows-tests -j2
ctest --test-dir build/file-rows-tests --output-on-failure
LCB_TEST_LLAMA_SOURCE=/absolute/path/to/llama.cpp \
  python3 -m unittest discover -s tests -p 'test_file_backed_rows*.py'
node --test tests/read-only-file.test.mjs
```

The native fixture compares actual eager/file-backed logits, metadata-only
first of three shards, IQ4_NL rows, EOS, microbatches, graph reuse, mixed inputs,
state restoration and a controlled read-failure recovery after clearing state.
The reader fixture compares actual ggml get_rows output at offsets above 4 GiB.
These checks do not establish browser WebGPU behavior, target-device speed,
memory usage for the full 90 GB model, or uninterrupted same-context recovery
following device loss. Those are separate artifact and hardware acceptance gates.

## Bounded packed-row reuse

The model-owned reader now retains quantized rows across gathers. The cache is
four-way set associative, uses per-set least-recently-used replacement, and
compares the complete row ID on every hit. It does not change n-gram hashing,
row order, dequantization, or the logits/sampling algorithm. Different models
never share cache entries. Context replacement and state restore may reuse
immutable model rows without restoring any token-dependent cache state.

Payload **and** entry metadata together are capped at 8 MiB per reader. Allocation
is lazy and scales down for small tables. For the 160-column IQ4_NL table this
selects 65,536 slots: a 96-byte aligned packed row and 8-byte entry, totaling
6.5 MiB. Rows stay quantized rather than retaining 640-byte dequantized vectors.
There is no full-table allocation, unbounded growth, or asynchronous prefetch.
Scratch is allocated only after a miss; a fully warm or empty gather has no
packed staging allocation. The existing 64 MiB output-staging limit remains.

An optional cache allocation failure leaves the uncached reader operational and
is not retried on every token. Misses read into separate scratch and publish an
entry only after the complete read succeeds. A failed miss cannot corrupt the
victim. The model cache is protected by a mutex for native callers using multiple
contexts; browser calls stay on the existing serialized worker lane.

Each nonempty gather checks the retained descriptor's size. A size change or an
inaccessible descriptor invalidates all entries and fails that gather. On the
browser mount this is metadata-only, not an extra source-data read. The backing
model is still required to be immutable: same-size in-place mutation is not
supported, just as already-loaded resident weights do not follow file changes.

This is a locality optimization, not a universal speedup. Repeated text, prefix
reevaluation and retry can save source reads. An all-new stream still reads each
row and adds hashing, cache insertion and one metadata query per gather. Native
reader microbenchmarks must report that cold-path cost separately; they do not
establish a browser/model tokens-per-second improvement. The constructor's
optional cache budget (zero disables it) is available to native regression
harnesses; it is not a new exported runtime API.
