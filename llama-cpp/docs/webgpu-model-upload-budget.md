# Bounded WebGPU model uploads

In this source tree, the normal build enables bounded synchronous model uploads
in every WebGPU profile through the shared source-preparation path. CPU profiles
retain the upstream loader. No additional build variant, workflow input or publication
route is required.

## Scope and accounting

Only the existing non-mmap, non-host synchronous fallback for a default WebGPU
device buffer is changed. Reads use at most an 8 MiB reusable vector with complete-block
chunk alignment. One `llama_webgpu_load_uploads` helper lives outside the tensor
loop for each `load_all_data` call. It owns one backend created from the actual
current tensor buffer's device, never a device inferred from `bufs[0]`.

Why bounded loading: reusable chunks bound explicit read staging, while a separate
charged budget limits queued loader work across tensors. Why not wait after every
write: batches amortize synchronization overhead. The current maximum is 8 MiB;
complete-block alignment may round it down and the final chunk can be smaller.
The 32 MiB budget applies to the sum of charged payloads, including variable tail
chunks, rather than a fixed number of writes. Browser transport thresholds are
version-specific implementation details, not a loader or memory-limit contract.

Before each upload it reserves `read_size + 24`. The reviewed WebGPU setter writes
at most the payload and two three-uint32 head/tail parameter blocks. The budget is
provisionally 32 MiB, and `charge > budget - pending` triggers synchronization
before enqueueing the next upload. Consequently four exact 8 MiB chunks do not fit
in one batch after auxiliary charges: the fourth begins a new batch. Oversized
reservations are rejected before addition, preventing size_t overflow. Rounding
full chunks down for Q4_K or Q8_0 alignment leaves enough room for four such
chunks plus their charges; smaller tails likewise change the number that fits.

The loader also considers a smaller aligned maximum of `32 MiB / 4 - 24` so
four charged full chunks fit in a batch. It uses that candidate only when the
tensor needs the same number of reads and uploads, and the projected final
`(wait count, pending charge)` is lexicographically no greater than with the
original maximum. The projection starts with the current pending charge for
the same device, or zero for a device switch. It is a constant-time integer
calculation that does not allocate, initialize a backend, read a file, or wait.
Exact multiples of the original maximum keep that maximum if reducing it would
add another read. Equal read counts alone are insufficient: a larger final tail
can otherwise increase waits when it meets earlier tensors' pending uploads.

This ordering is preserved by subsequent charged uploads and the final drain,
so choosing the candidate does not increase charged wait counts across tensors.
It does not guarantee lower wall time. Selection leaves the read, reservation,
and upload order unchanged, and each read vector is no larger than before.

Accounting spans tensor boundaries. A device change drains and releases the old
backend before acquiring the new one, so multiple queues do not each claim the
full budget. Normal completion drains the partial batch before final completion
reporting. The destructor drains on ordinary cancellation and C++ unwinding,
then frees the helper backend. Reservation happens before `tensor_set` so partial
submission followed by an ordinary exception is included in cleanup.

## What this does not guarantee

The budget is charged bytes for this loader path, not a hard memory bound for the
browser process, Wasm heap, GPU allocations, driver staging or other queue users.
Filling four chunks instead of three can increase the actual pending peak from
about 24 MiB to nearly 32 MiB while remaining within the same budget. An unchanged
cap does not imply unchanged peak memory: Wasm capacity, driver staging and
retained pools must be measured separately, including after model release.
It does not cap retained browser transport pools: alignment, fragmentation and
allocation history can affect retained capacity even after queue completion.
The backend synchronization uses the existing Wasm yield mechanism (JSPI or Asyncify,
according to the selected profile); this is not a new spin wait. Its upstream
completion and device-loss handling are unchanged. A resolved wait does not
prove upload success after device loss. A hung queue can stall both cancellation
and cleanup. JS aborts, traps, graceful OOM recovery and interruptible waits are
not covered by ordinary C++ RAII. No generic setter or inference path is changed.

## Source and validation

Reviewed upstream is 8642541b255bfb5dc39cd351a178be09013a55d6 (v0.8.0).
Exact source, API, file-reader, patch and generated-output identities are checked
by the existing preparation/provenance flow; input drift fails closed. The
backend overlay and normal build/publication route remain unchanged.

Run the focused tests with `LCB_TEST_LLAMA_SOURCE` pointing at that pristine
checkout, and optionally `LCB_TEST_DAWN_PACKAGE` at the pinned Dawn package:

- `python3 -m unittest discover -s llama-cpp/tests -p test_webgpu_model_upload_budget.py`
- `python3 -m unittest discover -s llama-cpp/tests -p test_webgpu_source_overlay.py`

The budget suite compiles the actual generated helper against deterministic CPU
queue stubs, asserts actual loader wiring, and syntax-checks the complete loader
against upstream headers. Integer-width fixtures exercise the same prepared
helper with 32-bit and 64-bit arithmetic, including independent sequential and
cycle-skipping reference states, size limits, pure device-switch selection,
exact-multiple fallback, cross-tensor counterexamples, and a positive adoption
case that prevents an always-original policy from passing. These integer-width
fixtures do not execute Wasm ABIs. Overlay tests configure real CMake targets. These
focused checks establish source/native contracts only. Browser model-load tests
are required to establish runtime behavior, peak memory, throughput, cancellation
and failure behavior; compilation alone does not establish those properties.
