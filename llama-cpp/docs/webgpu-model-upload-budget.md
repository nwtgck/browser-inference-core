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

Accounting spans tensor boundaries. A device change drains and releases the old
backend before acquiring the new one, so multiple queues do not each claim the
full budget. Normal completion drains the partial batch before final completion
reporting. The destructor drains on ordinary cancellation and C++ unwinding,
then frees the helper backend. Reservation happens before `tensor_set` so partial
submission followed by an ordinary exception is included in cleanup.

## What this does not guarantee

The budget is charged bytes for this loader path, not a hard memory bound for the
browser process, Wasm heap, GPU allocations, driver staging or other queue users.
It does not cap retained browser transport pools: alignment, fragmentation and
allocation history can affect retained capacity even after queue completion.
The backend synchronization uses the existing Wasm yield mechanism (JSPI or Asyncify,
according to the selected profile); this is not a new spin wait. Its upstream
completion and device-loss handling are unchanged. A resolved wait does not
prove upload success after device loss. A hung queue can stall both cancellation
and cleanup. JS aborts, traps, graceful OOM recovery and interruptible waits are
not covered by ordinary C++ RAII. No generic setter or inference path is changed.

## Source and validation

Reviewed upstream is d81235049384534c167caea52b85a694f6103d14 (b11429).
Exact source, API, file-reader, patch and generated-output identities are checked
by the existing preparation/provenance flow; input drift fails closed. The
backend overlay and normal build/publication route remain unchanged.

Run the focused tests with `LCB_TEST_LLAMA_SOURCE` pointing at that pristine
checkout, and optionally `LCB_TEST_DAWN_PACKAGE` at the pinned Dawn package:

- `python3 -m unittest discover -s llama-cpp/tests -p test_webgpu_model_upload_budget.py`
- `python3 -m unittest discover -s llama-cpp/tests -p test_webgpu_source_overlay.py`

The budget suite compiles the actual generated helper against deterministic CPU
queue stubs, asserts actual loader wiring, and syntax-checks the complete loader
against upstream headers. Overlay tests configure real CMake targets. These
focused checks establish source/native contracts only. Browser model-load tests
are required to establish runtime behavior, peak memory, throughput, cancellation
and failure behavior; compilation alone does not establish those properties.
