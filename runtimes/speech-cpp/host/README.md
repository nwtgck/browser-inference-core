# Thin speech host bindings (ABI 3)

Import `attachCore`, `schema`, and `mountReadOnlyFile` from
`browser-inference-core/speech-cpp/host`. Instantiate the selected
`profiles/webgpu-wasm32-jspi/browser/core.mjs` factory, then call
`attachCore(module, schema, { suspension: 'jspi' })`. The browser profile targets
desktop Chrome with WebGPU, Wasm32 SIMD, and JavaScript Promise Integration.
It does not require memory64.

Type-only `CoreFactory` and `FactoryByProfile` exports derive from the packaged
module factories, so dynamic loaders can use their actual option and result
types without importing a particular profile path.

The generated API exposes the pinned `speech.h` C functions directly. Handles,
pointer out-parameters, pointer-sized counts, and 64-bit integers use bigint in
the public host API. Raw Wasm32 pointer/size_t exports and callbacks use Number,
while uint64_t/int64_t values remain bigint. `api` calls return
Promises and serialize native access. The generated schema and compiled
fingerprint must match. ABI 3 describes the wire convention; upstream speech API
compatibility is separately reported by `speech_api_version_major/minor()`.

Load with `speech_load_params_new`, explicit load setters, and
`speech_model_load`. Request options keep upstream validation and defaults;
`speech_model_info_*` reports the model task and supported options. Allocate
pointer out-parameters with `core.alloc(core.pointerBytes)` and read them using a
pointer-width DataView. Refresh views after awaited calls because Wasm memory
may have grown. Release each handle with its matching `speech_*_free` function;
use `core.free` only for caller allocations. Free requests and conversations
before their model.

Failures return a speech status. Read `speech_last_error` and
`speech_last_error_option` before another library call replaces them. The error
strings are borrowed. The host does not convert statuses into exceptions.

## Full-duplex dialogue

For a dialogue model, create a request, set its character/voice/sampling options,
and call `speech_dialogue_start`. Give microphone PCM to
`speech_dialogue_push` as it arrives. Start emits the first output frame before
returning; each complete input frame produces the next output frame. Obtain the
model's sample rate and frame length through `speech_model_info_sample_rate` and
`speech_model_info_frame_samples`. Upstream resamples input PCM from the rate
passed to start. `speech_dialogue_end` drops input shorter than a complete frame.

Register a synchronous frame callback with `module.addFunction(callback, 'ipppp')`
and pass its table pointer to `speech_dialogue_start`. The callback receives
`samples`, `n_samples`, UTF-8 `text`, and caller data. Copy needed audio/text
before returning; both are borrowed for that callback only. Return zero to
continue or nonzero to stop. Use the native pointer signature `p` for pointers
and `size_t`; Wasm32 callbacks receive native Number values. Normalize borrowed
pointers with `fromNativePointer` before using them as public handles. Callback table entries must stay
registered until their conversation is freed. Callbacks must catch their own
errors and must not invoke the library again with the same model.

The same public API includes synthesis, transcription, detection, diarization,
and enhancement; available operations depend on the loaded model's task.
Audio callbacks for synthesis/enhancement use signature `ippp`. Native
`speech_request_cancel` is a synchronous leaf for request cancellation. The
generated `core.api.speech_request_cancel` still uses the host's asynchronous
serialization guard and rejects while another native call is pending. To cancel
a suspended request, create a separate ABI caller while idle, then use its
declared synchronous leaf:

```js
import { createAbiCaller } from 'browser-inference-core/speech-cpp/host';

const cancellation = createAbiCaller({
  module: core.module,
  schema: core.schema,
  suspension: 'jspi',
  gate: undefined,
});
// request is the still-live speech_request handle for the pending operation.
cancellation.callSync({ name: 'speech_request_cancel', args: [request] });
```

The raw export `core.module._speech_request_cancel` is also synchronous, but
requires a native Wasm32 Number pointer. Use
`toNativePointer({ value: request, pointerBytes: core.pointerBytes })` to convert
a public bigint handle with range checking. Only explicitly declared synchronous leaves may bypass the guard;
keep the request alive until the pending call settles. Dialogue
stops via the frame callback or by ending/freeing it between calls. A Worker
running native code cannot receive application messages until it yields; the
caller owns scheduling, audio playback/capture, queues, and interruption policy.

## Model files

`mountReadOnlyFile(core, path, { size, read(destination, offset) }, { maxChunkBytes })`
mounts a synchronous random-access source in Emscripten's FS. Reads are bounded,
short reads are retried, async reads and mmap are rejected. This does not copy the
whole model into Wasm memory. The upstream loader reads metadata and uploads
weights in bounded pieces, using 64-bit file offsets. FileReaderSync with
Blob.slice or an OPFS sync handle can implement the caller-owned source.

Keep the source consistent and mounted until all native users are freed. The
core does not own downloads, persistence, locks, model selection, a Worker, or
device/memory budgets. The 4 GiB linear-memory ceiling does not enlarge WebGPU
device limits, and API
availability does not establish that a model fits or meets real-time latency.
Device memory queries retain the pinned backend's meaning; its WebGPU result
currently describes a buffer-size limit, not measured free or total GPU memory.
