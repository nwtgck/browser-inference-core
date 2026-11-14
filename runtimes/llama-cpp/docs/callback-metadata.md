# Synchronous metadata getters inside native evaluation callbacks

An evaluation callback needs synchronous metadata while native evaluation is in
progress. Awaiting a getter there would violate the native callback contract.
For native evaluation callbacks, the ABI keeps these reviewed public C getters
outside the JSPI suspending-export list:

- `ggml_op_desc`
- `ggml_backend_buffer_name`
- `ggml_backend_buffer_is_host`
- `ggml_backend_dev_count`
- `ggml_backend_dev_get`
- `ggml_backend_dev_name`
- `ggml_backend_dev_supports_op`

These are the original C exports. The generated ABI 3 schema identifies them in
`callbackMetadata`, whose metadata-contract version is `2`. Record adapters and
ordinary model operations do not acquire synchronous callback access. The
ordinary generated API remains asynchronous and serialized. Other reviewed
synchronous leaves, such as the file-backed-row handshake, have their own idle
state and lifetime requirements; their presence is not permission to call them
from evaluation callbacks.

## Calling the reviewed leaves

The host exports `createAbiCaller`. Use the same module and selected schema as
the active core, and the suspension mode of that profile:

```js
import { createAbiCaller } from 'browser-inference-core/llama-cpp/host';

const caller = createAbiCaller({
  module: core.module,
  schema: core.schema,
  suspension: 'jspi',
  gate: undefined,
});
// Inside the native callback; tensorPointer belongs to this invocation.
const descriptionPointer = caller.callSync({
  name: 'ggml_op_desc',
  args: [tensorPointer],
});
```

`callSync` permits only the schema's reviewed leaves, performs the target's
scalar/pointer conversion, and rejects a Promise result. Select the schema using
the actual module's ABI version and layout fingerprint; do not combine schema
and module files from different artifacts. See the [host API](../host/README.md)
for pointer conversion and UTF-8 reading.

## Scope and lifetime

The getters read operation descriptions, backend registry/device capability,
and destination-buffer metadata. They do not submit work, map GPU buffers, read
tensor values, create or free contexts, or implement model operations. Pointer
arguments must be live native objects belonging to the active module. A tensor
pointer is valid only while the evaluation callback owns it. Do not retain it or
use these leaves as a general reentrant API.

Buffer location and a device's `supports_op` result do not identify the actual
scheduler-selected execution backend, a CPU fallback reason, or GPU time.
Installing an evaluation callback can alter synchronization; isolate that run
from ordinary performance measurements. Exposing these leaves installs no
callback and changes no scheduling by itself.

## Compatibility and validation

`runtimes/llama-cpp/tools/bindings.ts` owns the reviewed name set. ABI generation
requires those declarations to exist and be direct scalar exports. The ordinary
`native/abi/metadata.cpp` translation unit checks their C signatures. Inspect the
upstream implementations at each update: a compatible signature cannot prove
that a function remains synchronous and non-submitting. Add only reviewed
metadata leaves, never a wildcard exemption.

The ABI generator tests under `tests/tools/abi/` cover the actual public-header
surface, export split, generated declarations and Wasm adapter behavior. Host
tests cover rejected synchronous calls and pointer conversion. Run the relevant
emitted Node tests as described in [verification](../../../docs/verification.md).
Source/native fixtures, tiny Wasm regressions, browser execution and real-model
performance are separate evidence; a fixture result does not establish the last
two. Consumers of older experimental artifacts must rebuild and validate a
matching module/schema pair rather than infer this surface from a legacy alias.
