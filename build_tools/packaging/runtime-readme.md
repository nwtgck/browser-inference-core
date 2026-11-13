# Browser Inference Core

This package contains independently built llama.cpp and stable-diffusion.cpp
browser runtimes, generated declarations, schemas, license notices and build
provenance. Applications choose a profile and create each native module separately.

## Import paths

| Public path | Contents |
| --- | --- |
| `browser-inference-core/llama-cpp/host` | Typed llama module creation, schema attachment and read-only file mounting |
| `browser-inference-core/stable-diffusion-cpp/host` | Typed image module attachment and read-only file mounting |
| `browser-inference-core/<runtime>/api/functions` | Generated function declarations; use a type-only import |
| `browser-inference-core/<runtime>/api/schema.mjs` | Runtime-owned binding schema |
| `browser-inference-core/<runtime>/profiles/<profile>/<variant>/core.mjs` | Module factory and the declarations for that exact compiled variant |
| `browser-inference-core/<runtime>/profiles/<profile>/<variant>/core.wasm` | Matching WebAssembly binary |
| `browser-inference-core/manifest.json` | Complete package identities and links to each runtime manifest |

Replace `<runtime>` with `llama-cpp` or `stable-diffusion-cpp`. Select an actual
profile listed in its manifest and either the `browser` or `test` variant.
Keep the module, WebAssembly binary and schema from the same artifact commit.

## llama.cpp

```js
import { createCore, mountReadOnlyFile } from 'browser-inference-core/llama-cpp/host';

const core = await createCore({ profile: 'cpu-wasm32', variant: 'browser' });
```

`createCore` retains the native module type for the selected profile and selects
the appropriate suspension mechanism. `mountReadOnlyFile` attaches caller-owned
synchronous random-access storage. The application supplies the model, serializes
native operations, and releases native handles before unmounting their sources.

## stable-diffusion.cpp

```js
import createModule from 'browser-inference-core/stable-diffusion-cpp/profiles/webgpu-wasm32-jspi/browser/core.mjs';
import { attachCore, schema } from 'browser-inference-core/stable-diffusion-cpp/host';

const module = await createModule();
const core = attachCore(module, schema, { suspension: 'direct' });
```

Choose `suspension: 'asyncify'` when attaching an Asyncify module; the host waits
for the Emscripten completion contract. A JavaScript Promise Integration (JSPI)
profile uses `direct`. Browser support and an appropriate WebGPU device remain
requirements of the selected profile.

The application owns model files, generation parameters, context lifetime,
callbacks, cancellation, output conversion and Worker messages. Native modules
and upstream version pins remain independent between the two runtimes.

## Artifact layout and provenance

The format-4 root manifest covers the complete installed tree. Physical runtime
files live under `runtimes/llama-cpp-browser-core/` and `runtimes/stable-diffusion-cpp-browser-core/`; each
retains its own format-2 manifest and validation scope. Use the public import
paths above when consuming modules and types.

The source repository and build tools are maintained separately from this runtime
package. Pin the exact artifact commit emitted by the source repository's successful
publication workflow. Compilation, synthetic browser checks and real-model
inference are distinct evidence; consult the recorded validation fields.
