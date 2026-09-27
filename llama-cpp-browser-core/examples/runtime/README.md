# Example runtime

These files are always shipped with the artifact as tested reference code for
application TypeScript implementations. Importing the example is optional;
changing application host code does not require rebuilding Wasm.

```js
import { createCore } from 'llama-cpp-browser-core/examples/runtime';
const core = await createCore({ profile: 'cpu-wasm32' });
await core.api.llama_backend_init();
console.log(core.readUtf8(await core.api.llama_version()));
await core.api.llama_backend_free();
```

`index.mjs` loads a profile and variant; `bindings.mjs` demonstrates generated C schema checks,
record access, memory helpers, and serialized Promise calls; `read-only-file.mjs`
adapts synchronous range reads to Emscripten's filesystem. None owns a model,
conversation, tool loop, or worker. CPU Chromium tests use these shipped files for
synthetic model loading/decode/state operations; host tests cover the helpers.

The default `variant: 'browser'` omits Node.js support and Emscripten assertions.
Use `createCore({ profile: 'cpu-wasm32', variant: 'test' })` for Node.js tests;
that variant retains both. The loader selects JavaScript and Wasm from the same
`profiles/<profile>/<variant>/` directory. It does not detect an environment and
switch variants automatically. Keep these files together when copying artifacts.

`webgpu-wasm32-asyncify` provides WebGPU without memory64 or JSPI. The loader
selects its `ccall(..., { async: true })` completion path automatically. When
using `attachCore` directly, pass `{ suspension: 'asyncify' }` as its third argument.
Keep native calls serialized until the returned Promise settles; a raw Asyncify
export can return while the GPU operation is still pending. Callback functions
remain synchronous. The 32-bit profile's linear-memory ceiling is 4 GiB; actual
model capacity also depends on device limits and native staging allocations.

`webgpu-wasm32-jspi` uses the same 4 GiB linear-memory ceiling without requiring
memory64. It requires JSPI instead of Asyncify; its Promise exports use the same
default `attachCore` path as `webgpu-wasm64-jspi`. Normalized `lcb_` pointer and
64-bit arguments remain `bigint` on both pointer widths. The application selects
an explicit profile after checking capabilities; the example performs no fallback.

Alternatively import `llama-cpp-browser-core/profiles/cpu-wasm32/browser/core.mjs` directly.
Its Embind chat types and C exports work without these helpers. See the artifact's
[native binding guide](../../chat-and-multimodal.md) for ownership and limitations.
