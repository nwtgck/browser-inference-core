# llama-cpp-browser-core

This branch distributes a prebuilt runtime. Source code and build instructions are on the source branches of the same repository. Installing this package does not install Emscripten or compile C/C++.

```sh
npm install github:nwtgck/browser-inference-core#ARTIFACT_COMMIT_SHA
```

Replace `ARTIFACT_COMMIT_SHA` with the complete commit hash containing the chosen runtime artifacts, not a source commit. No npm registry publication is required.

```js
import { createCore } from 'llama-cpp-browser-core/examples/runtime';

const core = await createCore({ profile: 'cpu-wasm32' });
await core.api.llama_backend_init();
const version = await core.api.llama_version();
console.log(core.readUtf8(version));
await core.api.llama_backend_free();
```

The example defaults to `variant: 'browser'` (`ASSERTIONS=0`, `ENVIRONMENT=web,worker`). For Node.js tests use `createCore({ profile: 'cpu-wasm32', variant: 'test' })`; this variant retains assertions and Node.js support. Direct imports use `llama-cpp-browser-core/profiles/cpu-wasm32/browser/core.mjs` or the matching `test/` path. Variants never switch automatically based on the execution environment.

The dynamically imported modules and their neighboring Wasm assets must be served together. A bundler may not discover them automatically. After copying `profiles/` into a public directory, set `baseURL`, for example `new URL('/runtime/profiles/', location.origin)`. This is an example path, not a required deployment layout. Do not mix generated JavaScript and Wasm from different profiles, variants, or artifact commits.

Generated-module types are in each `profiles/<profile>/<variant>/core.d.ts`; normalized C function types are in `api/functions.d.ts`, with layout identifiers in `api/schema.json`. The native core reports actual memory layouts. Manifest format 2 records build settings and performed checks under `profiles[profile].variants[variant]`, plus source commits and all payload hashes. Passing a test-variant check does not establish that the browser variant passed it. The `lcb_` C bindings normalize pointers and sizes to JavaScript `bigint` and replace structure return values with explicit return-storage pointers.

This is not a high-level chat library. Applications own files, workers, generation loops, conversations, cancellation, and data formats. `mountReadOnlyFile` is an optional adapter for synchronous range reads; it does not impose a storage backend.

`examples/runtime/` is always shipped as tested reference code for application
TypeScript implementations; importing it is optional. Applications may import a
profile's generated `core.mjs` directly for Embind and C exports.
See [native chat and multimodal bindings](chat-and-multimodal.md). The public
bindings may change with the pinned upstream version.

All profiles are single-threaded. `webgpu-wasm64-jspi`, `webgpu-wasm32-jspi`, and `webgpu-wasm32-asyncify` are experimental and require device-specific testing. Both wasm32 profiles have a 4 GiB linear-memory ceiling and do not require memory64. `webgpu-wasm32-jspi` uses JSPI without Asyncify; its native Promise exports work with the example loader's default completion path. The Asyncify profile does not require JSPI; use its example loader or `ccall(..., { async: true })` to await suspended native calls. Feature-detect WebGPU and Wasm support in the application; a profile does not certify any browser or device. A successful build, a synthetic-model smoke test, and support for a large production model are separate claims; consult the manifest for the checks actually performed.

The upstream GPU loader can allocate a tensor-sized staging buffer. This runtime has not replaced that path with bounded staging. Limiting individual file reads does not bound every allocation in the loader.

Project code is MIT-licensed. Upstream and toolchain notices are under `licenses/`.
Original headers retained as `.txt` under `licenses/embedded/` preserve embedded
notices verbatim; they are not build inputs. Model weights are not included.
