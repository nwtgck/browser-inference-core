# llama-cpp-browser-core

## Upstream compatibility policy

Keep this core close to upstream. Downstream patches are a last resort and require
explicit user approval for each new or expanded scope; a feature request alone
is not that approval. See [the repository instructions](AGENTS.md) and the
[exception policy and register](upstream-patches-only-as-a-last-resort-with-explicit-user-approval/README.md).
The retained vision and single-thread audio fixes are independent. Optional
waveform-tail optimization and automatic-language extensions are not carried.


A low-level [llama.cpp](https://github.com/ggml-org/llama.cpp) Wasm build. This repository keeps the heavy WebAssembly build and artifact publication independent of application builds.

Applications own model downloads, storage, workers, generation loops, and conversation formats. The generated Emscripten module exposes upstream operations. `examples/runtime/` is always included as tested reference code for application host implementations; importing it is optional. The public bindings may change with the pinned upstream version.

## Install a runtime

Install a specific **artifact commit**, not a commit from a source branch:

```sh
npm install github:nwtgck/browser-inference-core#ARTIFACT_COMMIT_SHA
```

Replace `ARTIFACT_COMMIT_SHA` with the complete commit hash of the chosen runtime artifacts. No npm registry publication or install-time C/C++ compilation is required.

```js
import { createCore } from 'llama-cpp-browser-core/examples/runtime';

const core = await createCore({ profile: 'cpu-wasm32' });
await core.api.llama_backend_init();
const version = await core.api.llama_version();
console.log(core.readUtf8(version));
await core.api.llama_backend_free();
```

Each profile ships two variants under `profiles/<profile>/<variant>/`: `browser` uses `ASSERTIONS=0` and `ENVIRONMENT=web,worker`; `test` retains assertions and Node.js support with `ENVIRONMENT=web,worker,node`. The example defaults to `browser`; Node.js tests must explicitly select `variant: 'test'`. Each variant has its own matching JavaScript, Wasm, and generated types. Never mix files between variants, even within one artifact.

Serve the selected profile and variant's JavaScript, Wasm, and any auxiliary files together. Bundlers may not discover dynamically imported assets automatically. When copying `profiles/` to a public directory, pass its URL as `baseURL`:

```js
const core = await createCore({
  profile: 'cpu-wasm32',
  baseURL: new URL('/runtime/profiles/', location.origin),
});
```

See the [API and ownership contract](docs/core-contract.md) and the [OPFS worker example](examples/opfs-model.mjs).

## Build from source

Prerequisites: Linux, Git, Python 3.11 or later, Clang, CMake 3.24 or later, Ninja, and Node.js 22. Downloading the toolchain requires network access.

Run these commands from `llama-cpp/`:

```sh
git submodule update --init --recursive
python3 ../scripts/setup_toolchain.py
source ../.tools/emsdk/emsdk_env.sh
for profile in cpu-wasm32 cpu-wasm64 webgpu-wasm32-asyncify webgpu-wasm32-jspi webgpu-wasm64-jspi; do
  for variant in browser test; do
    python3 scripts/build.py --profile "$profile" --variant "$variant"
  done
done
python3 scripts/package_runtime.py
```

Build outputs go to `build/<profile>/<variant>/runtime/`; the assembled package goes to `dist/package/`. Do not commit generated binaries to source branches. The shared compiler/Dawn pins are in `../toolchain/config.json`; the llama upstream pin is in `config/toolchain.json`; profile and variant settings are in `config/profiles.json` and `config/variants.json`.

| Profile | Pointer width | Maximum linear memory | Backends |
|---|---:|---:|---|
| `cpu-wasm32` | 32-bit | 4 GiB | CPU |
| `cpu-wasm64` | 64-bit | 16 GiB | CPU |
| `webgpu-wasm32-asyncify` | 32-bit | 4 GiB | CPU and WebGPU without JSPI; experimental |
| `webgpu-wasm32-jspi` | 32-bit | 4 GiB | CPU and WebGPU with JSPI; experimental |
| `webgpu-wasm64-jspi` | 64-bit | 16 GiB | CPU and WebGPU; experimental |

All profiles are single-threaded. The Asyncify profile uses transformed Wasm and JavaScript exception handling; it does not require memory64 or JSPI. It still requires Wasm SIMD and a usable WebGPU adapter/device. Asyncify adds code-size and execution overhead; applications should prefer a JSPI profile when supported and feature-detect their choice rather than use browser names. `webgpu-wasm32-jspi` requires JSPI but not memory64 and does not use Asyncify. Memory limits are build-time ceilings, not guarantees that a browser can allocate that memory or run a model of that size. Browser and device support must be checked for the chosen profile.

## Test

```sh
npm test
python3 -m unittest discover -s tests -p 'test_*.py'
```

Native integration tests can run without Emscripten:

```sh
cmake -S . -B build/native -G Ninja -DLCB_NATIVE_CHECK=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --target core -j 4
python3 scripts/make_test_model.py build/fixture.gguf
python3 tests/native_smoke.py --library build/native/libcore.so \
  --schema build/native/generated/schema.json --model build/fixture.gguf \
  --output build/native-results.json
```

The generated model is a small, deterministic, untrained GGUF fixture. Its purpose is to exercise loading and execution, not language quality or compatibility with large trained models. Native tests, host-side mocks, and browser Wasm tests cover different execution paths.

## Distribution

This runtime is one component of [Browser Inference Core](../README.md). Its
package and browser checks run independently of the image runtime. Only final
publication joins both. See the [common cache policy](../toolchain/README.md).


Source branches contain the pinned llama.cpp submodule and build tooling. The `artifacts` branch contains only the installable runtime, types, schemas, manifest, and license notices.

GitHub Actions builds on pushes outside `artifacts` and `artifacts/**`, and on PR opened/reopened/synchronize events. PR builds use the exact head commit rather than a synthetic merge commit. Same-repository PRs can publish before merge; fork PRs build/test without publication credentials. Duplicate push/PR builds are intentionally allowed. It runs host tests and builds all five profiles in both variants on separate runners. After all succeed, one assembly job runs Chromium smoke tests for both variants of the two CPU profiles and all three WebGPU profiles with JSPI/Asyncify suspension with a mocked unavailable GPU adapter. A Node.js regression exercises the test variant's Asyncify suspension. The browser variant is tested directly; passing test-variant checks alone does not validate browser artifacts. The combined package is verified before publishing an append-only artifact commit. Within the runtime build workflow, only the publication job has repository write permission. Repository rules must permit that job to update the artifact branch.

For browser-only upstream updates, **Actions > Update llama.cpp** prepares and
pushes a candidate branch, then shows a prefilled PR form link. The updater does
not create the PR or dispatch a build. Submitting the PR manually starts the
ordinary runtime workflow. Its summary and PR comment contain the artifact pin,
installation command, and consumer metadata. See [update automation](docs/update-automation.md).

The artifact branch tip is the most recently published result, not necessarily a build from `main`. Consumers should pin the complete artifact commit and commit their lockfile. See the [distribution contract](docs/distribution.md).

## Scope and limitations

The C bindings generator exposes non-deprecated, non-variadic llama.cpp, GGUF, selected backend, and multimodal functions. Standard Embind registrations expose upstream chat types, Jinja, tool parsing, grammar conversion, and reasoning-budget primitives. See [native chat and multimodal bindings](docs/chat-and-multimodal.md). The generated C schema and layout queries use normalized `bigint` pointers/sizes; Embind uses its standard type and lifetime conventions.

`mountReadOnlyFile` connects synchronous range reads to the ordinary model loader without creating an additional file-wide JavaScript buffer. It does not eliminate the model's resident memory requirements or the upstream GPU loader's tensor-sized staging buffers. General bounded GPU staging and pthread profiles are not implemented. WebGPU vision BF16 weights have a limited [bounded F32 expansion workaround](docs/webgpu-bf16-projector.md); it increases resident weight memory and does not guarantee GPU placement or full-F32 arithmetic. Multimodal video subprocess helpers are excluded; image/audio model execution needs separate validation, and some upstream audio paths require threads.

WebGPU is experimental; the workflow compiles all three GPU profiles but does not certify GPU inference on real devices. Small synthetic-model tests do not establish multi-GiB model support or broad browser compatibility.

## License

Project code is licensed under the [MIT License](LICENSE). Runtime packages collect upstream and toolchain notices under `licenses/`. Model weights are not included and have separate licensing requirements.
