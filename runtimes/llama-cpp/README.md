# llama.cpp browser runtime

## Upstream compatibility policy

Keep this core close to upstream. Downstream patches are a last resort and require
explicit user approval for each new or expanded scope; a feature request alone
is not that approval. See [the repository instructions](AGENTS.md) and the
[exception policy and register](upstream-patches-only-as-a-last-resort-with-explicit-user-approval/README.md).
The retained vision and single-thread audio fixes are independent. Optional
waveform-tail optimization and automatic-language extensions are not carried.


A low-level [llama.cpp](https://github.com/ggml-org/llama.cpp) Wasm build. This repository keeps the heavy WebAssembly build and artifact publication independent of application builds.

Applications own model downloads, storage, workers, generation loops, and conversation formats. The generated Emscripten module exposes upstream operations. `host/` is always included as tested reference code for application host implementations; importing it is optional. The public bindings may change with the pinned upstream version.

## Install a runtime

Install a specific **artifact commit**, not a commit from a source branch:

```sh
npm install github:nwtgck/browser-inference-core#ARTIFACT_COMMIT_SHA
```

Replace `ARTIFACT_COMMIT_SHA` with the complete commit hash of the chosen runtime artifacts. No npm registry publication or install-time C/C++ compilation is required.

```js
import { createCore } from 'browser-inference-core/llama-cpp/host';

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

Prerequisites: Linux, Git, Python 3.11 or later, Clang, CMake 3.24 or later, Ninja, jq, and Node.js 22.12 or later. Python runs the unchanged upstream WGSL generator; project-owned tools run as emitted TypeScript. Toolchain acquisition requires network access.

Acquire the pinned compiler and Dawn as described in the root
[toolchain contract](../../toolchain/README.md), then run from the repository root:

```sh
git submodule update --init --recursive
npm ci
npm run tooling:build
node build/tooling/tools/toolchain/cli.js prepare --checkout-root "$PWD" --tools-root "$PWD/.tools"
BICORE_TOOL_NODE="$(command -v node)"
source .tools/emsdk/emsdk_env.sh
for profile in cpu-wasm32 webgpu-wasm32-asyncify webgpu-wasm32-jspi webgpu-wasm64-jspi webgpu-wasm64-jspi-gpu-profile; do
  for variant in browser test; do
    "$BICORE_TOOL_NODE" build/tooling/runtimes/llama-cpp/tools/cli.js build \
      --checkout-root "$PWD" --tooling-root "$PWD/build/tooling" --local \
      --output-root /tmp/browser-inference-builds/llama-cpp \
      --toolchain-root "$PWD/.tools" --profile "$profile" --variant "$variant" --kind wasm \
      --jobs 4 --python "$(command -v python3)"
  done
done
```

Build outputs go to `<output-root>/build/<kind>/<profile>/<variant>/runtime/`, and Wasm builds stage matching runtime assets and provenance under `<output-root>/stage/<profile>/<variant>/`. The `--local` mode captures owned working files and remains unpublishable; release builds require a clean checkout whose HEAD matches `--expected-commit` instead. Do not commit generated binaries to source branches. The shared compiler/Dawn pins are in `../../toolchain/config.json`; the llama upstream pin is the Git submodule entry. Profile and variant settings are in `config/profiles.json` and `config/variants.json`.

| Profile | Pointer width | Maximum linear memory | Backends |
|---|---:|---:|---|
| `cpu-wasm32` | 32-bit | 4 GiB | CPU |
| `webgpu-wasm64-jspi-gpu-profile` | 64-bit | 16 GiB | CPU and WebGPU; diagnostic GPU timestamps |
| `webgpu-wasm32-asyncify` | 32-bit | 4 GiB | CPU and WebGPU without JSPI; experimental |
| `webgpu-wasm32-jspi` | 32-bit | 4 GiB | CPU and WebGPU with JSPI; experimental |
| `webgpu-wasm64-jspi` | 64-bit | 16 GiB | CPU and WebGPU; experimental |

All profiles are single-threaded. The Asyncify profile uses transformed Wasm and JavaScript exception handling; it does not require memory64 or JSPI. It still requires Wasm SIMD and a usable WebGPU adapter/device. Asyncify adds code-size and execution overhead; applications should prefer a JSPI profile when supported and feature-detect their choice rather than use browser names. `webgpu-wasm32-jspi` requires JSPI but not memory64 and does not use Asyncify. Memory limits are build-time ceilings, not guarantees that a browser can allocate that memory or run a model of that size. Browser and device support must be checked for the chosen profile.

The diagnostic profile uses the same upstream and settings as `webgpu-wasm64-jspi`,
with `GGML_WEBGPU_GPU_PROFILE=ON`. It additionally requires the optional WebGPU
`timestamp-query` feature. Upstream separates dispatches into compute passes and
reads timestamps back, so its generation rates are **not** normal-chat performance
measurements. The build checks the actual backend compile definition and keeps it
absent from all ordinary profiles. Consumers must opt in explicitly and should
load this asset only for diagnostics; it is not an automatic fallback. The retired
`cpu-wasm64` profile is no longer published. CPU fallback uses `cpu-wasm32`, with
its 4 GiB linear-memory ceiling.

## Test

From the repository root:

```sh
npm run tooling:build
node --test build/tooling/tests/runtimes/llama-cpp/tools/*.test.js
node --test tests/runtimes/llama_cpp/*.test.mjs
```

Native integration tests can run without Emscripten:

```sh
cmake -S runtimes/llama-cpp -B /tmp/browser-inference-native -G Ninja \
  -DLCB_NATIVE_CHECK=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DLCB_CHECKOUT_ROOT="$PWD" -DLCB_COMPILE_ROOT="$PWD" \
  -DLCB_TOOLING_ROOT="$PWD/build/tooling" \
  -DLCB_NODE="$(command -v node)"
cmake --build /tmp/browser-inference-native --target core -j 4
```

Native fixtures use small synthetic inputs to exercise typed boundaries, ownership and numerical behavior. They do not establish language quality or compatibility with large trained models. Native tests, host-side mocks, and browser Wasm tests cover different execution paths.

## Distribution

This runtime is one component of [Browser Inference Core](../../README.md). Its
package and browser checks run independently of the image runtime. Only final
publication joins both. See the [common cache policy](../../toolchain/README.md).


Source branches contain the pinned llama.cpp submodule and build tooling. The `artifacts` branch contains only the installable runtime, types, schemas, manifest, and license notices.

GitHub Actions builds on pushes outside `artifacts` and `artifacts/**`, and on PR opened/reopened/synchronize events. PR builds use the exact head commit rather than a synthetic merge commit. Same-repository PRs can publish before merge; fork PRs build/test without publication credentials. Duplicate push/PR builds are intentionally allowed. It runs host tests and builds all five profiles in both variants on separate runners. After all succeed, one assembly job runs Chromium smoke tests for both variants of the CPU profile and all four WebGPU profiles with JSPI/Asyncify suspension with a mocked unavailable GPU adapter. A Node.js regression exercises the test variant's Asyncify suspension. The browser variant is tested directly; passing test-variant checks alone does not validate browser artifacts. The combined package is verified before publishing an append-only artifact commit. Within the runtime build workflow, only the publication job has repository write permission. Repository rules must permit that job to update the artifact branch.

For browser-only upstream updates, **Actions > Update llama.cpp** prepares and
pushes a candidate branch, then shows a prefilled PR form link. The updater does
not create the PR or dispatch a build. Submitting the PR manually starts the
ordinary runtime workflow. Its summary and PR comment contain the artifact pin,
installation command, and consumer metadata. See [update automation](docs/update-automation.md).

The artifact branch tip is the most recently published result, not necessarily a build from `main`. Consumers should pin the complete artifact commit and commit their lockfile. See the [distribution contract](docs/distribution.md).

## Scope and limitations

The C bindings generator exposes non-deprecated, non-variadic llama.cpp, GGUF, selected backend, and multimodal functions. Standard Embind registrations expose upstream chat types, Jinja, tool parsing, grammar conversion, and reasoning-budget primitives. See [native chat and multimodal bindings](docs/chat-and-multimodal.md). The generated C schema and layout queries use normalized `bigint` pointers/sizes; Embind uses its standard type and lifetime conventions.

`mountReadOnlyFile` connects synchronous range reads to the ordinary model loader without creating an additional file-wide JavaScript buffer. It does not eliminate the model's resident memory requirements or the upstream GPU loader's tensor-sized staging buffers. General bounded GPU staging and pthread profiles are not implemented. WebGPU vision BF16 weights have a limited [bounded F32 expansion workaround](docs/webgpu-bf16-projector.md); it increases resident weight memory and does not guarantee GPU placement or full-F32 arithmetic. Multimodal video subprocess helpers are excluded; image/audio model execution needs separate validation, and some upstream audio paths require threads.

WebGPU is experimental; the workflow compiles all four GPU profiles but does not certify GPU inference on real devices. Small synthetic-model tests do not establish multi-GiB model support or broad browser compatibility.

## License

Project code is licensed under the [MIT License](LICENSE). Runtime packages collect upstream and toolchain notices under `licenses/`. Model weights are not included and have separate licensing requirements.
