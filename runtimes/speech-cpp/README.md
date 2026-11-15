# speech.cpp browser runtime (experimental)

An independent [speech.cpp](https://github.com/nyosegawa/speech.cpp) WebAssembly
runtime in Browser Inference Core. It exposes the upstream C API for speech
synthesis, recognition, detection, diarization, enhancement, and full-duplex
dialogue. Available operations and options depend on the loaded model.

The initial target is a hosted browser application on desktop Chrome with
WebGPU, Wasm32, SIMD, and JavaScript Promise Integration (JSPI). Standalone
application distribution is outside this runtime's initial integration scope.
This is an experimental target, not a claim that a particular model fits a
device or runs in real time.

## Runtime and application ownership

This runtime has its own Wasm instance, speech.cpp pin, and ggml pin. It does not
link another runtime's model implementation or ggml. Upstream changes can be
prepared and checked independently of the llama.cpp and image runtimes.

The generated ABI exposes `speech_*` functions from the pinned
[`speech.h`](https://github.com/nyosegawa/speech.cpp/blob/2f51f66164a906b322a48303bfbcdd6936833302/include/speech.h).
Scalar and pointer calls use their original C exports; handles remain opaque.
ABI 3 describes the browser calling convention. The separate
`speech_api_version_major/minor()` functions report upstream API compatibility.
Always use the JavaScript, Wasm, types, and schema from the same artifact.

Applications own model acquisition and storage, Workers, audio capture and
playback, queues, request options, cancellation policy, and resource lifetimes.
Optional host helpers provide typed calls, pointer-width conversion, serialization,
and caller-owned random-access file mounting. See the
[host and ownership contract](host/README.md).

For a dialogue model such as Haru, `speech_dialogue_start/push/end` streams input
PCM and output audio/text frames. The caller obtains sample rate and frame size
from model information and copies borrowed callback data before returning.
The core does not implement a conversation UI or audio scheduling policy.

## Use a published artifact

Install a complete artifact commit, rather than a source-branch commit:

```sh
npm install github:nwtgck/browser-inference-core#ARTIFACT_COMMIT_SHA
```

Replace `ARTIFACT_COMMIT_SHA` with the chosen artifact's full commit hash. Then
instantiate the selected module and attach its matching schema:

```js
import createSpeechCore from 'browser-inference-core/speech-cpp/profiles/webgpu-wasm32-jspi/browser/core.mjs';
import { attachCore, schema } from 'browser-inference-core/speech-cpp/host';

const module = await createSpeechCore();
const core = attachCore(module, schema, { suspension: 'jspi' });
console.log(core.readUtf8(await core.api.speech_version()));
```

Serve the selected module and its Wasm together. A bundler may require explicit
asset copying or a `locateFile` factory option. The host API returns Promises and
represents pointers and 64-bit integers as bigint. Callers must serialize access
to an instance and keep callback table entries alive until their native users
are released.

The `webgpu-wasm32-jspi` profile is single-threaded and has a 4 GiB maximum Wasm
linear-memory setting. This ceiling is not an allocation guarantee and does not
enlarge WebGPU buffer, binding, or physical-memory limits. Feature-detect the
required browser capabilities before loading a model; memory64 is not required.

| Variant | Assertions | Emscripten environments |
|---|---:|---|
| `browser` | 0 | `web,worker` |
| `test` | 1 | `web,worker,node` |

## Source pins and preparation

The source checkout records independent submodule gitlinks:

| Component | Repository | Initial commit |
|---|---|---|
| speech.cpp | `nyosegawa/speech.cpp` | `2f51f66164a906b322a48303bfbcdd6936833302` |
| ggml | `nyosegawa/ggml` | `e1a9ac12edbb57402618fd6936598603f5ee6266` |

The ggml pin matches the dependency selected by that speech.cpp revision.
Artifact manifests and preparation receipts record the actual source and patch
identities of each build. Source preparation exports the selected commits into
private build trees and applies reviewable patches there, leaving vendor
checkouts untouched. The external-ggml CMake patch selects this runtime's prepared
ggml tree. The owned build links `speech-static`; upstream CLI, server, and check
executables are excluded from its build target.

## Build and check from source

Native checks require Git, Node.js 22.12 or later, Clang, CMake 3.24 or later,
Ninja, and the initialized speech.cpp and ggml submodules. From the repository
root:

```sh
npm ci
npm run tooling:build
node build/tooling/runtimes/speech-cpp/tools/cli.js build \
  --checkout-root "$PWD" --tooling-root "$PWD/build/tooling" --local \
  --output-root /tmp/browser-inference-builds/speech-cpp \
  --toolchain-root "$PWD/.tools" \
  --profile webgpu-wasm32-jspi --variant test --kind native --jobs 4
ctest --test-dir /tmp/browser-inference-builds/speech-cpp/build/native/webgpu-wasm32-jspi/test \
  --output-on-failure
node --test build/tooling/tests/runtimes/speech-cpp/tools/*.test.js
node --test tests/runtimes/speech-cpp/read-only-file.test.mjs
```

Native checks do not require Emscripten or a GPU. For a Wasm build, first prepare
the repository's pinned Emscripten and Dawn toolchain, activate its Emscripten
environment, and use `--kind wasm`. Python is also required by upstream's WGSL
generator. Build outputs are placed under
`<output-root>/build/<kind>/<profile>/<variant>/`. Local builds are not
publishable; release builds require clean, source-bound inputs.

Native contracts cover public API ownership, errors, options, and callbacks. Host
tests cover bounded synchronous reads, large file offsets, and mounted-file
ownership. The test variant's Node ABI probe exercises generated JavaScript/Wasm
calls and callbacks without a model or GPU. These checks cover different paths:
native or host success does not establish WebGPU model inference, speech quality,
full-duplex latency, or browser/device compatibility. Real-model WebGPU dialogue
requires separate validation.

## Licenses

`LICENSE` covers the Browser Inference Core wrapper under the MIT license, using
the repository's copyright notice. It does not replace upstream attribution.
Packaged artifacts retain speech.cpp notices under `licenses/speech/`, ggml
notices under `licenses/ggml/`, and compiler/Dawn notices under
`licenses/toolchain/`. `native-license-index.json` records those components and
their notices. Model weights and voice files are separate inputs with their own
terms; this runtime does not distribute them.
