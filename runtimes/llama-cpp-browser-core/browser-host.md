# Browser WebGPU host interface

WebGPU **browser** variants accept an optional `hostNavigator` factory option.
It supplies a navigator or a forwarding facade for this module's WebGPU
acquisition. Omitting it uses `globalThis.navigator`.

```js
const module = await createLlamaCppCore({
  wasmBinary,
  hostNavigator: scopedNavigator,
  print,
  printErr,
});
```

The host loader forwards the same option through `moduleOptions`:

```js
const core = await createCore({
  profile: 'webgpu-wasm64-jspi',
  variant: 'browser',
  moduleOptions: { hostNavigator: scopedNavigator },
});
```

`hostNavigator` is a browser-core API, not an existing Emscripten or Dawn option.
The source-owned `CoreModuleOptions` declaration in
`host/index.d.ts` describes its opaque host handle without adding a
WebGPU types dependency. The generated low-level factory declaration continues
to accept `unknown`; custom pre-JavaScript does not generate an option type.

## Contract

The value is captured once per factory invocation. It remains available for
later native backend initialization and device acquisition, after the factory
promise resolves. Replacing the option afterward is unsupported. Independent
factory invocations capture independent values.

The supplied object must provide a WebGPU-compatible `gpu` when WebGPU is
available. The current port reads `gpu.requestAdapter` and
`gpu.wgslLanguageFeatures`. A forwarding facade must retain native receivers and
preserve the acquisition options and device descriptors. The opaque TypeScript
type does not validate that behavior.

The core does not create an adapter or device in advance. The native backend
still selects the adapter, requests features and limits, owns its device and
queue, and installs its device-loss and error callbacks. Hosts can return
forwarding facades around those objects. Dispatch adaptation, shader changes,
diagnostics, model policy and worker management remain host responsibilities.

The interface does not change global navigator/GPU objects or their prototypes.
It only supplies the WebGPU port's unqualified `navigator` lookup inside this
factory. It does not intercept explicit `globalThis.navigator` reads, such as
the runtime's language lookup, or promise control over all navigator APIs.

## Build scope and update checks

`host/emscripten/host-navigator-pre.js` is linked with Emscripten's standard `--pre-js`
extension inside the `MODULARIZE=1` factory and optimized with the generated
code. Its contents are a link dependency. No generated JavaScript replacement,
vendor edit, native device setter or custom `INCOMING_MODULE_JS_API` entry is
needed.

The input is linked only for the four WebGPU browser profiles: wasm32 Asyncify,
wasm32 JSPI, wasm64 JSPI and wasm64 JSPI with GPU profiling. The CPU profile,
native checks and all test variants do
not consume `hostNavigator`.

This restriction preserves the test variant's browser-version checks. The
Emscripten assertion prologue reads `navigator` before pre-JavaScript runs. A
function-scoped `var navigator` would be hoisted over that check while still
uninitialized, causing it to skip the browser check even with no supplied
option. A lexical `let` or `const` would instead introduce a temporal dead zone.
Browser variants currently disable assertions; configuration fails if their
assertions are enabled without reviewing this interface's scope.

This is a supported compiler extension implementing a core-owned contract; it
is not a Dawn injection API. Updates must check that WebGPU acquisition and WGSL
feature queries still resolve the local navigator. A move to
`globalThis.navigator.gpu`, an earlier lookup, or another acquisition path needs
review. Browser/test support must not be made uniform by silently weakening the
test prologue or rewriting its checks.

Validation must use the shipped browser variants for explicit injection,
omitted-option fallback, independent factory instances, and delayed native
acquisition. Test-variant mocks alone do not verify this interface. Mocked
adapter suspension does not establish successful device creation, GPU inference
or numerical correctness. A new build and its actual runtime hashes are needed
before publishing; no prior artifact gains this API retroactively.

## Separate loader concerns

`hostNavigator` does not change Wasm loading. `wasmBinary`, `instantiateWasm`
and `locateFile` remain separate Emscripten controls.

The runtime build CLI prepares the generated sibling `core.wasm` URL after
linking by adding one `/* @vite-ignore */` comment immediately before its first
argument. This applies to every browser profile, including CPU. Vite skips
automatic Wasm discovery at this expression; other bundlers may differ. The test
variant uses the compiler's original expression. Runtime URL acquisition and
supplied-byte fallback behavior are preserved.

From a source checkout with Node dependencies installed and the pinned toolchain
prepared under `.tools/`, run from the repository root. The source checkout's
`toolchain/README.md` describes toolchain acquisition. Keep the project Node
executable when loading the SDK environment:

```sh
npm run tooling:build
BICORE_TOOL_NODE="$(command -v node)"
source .tools/emsdk/emsdk_env.sh
"$BICORE_TOOL_NODE" build/tooling/runtimes/llama-cpp/tools/cli.js build \
  --checkout-root "$PWD" --tooling-root "$PWD/build/tooling" \
  --output-root /tmp/browser-inference-builds/llama-cpp \
  --toolchain-root "$PWD/.tools" --local \
  --profile webgpu-wasm64-jspi --variant browser --kind wasm --jobs 4
```

The CLI stages the prepared runtime and matching generated declarations under
`<output-root>/stage/<profile>/<variant>/`. Local builds remain unpublishable;
release builds require a clean checkout matching `--expected-commit` instead of
`--local`. Direct CMake linking leaves the generated URL in the compiler's
original form; use the runtime build CLI for prepared stage artifacts.

Hosts explicitly copy/deploy the matching Wasm alongside its JavaScript, select
another deployed URL through `locateFile`, or provide `wasmBinary`. A URL callback
does not copy or deploy an asset by itself. A supplied whole `ArrayBuffer`
becomes a byte view in the generated loader without an extra JavaScript payload
copy; a supplied typed-array view is copied by that loader. Never pass a partial
view's entire backing buffer: it also includes bytes outside the selected span.
`instantiateWasm` is another existing option when hosts need to own compilation
or preserve an arbitrary view without this copy.

Per-build provenance records the preparation's input/output byte counts and
hashes, and whether the input already contained the annotation. Asset deployment,
compression and packaging remain separate host responsibilities.

The build CLI also prepares the SDK-generated `core.d.ts` native export
declarations. With `WASM_BIGINT`, native i64 arguments and results are JavaScript
`bigint` primitives. The pinned SDK spells these types as the boxed `BigInt`
interface. The preparation uses TypeScript's syntax tree and symbol resolution
to correct global `BigInt` type references in native export methods while
preserving record types, value identifiers, and Embind declarations. Provenance
records the declaration input/output identities and number of corrected type
references. This changes declarations only; the JavaScript and Wasm ABI remain
the compiler's generated artifacts.
