# stable-diffusion.cpp browser core (experimental)

An independently pinned image runtime in Browser Inference Core. It shares the
compiler and publication machinery, not llama-cpp's model implementation or
upstream patch policy. WebAssembly size optimization is deliberately separate
from functional browser bring-up.

## Boundary: native capability, application policy

ABI 3 is generated from the patched `include/stable-diffusion.h` and the pinned
public `ggml.h`, `ggml-backend.h`, `ggml-cpu.h`, `ggml-alloc.h` and `gguf.h` headers.
Supported public declarations and complete public records are exposed, including
tensor/graph construction, allocation, transfer, execution and CPU cancellation.
Variadic calls, FILE-pointer APIs without a matching host ownership surface and
the pinned upstream's unimplemented thread-count getter remain explicitly excluded.
The generated schema records each exclusion. Raw exports use the native target
calling convention: Wasm32 pointers use `number`, while Wasm64 pointers and
64-bit integers use `bigint`. The host API presents pointers and 64-bit values
as `bigint`, checks their ranges and converts to the target's native width.
Record sizes, offsets and constants are derived using the actual target compiler
inputs. The Wasm exports its pointer width and matching SHA-256 schema fingerprint,
which selects that target's layouts from the packaged ABI catalog.
New public declarations in these headers do not need a per-function export-list
change. Required names and representative signature tests detect incompatible
pin updates; unsupported C types fail generation instead of receiving guessed types.

There is **no** `sdb_load(JSON)`/`sdb_generate(JSON)` dispatcher, singleton model,
fixed model directory, preset sampler, resolution limit, memory budget or model
specific configuration in the bridge. Upstream parameter initialization retains
upstream defaults. ABI 3 changes the experimental ABI 1/2 wire convention;
artifacts built for those versions must be regenerated.

The application owns model selection, file acquisition/storage, mounting paths,
profile selection, resource budgets, sampler/scheduler, generation settings,
context lifetime, callbacks, image encoding, Worker protocol and cancellation.
Scalar and pointer functions retain their original C exports. Ordinary native
bridges handle records passed or returned by value, using caller-owned record
storage. Optional helpers in `host/` add typed record access, width conversion
and serialization. They do not select application policy.
See [host contract](host/README.md).

For example, the caller initializes `sd_ctx_params_t`, fills its chosen file
paths, calls `new_sd_ctx`, initializes/fills `sd_img_gen_params_t`, then calls
`generate_image` with an output pointer for `sd_image_generation_result_t`.
Copy each item's image data and any needed generation metadata before calling
`free_sd_image_generation_result`. This pinned upstream replaces the previous
image-array/count result API; ABI 3 alone does not imply that old calls remain
compatible. VAE tile dimensions now use `tile_size_w`/`tile_size_h` in image pixels.
The caller frees the context before releasing its disk-backed file sources.
Log, legacy progress, and preview registrations are upstream module-global state;
the generation-progress callback is context-local. Callbacks must not throw or
re-enter a suspended native operation. The application must unregister callbacks
before removing their function-table entries.

## Repository-native model files

`mountReadOnlyFile(core, path, {size, read(destination, offset)})` accepts caller-
owned synchronous random access. No Blob, OPFS, download, persistence, locking,
or ownership policy is built into it. The source receives safe JavaScript
integer offsets, not truncated 32-bit integers. Reads are chunked; whole files
are not copied into JavaScript or Wasm memory. File names have no special policy
in the core. The consumer controls which complete files and relative paths it mounts.

The image-specific upstream patch keeps file positions in `uint64_t`, and parses
GGUF v2/v3 metadata without accumulating the complete tensor payload in `size_t`.
It validates dimensions, type/block sizes, alignment, arrays and file bounds,
and preserves the upstream extended-GGUF higher-rank flattening with checked
arithmetic. Tensor bytes remain lazily read. The generic reader rejects mmap;
applications using it must set `enable_mmap=false` and avoid prefetch threads.

**File size is not resident memory usage.** A file larger than 4 GiB may work in
Wasm32, but individual tensors, staging buffers and actual live allocations still
have that runtime's limits. Wasm64 does not remove GPU buffer or physical-memory
limits. Several component files (diffusion/text encoder/VAE) are not GGUF shards;
each can be a complete unsplit GGUF. Unsupported quantization or architecture
remains unsupported even when its file is readable.

### Safetensors and already-sharded repositories

`module._bicore_model_io_capabilities()` reports compiled loader capabilities. Bit 0
means 64-bit safetensors file positions with bounded metadata validation; bit 1
means complete standard GGUF shard-group loading; bit 2 (value `4`) means bounded
Comfy GGUF original-shape restoration. This build returns `7`. A
consumer must check these capabilities separately from the ABI transport version.
The image manifest also records
`safetensorsFileOffsetBits: 64`, `ggufShards: true`, and `ggufOriginalShapes: true`.

`comfy.gguf.orig_shape.<tensor-name>` stores PyTorch-order integer dimensions.
The model-loader reader reverses them to ggml order and restores a flattened descriptor only
when the element count, dtype byte size and quantized row alignment agree.
Already multidimensional descriptors must match exactly. Dimensions must be
positive signed-64-bit values; ranks are limited to `GGML_MAX_DIMS`, with at most
65,536 original-shape entries and 16 MiB of retained names/dimensions per file.
Invalid types, duplicate keys and conflicting dimensions are rejected. Valid
entries for tensors in another shard are ignored after validation. Payloads,
dtypes, file offsets and standard shard ownership remain unchanged. The generic
native parser cannot silently bypass recognized original-shape validation.

Single safetensors files retain their original bytes, dtype and file size.
Metadata is bounded at 100 MiB; positions and aggregate payload size use 64-bit
integers, while each individual decoded tensor still must fit the address space.
Duplicate keys, invalid ranges, holes/overlaps and unsafe index references are
rejected. Index files are limited to 16 MiB and reference only local safetensors
siblings; nested indices, traversal and duplicate shard tensors are rejected.

Standard GGUF shards are resolved using their original five-digit filenames and
`split.no`, `split.count`, `split.tensors.count`. All members and tensor identities
are validated before the group is accepted. No file is split, concatenated or
converted. This is a loader capability, not repository discovery: the application
must mount every required sibling before passing the chosen component path.

The upstream quantization/backend restrictions still apply (in particular,
INT8 tensorwise/convrot is not enabled by supporting the safetensors container).
A missing model component or unsupported operation is not repaired by changing
file formats. Model-family detection, compatible-candidate selection, optional
features and presets remain application responsibilities.

## Explicit BF16 WebGPU weight retention

`sd_browser_bf16_retention_version()` returns `1` for this browser-enabled
implementation (`0` in a non-browser upstream build). The source-bound image
manifest records `bf16WeightRetention: true` separately from the unchanged
model-I/O capability mask. Callers must check the feature before selecting
`sd_ctx_params_t.webgpu_bf16_type = SD_TYPE_BF16`; the ABI version alone is insufficient.
F32 remains the initializer's default, and explicit F16 keeps its existing meaning.
No profile, precision, memory budget or application preset is selected automatically.

Retention applies to original contiguous two-dimensional BF16 parameters with
registered Linear-weight or Embedding-weight roles and WebGPU compute. Unknown
roles, norms and biases keep the F32 conversion path and are reported separately;
tensor rank or a suggestive name alone does not establish a role. Unsupported
device operations or an incompatible actual graph fail explicitly rather than
silently retrying retained weights on CPU or converting them to F32. Destructive
LoRA weight updates are rejected when retained weights are present.
Caller-owned source files and existing quantized/F16/F32 parameter bytes are unchanged.

The image-only patch inventory combines four reviewable source patches with
observed identities and regression coverage:

- `0036` backports the upstream BF16-left WebGPU matrix and row-lookup kernels.
  Remove it when the image ggml pin supplies equivalent kernels and admission.
- `0037` bounds parameter uploads while preserving existing queue order and
  completion ownership. Remove it when upstream provides equivalent bounded uploads.
- `0038` reads raw device tensors and BF16-to-F32/F16 conversions through bounded
  loader scratch instead of a whole-tensor host mirror. Remove it when upstream
  provides the same checked reader, byte and lifetime contracts.
- `0039` adds the explicit parameter policy, registered roles and actual-graph
  validation. Remove it only when upstream provides an equivalent opt-in API with
  preserved defaults and failure/ownership contracts.

Packaging requires these four patches in canonical order with matching targets
and SHA-256 identities across every profile/variant, as well as the generated
version API contract. A version declaration alone does not establish the feature.

Native CPU arithmetic and CPU-only Wasm contracts do not prove
GPU shader execution, trained-model image quality, peak memory or browser stability.
The feature does not imply that every BF16 parameter or operation can be retained.

## Build and distribution

Acquire the compiler and Dawn using the root [toolchain contract](../../toolchain/README.md).
Then run from the repository root:

```sh
node node_modules/typescript/bin/tsc --project tsconfig.stable-diffusion-cpp.json
. .tools/emsdk/emsdk_env.sh
node build/tooling/runtimes/stable-diffusion-cpp/tools/cli.js build \
  --checkout-root "$PWD" --tooling-root "$PWD/build/tooling" \
  --toolchain-root "$PWD/.tools" --output-root /tmp/image-runtime-build \
  --local --kind wasm --profile webgpu-wasm32-jspi --variant browser
```

Initialize submodules first. Profiles are `webgpu-wasm32-asyncify`,
`webgpu-wasm32-jspi`, and `webgpu-wasm64-jspi`, each with separate `browser` and
`test` variants. They are single-threaded. Application-created Workers remain
possible and are not the same thing as native pthreads.

Shared toolchain pins are in repository-root `toolchain/config.json`. Image
source commits in a release build come from the clean checkout HEAD's gitlinks.
Local development uses `--local` for owned working-tree inputs and staged gitlinks;
release builds require `--expected-commit <full-commit>` to match checkout HEAD.
`config/upstreams.json` declares the independent upstream roles and paths. Preparation reads the
selected Git objects. The build snapshots its CMake, native, template and patch
inputs before compilation and verifies their identities before staging. Configuration and
recipe values come from the running checkout's static imports. All
nested checkouts selected by the build graph must be initialized before offline preparation.
Disabled CUDA, WebP, WebM and example projects do not need initialized checkouts.

The independent `vendor/ggml-webgpu-source` checkout contributes **only** `ggml/`;
stable-diffusion.cpp's nested ggml is not linked. `tools/recipe.ts`
declares each patch's target files, order and purpose. The generated
`prepared/patch-provenance.json` records the actual source and patch identities;
these observations do not prove numerical or ownership correctness. Changes to
the image-runtime pins are deliberate, not inherited from llama-cpp updates.

Prepared sources and their receipt become available together. Without `--fresh`,
an unchanged composition preserves file timestamps and avoids recompilation of
identical native inputs. A preparation failure does not replace the previous
complete output, and a changed source commit during compilation prevents staging
publishable artifacts. Explicit `--fresh` discards only the selected build tree.
The output directory holds private compiler inputs and upstream trees. The
checkout's tooling is compiled once and calls preparation, configuration,
compilation and staging in the same Node process; dependencies and tooling are
not copied. Local and native builds remain private; release Wasm builds alone
produce a CI upload candidate.

Compiler output is under `build/<kind>/<profile>/<variant>` in the selected
output root. A release Wasm build stages its candidate separately under
`stage/<profile>/<variant>`. The local example above compiles private output
without creating a publishable candidate.

Compile/packaging jobs are parallel to llama's jobs; final aggregate publication
waits for both. Toolchain and license transfer is independent. See the shared
[cache policy](../../toolchain/README.md). The inner image manifest is format 2,
ABI 3, with large-file capabilities. Each packaged profile/variant has its own
`abi.json`; `api/schema.json` and `api/schema.mjs` contain the catalog indexed by
compiled target fingerprint. The manifest's `schemaSha256` is the catalog's
canonical schema fingerprint; its file inventory separately binds the actual
`api/schema.json` bytes. The format-5 parent
package uses the name `browser-inference-core` and stores this component under
`runtimes/stable-diffusion-cpp-browser-core/`. Public imports use
`browser-inference-core/stable-diffusion-cpp/host` and the corresponding
`/profiles/<profile>/<variant>/core.mjs` path. See the
[root package layout](../../README.md).

## Verification scope

Native checks compile the prepared SD/GGML source and ordinary C++ fixtures. They
use sparse unsplit GGUF files whose last
tensor starts beyond 2/4/8 GiB, actual C++ file reads, malformed metadata, callback
registration and numerical comparison of patched normalization to upstream CPU.
They also read sparse safetensors beyond 20 GiB and validate complete/missing/
duplicate GGUF groups and local safetensors indices, plus the Qwen timestep
allocation/arithmetic regression described below.
Synthetic tensors are **not** a trained model or an image-generation benchmark.

CI browser smoke uses real modules in dedicated Workers for all six variants:
parameter round trips (including >2^53 seeds), caller-owned file reads, native
GGUF indexing/reading across 4/8 GiB in test builds, and log/progress registration,
notification and unregistration. Additional test-variant probes exercise
safetensors beyond 20 GiB, complete/missing GGUF shard groups, and a tiny
synthetic CPU Qwen timestep graph. A metadata-only graph probe also exercises
32,768-node dependency chains, selected-branch compute propagation, both source
orders and shared operand counts. Test-only probes are absent from browser
artifacts. The fixture never allocates a multi-gigabyte JavaScript array.

Default smoke does not request a GPU device. The optional WebGPU timestep
check below is still synthetic, not real-model image inference.
`validation.realModelInference` stays false. Model/operator/dispatch limits,
precision, peak memory, repeated generations and performance require actual
browser/model validation. Exposed APIs are not claims that every feature has
been verified. Initial application trials should use a small model and output.

## Qwen timestep / mixed-backend regression

`upstream-patches/0007-webgpu-qwen-timestep-out-of-place.patch` fixes the
browser build's Qwen timestep SiLU output allocation. In Qwen Image 2.1 the
first BF16 linear can execute on CPU and its supported activation on WebGPU.
Copying `src[0]` across backends does not make an in-place CPU `view_src` a GPU
buffer. The activation therefore uses its own result allocation; the formula,
weights, quantization, sampling options and backend preferences are unchanged.
Non-browser upstream builds retain the original in-place path.

`../../tests/runtimes/stable_diffusion_cpp/qwen-timestep-probe.cpp` builds the actual, bias-free Qwen timestep block
with tiny synthetic identity weights (first linear BF16, second linear F16).
Native checks and browser smoke **test variants** assert the absence of the
unsafe alias and compare CPU results with the SiLU formula. To additionally run
the mixed-backend placement and numerical check on a WebGPU-capable host, set
`SDCB_TEST_WEBGPU=1` when running `../../tests/runtimes/stable_diffusion_cpp/browser-smoke.mjs` against the freshly
packaged runtime. This opt-in fails rather than silently skipping an unavailable
GPU; ordinary smoke does not require one. The suspending test export is absent
from browser variants and does not change the public application API.

These are bounded synthetic graph checks, not a claim that full Qwen model
inference or image quality was validated on every browser/device. A source-only
patch does not update an installed Wasm: rebuild/publish through the existing
source-bound artifact workflow, then update the application's artifact pin.


From the repository root, compile the tooling and run the image-runtime tool tests with:

```sh
npm run tooling:build
node --test build/tooling/tests/runtimes/stable-diffusion-cpp/tools/*.test.js
```

The selected-runtime TypeScript project builds independently of other runtime
checkouts. CMake receives separate checkout, compiler-input and emitted-tooling
roots and invokes the selected Node ABI tooling against its actual target compile
commands. The raw `core.d.ts`, its `module-options.d.ts`, exports and target schema
are generated or copied together. Native
checks use ordinary complete C++ fixtures. Upstream-owned WGSL generation remains
with the unchanged upstream generator.
