# Accepted upstream exceptions

Read [AGENTS.md](AGENTS.md) before changing an exception. This is an inventory of
existing, deliberately limited deviations, not permission to add another one.
The original reason for carrying source patches was required multimodal
performance, not general feature expansion. The project owner decides which
requirements justify upstream maintenance work.

The vision/audio patches take the pristine pinned upstream tree as input;
neither depends on a TTS-generation copy. Other accepted overlays compose only as
specified below. The runtime source gitlink selects the upstream revision.
Re-evaluate compatibility at each pin update rather than treating a revision or
these workarounds as permanent.

## Current preparation contract

The production recipe is `runtimes/llama-cpp/tools/recipe.ts`, executed by
`tools/prepare.ts` in this runtime. It applies individual reviewable patches to a
private coherent source tree and then runs the unchanged upstream shader emitter.
Input/output digests identify the selected inputs and results; they are not
whole-file compatibility gates. C++ compilation and focused native/behavioral
checks establish separate contracts. See
[the source-preparation contract](../../../docs/source-preparation.md).

The entries below describe each retained change's supported behavior, coupling,
limits and removal condition. Current compilation uses the final composition,
not a chain of historical generated-header fixtures. Historical numerical
comparisons, when retained by tests, own their reference inputs separately.

## Portable explicit subgroup matvec selection

- **Patches:** `ggml-webgpu-subgroup-size-control.patch`,
  `ggml-webgpu-subgroup-size-control-profile.patch`,
  `ggml-webgpu-subgroup-size-control-shared.patch`, and
  `ggml-webgpu-subgroup-size-control-wgsl.patch`. The WGSL patch represents the
  existing optional-Q8 annotations at their shader source. The shared patch
  retains its host helpers; its redundant generated Q8 header hunk has been
  removed. The source-equivalent WGSL patch and unchanged upstream emitter
  produce the same shader without patching generated C++ strings.
- **Accepted scope:** request real
  subgroup-size-control and S16 for existing guarded ordinary Q8_0/Q4_0/Q4_K
  matvecs when a variable advertised range contains16. Existing fixed AUTO,
  shape/layout guards, accumulators, indexed paths, bindings and ABI remain.
- **Integration:** append a checked final stage after shared byte loaders and
  checkpoint reads. Keep actual advertised capabilities. Typed healthy optional
  pipeline errors return canonical pipeline/decisions; other errors stop the
  device generation. Requested keys, callback lifetimes and device ownership are
  explicit, and warm hits add no optional scope/wait. Checkpoint GPU APIs and
  post-wait copies observe the same terminal state; empty reads remain no-ops.
- **Coupling:** pinned WebGPU SDK future/callback/error-scope APIs, private
  shader-library keys, device initialization and host launch decisions. Profile
  AUTO requested S16 uses codegen revision2 in the unchanged diagnostic v1 ABI.
- **No-patch alternative:** preserve existing fixed-width or cooperative kernels
  and omit explicit subgroup requests. No public force flag or artifact variant
  is added.
- **Validation and limits:** source/CPU contracts, SDK callback controls and small
  software-GPU numeric checks do not establish real variable-adapter, browser,
  target-model or target-device performance. Partial subgroups remain legal.
- **Removal:** upstream supplies portable explicit-width selection and safe
  optional-pipeline fallback, or the owner withdraws this performance requirement.

See [the v0.8.0 reference review](../docs/llama-v0.8.0-review.md) for previously reviewed
contracts and validation boundaries. The [338fce review](../docs/llama-338fce-review.md)
and [v0.7.0 review](../docs/llama-v0.7.0-review.md) remain historical context.

## Vision BF16 compatibility/performance

- **Patches:** `mtmd-webgpu-bf16.patch` and `mtmd-vision-bf16-runtime.patch`.
- **Accepted scope:** per-load immutable storage selection in the same runtime. Existing entries remain
  legacy-expand; explicit retain-checked uses original BF16 storage and rejects
  unreviewed roles/graphs/placement instead of retrying in F32 or on CPU.
- **Review and failure reporting:** a metadata-only candidate classifier and a
  caller-owned fixed-width report accompany a strict single-load entry point.
  The core never retries. Only dedicated preupload role-review refusals report
  completed cleanup for a host to consider one legacy retry. General I/O,
  allocation, backend, layout, cancellation and late graph errors are ordinary
  failures. An exhaustive inventory of the real upstream projector enum is
  compiled into the loader; new identifiers require an explicit review decision.
  Existing family changes still require role and graph review, not just enum coverage.
- **Muse Glimmer scope:** shared transformer matrices and the three direct
  `mm_0_w`/`mm_1_w`/`mm_2_w` projection operands are reviewed. BF16 interpolated
  positions, patch, normalization and bias weights remain excluded. Synthetic
  tests use the real upstream builder; no builder or shader change is introduced.
- **Gemma4UV scope:** `mtmd-retained-roles.h` accepts only the image `mm_input_proj_w` for this
  projector, not common layer weights, patch/position/norm weights or audio.
  The private-role contract and `models/gemma4uv.cpp` require review on
  upstream changes. No shader or upstream model-builder change is included;
  `legacy-expand` remains the no-new-role alternative. Native contract tests
  are not real-browser/12B model validation.
- **Integration:** one checked copy of the complete mtmd source/header subtree.
  All target translation units, including unity batches, read the same private
  model definition. Original non-unity source properties are retained; model
  builders are byte-identical copies. Thin API plumbing exists in every profile;
  WebGPU loader handling remains under `LCB_WEBGPU_BF16_PROJECTOR`. Immutable
  selection is model-owned and shared through const subcontexts. Unsupported
  GPU-retain vision fails at existing modality classification before allocation;
  audio-only and explicit CPU storage preserve upstream behavior.
- **Coupling:** retained upload/graph helpers use public ggml interfaces;
  `mtmd-retained-roles.h` enumerates pinned private model fields without accepting
  new roles automatically. Because weights load before compute subcontexts,
  retained uploads temporarily initialize a backend from the buffer-type device.
  The pinned WebGPU implementation shares that device's global queue with weight
  writes, so draining it preserves the existing 32 MiB charged upload budget.
  This same-queue property is not assumed for other backends. No-allocation,
  legacy, CPU and rejected initial callbacks do not create that backend. Queue
  drain precedes backend free and weight-buffer destruction on every exit.
  Extra backend bookkeeping is not a physical peak-memory improvement claim.
  The shared compute owner rolls back a partial constructor by resetting its
  scheduler before freeing distinct backends. Subcontexts borrow that owner and
  never free shared resources. Upstream shared warmup, compute locking and
  single-count compute memory accounting remain unchanged.
  Source contracts, compile-target checks, generated ABI and provenance expose
  this maintenance cost. Vendor sources and upstream shaders remain intact.
- **Compatibility:** ordinary assets contain both modes without extra variants
  or build directories. No global/TLS selector or post-load mutator is added.
  Upstream owns the graph-allocation failure check; preserve it unchanged.
- **No-patch alternatives:** keep established expansion, compatible weight
  formats, the existing CPU path or defer retention. Disabling the whole overlay
  also removes independent safety/diagnostics and can restore full-tensor scratch.
- **Validation and limits:** [storage design](../docs/webgpu-bf16-projector.md).
  CPU and synthetic native tests do not establish actual browser/model
  correctness, physical peak or equal speed. Defaults are not switched on that
  evidence alone. New shaders and wider role acceptance are separate scope.
- **Removal:** upstream supplies the required storage, safety and performance
  contract, or the owner withdraws the requirement; retire API/plumbing/tests
  coherently without changing immutable artifacts or unrelated overlays.

## Non-pthread Wasm mtmd threadpool compatibility

- **Patch:** `mtmd-non-pthread-threadpool.patch`.
- **Scope:** only non-pthread Emscripten normalizes the private shared mtmd
  threadpool request to one, taking the existing no-pool branch. Native and
  pthread requests are unchanged. This preserves single-thread model loading
  after upstream moved audio preprocessing into scheduler graphs. Public record
  layouts, audio arithmetic and model pipelines stay upstream.
- **Integration:** the final checked stage in the coherent mtmd subtree, applied
  in CPU and WebGPU modes. Guard actual `mtmd_threadpool_init` and CPU pool source;
  bridge-only normalization would miss direct generated exports.
- **No-patch alternatives:** require a threaded runtime or omit affected loads;
  neither preserves the existing non-pthread profile contract.
- **Validation and limits:** actual-helper controlled registry tests distinguish
  non-pthread, native and pthread branches. They do not establish successful
  model loading, browser audio quality or actual pthread worker execution.
- **Removal:** upstream safely handles the non-pthread pool request, or the owner
  changes the affected profile requirement. Remove patch, stage and report
  together while retaining unrelated BF16 safeguards.

The former `mtmd-audio-single-thread.patch` is retired: upstream removed its
shared and Parakeet worker loops in favor of scheduler-backed log-mel graphs.
The new DFT/F32 graph arithmetic is preserved. The former
`mtmd-vision-bf16-clip-runtime.patch` is also retired; its unsupported GPU-retain
check is consolidated into the common model-policy/classification overlay.
The active preparation recipe lists the current compiled overlays; previously
published artifacts retain their original bytes and identities.

## Emscripten WebGPU batched-write fallback

- **Patch:** `ggml-webgpu-emscripten-scalar-tensor-writes.patch`.
- **Scope:** leave only the optional buffer `set_tensor_n` hook NULL under
  Emscripten. The public batch API uses existing generic scalar/2D fallback;
  native hooks, upstream batch implementations, shaders and eager pipeline
  creation remain intact.
- **Reason:** the pinned batch shader binds the entire destination buffer and
  packs whole inputs, which can exceed storage-binding limits even for small
  writes into a legal larger buffer. Its parallel overlapping writes also lack
  the generic fallback's same-buffer-type input ordering.
- **Integration:** an independent checked final WebGPU stage after buffer-size
  compatibility. Scalar writes retain their queue order and unaligned-edge
  handling. This adds no global queue budget and does not replace the separate
  loader upload bound or 256-entry/4 MiB checkpoint readback contract. Existing
  grouping across different buffer types is unchanged.
- **No-patch alternatives:** defer this upstream update or restrict affected
  writes; neither preserves existing browser checkpoint behavior generally.
- **Validation and limits:** source-bound CPU controls check the hook, generic
  fallback, ordering and failures. Runtime build/smoke checks are recorded
  separately; they do not establish real-browser or model correctness.
- **Removal:** upstream batch writes meet binding, scratch, representation and
  overlapping-write contracts, or the compatibility requirement is withdrawn.

## Upstream audio generation

Generation, graph/state layout and public audio-helper headers follow the
selected upstream. The runtime adds no downstream final-frame waveform
optimization, automatic-language conditioning or capability query for those
extensions. Expanding that boundary requires a scope-specific upstream-patch
review. Future upstream support follows ordinary update review rather than a
patch that freezes older behavior.

## Other existing deviations

The Emscripten Asyncify/BigInt correction is an individual repository-root
`toolchain/patches/emscripten-asyncify-bigint.patch`, applied through the owned SDK
preparation boundary. It uses the same atomic, context-based file operation as
the declaration correction below; its behavior remains a separate patch policy.
The pinned link-option correction in `CMakeLists.txt` is another build workaround.
The same approval policy applies to expanding these changes. The Asyncify
correction remains separately reported.

### SDK declaration signatures

- **Patch:** repository-root `toolchain/patches/emscripten-typescript-signatures.patch`.
- **Scope:** correct SDK 6.0.9 `--emit-tsd` to use the same final `native_sigs`
  mapping as its JS generation, including explicit signature conversions and
  SDK helpers. MEMORY64 p/P slots use Number; non-converted i64 uses primitive
  bigint. Existing JSPI export selection and all generated runtime behavior
  remain unchanged.
- **Coupling:** the SDK declaration phase follows JS generation, which populates
  the conversion mapping. Preparation uses normal forward/reverse patch checks
  and atomic installation into an owned SDK. Patch content identity participates
  in cache partitions and toolchain provenance. Standard CMake link dependencies
  bind warm rebuilds to the SDK generator and signature definitions.
- **No-patch alternative:** defer the affected generated-type contract until
  upstream corrects it. Consumer casts, replacement types or changing the SDK's
  runtime Number conventions would not establish correct public declarations.
- **Evidence and limits:** an actual SDK C fixture, strict imports of its emitted
  declarations, and Node runtime checks cover 32-bit Asyncify, 64-bit direct and
  64-bit JSPI exports, including actual i64 rewind. These are not browser/GPU or
  model-inference tests.
- **Removal:** retire the patch and preparation once upstream generates the
  matching signatures; keep the actual generated-type/runtime regression cases.

## Inventory and publication

The runtime preparation recipe, CMake hooks and updater preflight select these
patches. Preparation receipts record the actual patch paths/digests, affected
files, selected inputs and final outputs. Publication compares those facts with
the source-bound build inputs. Keep source changes in this reviewable inventory;
do not hide a divergence in an unrelated generator or helper. A successful
preflight is not proof of numerical or browser compatibility.

Removing source changes cannot alter an already published artifact. Rebuild,
validate, and publish the cleaned source; update consumers to that actual new
artifact through their normal verified update flow. Do not republish an existing
commit with different bytes or invent a future artifact identity.

## Generated browser Wasm URL annotation

- **Accepted scope:** every ordinary browser profile, including CPU, receives one `/* @vite-ignore */` comment
  immediately before the first argument of the generated sibling Wasm URL.
- **Integration:** `annotateBrowserModule` in `runtimes/llama-cpp/tools/build.ts`
  runs after the normal link. A JavaScript syntax tree identifies the single
  literal `new URL("core.wasm", import.meta.url)` boundary, allowing equivalent
  parentheses and literal spelling. Comments, strings and unrelated asset URLs
  are not Wasm targets. Unsupported or multiple Wasm URL expressions require
  review. The annotation is inserted at the original argument offset without
  printing the syntax tree; every other generated byte is preserved. An existing
  directive comment makes a skipped-link rerun a no-op. This recognizes the known
  SDK syntax, rather than resolving arbitrary JavaScript URL construction.
  Test variants, native builds, upstream sources and the SDK are unchanged.
- **Behavior and cost:** Vite no longer discovers/copies this Wasm from that URL
  in the default browser entry. Hosts explicitly deploy it, provide `locateFile`,
  or supply `wasmBinary`. Runtime acquisition/fallback code is unchanged. An
  Emscripten output-shape change requires review of this boundary, without a
  whole-module hash allowlist or a new distribution variant.
- **Evidence:** per-build provenance records actual input/output byte counts and
  hashes and whether preparation was a no-op.
  Publication inventories and consumer reports retain that generated processing;
  unit fixtures do not establish compilation or browser/GPU success.
- **No-annotation alternatives:** external raw asset deployment, a build-tool
  integration that handles the generated module as a prebuilt chunk, or retaining
  bundler-managed sibling copying. These have different output-ownership costs.
- **Removal:** remove the preparation and its build/report/cache integration if
  upstream emits the needed hint, or if hosts no longer need this asset boundary.
  See [browser host integration](../docs/browser-host.md).

## Retired WebGPU MoE direct selected-slot overlay

- **Retired patch:** `ggml-webgpu-moe-direct-slot.patch`.
- **Reason:** v0.7.0 implements the same selected-slot vector-kernel dispatch
  upstream, including broadcast addressing and trailing-workgroup guards.
- **Removal:** patch, option, preparation, CMake include override and old overlay
  tests were removed together. The upstream-generated shader header is used by
  every WebGPU profile. Separate source/dispatch and stale-header tests remain.
- **Reporting:** the retired patch is absent from the active preparation recipe.
  Existing published manifests and identities are not rewritten.
- **Review and limits:** [MoE retirement](../docs/moe-direct-slot.md) and
  [v0.7.0 source review](../docs/llama-v0.7.0-review.md). No v0.7.0 hardware-GPU
  throughput claim accompanies the removal.

## Same-device WebGPU tensor copy

- **Patch:** `ggml-webgpu-same-device-tensor-copy.patch`.
- **Purpose:** eligible buffer copies avoid the generic host read/write fallback.
- **Scope:** existing optional copy hook with same-device/layout, byte-range,
  resource-identity, alignment and usage checks. No model or checkpoint policy.
- **Integration:** shared checked build-tree source for every WebGPU profile;
  ordinary build and publication paths, without a separate optimization option.
- **Inputs, tests and limits:** reviewed v0.7.0 backend/queue and fallback contracts
  and pinned Dawn headers. See [design and tests](../docs/webgpu-tensor-copy.md).
  Mock/header checks do not certify browser/GPU correctness or performance.
- **No-patch alternative:** generic host copies or upstream support.
- **Removal:** equivalent upstream behavior or withdrawal of the requirement;
  remove patch, preparation, hook, tests and provenance entry together.

## WebGPU parameter upload batching

- **Patch:** `ggml-webgpu-batch-param-uploads.patch`.
- **Scope:** aligned parameter uploads at existing submission boundaries; no
  tensor/weight upload, synchronization or shader changes.
- **Integration:** composed with tensor copy in the shared build-tree translation
  unit for every WebGPU profile. CPU profiles remain unchanged.
- **Cost:** bounded CPU mirror and padded upload bytes. Fewer calls alone are not
  a speedup claim. Source, patch and combined-output identities are checked.
- **Tests, alternatives and removal:** [parameter upload batching](../docs/webgpu-param-upload-batching.md).

## Same-queue blocking WebGPU readback

- **Patch:** `ggml-webgpu-ordered-readback.patch`.
- **Scope:** register a blocking `get_tensor_async` hook. For a nonempty read from a WebGPU buffer on the backend's own queue, skip the generic fallback's preliminary queue wait and use the existing copy, submit and blocking map. Foreign buffers and zero-byte reads retain the preliminary synchronization. The hook returns only after the destination has been filled; it is not a nonblocking readback API.
- **Integration:** apply after the other WebGPU source overlays to the checked build-tree translation unit for every WebGPU profile. CPU profiles and the pristine upstream checkout are unchanged. Review the generic backend and queue contracts and selected SDK declarations. Preparation records the source, patch and final output identities.
- **No-patch alternative:** retain the existing extra synchronization before each async fallback read or wait for an equivalent upstream implementation.
- **Validation and limitations:** source-bound deferred queue tests exercise ordering, ranges, destination lifetime and failure paths; full source syntax checks use the pinned headers. These checks do not establish browser correctness, device-loss recovery or target-device speedup. The queue-order assumption must be validated on the target browser and GPU.
- **Removal:** upstream provides equivalent behavior or the owner drops this optimization; remove the patch, preparation, build wiring, tests and provenance entry together.

## Ordered WebGPU async tensor copies

- **Patch:** `ggml-webgpu-async-tensor-copy.patch`.
- **Scope:** implement the existing optional copy hook for checked same-queue WebGPU copies and CPU-to-WebGPU uploads. CPU producers are synchronized before their host bytes are uploaded; unsupported devices, layouts, ranges, aliases or usages retain generic fallback. Queue ordering and the existing buffer-copy checks remain in force.
- **Integration:** apply in the checked build-tree WebGPU translation unit for WebGPU profiles; review the generic backend, buffer and scheduler contracts and record patch, composition input and output identities. CPU profiles and upstream source are unchanged.
- **Validation and limits:** source-bound CPU queue and scheduler fixtures cover accepted and rejected paths, producer completion, offsets, alias and fallback, with full source syntax checks against pinned headers. Software GPU checks do not establish browser correctness or target-device speedup.
- **No-patch alternative:** use the existing generic copy fallback or wait for equivalent upstream support.
- **Removal:** upstream supplies equivalent behavior or the owner drops the optimization; remove patch, preparation, build wiring, tests and provenance together.

## Optional WebGPU host-input snapshot

- **Patch:** `ggml-webgpu-host-input-snapshot.patch`.
- **Scope:** add a versioned optional registry contract between the scheduler and the WebGPU provider. For a single-copy, event-free split with a non-weight host input, the scheduler may try the callback before its existing wait. The first input in each split retains the original wait path, bounding queued work even across repeated prefill calls without output readback; only later inputs may skip their waits. Success synchronizes the CPU producer, snapshots the source bytes before returning, and orders the upload on the destination compute queue. Unsupported or absent callbacks retain the existing scheduler path. Host weights, multi-copy rotation and event paths retain their original behavior.
- **Integration:** every WebGPU profile compiles checked build-tree copies of both `ggml-webgpu.cpp` and the scheduler's `ggml-backend.cpp`, plus their shared private contract header. The latter is a downstream registry semantic ABI and affects the scheduler in the WebGPU-enabled build; public backend structures and bindings are unchanged. CPU-only profiles and pristine upstream sources remain unchanged. Pinned source, registry, allocator and backend contracts, patch and all output identities require review on drift.
- **Validation and limits:** source-bound scheduler/provider tests and target-configuration checks cover callback discovery, eligibility, fallback, ordering and producer lifetime. These checks do not establish browser/device-loss behavior, production model quality or target-device speedup. Existing backend and scheduler lifetime and synchronization obligations remain in force.
- **No-patch alternative:** retain the existing scheduler waits and narrow async-copy hook or wait for equivalent upstream support.
- **Removal:** equivalent upstream support or withdrawal of the optimization; remove patch, preparation, target replacement, tests and provenance together.

## Bounded WebGPU bind-group reuse

- **Patch:** `ggml-webgpu-bind-group-cache.patch`.
- **Scope:** retain up to 8192 bind groups per backend context by graph dispatch position, with at most 16 buffer entries per group. Reuse requires matching pipeline, group index, every binding number, buffer identity, offset and size; the parameter arena binding is included. Unsupported or larger bindings bypass the cache, and parameter uploads and dispatches remain unchanged. Shorter graphs trim unused tail entries.
- **Lifetime:** a device-global generation advances before backend buffer destruction; each context clears stale entries before its next lookup. Cache entries hold WebGPU references, but do not change the existing scheduler's obligation to keep in-flight resources valid. The limits bound counts, not retained GPU memory in bytes; references may keep resources alive until eviction, invalidation or context teardown. Device loss and allocation failure recovery are unchanged.
- **Integration:** apply to the build-tree WebGPU translation unit after ordered readback. Review upstream source, generic backend contracts and Dawn declarations; record patch and composition identities. CPU profiles and the upstream checkout are unchanged.
- **No-patch alternative:** create a bind group for each dispatch as upstream does, or wait for equivalent upstream reuse.
- **Validation and limits:** complete-source native queue and lifetime fixtures cover key equality, position, generation, bounds, fallback and parameter uploads, and the complete composed source is syntax-checked against pinned headers. These checks do not establish browser correctness, actual model hit rate, VRAM savings or target-device speedup.
- **Removal:** upstream supplies equivalent reuse or the owner drops this optimization; remove patch, preparation, build wiring, tests and provenance entry together.

## Single synchronization for WebGPU sampler reads

- **Patch:** `llama-sampler-single-sync.patch`.
- **Scope:** in `llama_sampler_sample`, synchronize the context once before reading its internal sampled outputs and counts. If sampled logits are absent, use the existing ordinary logits fallback after that synchronization. Public getters keep their original behavior. CPU profiles retain the upstream sampler.
- **Integration:** for every WebGPU profile, prepare a checked build-tree sampler translation unit and replace its source on the existing llama target. The selected upstream commit and accepted patch identify its inputs; actual input/output bytes are recorded as provenance. Public upstream declarations and complete prepared translation-unit compilation check type compatibility. The sampler remains independent of the model, loader and mtmd overlays; upstream source is unchanged.
- **Evidence and limits:** the baseline tiny-model Wasm observation found five queue-completion calls during one sampling operation with no intervening enqueue. Typed native fixtures exercise 107 output, sampler and scheduler cases, including public getters and the logits fallback; the complete prepared sampler is compiled with actual upstream headers. Harmless source comments remain accepted, while patch conflicts and incompatible dependency types fail. These checks do not establish target-browser correctness, model quality or throughput improvement.
- **No-patch alternative:** retain the upstream public getter synchronization behavior or wait for an equivalent upstream implementation.
- **Removal:** upstream provides equivalent synchronization or the owner drops the optimization; remove patch, preparation, target replacement, tests and provenance entry together.

## Bounded synchronous WebGPU model upload

- **Patch:** `llama-model-loader-webgpu-chunked-upload.patch`.
- **Scope:** only the existing non-mmap/non-host synchronous fallback for the
  default WebGPU buffer uses bounded reads. In this source tree, the normal build
  enables this path in every WebGPU profile; CPU profiles retain the upstream
  loader.
- **Approval policy:** the user decides whether adoption justifies the continuing
  upstream-maintenance cost; normal build routing does not grant that approval.
- **Why a patch:** the current loader allocates one read vector per whole tensor;
  backend offset uploads already work, but no upstream loader setting bounds it.
  No-patch alternatives are retaining that allocation or waiting for upstream.
- **Integration:** reuse ordinary WebGPU source preparation and its CMake hook to
  replace only `src/llama-model-loader.cpp` on `llama`. No additional options,
  variants, workflow inputs or publication gates. CPU builds stay upstream.
- **Dependencies:** the selected loader, file-reader implementation, backend/API
  declarations and target widths. The private coherent source tree keeps these
  consumers consistent. Updater preflight and publication use the same typed
  preparation. Review changed ownership, read and upload behavior rather than
  treating every changed input digest as incompatibility.
- **Behavior:** retain at most 8 MiB of explicit read-vector storage, complete
  quantization blocks and four-byte intermediate upload alignment. A loader-local
  helper carries a provisional 32 MiB charged upload budget across tensors. Each
  chunk, including a smaller aligned or tail chunk, reserves its actual payload
  plus 24 bytes before `tensor_set`; exceeding the
  remaining budget drains the actual destination device first. Device handoff,
  final partial batches, callback cancellation and ordinary C++ exceptions drain
  pending work, and the helper releases its backend through RAII. No global
  tensor-set, inference or WEIGHTS-buffer waits are added. Other buffer types and
  backends retain their existing upload behavior and optional validation. A
  smaller aligned chunk can admit four charged uploads per batch, but is selected
  only when it preserves read/upload counts and does not increase the projected
  `(wait count, pending charge)` state. Pure constant-time selection accounts for
  the current queue or a device handoff; it does not acquire or wait for a backend.
- **Validation/limits:** compilation checks the actual complete loader/helper
  against the selected upstream headers. Important behavioral properties are
  pre-write reservation, cross-tensor/device accounting, auxiliary charges,
  overflow, cleanup and initialization failure. A copied helper is not evidence
  that the production loader uses it. The 32 MiB bound covers charged loader
  uploads, not total Wasm, process, driver or GPU memory. Upstream direct-I/O
  bounce buffers are outside both stated bounds; the Windows branch is not
  compiled in browser builds. Actual pending peaks can rise from three to four
  chunks within the cap. Lower read-vector maxima do not establish unchanged
  physical or Wasm high-water memory. Cancellation cannot interrupt a hung wait;
  graceful OOM/device-loss recovery is not claimed. See
  [bounded WebGPU model uploads](../docs/webgpu-model-upload-budget.md).
- **Removal:** equivalent upstream support, withdrawal of the requirement, or costs
  exceeding measured benefit; remove this patch and its entries in the shared
  preparation, hook, tests and provenance without disturbing other overlays.


## Single-token WebGPU SSM convolution

- **Patch:** `ggml-webgpu-ssm-conv-single-token.patch`.
- **Scope:** select one token lane for SSM_CONV when output tokens per sequence equal one; retain eight lanes otherwise. Channel block size, scalar/length-four arithmetic, upstream WGSL source-file bytes and bounds checks remain unchanged. The preprocessed shader workgroup size changes for single-token input. Include the selected width in the existing pipeline cache key, equality/hash and dispatch decisions.
- **Integration:** the ordinary `LCB_WEBGPU` source preparation emits a checked `ggml-webgpu-shader-lib.hpp` beside its existing WebGPU source copy. It does not emit `ggml-wgsl-shaders.hpp`; the upstream-generated header remains authoritative after MoE overlay retirement. No new workflow, profile, variant or feature switch.
- **Dependencies:** the selected WebGPU factory/header and `ssm_conv.wgsl`. Review token-width specialization, source bounds and key ownership on upstream changes. Preparation records actual patch/input/output identities.
- **Evidence and limits:** isolated shader correctness and tiny LFM full-runtime tests pass on software WebGPU adapters, including both pipeline creation orders and full/partial checkpoint restoration. Those tests do not establish production model quality, physical GPU occupancy or a Mac throughput improvement. Other MoE operators are not modified.
- **No-patch alternative:** retain the existing eight-token launch for decode. The change removes inactive logical token lanes; it does not guarantee a performance improvement.
- **Removal condition:** remove this preparation/header patch when upstream selects an equivalent single-token workgroup and the same supported profiles pass correctness and device validation, or when the owner drops the optimization.

## Qwen4Exp file-backed per-layer embedding rows

- **Patch:** `qwen4exp-file-backed-rows.patch`.
- **Approval scope:** the project owner's request to implement the reviewed
  Qwen3.8 browser design after explicitly discussing llama.cpp patch size.
  This authorizes this Qwen4Exp table path, not unrelated model/backend work.
- **Base:** v0.7.0, `620d2225cc436ab0b001e3200e590e3b286ef9ab`.
  Preparation records the selected source and final coherent outputs.
- **Reason / alternatives:** this table is 26.82 GiB in the investigated model,
  larger than the current 16 GiB Wasm64 linear memory limit. Existing mmap-based
  lazy reading is not provided by the browser file adapter. Deferring the model,
  changing its quantization, or using native inference avoids a downstream patch,
  but does not implement the requested browser path. Native threaded pread work
  cannot be copied unchanged to a non-pthread browser runtime.
- **Scope:** Qwen4Exp PLE (Per-Layer Embedding), 64-bit profiles,
  `LCB_FILE_BACKED_PLE` plus per-load `LLAMA_LOAD_MODE_NONE` /
  `LLAMA_LAZY_MODE_ON`. Non-Qwen4Exp stays eager for this combination; explicit
  mmap/AUTO behavior is unchanged. Export of a file-backed model is rejected
  before creating an incomplete output. Other weights,
  n-gram hash computation, downstream graph, GGML kernels and public llama.h
  remain unchanged. Shared model header users compile coherent build-tree copies.
  Before publication, a retained descriptor is matched by stable filesystem
  identity to exactly one entry in the original public file array. Missing or
  ambiguous matches fail instead of exposing a compacted internal file index.
  The leaf `file-backed-identity.h` is copied into that same header universe and
  included in compiled-source, coherent-set and supporting-source provenance.
  The private loader layout and public source-index contract are unchanged.
- **Validation / limitations:** see `docs/file-backed-rows.md`. Synthetic native
  graph equivalence is not a real-model WebGPU inference claim.
- **Removal condition:** pinned upstream offers a tested non-mmap bounded row
  reader with compatible ownership and browser behavior, or the owner drops
  the requirement. Re-run lifetime, split, graph equivalence and browser tests
  before removing the overlay and retiring the versioned consumer handshake.


## QSA all-pools selection

- **Patch:** `qwen4exp-all-pools-selection.patch`.
- **Accepted scope:** skip redundant ranking when every padded candidate pool
  fits the selection budget, preserving the exact visibility and index-range
  guards. No model repository, quantization-name or device-name branch is used;
  no longer-context GPU kernel is bundled here.
- **Why / alternatives:** the pinned graph scores and ranks candidates even when
  every padded pool fits its budget. No existing runtime option removes that
  redundant work. Keeping the upstream graph, or deferring until equivalent
  upstream support, avoids this maintenance cost.
- **Dependencies:** the selected upstream Qwen graph, model declarations and
  cache/mask behavior. Positive rectified scores, zero/-infinity visibility, and set-valued use of
  selection are required. Raw/pooled key updates and the ranked path stay intact.
  Selection sharing requires the same graph-local pool input and mask type;
  upstream's F16/F32 selection, fused ranked mask and scatter fallback are retained.
- **Integration:** `LCB_QSA_ALL_POOLS` is independent of file-backed PLE. A single
  private `qwen4exp.cpp` is composed after that optional overlay, keeping its
  model header layout. Source is not edited in the vendor checkout. Build commands,
  updater preflight and per-profile final-source provenance cover both variants.
- **Validation / limits:** native full-model parity on small Q3_K/Q4_K/F16
  fixtures, operation coverage, state/sequence/mixed-input cases, configuration,
  provenance and negative controls. These are not real Q3/Q4 weights or an Apple
  WebGPU throughput claim. See `docs/qsa-all-pools-selection.md`.
- **Removal condition:** equivalent upstream selection, semantic inputs no longer
  meeting the proof, or withdrawal by the owner. Remove the option, preparation,
  hook, exception and reporting together without disturbing file-backed rows.

## Four-stream HC PRE/POST fusion

The narrow overlay `ggml-webgpu-hc-pre-post.patch`
adds WebGPU execution of existing gated PRE and identity POST operations for
contiguous F32 HC4 only. No graph rewrite or extra profile is introduced. The
no-patch option is the existing staged graph. See
[`webgpu-hc-pre-post.md`](../docs/webgpu-hc-pre-post.md) for scope, maintenance,
removal, continuous software-GPU validation and remaining physical-GPU/model
measurements. Local preparation, host tests and standalone numerical checks do
not establish target-device speed.

## Q8_0 short-input matvec workgroups

- **Patch:** `ggml-webgpu-q8-short-workgroup.patch`.
- **Purpose:** choose the existing upstream workgroup specialization for
  guarded short-input matvecs. No model change is included.
- **Scope:** subgroup Q8_0 x F32 single-column matvecs with K <= 1024 and at least
  1024 output rows. Preserve other paths, WGSL, arithmetic, quantization, tiling
  and dispatch count. Include the workgroup in the pipeline cache identity.
- **Integration:** one checked shader-library header composed after HC; no
  modified vendor checkout or copied/modified shader implementation. The
  existing WebGPU profile switch is the only build switch.
- **No-patch alternative:** retain upstream's 256-lane workgroup; public runtime
  settings do not offer this shape-specific selection.
- **Coupling and validation:** the cache key/factory and the transitive shader
  contract are pinned. See [design and validation](../docs/webgpu-q8-short-workgroup.md).
  Software-GPU checks do not establish Mac/model speed or universal bit identity.
- **Removal condition:** upstream provides a suitable selection, the owner drops
  the goal, or target-device measurements reject the change. Remove preparation,
  CMake inputs, provenance and this exception together.

## Ordered expert-output sums

- Patch: `ggml-webgpu-moe-expert-sum.patch`.
- Scope: the owner's continuing Qwen acceleration and ordered expert-sum candidate.
- Replace only eligible complete F32 expert-output ADD chains with one dispatch.
- Preserve slot/order, staged fallback, backend ownership, disjoint aligned bindings.
- Alternative: retain upstream staged ADDs; no model pipeline or variant changes.
- Maintenance: pinned graph, allocator, fusion, scheduler and shader contracts.
- Tests, limits and removal: [ordered sums](../docs/webgpu-moe-expert-sum.md).

## Short Q4_K matvec workgroups

- **Patch:** `ggml-webgpu-q4-short-workgroup.patch`.
- **Scope:** choose an existing 32/64/128/256-lane subgroup workgroup for eligible Q4_K x F32 single-column ordinary and indexed matvecs with complete K blocks through 4096. Shape, subgroup, workgroup-limit, overlap and indexed-slot guards preserve the upstream 256-lane fallback elsewhere. WGSL arithmetic, quantization, output tiling and dispatch grid stay unchanged; existing pipeline keys include the selected width.
- **Integration:** apply after the other WebGPU host overlays to the checked build-tree shader library for every WebGPU profile. Review the upstream shader/host contracts; record the selected patch, input and final header identities. No shader file or upstream checkout is rewritten.
- **No-patch alternative:** retain the upstream fixed workgroup or wait for upstream shape-dependent selection.
- **Validation and limits:** source-bound factory and dispatcher tests exercise selection, cache keys and fallback. The opt-in standalone GPU fixture compares actual generated shaders with the upstream path and an independent CPU reference. These checks do not establish browser model quality or target-device performance.
- **Removal:** upstream supplies equivalent selection or the owner drops the optimization; remove its patch, preparation, build wiring, tests and provenance entry together.

## Fused WebGPU scale and unary operations

- **Patch:** `ggml-webgpu-scale-unary.patch`.
- **Scope:** fuse eligible contiguous F32 scale followed by SILU or sigmoid, with an optional final scale, into one dispatch. Admission checks graph usage, supported layout and binding ranges; unsupported or unsafe combinations retain upstream dispatches. The fusion preserves the operation order and does not change model construction.
- **Integration:** compose after existing WebGPU host overlays in every WebGPU profile, generating an isolated shader header and host helper. Reviewed graph, allocator, backend and upstream shader contracts, patch and final output identities are recorded. CPU profiles and upstream source are unchanged.
- **Validation and limits:** source-bound graph, allocator, dispatch and shader factory tests cover accepted and rejected graphs and fallback. Software GPU checks do not establish model quality or target-device speedup.
- **No-patch alternative:** use the existing separate dispatches or wait for upstream fusion.
- **Removal:** equivalent upstream fusion or withdrawal of the optimization; remove patch, preparation, build wiring, tests and provenance together.

## Fused WebGPU RMS normalization and scale

- **Patch:** `ggml-webgpu-rms-scale.patch`.
- **Scope:** fuse eligible F32 RMS normalization followed by scale into one dispatch, retaining upstream reduction and normalization before the scale. Existing RMS-multiply fusion keeps priority. If the following scale begins an eligible scale-unary-scale chain, RMS fusion declines so the existing three-operation fusion remains available; RMS-scale takes priority over the two-operation scale-unary case. Unsupported graphs, layouts and binding ranges retain upstream behavior.
- **Integration:** compose the checked host and shader overlays in every WebGPU profile. Embed the modified row-normalization shader separately so the canonical upstream shader remains authoritative for other paths. Review graph, allocator, shader and backend contracts and record the generated header identity. CPU profiles and upstream source are unchanged.
- **Validation and limits:** source-bound graph, allocator, alias and dispatch tests and 128 software WebGPU cases passed. These checks do not establish production model quality or target-device throughput.
- **No-patch alternative:** retain separate RMS and scale dispatches or wait for upstream fusion.
- **Removal:** equivalent upstream fusion or withdrawal of the optimization; remove patch, preparation, build wiring, tests and provenance together.

## Fused WebGPU F32 L2 clipping

- **Patch** `ggml-webgpu-l2-clip.patch`.
- **Scope** Fuse an eligible square, row sum, square root, epsilon addition, threshold fill, division, clamp and multiplication into one dispatch. Admission requires dense F32 tensors of shape `[128,24,T,1]`, `[128,8,T,1]` or `[16,24,T,1]` for `1 ≤ T ≤ 128`, the original vector as the final multiplication input, and the checked coefficients and clip bounds. The dispatch uses one workgroup per head and token. Graph output, use-count and graph-wide raw view checks protect observable intermediate values across scheduler slices. Overlapping binding ranges use the separate operations unless input and output have identical tensor offsets and layouts.
- **Integration** Normal and GPU-profiling WebGPU builds embed the same canonical shader. The patch uses the existing RMS-scale layout and binding helpers and the upstream `SUM_ROWS` reduction order for the populated lanes. GPU-profiling builds record the fused dispatch through the existing timestamp path. CPU builds do not apply the patch. The initial pin is llama.cpp `c212d25ad`. Review the graph view, allocator, `SUM_ROWS` and shader embedding contracts when updating it.
- **Validation and limits** Complete-provider tests exercise single-token and batched admission, actual dispatch encoding, exact allocation alias, and raw views crossing a graph slice. Linked Wasm64 software WebGPU checks compare output words with the separate operations, including authored exceptional values in the single-token tests. A 128-token synthetic GGML graph and an authored Inkling model with a five-token prompt both produced identical output words. Physical-device throughput and model quality remain unmeasured.
- **No-patch alternative** Keep the existing individual WebGPU dispatches until upstream provides equivalent fusion.
- **Removal** Remove the patch, shader preparation, and focused provider checks when upstream provides equivalent behavior or the optimization is retired.

## Fixed-subgroup Q8_0 matvec

- **Patch:** `ggml-webgpu-q8-subgroup-matvec.patch`.
- **Scope:** for guarded long-K Q8_0 x F32 single-column matvecs with a fixed reported subgroup size, assign output rows to independent subgroups in one workgroup. Preserve the pinned upstream Q8 decoder and accumulator; unsupported shapes, device limits and varying subgroup ranges use the existing path. No model-specific selection is added.
- **Integration:** compose after scale/unary fusion in the shared WebGPU header, emitting an isolated shader header using the pinned upstream include expander. Review upstream shader/host contracts and record the selected patch, shader inputs and final outputs. CPU profiles and upstream source are unchanged.
- **Partial subgroups:** even with a fixed reported width, a subgroup can be partial. Each active invocation covers additional logical lanes, preserving coverage without shared scratch; the full-packed path retains its previous calculation. Geometry probing and readback are test-only and add no production CPU readback.
- **Validation and limits:** native selector, cache-key and dispatcher contracts are separate from GPU arithmetic checks. Numerical comparisons must include partial geometries and the supported fixed subgroup widths, retain input/guard checks and report error against the canonical shader. Arithmetic and reduction order differ from the upstream baseline; production model quality and target-device speedup are not established.
- **No-patch alternative:** retain the existing matvec path or wait for equivalent upstream support.
- **Removal:** equivalent upstream implementation or withdrawal of the optimization; remove patch, preparation, build wiring, tests and provenance together.

## Fixed-subgroup indexed IQ3_S matvec

- **Patch:** `ggml-webgpu-iq3s-id-subgroup.patch`.
- **Scope:** assign rows of eligible single-token IQ3_S indexed matvecs to independent fixed-width subgroups. Shape, strides, complete blocks, binding ranges, device limits and subgroup width are checked; unsupported cases retain the upstream path. The upstream quantization loaders and accumulator are reused, and selection is independent of model identity.
- **Integration:** compose the checked WebGPU host and shader overlays after the other WebGPU changes. Embed the dedicated shader with the pinned upstream include expander; retain canonical upstream shader sources for other paths. Upstream contracts require review; the applied patch and generated header identities are recorded. CPU profiles and upstream source are unchanged.
- **Partial subgroups:** even with a fixed reported width, a subgroup can be partial. Active invocations cover additional logical lanes; the full-packed path retains its previous calculation without extra shared scratch. Geometry probing and readback are test-only and add no production CPU readback.
- **Validation and limits:** source-bound selector, cache and dispatch checks and software GPU numerical fixtures cover accepted and rejected cases, including partial geometries at reported widths 32/64. A full-packed width 16 matched the previous subgroup shader bit for bit in the tested cases. Reduction order differs from the upstream baseline; production model quality and target-device speed are not established.
- **No-patch alternative:** retain the existing indexed matvec path or wait for upstream support.
- **Removal:** equivalent upstream support or withdrawal of the optimization; remove patch, preparation, build wiring, tests and provenance together.

## Lane-selected fields in indexed IQ3_S matvec

- **Patch:** `ggml-webgpu-iq3s-field-loads.patch`.
- **Scope:** replace only the existing independent indexed IQ3_S embedded shader. Each half-lane reads its selected qs word, and qh/sign/scale fields use their containing aligned word. Keep the 110-byte block representation, decoder arithmetic and addition order, partial-subgroup main, IDs, bindings, selector, cache key and fallback paths unchanged. Canonical templates and other quantization/ordinary/cooperative shaders are unchanged.
- **Integration:** append a source-bound stage after the original IQ3 stage. Patch only a private accumulator copy and regenerate the dedicated header with the pinned upstream emitter. Record the actual final compiled header in provenance. No additional runtime buffers, probes or synchronization are introduced.
- **Validation and limits:** actual factory output for logical widths 16/32/64 is source-bound. Synthetic numerical comparisons covered width16 full and reported-width32 partial execution, input updates, skipped IDs, binding ends and guards. Target-device performance, full hardware width32/64, Mac/Metal and complete-model quality/performance are not established. Source read-expression counts are not physical bandwidth measurements.
- **No-patch alternative:** retain the existing independent indexed matvec or wait for upstream equivalent field loads.
- **Removal:** upstream equivalent support, withdrawal of this optimization, or removal of its parent indexed IQ3 implementation; remove this stage, patch, build dependencies, tests and provenance together.

## Fixed-subgroup indexed IQ4_NL and NVFP4 decode

- **Patch:** `ggml-webgpu-quantized-expert-subgroup-decode.patch`.
- **Scope:** use independent 16- or 32-lane subgroups for single-token IQ4_NL or NVFP4 indexed matvec with F32 activations, K1152 or K2304, M2304 and eight selected experts. Select by tensor shape, physical binding bounds and device limits. Preserve the existing IQ3_S scope and the canonical fallback.
- **Integration:** both WebGPU modes share the existing independent subgroup shader, quantization loaders and partial-subgroup arithmetic. This stage reuses its buffer and dispatch checks with the corresponding GGML block traits. The upstream revision and applied patch remain in the preparation receipt.
- **Validation and limits:** complete provider tests cover fixed and varying widths, boundary binding limits and unchanged IQ3_S selection. Software WebGPU replay of representative packed tensors covers the two K sizes and fixed widths. Independent reductions change F32 addition order. Physical-device throughput and model-level quality require measurement.
- **No-patch alternative:** retain the existing cooperative indexed decode path.
- **Removal:** upstream equivalent support or withdrawal of the optimization. Remove this stage, tests and provenance together.

## Short Q5_1 / IQ4_NL expert matvec workgroups

- Patch: `ggml-webgpu-expert-short-workgroup.patch`.
- Scope: the continuing user-authorized Qwen speed investigation, following the
  separately reported expert-down experiment. Reuse upstream `WG_SIZE` for a
  guarded 128-lane single-token path; no new shader, model graph or quantization.
- No-patch alternative: keep upstream's fixed 256-lane path. There is no public
  upstream setting for shape-dependent workgroup selection. This avoids the
  maintenance cost but keeps lanes without input work in the measured K=640 path.
- Coupling: two narrow C++ changes (factory/cache and subgroup-limit forwarding),
  pinned to llama.cpp `620d2225cc436ab0b001e3200e590e3b286ef9ab`. The selected-slot
  shader, lane mapping, reduction, preprocessor and composed source are checked.
  Existing HC/Q8/expert-sum composition ordering is explicit, not bypassed.
- Limits and validation: see `docs/webgpu-expert-short-workgroup.md`. Retain the
  unchanged 256-lane fallback for all unreviewed shapes, types and device limits.
  Software-adapter correctness/timing does not establish real-model Mac speed.
- Removal condition: upstream supplies equivalent guarded selection, or the user
  chooses to drop it; remove preparation, cache field if no longer needed,
  build/provenance plumbing and tests together, leaving other overlays independent.


## Fused WebGPU SSM convolution and SiLU

- **Patch:** `ggml-webgpu-ssm-conv-silu.patch`.
- **Scope:** fuse only eligible adjacent F32 SSM_CONV/UNARY-SILU graphs with one use of the convolution result. Preserve canonical scalar/vec4 dot arithmetic, the SiLU expression and existing single-token workgroup specialization. No model graph or CPU behavior changes.
- **Safety and integration:** compose after the other checked WebGPU overlays, for existing WebGPU profiles. Preserve intermediate observers, outputs and callback graph slices through shared fusion checks; validate shapes, u32/grid/device bounds, view roots, buffer ownership and disjoint payload/aligned bindings. The final SiLU may reuse the unused convolution allocation. A dedicated embedded shader keeps canonical sources unchanged; profiling uses only the existing optional profile.
- **No-patch alternative:** retain the existing separate dispatches or wait for upstream fusion. There is no existing runtime setting for this epilogue.
- **Maintenance and validation:** review the graph, allocator, backend, shader and generator contracts, and record the selected patch and output identities. Source-bound native tests exercise graph admission, actual allocator reuse, runtime bindings, dispatch, cache variants and both profile syntax configurations. These tests do not establish browser/model quality or target-device speed.
- **Removal:** upstream implements equivalent guarded fusion or the owner withdraws it; remove the patch, preparation, build wiring, tests and provenance entry together.


## WebGPU MoE sparse columns

- **Patch:** `ggml-webgpu-moe-sparse-columns.patch`. Applied after the inactive column groups and NVFP4 accumulator patches.
- **Scope:** Indexed IQ3_S and IQ4_NL matrix multiplication with scalar F32 input. For tiles of up to eight active columns, each participating lane accumulates one column. Larger tiles use the original grouped assignment.
- **Contract:** Keep the gather metadata, WebGPU bindings, uniform layout, dispatch dimensions, and F16 accumulation order for each output. The existing pipeline key covers weight type, input type and vectorization. The separate indexed single-token kernel keeps its existing selection.
- **Validation and limits:** Compare emitted normal and profile shaders and run numerical and timing comparisons for sparse, mixed and full tiles. Software tests cover authored routing and do not establish end-to-end prefill latency.
- **Removal:** Remove when upstream provides an equivalent sparse tile, or revisit the patch when the indexed matrix tile shape or gather contract changes.

## WebGPU MoE matrix inactive column groups

- **Patch:** `ggml-webgpu-moe-group-skip.patch`.
- **Scope:** skip only matrix register-tile calculation for a local
  four-column group whose first column is outside the gathered tile. Retain the
  fixed column loop and its final partial group; do not change shared loads,
  workgroup barriers, parameters, routing/gather, tiling, grid or final stores.
  No early workgroup return, matvec, CPU or model-specific behavior changes.
- **Integration:** compose after the SSM convolution/SiLU stage. Copy the pinned
  upstream matrix shader into an isolated work tree, apply the one-line guard,
  and use the upstream include expander/emitter for one dedicated header. Only
  the matrix factory selects that symbol; the canonical full shader header,
  gather and selected-slot matvec shaders remain authoritative upstream.
- **Alternatives:** retain upstream padding work or wait for equivalent upstream
  handling. Dynamic column-loop bounds and per-column guards remain unselected;
  they can change code generation and regress fully occupied tiles.
- **Validation and maintenance:** review host/gather/shader contracts; record
  the selected patch, expanded embedding and final outputs. Tests exercise normal preparation,
  actual factory/preprocessor and cache behavior, canonical byte comparisons and
  both profile syntax configurations. Opt-in synthetic GPU checks compare every
  output byte with canonical shaders and verify routing metadata, guards and
  immutable inputs. Software-adapter results do not establish Mac/model speed.
- **Removal:** upstream provides equivalent inactive-group handling, or the owner
  withdraws it; remove the patch, preparation, dedicated embedding, build wiring,
  tests and provenance entry together.

## WebGPU NVFP4 expert matrix accumulator precision

- **Patch:** `ggml-webgpu-nvfp4-expert-accumulation.patch`.
- **Upstream:** introduced against llama.cpp `c212d25ad`.
- **Scope:** the WebGPU expert matrix shader uses F32 products and accumulators
  when an NVFP4 operation requests a BF16 or F32 accumulator. F16 accumulation
  can overflow before the graph applies its weight scale. Shared staging,
  tensor storage and the single-token expert matvec keep their existing layouts.
- **Integration:** apply in ordinary and profiling WebGPU builds, extending the
  existing derived expert matrix shader and separating its pipeline cache keys
  by accumulator precision. The default shader keeps its F16 arithmetic.
- **Alternative:** defer to upstream precision support or use a backend that
  honors the requested range for this model.
- **Validation and maintenance:** exercise the provider's precision selection
  and cache reuse, and compare a synthetic NVFP4 graph against scalar results
  on a software WebGPU adapter. Recheck the GGML precision contract and
  expert matrix paths when rebasing.
- **Removal:** remove the patch and its test when the upstream WebGPU shader
  honors the NVFP4 accumulator request in the derived matrix path.

## Fused WebGPU gather normalization tail

- **Patch:** `ggml-webgpu-router-tail.patch`.
- **Scope:** generic F32 GET_ROWS/RESHAPE/SUM_ROWS/CLAMP/DIV/RESHAPE fusion. Preserve the selected IDs, original maximum-workgroup reduction tree, clamp and direct division. No softmax, sorting, quantization or model-specific behavior changes.
- **Aliasing:** disjoint bindings use three storage buffers. Only one-token, one-workgroup execution may merge overlapping probability/output ranges into one storage binding; all selected loads precede a uniform storage barrier and every store. IDs stay disjoint from the full union. Multi-token overlap retains the existing staged path, with no extra model scratch or weight copy.
- **Integration:** compose near the end of the ordinary WebGPU source preparation, before host assembly and the optional profile-only Q8 diagnostic. Emit only dedicated helper/shader headers. CPU profiles and the base bindings/schema remain unchanged.
- **Coupling:** review the six-node graph and view/use-count contracts, allocator behavior, typed buffers/device limits, WGSL sources and upstream generator. Record the selected host/header inputs, patch and final outputs. Reject intermediate OUTPUT/side users, unsupported views, callback subgraphs and invalid or overlapping ranges before pipeline creation.
- **No-patch alternative:** retain the four staged dispatches or wait for equivalent upstream graph fusion.
- **Validation and limits:** actual-ggml graph/allocator and complete-source dispatcher/factory tests, normal/profile syntax checks and synthetic software-GPU byte comparisons cover the guarded paths. These do not establish target-device speed or full-model quality. See `../../tests/runtimes/llama_cpp/webgpu-router-tail/README.md`.
- **Removal:** remove the stage when upstream provides equivalent ordered gather normalization and alias handling, or the owner drops the optimization.

## WebGPU router epsilon normalization

- **Patch:** `ggml-webgpu-router-tail-epsilon.patch`, introduced against llama.cpp `c212d25ad`.
- **Scope:** admit `SCALE(1, 1e-20)` as the denominator of the existing F32 gather normalization fusion. The workgroup reads the same ordered IDs, reduces the gathered weights with the same tree as `SUM_ROWS`, adds the epsilon and divides directly. Both the disjoint and single-token overlapping variants use separate pipeline keys for epsilon and clamp.
- **Integration:** apply after the gather normalization tail in normal and profiling builds. The existing shader emission list generates both variants from the updated WGSL. The graph, tensor, view and binding guards of the existing fusion still apply.
- **No-patch alternative:** execute the original gather, reduction, scale and division as four dispatches.
- **Validation and maintenance:** the complete-provider test exercises both allocation layouts, clamp and epsilon pipeline selection, uploaded epsilon parameters, and fallback for an exposed denominator or different SCALE parameters. The initial software WebGPU comparison checked sorted IDs and normalized weights for tiny denominators, cancellation and nonfinite values. Confirm the SCALE parameter order and reduction tree at an upstream update. Isolated comparison does not establish target model throughput.
- **Removal:** remove this stage when upstream fuses the ordered gather with epsilon normalization while preserving the existing alias contract.

## Raw-view readers of fused convolution and router intermediates

The `ggml-webgpu-conv-router-view-sources.patch` stage protects intermediates whose storage is observed through a raw GGML view. The upstream fusion check counts `src` edges, while `ggml_view_tensor` can expose a storage root through `view_src` alone. A scheduler slice can omit the reader while sharing the original graph visit set. The convolution and router encoders reuse the existing lazy index of those readers and leave the original dispatches in place when a discarded tensor remains observable. A raw view of the final result continues to observe storage written by the fusion.

Apply the stage in ordinary and profiling builds after the shared encoder and fusion view index are installed. The native complete-provider cases cover clamp and epsilon routing, short-convolution residuals, single-token and batched convolution with SiLU, and readers outside the encoded slice. The recurrent copy still writes the full concatenation and the persistent cell. Remove the stage when upstream reports these readers in fusion admission or the corresponding fusions are removed.

## WebGPU single-token short convolution and state store

- **Patch:** `ggml-webgpu-short-conv.patch`, introduced against llama.cpp `c212d25ad`.
- **Scope:** fuse four-tap F32 `SSM_CONV → RESHAPE → ADD → SCALE` into one dispatch for a single token and sequence. Preserve the tap order and evaluate the residual and scale in their original order. A separate fusion materializes `CONCAT` and writes the usual three-column recurrent state copy in one dispatch using raw 32-bit loads and stores.
- **Admission:** the epilogue accepts disjoint tensors and exact per-channel residual/output reuse. State fusion requires four disjoint aligned binding ranges and keeps both destinations visible. Graph closure checks protect observed intermediates and side consumers. Dynamic taps retain their existing epilogue. Rollback `SET_ROWS`, multi-token shapes and unsupported layouts retain their existing state update.
- **Integration:** compose in normal and profiling builds, emitting both WGSL sources through the upstream generator. The existing convolution/SiLU fusion continues to be tried first.
- **No-patch alternative:** retain the staged convolution, residual math and recurrent copy.
- **Validation and maintenance:** complete-provider cases cover both widths and allocation layouts, independent selection of the two fusions, observed intermediates and overlapping bindings. All 28 llvmpipe fixture pairs matched bit-for-bit for outputs and recurrent state. The fused source retains the F32 operation order, and adapter compilers can reassociate expressions across the former dispatch boundaries. These checks do not establish physical-device model throughput. Recheck the GGML copy destination view, the four-tap expression and the storage access modes when updating upstream.
- **Removal:** remove this stage when upstream provides equivalent convolution and recurrent-state fusion, or the owner drops the optimization.

## Profile-only Q8 kernel diagnostic

- **Patch:** `ggml-webgpu-q8-diagnostic.patch`.
- **Scope:** only `GGML_WEBGPU_GPU_PROFILE` builds add a versioned typed registry provider, fixed-width bridge, separate generated extension schema and explicit Q8 variants. AUTO preserves real production selection; forced variants never replace normal selection. No model data or host application policy is added.
- **No-patch alternative:** compare separate immutable artifacts or defer the diagnostic; neither supplies explicit same-device/same-buffer selection in one runtime.
- **Safety:** bounded contiguous, non-view Q8_0/F32/F32 single-vector tensors, checked physical/logical bindings and device/grid limits; reject unsupported requests before enqueue. PREPARE creates pipelines and records actual processed-WGSL SHA-256; EXECUTE only looks up prepared keys. Same-context operation guards span pipeline and profile-map suspension. Borrowed tensors are not retained; existing bind-group buffer references follow their generation cleanup contract. Hosts serialize all calls and destroy a poisoned runtime after unexpected rejection rather than freeing pending resources.
- **Integration:** apply a final profile-only stage after the other overlays. Normal final source and common generated ABI remain byte-identical. Pin upstream factory/decoder/submit contracts, patch, stage outputs, bridge contract and the owned public-domain SHA-256 header and its attribution. Separate extension records use generated native layout queries and a compiled schema fingerprint; absent exports remain optional for older artifacts.
- **Validation and limits:** focused source-bound native tests cover the ABI, cache, shader identity, dispatch, lifetime and guarded fallback; these do not establish real-device speed or model quality. Only the existing profiler supplies GPU timestamps.
- **Removal:** when upstream provides an equivalent optional diagnostic or the owner withdraws it, remove its profile stage, bridge, schema generator, build/provenance wiring and tests together.

## Synchronous host dispatch assembly

- **Patch:** `ggml-webgpu-host-encode-assembly.patch`.
- **Scope:** reduce temporary owning copies in the existing single/multi WebGPU builders. No new ABI, graph rewrite, kernel, selector, profile or host-application policy.
- **No-patch alternative:** keep the current builders and their temporary allocations; the existing exact bind-group cache already avoids warm creation calls. This change does not improve cache hit rate.
- **Contract:** a stack view borrows single-op parameters, entries and pipeline only until synchronous staging/encoding finishes. Cache and queued command ownership remain unchanged. Profile results retain owning pipeline names. Reserve temporary vectors within the existing arena slot bound and retain original parameter padding, binding key, upload/dispatch order and batch boundaries.
- **Caller observation:** the internal single builder takes const references. Moving named vectors into it no longer empties them. Every current caller returns immediately or only records the fused operation count before returning; none reads the vectors afterward. Recheck those call sites on upstream updates.
- **Integration:** append a checked source stage after router-tail and before the optional profile diagnostic. Review upstream contracts and record the composed input, patch and output. No shader/header or common ABI changes.
- **Validation and limits:** complete-source native tests observe bounded warm allocations and existing cache, queue, free and failure behavior. Normal and diagnostic providers compile the complete final source. Native mock allocations are not browser/GPU timings and do not establish model throughput or target-device improvement.
- **Removal condition:** remove the stage, provenance/build wiring and tests when upstream provides equivalent synchronous assembly without the copies, or the owner drops the optimization.

## Profile query capacity with multi-dispatch operations

- **Patch:** `ggml-webgpu-profile-query-budget.patch`.
- **Scope:** This profile-only correctness repair follows a source-bound counterexample showing that a valid multi-dispatch operation could exceed the existing profile QuerySet before the next collection. It is separate from normal submission tuning.
- **No-patch alternative:** leave profiling disabled or restrict profiled graphs to avoid the problematic boundary. Neither makes arbitrary accepted profiled batches safe.
- **Contract:** every timestamped dispatch consumes one arena slot. Reserve queries using that actual slot capacity, including nominal-batch overshoot, and assert during profile initialization that the capacity is at most half the query limit. Keep collection after complete graph batches; do not flush inside the builder or introduce new encoder/name ownership.
- **Integration:** run a checked profile-only stage immediately before the optional Q8 diagnostic. Review upstream contracts and record the selected input, patch and final output. Normal prepared source and the shared ABI/schema/shaders remain byte-identical.
- **Validation and limits:** the native diagnostic provider executes the actual graph loop and query collector with recorded WebGPU calls. A typed encoding callback supplies controlled multi-dispatch counts; capacity and collection assertions do not depend on a copied graph-loop implementation. These cases do not execute GPU kernels or validate the complete device initializer. Actual profile runtime validation remains a separate check.
- **Removal condition:** remove the stage and its preparation/build/provenance/test wiring when upstream reserves queries for every accepted batch or supplies an equivalent safe collection boundary.


## F32 small-M prefill register tiles

- **Patch:** `ggml-webgpu-f32-small-m.patch`.
- **Scope:** Only F32/F32/F32 matrices with K>=128, M1..4 and N>=5, without subgroup matrices or source overlap, use narrower existing register tiles. Existing N<=4 matvec routing, other types and fallback geometries remain unchanged.
- **Contract:** use the pinned upstream register-tile WGSL and TK8 accumulation unchanged. M1 uses tileM1; M2..4 use tileM4. Both use tileN1 and a 1x32 workgroup. Effective geometry is part of pipeline equality/hash before lookup, and profile names distinguish it. Parameter bytes, binding layout and dispatch grid remain unchanged.
- **Integration:** a checked header-only stage follows host assembly and precedes optional profile stages. Normal/profile shader-library identities change; host source, shared bindings/schema and canonical WGSL inputs do not. No extra runtime profile or native API is added.
- **Validation and limits:** source-bound admission/cache/factory/dispatch checks cover boundaries and guarded fallbacks. Private software-GPU geometry comparisons informed the scope; target-device/model speed and model quality are not established by those results. Revalidate arithmetic, layouts and performance before widening the scope.
- **Removal:** remove this stage and its metadata/tests when upstream provides a suitable equivalent or the specialization is no longer useful. Re-review upstream factory, key, geometry decisions, reg-tile loads/accumulation and dispatch contracts on pin changes.


## MoE gather logical lane groups

- **Patch:** `ggml-webgpu-moe-gather-logical32.patch`.
- **Scope:** Distribute only the existing skipped-slot zero loop among logical groups of at most 32 lanes; the grouping is independent of hardware subgroup features. Preserve expert counting, prefix/tile construction and slot-position atomics.
- **Contract:** shared storage and its expert capacity, existing barriers/atomics, seven parameter words, five bindings, gather grid, cache key and factory label remain unchanged. Partial final groups skip only zeroing; supported input/index/binding contracts are not widened. Matrix and matvec arithmetic are unchanged.
- **Integration:** a checked stage after F32 small-M embeds only the derived gather in a dedicated header and changes its factory source selection. Canonical upstream files and the existing matrix group-skip stage remain intact. Normal/profile library and gather identities are recorded separately from the base ABI/schema.
- **Validation and limits:** source-bound stage, ownership, factory/cache and unchanged-dispatch checks cover skipped/live slots and workgroup boundaries. Private software-GPU checks informed the candidate; benefits are not universal, and target-device/model performance is not established.
- **Removal:** remove the stage and its build/provenance plumbing when upstream supplies an equivalent or it ceases to be useful. Re-review gather ownership, metadata capacity, factory specialization and downstream dispatch ordering on upstream changes.

## Floating-point matmul vec4 layout guards

- **Patches:** `ggml-webgpu-matmul-vec4-layout.patch` and `ggml-webgpu-matmul-vec4-layout-profile.patch`.
- **Scope:** Existing F32/F16 ordinary and indexed matrix/vector factories selected vec4 on shape alone, allowing accepted padded or offset views to read rounded-down addresses. This repair uses the existing scalar fallback.
- **Contract:** test actual binding-relative input offsets, including merged source bindings, and active higher-dimensional strides before the factory cache lookup. Matrix stores also require the destination offset to align to four elements. Keep existing shape/type guards, scalar/vector cache dimensions, aligned shader bytes, uniforms, bindings, grids, quantized paths and gather behavior. Extent-one strides are unused.
- **Integration:** append a checked stage after the existing normal/profile prefixes. The profile patch accounts for the optional diagnostic's separate key builder. Record actual final compiled identities. No vendor files or shader source are modified.
- **No-patch alternative:** callers can materialize aligned contiguous inputs, or avoid the affected layouts. This does not repair accepted operations using existing views.
- **Validation and limits:** source-bound caller/factory checks compare both profile modes, alignment residues, active/unused strides, merged offsets, cache order and unchanged shader/parameter contracts. Full pinned-header syntax and actual runtime checks are distinct gates. Falling back changes the existing reduction partition; incorrectly addressed outputs are not a numerical oracle. This correctness repair makes no speed or model-quality claim.
- **Removal:** retire the stage and associated preparation/build/provenance/tests when upstream checks the same layout conditions or supplies unaligned-safe loads/stores. Re-review binding rebasing and factory/key ownership on upstream updates.

## Tight routing-independent MoE tile capacity

- **Patch:** `ggml-webgpu-moe-tight-tile-bound.patch`.
- **Scope:** Keep the current slot width L=32 and replace only its shared capacity helper after the checked normal/profile final stages. No shader, graph, parameter, key, indirect dispatch or runtime readback is added.
- **Bound:** with nominal slots S, experts E and A=min(E,S), use A+floor((S-A)/L). For a nonempty experts and V valid slots, sum ceil(c_e/L)=a+sum floor((c_e-1)/L)<=a+floor((V-a)/L)<=A+floor((S-A)/L). Assign one slot to A experts and concentrate the remainder to attain this generic bound. Duplicate IDs and skipped slots remain supported; real routing occupancy is not assumed.
- **Allocation and dispatch:** descriptor capacity, aligned binding size and direct matrix workgroups use the same helper. The result is at most S and does not add uint32 arithmetic overflow. Existing tensor-product, size_t, device capacity and grid admission limits remain unchanged. Single-token matvec dispatch is unchanged.
- **Integration:** apply the individual patch in the final normal/profile composition and record actual final compiled copies. Canonical templates and the upstream checkout remain untouched.
- **No-patch alternative:** retain the conservative existing allocation/dispatch upper bound or wait for an upstream equivalent.
- **Validation and limits:** source-bound integer/extremal routing, real ggml allocator, recorded binding/grid and normal/profile syntax checks cover the shared contract. Reduced empty workgroups or descriptor bytes are not measured speed or peak-memory claims; target-device/model performance remains unestablished.
- **Removal:** remove the stage, patch, preparation/build/provenance and related tests when upstream provides an equivalent bound or the owner drops it. Recheck gather descriptor emission and tensor/device arithmetic contracts on upstream updates.

## Independent IQ3 four-row loop splitting

- **Patch:** `ggml-webgpu-iq3s-row-split.patch`.
- **Scope:** Split only the existing independent indexed IQ3_S accumulator's row0..3 loop into row0..1 and row2..3, keeping its four accumulators. Do not change the canonical accumulator, ordinary/cooperative shaders, selector or supported subgroup widths.
- **Contract:** the existing factory supplies OUTPUTS_PER_WG=4. Preserve one source-level activation-block load, every row's decoder/FP expression and K order, return layout, main/partial-subgroup completion, reductions, parameters, bindings, geometry, cache and queue behavior. Pin and recheck this factory contract rather than applying the literal two-row split to another row count.
- **Integration:** append a checked dedicated-header stage after tight tile capacity. Copy the field-load stage's pinned derived accumulator, patch only its IQ3 branch and re-embed with the same main/common declarations. Provenance records the final compiled header. Normal/profile host source, other headers and the shared ABI/schema are unchanged.
- **No-patch alternative:** keep the existing single four-row loop or wait for an upstream equivalent. This avoids maintaining another source specialization.
- **Validation and limits:** source-bound inverse/row-order/tail checks and actual factory/cache tests cover fixed widths16/32/64. Private software hot-only comparisons informed the choice; native load/register behavior, streaming/model performance and target-device gains are not established. Reusing the same mathematical source order is not a universal cross-compiler FP-bit guarantee.
- **Removal:** remove this patch and its preparation/build/provenance/tests when upstream provides equivalent code generation or the owner drops the specialization. Re-review the derived field-load input, four-row factory, preprocessor and emitter after upstream changes.


## Four-row F32 matvec scale/sigmoid epilogue

- **Patches:** `ggml-webgpu-f32-matvec-epilogue.patch` and `ggml-webgpu-f32-matvec-epilogue-profile.patch`.
- **Scope:** Only dense F32/F32/F32 matvecs with K1..16384, M4, N1 and batch1, followed by SCALE, SIGMOID and SCALE, are eligible. No model branch or new public API is added.
- **No-patch alternative:** retain the existing matvec plus the existing three-operation scale/unary/scale fusion, for two dispatches in total.
- **Contract:** retain the original dot arithmetic, row assignment, reduction and workgroup geometry; append the three existing epilogue expressions only at the final store. Require graph use-count/callback visibility checks and all input/output payload and aligned binding ranges to be disjoint. A dead matvec allocation may be reused by the final output. Unsupported graphs, buffers, limits and kinds retain existing fallback.
- **Ownership and cache:** the default-none key and sixteen parameter words retain their original behavior. A distinct typed epilogue key carries no coefficients; four copied coefficient values extend only the fused path to eighty bytes. Check the actual uniform arena slot and legacy 128-byte binding. Buffer facts are local to one admission attempt; no permanent capability cache, runtime probe or timing resource is introduced.
- **Integration:** append a checked normal/profile source stage after the existing final stages, with a dedicated embedded main and canonical accumulator/declarations. Keep vendor sources, other dtype shaders and the generated public ABI unchanged. Profile diagnostics keep the default-none path.
- **Validation and limits:** source-bound graph/gallocr, physical range, factory/cache, params and full translation-unit tests cover admission and fallback. Private software comparisons informed this candidate; they do not establish model, target-device or cold-start improvement. Actual runtime correctness and performance remain separate gates.
- **Removal:** remove the stage and its provenance/build/tests when upstream supplies the equivalent fusion or the owner drops the specialization. Re-review graph fusion, allocator reuse, binding rebasing, parameter staging, factory keys and canonical shader arithmetic after upstream updates.


## Ordinary short-K Q6_K workgroups

- **Patch:** `ggml-webgpu-q6-short-workgroup.patch`.
- **Scope:** Reuse the existing short-workgroup helper for ordinary Q6_K x F32 single-column matvecs with complete 256-element blocks through K4096. Indexed Q6_K keeps its existing workgroup. The shared helper retains all shape, subgroup-range, device-limit, MMVQ and overlap guards.
- **Integration:** `runtimes/llama-cpp/tools/recipe.ts` appends a checked header-only stage after the F32 matvec epilogue. Record actual patch identities; provenance records the final normal/profile headers separately. Existing cache keys already include workgroup width.
- **Numerics and limits:** preserve the upstream decoder, dot order, reduction source, four-row tiling, parameters and dispatch grid. A different workgroup may change subgroup reduction topology, so source identity alone does not guarantee bit equality. Authored finite standalone checks use an independent upstream-dequantized Float64 oracle, full-output and input/guard validation, and separate same-workgroup controls. Software Vulkan measurements do not establish Windows, macOS, model-quality or end-to-end speed gains.
- **No-patch alternative:** retain the upstream 256-lane workgroup, including its idle lanes for these short inputs, or wait for upstream tuning.
- **Removal:** remove this stage and its metadata/tests when upstream supplies the required generic short-input selection, or when the owner withdraws it. Reassess at every upstream pin change.

## Paired-half shared storage for Q4 prefill

- **Patch:** `ggml-webgpu-quant-half-pairs.patch`.
- **Scope:** Only ordinary Q4_0/Q4_K x F32 to F32 matrices with N>=5, complete quantization blocks, no source overlap and no subgroup-matrix path are eligible. Keep the reviewed TK32, 4x4 register tiles and 8x8 workgroup geometry; other geometry, dtype, indexed and matvec paths retain canonical shaders.
- **Contract:** store adjacent half values as whole shared-memory pairs, load them into private pairs, and accumulate the two products in ascending K order. Preserve global scalar activation loads, packed weight decoders, F32 dequantization before conversion to half storage, output stores, two barriers per K tile, 4096-byte shared memory, bindings, sixteen uniform words and host dispatch. A typed mode selected before lookup participates in cache equality/hash and profile labels; no shape-sized cache key or runtime probe is added.
- **Integration:** append a checked shader-library stage after the Q6 selector, copy the pinned templates into an isolated preparation tree and generate a dedicated embedding with the canonical emitter. Keep vendor and canonical embeddings unchanged, report actual normal/profile compiled copies separately.
- **No-patch alternative:** retain canonical scalar shared storage or wait for an upstream equivalent. No supported runtime setting provides this code-generation choice.
- **Validation and limits:** complete-source native factory/cache/fallback/grid checks establish the host selection contract. Shader numerical checks establish a separate property. Finite authored GPU checks and software measurements are separate evidence; they do not establish physical peak memory, cold-start cost, model quality or Windows/macOS speed. Preserving source arithmetic is not a universal compiler bit-equivalence guarantee.
- **Removal:** remove this stage, embedding and associated build/provenance/tests when upstream supplies an equivalent or the owner drops it. Re-review dequantization precision, half conversions, pair ownership, geometry, keys and emitter/preprocessor contracts on upstream changes.

## Direct aligned mapped reads in Emdawn

- **Patch:** `ggml-webgpu-direct-mapped-read.patch`.
- **Scope:** Under `__EMSCRIPTEN__`, mapped tensor reads with zero local byte offset and four-byte size alignment use the pinned Emdawn `ReadMappedRange` API to copy directly into the final caller destination. Native builds and unaligned ranges retain pointer mapping plus memcpy.
- **Contract:** preserve source range rounding, staging allocation, mutex, command submission, map completion wait and successful Unmap. The destination is not retained asynchronously. Direct-copy failure unmaps then aborts without retrying through the old path. No shader, binding, uniform, dispatch, public ABI or checkpoint policy changes.
- **Integration:** append a checked normal/profile CPP-only stage after paired-half preparation. Report actual final compiled copies; the existing preparation CLI and CMake dependencies retain the pinned Dawn-header contract. Keep the upstream checkout untouched.
- **No-patch alternative:** retain Emdawn's pointer-returning mapping API and its temporary Wasm allocation/copy, or wait for an upstream equivalent.
- **Validation and limits:** exact inverse/source checks, both preprocessor arms in the actual-getter queue model, all byte offsets/tails, guarded destinations, ordering and error cleanup cover the source contract. Pinned SDK checks and small Wasm proofs are separate from full-runtime/browser validation. Removing a temporary does not shrink an already-grown Wasm linear memory; no physical-peak, TTFT or target-device speed guarantee is made.
- **Removal:** remove this stage and its build/provenance/tests when upstream or the pinned SDK provides equivalent direct behavior, or the owner drops it. Re-review mapped-range alignment, status/Unmap semantics and SDK implementation on updates.

## Wider-M paired-half Q4 prefill tiles

- **Patch:** `ggml-webgpu-quant-wide-m.patch`.
- **Scope:** Within every existing paired-half Q4_0/Q4_K x F32 prefill guard, select TM8/TN4 only when M>=64, M is a multiple of 64 and N>=32; otherwise retain TM4/TN4. Q6_K reuses C8 only for complete K256 blocks with the same base/type exclusions, positive output batch axes and total C8 grid `(M/64)*ceil(N/32)*batch2*batch3>=8`; a saturating predicate avoids overflow. Other Q6 cases retain scalar, with no Q6 C4 admission. Preserve complete quantization blocks and all dtype, overlap, subgroup-matrix and base-geometry exclusions. No model, OS or GPU-vendor condition is added.
- **Contract:** a third typed shared mode uniquely identifies the wider geometry before cache lookup and participates in existing equality/hash. Derive shader selection, TILE_M, profile label and decisions from that same mode. The existing host grid helper consumes those decisions for 64x32 output tiles. Reuse the paired embedding, ascending K arithmetic, decoders/global access expressions, TK32, WG64, two barriers and sixteen-word uniform ABI. Shared storage is 6144 bytes and each invocation has 32 scalar accumulators; no parameter, weight or readback buffer is added.
- **Integration:** append a checked header-only stage after direct mapped read. Preserve canonical, four-row, indexed, matvec, F32 small-M and other quantized fallbacks and other guarded fallbacks. Report actual final normal/profile compiled copies and the reused embedding identity.
- **No-patch alternative:** retain the four-row geometry or wait for upstream tuning.
- **Validation and limits:** actual normal/profile factory/cache/grid/source equivalence, ownership and pinned-header syntax are source/native gates. Finite authored software checks are separate from full runtime/model/hardware validation. Small-K and small-grid speed, cold compilation cost, physical peak and Windows/macOS improvements remain unmeasured; no additional K threshold is inferred from untested cases.
- **Removal:** retire this stage and its build/provenance/tests when upstream provides equivalent generic selection or the owner drops it. Recheck geometry, memory limits, per-row arithmetic, pair ownership and key/grid coupling on updates.

## Wider K tiles for indexed prefill

- **Patch:** `ggml-webgpu-indexed-wide-k.patch`.
- **Scope:** Allow K64 for indexed IQ4_NL with its existing F16 accumulation and for indexed NVFP4 with F32 accumulation. Require F32 input and output, at least 12 input tokens, and at least 8,448 bytes of workgroup storage. The K/M pairs are 1152/2304, 2304/2304, and 2304/512. Keep the original matrix-vector selection. Use K32 for this matrix path when the adapter reports vendor `apple` or the K64 requirements are unmet.
- **Contract:** Keep gather, workgroup grid, activation packing, K summation order, and accumulated precision. Include tile width in the actual pipeline cache key and distinguish it in profile labels. The sparse IQ4_NL column path remains active inside either tile width.
- **Validation:** Test adapter selection and the shape, token, storage and precision boundaries in both provider modes. Compare actual shader output, including sparse and dense column counts, using deterministic inputs and local samples of packed model weights. Software WebGPU timing covers the isolated tile. An isolated K32/K64 comparison on a physical adapter is pending.
- **Removal:** Remove when upstream supports equivalently fast indexed prefill for these layouts without changing precision or fallback behavior.

`ggml-webgpu-indexed-wide-k-apple.patch` applies the Apple rule after indexed tiling is configured in the ordinary and profiling recipes. In a long-prompt comparison between runtime versions on an Apple `metal-3` adapter, the K64 shader took more GPU time at matching dispatch grids. K32 lowers workgroup storage from 8,448 to 4,352 bytes for these tiles. The paired Q4_K and Q4_0 selections keep their own rules. Revisit the Apple rule when an isolated physical-adapter comparison supports a faster tile.

## Paired Q4_K loads for indexed prefill

- **Patch:** `ggml-webgpu-indexed-q4-k-paired.patch`.
- **Scope:** Use K64 for indexed Q4_K x F32 to F32 at K/M 512/2048 or 2048/512. Require at least 12 input tokens and 8,448 bytes of workgroup storage. Lower limits, other dimensions and single-token matvec use their existing selections.
- **Contract:** Decode the low and high nibbles of a packed Q4_K word together. Each half uses its original scale and minimum, F32 dequantization and F16 rounding before the original ascending K accumulation. Reuse gather, tile grid, binding layout and cache key. Use a dedicated indexed shader declaration input and the `tk64_paired` profile suffix.
- **Validation:** Exercise both shape, token, storage and dtype boundaries through the normal and GPU-profile factories. Compare the emitted WGSL and deterministic packed-weight outputs. Isolated software WebGPU comparisons do not establish physical-adapter or model throughput.
- **Removal:** Remove when upstream supplies equivalent indexed loads with the same precision and compatible fallback behavior.

## Wider output-row tiles for indexed prefill

- **Patch:** `ggml-webgpu-indexed-wide-m.patch`.
- **Scope:** Use 64 output rows per workgroup for the indexed F32-input and F32-output K32 matrix path with 512 stored experts, ten selected experts and 128 input tokens. Admit IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS and Q4_K at K2560/M640 with broadcast input, and IQ4_NL and Q5_1 at K640/M2560 with per-slot input. Require the matching I32 ID tensor, 64 compute invocations and at least 6,400 bytes of workgroup storage.
- **Contract:** Reuse WG64, the 32-slot gather, the existing sparse and grouped column assignments, ascending K accumulation and the original uniform and storage bindings. Include the row tile in the pipeline cache key and derive the shader, `_tm8` label and dispatch geometry from it. The existing K64, narrow matrix and indexed matvec routes keep their selection.
- **Integration:** Apply one shader-library header stage in the normal and GPU-profile recipes. Reuse the existing indexed shader embedding and context limits. Keep upstream source untouched.
- **No-patch alternative:** Keep the existing 32-row indexed tile or wait for upstream tuning.
- **Validation:** Complete-provider native tests exercise the cache, exact shape and limit boundaries, grid and uniform layout in both modes. Independent CPU references and authored software WebGPU inputs cover sparse and grouped routing. Warm software measurements are limited to isolated dispatches with a prefix of the expert weights. Physical-adapter and model first-token timing remain unmeasured.
- **Removal:** Remove the patch, recipe entry and tests when upstream provides an equivalent indexed tile without changing the numerical or fallback contracts.

## Paired Q4_0 blocks for indexed prefill

- **Patch** `ggml-webgpu-indexed-q4-0-paired.patch`.
- **Scope** Use a 64-row, K64 tile for single-binding indexed Q4_0 × F32 to F32 at K2816/M1408 with 48 or 128 experts, eight selected experts, 128 tokens and broadcast activation. Require contiguous, aligned root weights, matching I32 IDs, 64 invocations and 12,544 bytes of workgroup storage. Device features and tensor geometry determine eligibility.
- **Contract** Load two adjacent 18-byte blocks in nine aligned words. Preserve each block's F32 scale multiplication, F16 rounding before staging and ascending K accumulation. The existing K and row tile cache key identifies the variant. The existing host uses the selected row tile for its grid and keeps the 32-slot gather and buffer layouts.
- **Integration** Apply the checked normal and GPU-profile stage through the existing indexed shader embedding. Source selection and binding offset are derived from the existing WebGPU encoder. Ordinary placement and indexed matvec keep their current selection.
- **No-patch alternative** Keep the 32-row, K32 path or wait for upstream tuning.
- **Validation** The normal and GPU-profile native recorders exercise cache, alignment, resource limits and encoder dispatch. The E48 single-binding software WebGPU fixture included packed nibbles with half-float infinity and NaN bit patterns next to finite scales. Its outputs matched the CPU reference bitwise and isolated GPU time was 24–28% shorter in mirrored runs. The E128 single-binding case was checked at native selection with a device limit that admits its 285,474,816-byte weight tensor. Physical-adapter throughput and model first-token latency require separate measurements.
- **Removal** Remove this stage and its tests when upstream provides equivalent indexed loads with comparable precision and device performance.

## Paired-half Q8 prefill tiles

- **Patch:** `ggml-webgpu-q8-prefill-wide-m.patch`.
- **Scope:** Ordinary single-matrix Q8_0 x F32 to F32 with complete tiles. Use the 64x32 tile for K>=5120 with at least 640 output workgroups, or at N=128 for K/M pairs 2304/1024, 2304/3072 and 3072/2304. Use the 32x32 tile at K10240/M320/N128 to retain the original dispatch grid. Keep the overlap, subgroup-matrix, vector and base-geometry exclusions. Selection uses tensor shape and backend capabilities without model or device identifiers.
- **Contract:** Reuse the paired-half cache modes, labels, WG64/TK32 tiles and 16-word UBO. Decode signed packed Q8 bytes with the canonical 34-byte block layout and multiply the F32 scale before rounding the products to half. Store adjacent values as pairs and accumulate in the original K order. Preserve the existing bindings, offsets and global memory layout. No new buffers, exports, probe or host synchronization.
- **Validation:** Exercise actual factory dispatch and cache transitions in normal and profile providers. Check packed decoding, full output bits and guard regions in representative prefill shapes. At K3072/M2304/N128 the 64-row tile was faster than both 32-row candidates in mirrored software WebGPU comparisons. Physical-device prefill and ordinary first-token latency require model measurements.
- **Removal:** Delete the patch and its recipe entry when upstream supplies an equivalent Q8 prefill path with comparable numerical behavior and device performance.

## Wider paired-half Q2 prefill

- **Patch:** `ggml-webgpu-q2-prefill-wide-m.patch`.
- **Scope:** Ordinary contiguous Q2_0 x F32 to F32 GEMM at K5120/M17408/N128, with default input and accumulator precision, singleton batch axes, 64 invocations and at least 6,144 bytes of workgroup storage. The 1,088-workgroup grid must fit the X dimension. Explicit precision, source overlap, vector and subgroup-matrix paths retain their existing selection.
- **Contract:** Reuse the 64x32 paired-half tile, K32, unchanged buffer bindings and 16-word uniform. Pack adjacent decoded half values into the existing shared-memory pairs. Preserve all four Q2_0 codes, round the decoded weights and activations to half, and accumulate with F32 in ascending K order. Adapter checks apply to the paired selection alone.
- **Validation:** Cover ordinary and profiling provider selection, cache and limit fallbacks, and binding sizes that end after an 18-byte block. Replayed provider shaders agree with an independent arithmetic reference on the full shape and a guarded fixture with all four codes. The 64-row tile gave the larger gain over scalar M32 in mirrored software WebGPU comparisons. Physical-adapter performance and ordinary first-token latency remain unmeasured.
- **Removal:** Retire the patch, recipe entry and focused tests when upstream provides equivalent Q2_0 prefill behavior and device performance.

## Ordinary fixed-subgroup Q4 matvecs

- **Patch:** `ggml-webgpu-q4-subgroup-matvec.patch`.
- **Scope:** Only ordinary Q4_0/Q4_K x F32 to F32 matvecs with N1 and complete blocks are eligible. Admit K1024..4096 and M>=4096 with checked broadcast/u32 bounds. For singleton weight/input/output batches, admit 4096<K<=17408 and M>=2560, plus Q4_K at K7680..10752 and M=2048. M=2048 fills complete row groups at both supported subgroup widths. The additional shape rule depends on tensor dimensions and dtype. Actual strides and physical binding ranges remain caller contracts. Exclude indexed inputs, MMVQ, source overlap and epilogues. Subgroup width follows the existing fixed advertised S16/S32 and optional requested S16 policy above.
- **Contract:** reuse the existing independent-subgroup main and unchanged canonical accumulator/declarations. Each subgroup owns four rows in WG128; partial active subgroups retain the existing virtual-lane completion and non-owner collective participation. Existing subgroup-size/type keys and shader decisions define cache identity and output grid. Do not change Q8 selection, other dtypes, bindings, sixteen uniform words or host buffer-allocation/queue implementation. Pipeline cache variants and cold compilation still follow the existing ownership rules. Partitioning blocks among lanes changes floating-point addition/reduction order, so cross-path bit equality is not required.
- **Integration:** append a checked normal/profile header-only stage after wide-M prefill selection. Keep shape/range eligibility separate from the fixed-width policy, record actual compiled copies. Reuse the existing embedding without new shader files, profiles or public ABI.
- **No-patch alternative:** retain cooperative workgroup reduction or wait for equivalent upstream tuning. Upstream settings do not expose this per-type subgroup mapping.
- **Validation and limits:** source/native factory/cache/geometry and rejection checks cover the guarded selector. Finite authored numerical comparisons use the original-F32 dot bound and check guards and inputs. Software hot-buffer comparisons at M=2048 cover both K endpoints and fixed S16/S32. Intermediate K, requested-width device timing, cold compilation and model throughput remain unmeasured by those comparisons.
- **Removal:** remove the stage and associated build/provenance/tests when upstream supplies equivalent selection or the owner withdraws it. Re-review block lane stride, partial subgroup completion, shape arithmetic, factory keys and host grid coupling on updates.


## Aligned shared byte-to-word loaders

- **Patches:** `ggml-webgpu-aligned-word-load.patch` and `ggml-webgpu-aligned-word-load-integration.patch`. The integration file records the existing canonical include redirection for the corrected generated shader header.
- **Scope:** in `load_u32_at_src` and `load_u32_at_src0`, select the containing word for both reads at a four-byte-aligned offset and the following word for the hi read otherwise. A masked left shift returns lo|lo at alignment without a control-flow branch. The previous eager following-word read is unnecessary and reaches the binding end for a complete Q4_0 final word with a minimal binding. No observed GPU failure is claimed.
- **Contract:** keep the nonaligned two-word join, all other declarations, arithmetic, selectors, bindings, grids and ABI bytes. This correction does not establish all-dtype bounds safety or repair a nonaligned call without a complete four-byte span, including separate partial-field overreads. It adds no runtime allocation, conversion, callback or probe.
- **Integration:** preserve the upstream checkout. Apply the accepted helper patch to a private WGSL template **before** invoking the upstream embedding generator. Its canonical header explicitly replaces the original canonical include through the integration patch; never include both symbol sets in one translation unit. The generator owns raw-string chunks and reserve capacities, not a historical header digest. The five dedicated embeddings use their own original shader inputs plus this same corrected WGSL template, followed by the unchanged upstream generator. No C++ string-body extraction or replacement applies this correction. The optional-Q8 annotations are likewise expressed in the explicit WGSL patch listed above. Record observed template/generator inputs and the final compiled copies as provenance, not proof of behavior.
- **No-patch alternative:** retain the accepted early-return correction or wait for the equivalent upstream correction. Supported runtime settings do not change these helper statements.
- **Validation and limits:** compile all canonical shader constants and compare their decoded strings against the original shaders with only the two accepted helper changes. A small-chunk generator fixture checks functions spanning C++ string boundaries. Dedicated C++ consumers compare the decoded shader strings, including the optional-Q8 form, against the recorded numerical fixture; bounded scalar byte oracles cover all four offset residues, complete nonaligned spans and minimal aligned bindings. Source range traces cover the Q4_0 final word and GET_ROWS consumer. Native translation-unit/include tests and finite GPU checks remain distinct from full runtime, model-quality, performance and target-device validation.
- **Removal:** remove this patch, final preparation stage, renamed embedding and associated build/provenance/tests when upstream supplies the same behavior or the owner drops it. Re-review all generated consumers, chunk ownership and actual include composition at each upstream update.

## Bounded WebGPU host checkpoint readback

- **Patch:** `ggml-webgpu-checkpoint-batch-readback.patch`.
- **Accepted scope:** the owner's inference-performance work permits batching
  existing host checkpoint reads while preserving the host snapshot format,
  synchronization, caller ownership and normal large-read memory policy.
- **Integration:** every browser WebGPU profile prepares one checked context TU
  and the WebGPU provider plus a shared versioned registry-contract header.
  CMake replaces the context only under Emscripten and preserves its original
  quoted-header lookup on that one source. CPU-only and native context sources,
  ON_DEVICE and file writers, public ABI/schema, shaders and model graphs stay
  upstream. Pristine vendor source is never changed.
- **Contract:** group only consecutive small requests on the same device, at
  most 256 entries and 4 MiB of staging per batch. A request is never split.
  Single requests and large requests keep the scalar getter. The optional
  callback may decline only before enqueue, map or destination write; failures
  after work begins are terminal. Request order, including destination aliases,
  is preserved. An explicit one-shot flush is inside existing state API error
  handling; destructors do no readback or retry. Caller memory remains owned
  until the blocking call returns.
- **Coupling:** pinned context writer/callers, tensor/view buffer conventions,
  registry and WebGPU staging/map helpers. Normal/profile composition, shared
  headers, patch, source contracts and final copies are recorded. Reassess these
  boundaries when upstream changes rather than widening a hash allowlist.
- **Evidence and limits:** CPU packing/writer controls and a tiny authored model
  on normal Wasm64 verify exact checkpoint/restore/logit/token results, fewer
  maps for multi-read snapshots, unchanged single-read behavior and bounded
  destination lifetime on a copy failure. These do not establish browser,
  real-model performance, app cancellation, device-loss recovery or large-model
  memory peaks. No target-device speedup is claimed.
- **No-patch alternative:** keep upstream's per-tensor blocking getters or wait
  for a corresponding upstream batched state writer/backend contract.
- **Removal:** equivalent upstream support or withdrawal of this optimization;
  remove the checked patch, optional header/proc, context source replacement,
  preparation, tests and provenance together, preserving published artifacts.


## Ordinary Q6 independent and requested subgroup matvecs

- **Patch:** `ggml-webgpu-q6-subgroup-matvec.patch`.
- **Scope:** ordinary Q6_K x F32 to F32 matvecs with N1, complete K256 blocks, K1536..10752 and M>=2048. Preserve tensor-shape, broadcast and u32 checks; exclude indexed inputs, MMVQ, overlap and epilogues. No model, vendor or OS branch is added. Physical strides and binding capacity remain existing caller/device contracts.
- **Contract:** reuse the unchanged Q6 accumulator/210-byte signed-scale decoder and existing full/partial independent main at WG128 with four rows per subgroup. Fixed advertised S16/S32 share pure eligibility with requested S16 through the existing real-feature/range policy. Resolve the existing dtype/mode/requested-width key before lookup. Rejected optional keys rebuild complete canonical decisions from the current context, retaining the existing short-K WG128 and longer-K WG256 fallback; no silent recovery from terminal/device-lost failures is added.
- **Integration:** append a checked normal/profile header-only stage after subgroup-size control. Preserve all CPP, generated shader embeddings, optional SDK helper, ABI, sixteen uniform words, host grid, allocation and queue bytes. Report the actual final compiled composition, not an intermediate prefix.
- **No-patch alternative:** retain cooperative reduction or await equivalent upstream selection.
- **Validation and limits:** finite authored numerical bounds, source/factory/cache/grid and physical-range checks are separate gates. Floating-point partition and reduction may differ, so cross-path bit identity is diagnostic. Direct software hot evidence covers a down projection, two bounded heads and one larger resident test, not full vocabulary, all intermediate shapes, actual model quality or Windows/macOS speed. No new preload, dequantization copy, probe, clock or GPU wait is introduced.
- **Removal:** retire this stage and its build/provenance/tests when upstream supplies equivalent selection or the owner drops it. Re-review lane stride, partial-subgroup ownership, shape/u32 checks, typed keys, current-context fallback and device feature/limit contracts on updates.

## Long-K fixed-subgroup Q4_K and Q6_K matvecs

- **Patch:** `ggml-webgpu-long-k-matvec.patch`.
- **Scope:** Extend the existing independent Q4_K and Q6_K ordinary matvec paths to the complete K19968/M6656/N1 shape with F32 input and output, singleton batches and an advertised fixed subgroup width of 32. Keep the existing bounds for fixed SG16 and requested subgroup widths. The original layout, overlap, epilogue, indexed and MMVQ exclusions also apply.
- **Contract:** Reuse each type's existing shader, cache key and bindings. The fixed SG32 kernel produces 16 rows per 128-lane workgroup and dispatches 416 workgroups for this shape. Its reduction order can differ from the cooperative kernel.
- **Validation:** Exercise selection, dispatch, cache reuse and the existing fallback boundaries in both provider modes. Local software WebGPU replay checks output bounds and paired timing using complete rows sampled from the corresponding packed weights. Physical GPU generation speed requires model measurements.
- **Removal:** Drop the stage when upstream selects a comparably fast implementation with equivalent numerical and layout behavior.

## Independently owned model vocabulary and chat metadata

- **Scope:** reuse native vocabulary and chat metadata for a host-owned graph.
  This leaf boundary does not extend to llama.cpp model, memory or execution
  internals; expanding the upstream patch scope requires explicit approval.
- **Patch:** `common-chat-independent-vocabulary.patch`. Extract the existing chat
  initialization body into a vocabulary/metadata entry point; keep the native
  model entry point and its default/tool/task template, BOS/EOS and add-token
  semantics. The native generation loop, graph builders and kernels are untouched.
- **Leaf bindings:** `native/model-graph/model-graph.cpp` uses the existing metadata-only model
  loader and vocabulary implementation without constructing a dummy model. It
  also queries actual architecture recognition and supplies reusable candidates
  to existing CPU samplers for external logits. No external code, native function
  pointer, model-specific builder or model-file I/O policy is introduced.
- **Coupling:** private loader/vocabulary/architecture interfaces require review;
  the extension ABI is fingerprinted. The changed chat translation unit is prepared in an isolated build
  overlay, with checked inputs, patch and outputs; upstream remains pristine.
- **No-patch alternative:** require existing native architectures or separately
  implement tokenization/prompt handling. That does not satisfy the approved
  unknown-architecture reuse goal.
- **Validation limits:** compilation and authored boundary tests are not a claim
  of browser inference, real model equivalence or unchanged native throughput.
- **Removal:** retire the overlay and adapters when upstream exposes independent
  vocabulary/chat initialization, or when the model-graph extension is removed. Review API, ownership and semantic changes before expanding that boundary.

## Representable WebGPU allocation limit

- **Patch:** `ggml-webgpu-buffer-size-limit.patch`.
- **Accepted scope:** preserve upstream's `maxBufferSize` allocation limit while
  preventing its uint64-to-size_t conversion from wrapping on 32-bit hosts.
  Clamp the uint64 value to the graph allocator's existing `SIZE_MAX / 2` chunk
  limit before narrowing, only when size_t is narrower than uint64_t. A 64-bit
  host returns the full reported value, including the uint64 maximum.
- **Integration:** one checked final stage in the shared WebGPU build-tree source
  for existing normal and diagnostic profiles. Storage binding limits, upstream
  chunked memset, queue ordering, allocator and public ABI are unchanged.
- **Coupling:** the private buffer-type getter, device limit field and graph
  allocator width contract, pinned separately from tensor-copy exceptions.
- **No-patch alternative:** keep upstream conversion, which returns zero for a
  reported 4 GiB allocation limit on a 32-bit host, or defer this upstream pin.
- **Validation and limits:** source-bound getter checks at actual 32/64-bit widths
  use controlled limits without allocating huge buffers. They do not establish
  which adapters report such limits or browser behavior at those sizes.
- **Removal:** upstream preserves the representable allocation limit before
  narrowing, or supported profiles no longer include 32-bit WebGPU hosts. Remove
  this patch, preparation, provenance and independent tests together.

## Scalar Flash Attention masking and KV tiles

- **Patches:** `ggml-webgpu-flash-attn-maskskip.patch` and
  `ggml-webgpu-flash-attn-wide-scalar-kv.patch`, against llama.cpp `c212d25`.
- **Mask scope:** scalar prefill with F32 Q, F16 K/V in separate bindings, at
  least 32 queries and 768 keys, and an F16 mask with contiguous key rows and
  zero ALiBi max bias. Before loading K and V, skip a tile only when every
  in-bounds score for every active query has the F16 negative infinity mask.
  A partial subgroup conservatively retains the tile. Other variants retain
  their existing mask behavior.
- **Tile scope:** the same scalar prefill types and length bounds, with or
  without a mask, equal Q/K and V head dimensions of 128, 256 or 512, and at
  least 32 KiB of workgroup storage. Calculate the KV tile width using the
  shader's shared K/V staging, query and probability arrays and any mask votes,
  rounding each allocation for WebGPU validation.
  Cache the actual geometry before creating a pipeline. Different tile widths
  can change floating-point rounding in probability and output accumulation.
- **Integration:** both patches apply to normal and profiling source
  preparations. The aligned-word shader family embeds the patched scalar tile
  through the upstream generator. The source receipt records the patch and
  shader input identities. The width patch follows the mask patch to account
  for its shared votes. There is no public ABI or binding layout change.
- **Validation and limits:** native provider cases check admission, cache and
  shader specialization. Software WebGPU comparisons cover causal and sliding
  windows, head-specific masks, sinks, tails, and nonzero binding offsets.
  Exact skipping preserves the tested finite outputs at the same geometry and
  follows CPU mask semantics for nonfinite K/V in wholly masked tiles. Mixed
  tiles retain the existing shader arithmetic. Wider-tile comparisons use
  numerical tolerances. Measurements cover kernels on a software adapter.
- **No-patch alternative:** retain the upstream scalar kernel and tile sizing,
  or adopt corresponding upstream support when available.
- **Removal:** retire each patch when upstream provides equivalent behavior or
  its scope is withdrawn. Recheck the per-tile reduction, mask indexing and
  workgroup storage budget when either upstream shader or factory changes.

## Single-token depth residual mixing

- **Patches:** `ggml-webgpu-depth-mix.patch` and
  `ggml-webgpu-depth-mix-view-sources.patch`, with the normal and profiling
  `ggml-webgpu-fusion-view-sources-*-encode.patch` adapters, against llama.cpp
  `c212d25`.
- **Scope:** contiguous F32 states of width 2304 and depth 2 to 5, with 1, 3 or
  4 learned queries. Match the complete sequence from depth concatenation to
  RMS normalization, query scores, unmasked softmax and weighted output.
  Keep the decomposed path for an incomplete sequence, an exposed or shared
  intermediate, unsupported layouts or device limits.
- **View readers:** account for views that borrow storage without a `src`
  edge, including readers in another scheduler slice. Index the enclosing
  graph's view roots once per compute invocation when a candidate reaches the
  closure check. The planned transpose's explicit source edge and the final
  output's views are compatible with fusion.
- **Contract:** normalize states directly into reusable device scratch, write
  probabilities to a separate scratch buffer, then blend without a transpose
  copy. The output can reuse storage whose last read has finished. Fixed SG16
  and SG32 use the compact blend. Score reduction uses subgroups when their
  total capacity equals the workgroup size and uses a complete workgroup tree
  otherwise. The choice of reduction can change floating-point rounding.
- **Integration:** use both normal and profiling source preparations with the
  upstream WGSL generator. Model graph construction and public ABI are
  unchanged. The prepared source receipt covers the patch and shader inputs.
- **Validation and limits:** native contracts cover dependent mixes in one
  graph, scratch reuse, external consumers, raw views across graph slices and
  workgroup-storage boundaries.
  Software WebGPU comparisons cover dependent outputs and actual partial
  subgroup packing. Full-model output and physical throughput remain
  unmeasured.
- **No-patch alternative:** retain the decomposed graph and existing kernels.
- **Removal:** retire when upstream fuses this graph with equivalent admission,
  device limits and numerical behavior. Review graph order, RMS and matvec
  reductions and storage binding rules when updating the upstream revision.

## Packed expert SITU activation

- **Patch:** `ggml-webgpu-situ-packed.patch`, against llama.cpp `c212d25`.
- **Purpose:** evaluate the nine materialized pointwise operations of the
  packed expert SITU activation in one WebGPU dispatch on single-token graphs. The
  existing graph builders and indexed expert projections provide the inputs
  and consume the result through their current interfaces.
- **Scope:** contiguous F32 output of width 1152, eight selected experts, and
  one token. The gate and up views must describe the first and second halves
  of each row of the same dense F32 source. Admit the exact branch order,
  original sigmoid input, and final multiplication. Retain the separate
  dispatches for other operation sequences, unsupported shapes or layouts,
  device limits, unfinished graph slices, aliased binding ranges, and any
  discarded intermediate a different consumer can observe. Views of the
  final output remain valid. The shared graph view-source index detects raw
  view readers without explicit source edges, including readers in a later
  scheduler slice.
- **Contract:** keep the four scale/bias expressions, the standard TANH clamp,
  SIGMOID, and the original multiplication order. Both storage bindings use
  the same access mode when their non-overlapping ranges share an allocation.
  The shader reads packed gate/up and writes the final value directly, without
  scratch or a model/adapter allowlist. A supported wider scope must preserve
  the same view offsets, lifetimes, binding limits, and numerical expressions.
- **Integration:** the normal and profiling recipes use the same matcher and
  embed the dedicated shader with the upstream generator. The public ABI and
  the Q8 diagnostic encoder seam are unchanged.
- **Validation and limits:** native recorder checks cover the fused dispatch,
  raw view readers, supported final aliases, and overlapping storage fallback.
  Software WebGPU checks compared packed inputs and downstream outputs,
  including scale bias, non-finite inputs and shared allocations. Floating-point
  lowering can vary across adapters. Full-model physical throughput is unmeasured.
- **No-patch alternative:** retain the existing nine-operation WebGPU graph.
- **Removal:** retire when upstream admits and fuses the same graph with
  equivalent layout, alias and numerical behavior. Recheck operation order and
  storage binding rules at each upstream update.

## Paired single-token F32 loop controller

- **Patches** `ggml-webgpu-loop-controller.patch` and
  `ggml-webgpu-loop-controller-context-normal.patch`, against llama.cpp
  `c212d25`. The profiling recipe already exposes the stock factory context.
- **Scope** Two contiguous F32 K64/M2304/T1 matrix-vector projections sharing a
  gate, followed by the exact TANH, MUL, SUB, TANH, MUL and ADD data dependencies.
  Both projections must select the existing 256-invocation, four-row vec4
  geometry. Preserve the stock operations for unrecognized graphs, other layouts
  and limits, partial overlaps, or observable intermediates. The shared view index
  accounts for raw view readers in other scheduler slices.
- **Contract** Evaluate both projections with the same two-stage subgroup
  reduction or workgroup tree as the stock operation. Evaluate the stock TANH
  clamp and elementwise update formula. Bind disjoint ranges on the same device
  allocation using consistent storage access. When the current input and the
  result have identical ranges, a single binding reads and writes each element
  in place. The fused pass writes only the destination and uses workgroup-local
  storage. WGSL permits FMA contraction across the fused MUL, SUB and ADD, so
  intermediate rounding can differ from the storage boundaries of separate
  passes. Subgroup packing and floating-point results can vary by adapter.
- **Integration** The normal and profiling recipes embed the same dedicated
  WGSL using the upstream generator. The planner uses the existing graph and
  matvec factory within the WebGPU backend.
- **Validation and limits** Native encoder tests cover dispatch, bindings and
  parameters, ownership and closure. Dawn/llvmpipe replay compared the emitted
  WGSL with stock, including non-finite inputs and cancellation. The 48-chain
  replay shortened from 241.60 ms to 134.87 ms on llvmpipe reporting a subgroup
  width of 32 with 16 active lanes. The Apple Metal adapter reported width 32,
  and its active lane count, numerical results and model throughput remain
  unmeasured. A separate 64-invocation prototype was tested at fixed subgroup
  width 16. The production path uses the canonical 256-invocation geometry.
- **Alternative** Retain the two projections and six pointwise passes.
- **Removal** Retire when upstream fuses this graph with equivalent numerical,
  layout and lifetime behavior. Recheck the canonical matvec factory, reduction
  and pointwise operations when updating upstream.

## Indexed Q2_0 GPU-resident weights

- **Purpose:** execute indexed Q2_0 expert operations on WebGPU when model
  loading has explicitly placed the weights there. A contiguous tensor can be
  larger than a storage binding even when it fits in a single device buffer.
- **Scope:** admit resident Q2_0 weights for the checked vector and matrix
  paths. The loader's metadata probe continues to select its existing buffer
  types. An explicit tensor-buffer override uses the existing model parameters.
- **Contract:** bind one range when it fits, otherwise split it into two aligned,
  disjoint ranges on the same buffer. Route aligned word loads by the split
  offset. Validate limits, sizes, backing device, overlap, and matrix scratch.
  K640/M2560 matvecs can use a shorter workgroup; pipeline keys distinguish
  width, split mode and matrix tile geometry.
- **Discovery:** the existing WebGPU backend registry reports
  `ggml_backend_webgpu_indexed_q2_placement_v1` as an `int (*)()` returning 1.
  Its presence identifies the runtime implementation. Check the selected
  device's limits separately before requesting a weight placement override.
- **Integration:** apply the host, shader-library and WGSL stages in both WebGPU
  preparations at llama.cpp `c212d25ad`. Use the existing shader generator and
  argument buffer. The existing Wasm export list and CPU profiles are unchanged.
- **Validation and limits:** native contracts exercise placement rejection and
  real encoder ranges, including foreign allocations. Software WebGPU compares
  decode and prefill output with CPU references. Total GPU memory availability
  and browser model throughput require measurement on the target device.
- **No-patch alternative:** keep indexed Q2_0 on the CPU through default loading.
- **Removal:** retire the stages when upstream can admit and bind these weights
  with equivalent layout and device checks. Review Q2_0 packing, matrix scratch,
  view ownership and indexed cache keys when updating upstream.

## BF16 small-M prefill register tiles

- **Patch:** `ggml-webgpu-bf16-small-m.patch`, against llama.cpp `c212d25`.
- **Scope:** ordinary BF16 × F32 → F32 matrix multiplication with K=10240,
  M=4, N=5–128 and one batch. Source overlap and subgroup matrix paths retain
  their existing tiles. The host continues to route N≤4 through matvec.
- **Contract:** reuse the existing 4×32 output tile with a 32-invocation
  workgroup and TK8. Keep packed BF16 storage, f16 staging, ascending K
  accumulation in F32, binding layout, and dispatch dimensions. The existing
  pipeline key and label distinguish the geometry. The kernel requires 576
  bytes of workgroup storage and allocates no additional buffers.
- **Integration:** apply this header-only selection in the ordinary and
  profiling WebGPU recipes after the F32 small-M specialization.
- **Validation and limits:** the complete-provider recorder verifies selection,
  dispatch, cache isolation and fallback. The expanded software WebGPU
  comparisons covered bitwise output and guard equality, including padded
  strides and partial column tiles. Timing used N=32, 96 and 128. Physical-device
  and full-model throughput remain unmeasured.
- **Removal:** retire the extension when upstream supplies an equivalent BF16
  geometry. Review its conversion, accumulation order and dispatch on pin changes.

## Dense Q4_K SwiGLU projection fusion

- **Patch** `ggml-webgpu-dense-q4-swiglu.patch` against llama.cpp `c212d25ad`.
- **Scope** Fuse two dense Q4_K × F32 matvecs with the same F32 input and
  their split SwiGLU into one dispatch. Admission covers K6656/M19968/N1,
  one batch, and ordinary independent matvecs with fixed subgroup width 16
  or 32. The four-row reduction and the activation approximation follow the
  ordinary shaders. The input sums for Q4_K minimum offsets are shared across
  both projections without changing their accumulation order.
- **Contract** Check the graph closure, raw view readers, GPUBuffer ownership,
  alignment and live ranges. Equivalent separate operations are available
  for other shapes, observable intermediate values, overlapping live ranges
  and rejected optional pipelines. Recheck the graph after cold creation,
  invalidating the shared view index even when optional creation is rejected.
- **Integration** The normal and GPU-profiling recipes specialize the same
  WGSL with the selected fixed subgroup width. The optional cache uses the
  existing device generation and error scope handling. The public ABI and
  GGML tensor storage format are unchanged.
- **Validation and limits** The complete-provider contract checks dispatch,
  bindings, and fallback for raw views, aliasing and scope rejection. In a
  software WebGPU comparison over real-size packed Q4_K weights, the fused
  chain cut isolated GPU time by about 10–13% across both subgroup widths.
  The normal Wasm ABI used the fused dispatch and produced identical output
  in a synthetic graph with the same weights. Physical-device and full-model throughput
  remain unmeasured.
- **Alternative and removal** Use the two independent matvecs and separate
  SwiGLU until upstream can fuse them with equivalent numerical and lifetime
  behavior. Review Q4_K packing, graph closure and the independent subgroup
  reduction when updating upstream.

## Indirect dispatch for indexed prefill

- **Patch** `ggml-webgpu-indexed-live-grid.patch` against llama.cpp `c212d25ad`.
- **Scope** Use the existing 32-slot gather to write an indirect grid for
  K640/M2560/E512 Q2_0 with split bindings and K2816/M1408/E48 or E128 paired
  Q4_0 with a single binding. Require 128 input tokens, the original F32
  input and output, and room for the upper grid bound in one dimension.
  Reuse the original matrix shader and its precision and tensor layouts.
- **Lifetime** Share one 12-byte storage and indirect buffer on one device.
  Every gather overwrites all three arguments before its matrix dispatch.
  Submission batches keep each pair together and share an ordered queue.
  The first eligible graph issues scoped buffer creation and uses direct
  dispatch. Later graphs use the indirect path once all three scope callbacks
  successfully validate the buffer. Pending or rejected creation uses direct
  dispatch. No GPU completion or readback is added to graph encoding.
- **Integration** Use the same gather source and patch in ordinary and
  profiling builds. The cache distinguishes the direct and indirect gather
  variants. Profiling continues to time the gather and matrix dispatches
  separately. The public ABI and model weight placement stay the same.
- **Validation and limits** Check pending initialization, same-queue reuse,
  device and shape fallback, and zero valid experts. Software WebGPU tests
  compared both paths bit-for-bit with a CPU reference for the Q2_0 split and
  E48 Q4_0 fixtures. An E128 Q4_0 device still needs the original single weight
  binding. Prefill time on physical adapters depends on routing and how many
  empty workgroups the direct grid would submit.
- **Alternative and removal** Retain the direct upper-bound grid until
  upstream can dispatch only the active indexed tiles. Review the gather
  descriptor width, scope semantics and queue submission boundaries on updates.

## Raw-view readers of fused intermediates

- **Patch** `ggml-webgpu-fusion-intermediate-views.patch` against llama.cpp `c212d25ad`.
- **Scope** Preserve the stored values read through raw views of RMS norm and
  scale chains, F32 matvec epilogues, and intermediate MoE adds. A raw view
  can share storage without a source edge counted by upstream's fusion check.
- **Contract** Consult the existing lazy view-source index for the intermediate
  values that the selected shader would omit. A view of the value the shader
  writes remains eligible, including an in-place final MUL that writes the
  entire RMS result range with the same layout. A raw view of the unary result can still use the
  two-operation scale fusion, and the RMS preference checks the same eligible
  three-operation plan. Graph slices share the visit set, including later
  readers. The index enumerates occupied slots using the used bitset and keeps
  the existing invalidation at the cold optional-pipeline boundary.
- **Integration and validation** Apply the admission checks in both WebGPU
  preparations. Complete-provider fixtures check the same graphs as one batch
  and as separate producer and reader slices, including terminal view and
  shorter-fusion controls. Shader source, bindings, and the public ABI retain
  their existing contracts.
- **Alternative and removal** Use the separate operations until upstream
  checks view-only uses of discarded storage across scheduler splits. Review
  visited-set ownership, aliases and which outputs the fused shaders write when
  updating upstream.

## Compact workgroups for a single argsort tile

- **Patch** `ggml-webgpu-argsort-compact-workgroup.patch` against llama.cpp `c212d25ad`.
- **Scope** For 128, 256 or 512 source values that fit into a larger single tile,
  use one invocation per source value. Use the same rule for argsort and top-k.
  Bound the original network width by compute invocations, the X dimension and
  workgroup storage. Build the merge pipeline only if the dispatch needs it.
- **Contract** Compare the same live pairs in the same order, including later
  stages that can affect NaN and equal-value ordering. In those stages the
  omitted upper blocks contain only sentinels. The cache distinguishes the
  network width, the physical workgroup width, and the sort order. Preserve
  the existing global buffers, bindings, tile count, and multi-tile merge path.
- **Validation and removal** Exercise production selection, uniform layout,
  cache reuse and multi-tile fallback in both providers. Compare the emitted
  kernels against the original GPU output and a CPU implementation of the
  padded network. Use the pinned upstream sort when the overlay is omitted.
  Remove this patch when upstream prunes equivalent padding and loads merge
  pipelines on demand.

## Vector Flash Attention mask workgroups

- **Patch** `ggml-webgpu-flash-mask-block-workgroup.patch` against llama.cpp `c212d25ad`.
- **Scope** Cap the mask classifier workgroup at 32 invocations for vector
  attention tiles of up to 32 keys. Smaller device limits still apply. The
  cache distinguishes the mask tile and workgroup widths.
- **Contract** Use the existing shader and one workgroup per mask tile. The
  extra invocations on wider workgroups contributed only neutral reduction
  values. Preserve the three mask states, grid, uniform words, storage
  bindings, scratch region and the consuming attention pipeline.
- **Validation and removal** Check real mask dispatch selection in both
  providers and replay emitted kernels with complete tiles, tails, padding,
  separate heads and batches, and F16 special values. Use upstream's original
  width in the absence of the overlay. Remove the patch when upstream bounds
  classifier work by the tile width with equivalent outputs.

## Q8_0 matvec sigmoid gate

- **Patch** `ggml-webgpu-q8-sigmoid-mul.patch` against llama.cpp `c212d25ad`.
- **Scope** Fuse a dense K2304/M3072/N1 Q8_0 matvec, its F32 sigmoid, and
  multiplication by an independent contiguous F32 vector of length 3072.
  Require one batch and the ordinary Q8_0 pipeline with a fixed subgroup
  width of 16 or 32. Use the same accumulator and multiply in the existing
  operand order.
- **Contract** The current backend graph slice must contain all three nodes.
  Check direct and raw-view readers of the discarded tensors. Accept a raw
  contiguous view as the other MUL input. An exact physical alias between
  this input and the result uses a single binding and reads each element
  before writing it. Check the GPU buffer ownership, lengths, alignment,
  and binding limits. The Q4_K SwiGLU fusion shares these binding checks.
- **Integration** The normal and profiling providers specialize the same
  subgroup source. The two gate variants have separate pipeline cache keys.
  GGML layouts, model placement, and the public ABI are unchanged.
- **Validation and removal** Exercise both binding layouts, graph splits,
  additional raw-view readers, and varying subgroup widths through the
  complete provider. Replay the emitted shaders for both subgroup widths
  with packed Q8_0 inputs and compare the complete output against the three
  original dispatches. Keep the original dispatches for unsupported graphs
  or limits. Remove this patch when upstream supports an equivalent
  epilogue with the same lifetime, layout, and subgroup guarantees.

## Fixed-subgroup indexed Q8_0 and Q5_K decode

- **Patch** `ggml-webgpu-indexed-q8-q5-subgroup.patch` against llama.cpp `c212d25ad`.
- **Scope** Extend the fixed-subgroup indexed matvec to Q8_0 at K1152 or
  K2304 and Q5_K at K2304. Require M2304, eight selected experts, one input
  token, F32 activations and outputs, and fixed subgroup width 16 or 32.
- **Contract** Use the existing quantization loaders, packed weight layout,
  and physical binding checks. Select the 128-wide workgroup only when it
  fits the X dimension. The independent shader writes the same output
  layout and zeroes skipped expert slots. F32 reduction order changes.
- **Validation and removal** Check selection and fallbacks in both providers.
  Replay the emitted kernels with bounded slices of representative GGUF
  weights and check every selected output against an independent reference,
  including duplicate and skipped experts. Other shapes and capabilities use
  the existing backend selection and cooperative dispatch when supported.
  Remove this patch when upstream supplies equivalent selection for these
  packed formats.

## Packed row views in the Inkling graph

- **Patch** `inkling-packed-row-views.patch` against llama.cpp `c212d25ad`.
- **Scope** In WebGPU preparations, read packed F32 rows of loop messages
  and mixed attention inputs as views. Use the canonical contiguous layout
  for a single token and preserve the bank's token stride for batches.
  Other tensor formats use the existing extraction path.
- **Contract** Keep the row offsets, token order, tensor shapes, and model
  arithmetic. Both token counts use a view with an explicit source edge.
  GGML tracks the bank's lifetime across all consumers. Keep the ordinary
  COPY-based graph when this overlay is omitted.
- **Validation and removal** Compare the actual CPU and Wasm64 model graphs
  across prefill, decode, and append with an authored Inkling GGUF. Confirm
  matching logits, scheduler allocations, and the affected dispatches.
  Review consumer strides, aliasing, and placement when updating upstream.
  Remove this patch when upstream reads these rows without a materialized copy.

## WebGPU 1024-point Hadamard transform

- **Patch** `ggml-webgpu-fwht-1024.patch` against llama.cpp `c212d25ad`.
- **Scope** Use a butterfly transform for an explicitly Hadamard-hinted
  F32 matrix product with packed 1024-element columns, at least five columns,
  and default source and accumulator precision. Require 128 invocations and
  4 KiB of workgroup storage. Keep the ordinary matvec and subgroup-matrix
  routes for their existing inputs.
- **Contract** Preserve the register-tile path's half-precision input rounding
  and the normalized Sylvester coefficients. The F32 accumulation order
  changes. Each workgroup reads one input column and writes the matching output
  column. Accept contiguous input views and reject overlapping aligned storage
  ranges. The dispatch binds the input and output plus uniforms. The graph and
  loader retain the dense weights for fallback.
- **Validation and removal** Exercise the complete normal and profiling
  providers with contiguous views, shared buffers, explicit precision, and
  unsupported layouts and limits. Compare the emitted WGSL with the dense
  register-tile shader on representative decode and prefill shapes. Omitting
  this overlay uses the original matrix product. Remove it when upstream
  WebGPU accelerates the Hadamard hint with compatible layout and precision.
