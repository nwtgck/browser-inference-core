# Accepted upstream exceptions

Read [AGENTS.md](AGENTS.md) before changing an exception. This is an inventory of
existing, deliberately limited deviations, not permission to add another one.
The original reason for carrying source patches was required multimodal
performance, not general feature expansion. The project owner decides which
requirements justify upstream maintenance work.

The vision/audio patches take the pristine pinned upstream tree as input;
neither depends on a TTS-generation copy. Other accepted overlays compose only as
specified below. The current reference pin is
`8642541b255bfb5dc39cd351a178be09013a55d6`. Re-evaluate compatibility at each pin
update rather than treating that revision or these workarounds as permanent.

See [the current upstream review](../docs/llama-v0.8.0-review.md) for reviewed
contracts and validation boundaries. The [338fce review](../docs/llama-338fce-review.md)
and [v0.7.0 review](../docs/llama-v0.7.0-review.md) remain historical context.

## Vision BF16 compatibility/performance

- **Patches:** `mtmd-webgpu-bf16.patch`, `mtmd-vision-bf16-runtime.patch`,
  and `mtmd-vision-bf16-clip-runtime.patch`.
- **Accepted scope:** the owner's explicit BF16-memory request permits per-load
  immutable storage selection in the same runtime. Existing entries remain
  legacy-expand; explicit retain-checked uses original BF16 storage and rejects
  unreviewed roles/graphs/placement instead of retrying in F32 or on CPU.
- **Integration:** checked build-tree clip.cpp and mtmd.cpp copies, preserving
  upstream parameter-record layouts. The thin mtmd API plumbing exists in every
  profile; WebGPU loader handling remains under `LCB_WEBGPU_BF16_PROJECTOR`.
  Other profiles use a minimal clip copy that rejects unsupported retained GPU
  vision only at existing modality classification, before weight allocation;
  audio-only loads keep their original behavior without a second metadata parse.
  Explicit CPU loads retain upstream storage and are not WebGPU acceptance.
- **Coupling:** retained upload/graph helpers use public ggml interfaces;
  `mtmd-retained-roles.h` explicitly uses pinned private clip_model fields. The
  source contract, compile-target check, generated ABI and provenance expose
  this maintenance cost. Vendor, audio, model builders and shaders stay intact.
- **Compatibility:** ordinary assets contain both modes without extra variants
  or build directories. No global/TLS selector or post-load mutator is added.
  Upstream owns the graph-allocation failure check; preserve it unchanged.
- **No-patch alternatives:** keep established expansion, compatible weight
  formats, the existing CPU path or defer retention. Disabling the whole overlay
  also removes independent safety/diagnostics and can restore full-tensor scratch.
- **Validation and limits:** [storage design](../docs/webgpu-bf16-projector.md).
  CPU/synthetic/source-extracted tests do not establish actual browser/model
  correctness, physical peak or equal speed. Defaults are not switched on that
  evidence alone. New shaders and wider role acceptance are separate scope.
- **Removal:** upstream supplies the required storage, safety and performance
  contract, or the owner withdraws the requirement; retire API/plumbing/tests
  coherently without changing immutable artifacts or unrelated overlays.

## Non-pthread Wasm audio preprocessing

- **Patch:** `mtmd-audio-single-thread.patch`.
- **Retention decision:** keep the already-applied compatibility behavior during
  this cleanup; do not silently remove reference-audio support. This retention
  does not authorize additional audio pipelines or new model features.
- **Scope:** shared mel preprocessing and the existing separate Parakeet worker
  loop, only for non-pthread Emscripten builds. Parakeet is retained unchanged,
  not claimed to be necessary for Qwen synthesis or newly approved model support.
  Qwen synthesis without a speaker reference bypasses the affected reference
  preprocessing. Native/pthread worker behavior is unchanged.
- **Integration:** one independent `tools/mtmd/mtmd-audio.cpp` build-tree copy.
  It does not change inference backend selection or add a public interface.
- **No-patch alternatives:** omit the affected reference-audio path or redesign
  the browser profiles for threads; neither is a silent substitute for this
  compatibility behavior.
- **Validation and limits:** see [audio preprocessing checks](../docs/audio-single-thread.md).
  Host simulations do not certify real browser model execution.
- **Removal condition:** upstream handles this single-thread configuration, or
  the owner explicitly changes the affected feature/profile requirements.
  Reassess the separate Parakeet scope with the owner, not as incidental cleanup.

## Removed optional generation extensions

The previous `mtmd-tts-generation.patch` is removed: final-frame waveform
optimization, automatic-language conditioning, and the downstream capability
query are not required for basic TTS (Text-to-Speech) generation. No replacement
pipeline, new-file copy, or consumer-side reimplementation is introduced.
Generation, graph/state layout, and public audio-helper headers follow upstream.
Reintroducing a downstream version of either extension requires a new
scope-specific decision. A future upstream implementation follows normal
upstream-update review; do not patch it out to freeze the old behavior.

## Other existing deviations

The Emscripten Asyncify/BigInt correction in repository-root
`scripts/patch_emscripten.py` (moved unchanged to the shared toolchain layer) and the
pinned link-option correction in `CMakeLists.txt` are existing toolchain/build
workarounds, not new exceptions introduced by this cleanup. Their behavior is
unchanged. The same approval policy applies to expanding them even though they
are not `.patch` files. The Asyncify correction remains separately reported.

## Inventory and publication

`prepare_mtmd.py`, the CMake hooks, updater preflight, tests, and
`upstream_provenance.py` all use this directory. Provenance records the retained
source/compiled identities and inventories any other `.patch` files here, even
when unclassified. Do not move a divergence elsewhere to hide it from that report.
A rename or successful preflight does not prove numerical/browser compatibility.

Removing source changes cannot alter an already published artifact. Rebuild,
validate, and publish the cleaned source; update consumers to that actual new
artifact through their normal verified update flow. Do not republish an existing
commit with different bytes or invent a future artifact identity.

## Generated browser Wasm URL annotation

- **Accepted scope:** the owner authorized the narrow producer-side URL annotation
  after reviewing consumer-side alternatives and update costs. Every ordinary
  browser profile, including CPU, receives one `/* @vite-ignore */` comment
  immediately before the first argument of the generated sibling Wasm URL.
- **Integration:** `scripts/prepare_browser_module.py` runs after the normal link
  in `scripts/build.py`. It accepts exactly the known raw or already annotated
  sibling URL expression (including standard whitespace and either quote style),
  fails on drift or duplicate Wasm URL targets, and preserves
  every other generated byte. A skipped-link rerun does not add another comment.
  Test variants, native builds, upstream sources and the SDK are unchanged.
- **Behavior and cost:** Vite no longer discovers/copies this Wasm from that URL
  in the default browser entry. Hosts explicitly deploy it, provide `locateFile`,
  or supply `wasmBinary`. Runtime acquisition/fallback code is unchanged. An
  Emscripten output-shape change requires review of this boundary, without a
  whole-module hash allowlist or a new distribution variant.
- **Evidence:** per-build provenance records actual input/output byte counts and
  hashes, the preparation script's source identity and whether it was a no-op.
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
- **Reporting:** retirement is recorded in `retiredSourceOverlays`, never as an
  active source patch. Existing published manifests are not rewritten.
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
- **Integration:** apply after the other WebGPU source overlays to the checked build-tree translation unit for every WebGPU profile. CPU profiles and the pristine upstream checkout are unchanged. The generic backend and queue contracts, reviewed source and package headers, patch, composition input and final output are pinned; drift requires review.
- **No-patch alternative:** retain the existing extra synchronization before each async fallback read or wait for an equivalent upstream implementation.
- **Validation and limitations:** source-bound deferred queue tests exercise ordering, ranges, destination lifetime and failure paths; full source syntax checks use the pinned headers. These checks do not establish browser correctness, device-loss recovery or target-device speedup. The queue-order assumption must be validated on the target browser and GPU.
- **Removal:** upstream provides equivalent behavior or the owner drops this optimization; remove the patch, preparation, build wiring, tests and provenance entry together.

## Ordered WebGPU async tensor copies

- **Patch:** `ggml-webgpu-async-tensor-copy.patch`.
- **Scope:** implement the existing optional copy hook for checked same-queue WebGPU copies and CPU-to-WebGPU uploads. CPU producers are synchronized before their host bytes are uploaded; unsupported devices, layouts, ranges, aliases or usages retain generic fallback. Queue ordering and the existing buffer-copy checks remain in force.
- **Integration:** apply in the checked build-tree WebGPU translation unit for WebGPU profiles; pin the generic backend, buffer and scheduler contracts, patch, composition input and output. CPU profiles and upstream source are unchanged.
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
- **Integration:** apply to the build-tree WebGPU translation unit after ordered readback. Reviewed upstream source and generic backend contracts, Dawn headers, patch and composition input/output hashes are pinned. CPU profiles and the upstream checkout are unchanged.
- **No-patch alternative:** create a bind group for each dispatch as upstream does, or wait for equivalent upstream reuse.
- **Validation and limits:** source-extracted CPU queue and lifetime fixtures cover key equality, position, generation, bounds, fallback and parameter uploads, and the complete composed source is syntax-checked against pinned headers. These checks do not establish browser correctness, actual model hit rate, VRAM savings or target-device speedup.
- **Removal:** upstream supplies equivalent reuse or the owner drops this optimization; remove patch, preparation, build wiring, tests and provenance entry together.

## Single synchronization for WebGPU sampler reads

- **Patch:** `llama-sampler-single-sync.patch`.
- **Scope:** in `llama_sampler_sample`, synchronize the context once before reading its internal sampled outputs and counts. If sampled logits are absent, use the existing ordinary logits fallback after that synchronization. Public getters keep their original behavior. CPU profiles retain the upstream sampler.
- **Integration:** for every WebGPU profile, prepare a checked build-tree sampler translation unit and replace its source on the existing llama target. Input, patch and output hashes and the upstream context, sampler and backend contracts are pinned. The sampler remains independent of the model, loader and mtmd overlays; upstream source is unchanged.
- **Evidence and limits:** the baseline tiny-model Wasm observation found five queue-completion calls during one sampling operation with no intervening enqueue. Source-extracted CPU tests exercise 107 output, sampler and scheduler cases, including public getters and the logits fallback; the full prepared source is syntax-checked with upstream headers. These checks do not establish target-browser correctness, model quality or throughput improvement.
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
- **Reviewed inputs:** 8642541b255bfb5dc39cd351a178be09013a55d6; exact loader,
  file-reader and target-layout identities augment existing backend/API guards.
  Copies are isolated; upstream sources are never rewritten. Updater preflight
  and provenance use the same preparation. Input drift requires semantic review.
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
- **Validation/limits:** focused deterministic CPU queue tests extract the actual
  generated helper, checking pre-write waits, cross-tensor/device accounting,
  auxiliary charges, overflow, cleanup and initialization failure. Independent
  32-bit and 64-bit integer references cover chunk selection, exact-multiple
  fallback, cross-tensor regressions and positive adoption; these are not Wasm
  ABI executions. The complete generated loader passes native syntax checking
  against pinned upstream headers;
  actual overlay/target/provenance contracts remain tested. These are not full
  Wasm builds or browser/GPU tests. The 32 MiB bound covers charged loader uploads,
  not total Wasm, process, driver or GPU memory. Actual pending peaks can rise
  from three to four chunks within that unchanged cap; lower read-vector maxima
  do not establish unchanged physical or Wasm high-water memory. Existing
  completion/device-loss behavior is unchanged; cancellation cannot interrupt a hung wait, and graceful
  OOM/device-loss recovery is not claimed. See
  [bounded WebGPU model uploads](../docs/webgpu-model-upload-budget.md).
- **Removal:** equivalent upstream support, withdrawal of the requirement, or costs
  exceeding measured benefit; remove this patch and its entries in the shared
  preparation, hook, tests and provenance without disturbing other overlays.


## Single-token WebGPU SSM convolution

- **Patch:** `ggml-webgpu-ssm-conv-single-token.patch`.
- **Scope:** select one token lane for SSM_CONV when output tokens per sequence equal one; retain eight lanes otherwise. Channel block size, scalar/length-four arithmetic, upstream WGSL source-file bytes and bounds checks remain unchanged. The preprocessed shader workgroup size changes for single-token input. Include the selected width in the existing pipeline cache key, equality/hash and dispatch decisions.
- **Integration:** the ordinary `LCB_WEBGPU` source preparation emits a checked `ggml-webgpu-shader-lib.hpp` beside its existing WebGPU source copy. It does not emit `ggml-wgsl-shaders.hpp`; the upstream-generated header remains authoritative after MoE overlay retirement. No new workflow, profile, variant or feature switch.
- **Reviewed inputs:** pinned `620d2225cc436ab0b001e3200e590e3b286ef9ab` host WebGPU source/header and `ssm_conv.wgsl`; strict patch/output identities fail closed on unknown inputs. The v0.7.0 review refreshes the source/output identities; the accepted single-token behavior and patch bytes are unchanged.
- **Evidence and limits:** isolated shader correctness and tiny LFM full-runtime tests pass on software WebGPU adapters, including both pipeline creation orders and full/partial checkpoint restoration. Those tests do not establish production model quality, physical GPU occupancy or a Mac throughput improvement. Other MoE operators are not modified.
- **No-patch alternative:** retain the existing eight-token launch for decode. The change removes inactive logical token lanes; it does not guarantee a performance improvement.
- **Removal condition:** remove this preparation/header patch when upstream selects an equivalent single-token workgroup and the same supported profiles pass correctness and device validation, or when the owner drops the optimization.

## Qwen4Exp file-backed per-layer embedding rows

- **Patch:** `qwen4exp-file-backed-rows.patch`.
- **Approval scope:** the project owner's request to implement the reviewed
  Qwen3.8 browser design after explicitly discussing llama.cpp patch size.
  This authorizes this Qwen4Exp table path, not unrelated model/backend work.
- **Base:** v0.7.0, `620d2225cc436ab0b001e3200e590e3b286ef9ab`.
  Input hashes are checked in `tests/file-backed-rows/inputs.json`.
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
- **Validation / limitations:** see `docs/file-backed-rows.md`. Synthetic native
  graph equivalence is not a real-model WebGPU inference claim.
- **Removal condition:** pinned upstream offers a tested non-mmap bounded row
  reader with compatible ownership and browser behavior, or the owner drops
  the requirement. Re-run lifetime, split, graph equivalence and browser tests
  before removing the overlay and retiring the versioned consumer handshake.


## QSA all-pools selection

- **Patch:** `qwen4exp-all-pools-selection.patch`.
- **Accepted scope:** continuation of the reviewed 008/009 short-context QSA
  optimization requested by the owner. No model repository, quantization name
  or device-name branch; no new longer-context GPU kernel is bundled here.
- **Why / alternatives:** the pinned graph scores and ranks candidates even when
  every padded pool fits its budget. No existing runtime option removes that
  redundant work. Keeping the upstream graph, or deferring until equivalent
  upstream support, avoids this maintenance cost.
- **Base / dependencies:** `338fce6511c744893cc0d3d958b275f40552c7a4` and the
  explicit semantic and source/output hashes in `tests/qsa-selection/inputs.json`.
  Positive rectified scores, zero/-infinity visibility, and set-valued use of
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

## Four-stream HC PRE/POST fusion (2026-10-19)

The user explicitly authorized bicore/Naidan patches for the Qwen3.8-Flash-Next
performance investigation. The narrow overlay `ggml-webgpu-hc-pre-post.patch`
adds WebGPU execution of existing gated PRE and identity POST operations for
contiguous F32 HC4 only. No graph rewrite or extra profile is introduced. The
no-patch option is the existing staged graph. See
[`webgpu-hc-pre-post.md`](../docs/webgpu-hc-pre-post.md) for scope, maintenance,
removal, continuous software-GPU validation and remaining physical-GPU/model
measurements. Local preparation, host tests and standalone numerical checks do
not establish target-device speed.

## Q8_0 short-input matvec workgroups

- **Patch:** `ggml-webgpu-q8-short-workgroup.patch`.
- **Purpose and approval:** the owner's explicitly authorized Qwen Next
  performance work, based on the supplied post-HC profiles. Limit this exception
  to choosing the existing upstream workgroup specialization; no model change.
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
- **Integration:** apply after the other WebGPU host overlays to the checked build-tree shader library for every WebGPU profile. Reviewed upstream shader/host contracts, patch, composition input and final header identities are pinned. No shader file or upstream checkout is rewritten.
- **No-patch alternative:** retain the upstream fixed workgroup or wait for upstream shape-dependent selection.
- **Validation and limits:** source-bound factory and dispatcher tests exercise selection, cache keys and fallback. The opt-in standalone GPU fixture compares actual generated shaders with the upstream path and an independent CPU reference. These checks do not establish browser model quality or target-device performance.
- **Removal:** upstream supplies equivalent selection or the owner drops the optimization; remove its patch, preparation, build wiring, tests and provenance entry together.

## Fused WebGPU scale and unary operations

- **Patch:** `ggml-webgpu-scale-unary.patch`.
- **Scope:** fuse eligible contiguous F32 scale followed by SILU or sigmoid, with an optional final scale, into one dispatch. Admission checks graph usage, supported layout and binding ranges; unsupported or unsafe combinations retain upstream dispatches. The fusion preserves the operation order and does not change model construction.
- **Integration:** compose after existing WebGPU host overlays in every WebGPU profile, generating an isolated shader header and host helper. Reviewed graph, allocator, backend and upstream shader contracts, patch and stage output identities are pinned. CPU profiles and upstream source are unchanged.
- **Validation and limits:** source-bound graph, allocator, dispatch and shader factory tests cover accepted and rejected graphs and fallback. Software GPU checks do not establish model quality or target-device speedup.
- **No-patch alternative:** use the existing separate dispatches or wait for upstream fusion.
- **Removal:** equivalent upstream fusion or withdrawal of the optimization; remove patch, preparation, build wiring, tests and provenance together.

## Fused WebGPU RMS normalization and scale

- **Patch:** `ggml-webgpu-rms-scale.patch`.
- **Scope:** fuse eligible F32 RMS normalization followed by scale into one dispatch, retaining upstream reduction and normalization before the scale. Existing RMS-multiply fusion keeps priority. If the following scale begins an eligible scale-unary-scale chain, RMS fusion declines so the existing three-operation fusion remains available; RMS-scale takes priority over the two-operation scale-unary case. Unsupported graphs, layouts and binding ranges retain upstream behavior.
- **Integration:** compose the checked host and shader overlays in every WebGPU profile. Embed the modified row-normalization shader separately so the canonical upstream shader remains authoritative for other paths. Pin graph, allocator, shader and backend contracts, stage identities and generated header. CPU profiles and upstream source are unchanged.
- **Validation and limits:** source-bound graph, allocator, alias and dispatch tests and 128 software WebGPU cases passed. These checks do not establish production model quality or target-device throughput.
- **No-patch alternative:** retain separate RMS and scale dispatches or wait for upstream fusion.
- **Removal:** equivalent upstream fusion or withdrawal of the optimization; remove patch, preparation, build wiring, tests and provenance together.

## Fixed-subgroup Q8_0 matvec

- **Patch:** `ggml-webgpu-q8-subgroup-matvec.patch`.
- **Scope:** for guarded long-K Q8_0 x F32 single-column matvecs with a fixed reported subgroup size, assign output rows to independent subgroups in one workgroup. Preserve the pinned upstream Q8 decoder and accumulator; unsupported shapes, device limits and varying subgroup ranges use the existing path. No model-specific selection is added.
- **Integration:** compose after scale/unary fusion in the shared WebGPU header, emitting an isolated shader header using the pinned upstream include expander. Patch, both intermediate inputs and final outputs, and upstream shader/host contracts are pinned. CPU profiles and upstream source are unchanged.
- **Partial subgroups:** even with a fixed reported width, a subgroup can be partial. Each active invocation covers additional logical lanes, preserving coverage without shared scratch; the full-packed path retains its previous calculation. Geometry probing and readback are test-only and add no production CPU readback.
- **Validation and limits:** source-bound selector, cache key and dispatcher tests passed. Software WebGPU tests covered partial geometries at reported widths 32/64 and compared full-packed widths 4/8/16 to the previous subgroup shader bit for bit. Earlier checked cases reached approximately 1e-7 normalized error, with increased RMSE relative to baseline. Arithmetic and reduction order differ from the upstream baseline; production model quality and target-device speedup are not established.
- **No-patch alternative:** retain the existing matvec path or wait for equivalent upstream support.
- **Removal:** equivalent upstream implementation or withdrawal of the optimization; remove patch, preparation, build wiring, tests and provenance together.

## Fixed-subgroup indexed IQ3_S matvec

- **Patch:** `ggml-webgpu-iq3s-id-subgroup.patch`.
- **Scope:** assign rows of eligible single-token IQ3_S indexed matvecs to independent fixed-width subgroups. Shape, strides, complete blocks, binding ranges, device limits and subgroup width are checked; unsupported cases retain the upstream path. The upstream quantization loaders and accumulator are reused, and selection is independent of model identity.
- **Integration:** compose the checked WebGPU host and shader overlays after the other WebGPU changes. Embed the dedicated shader with the pinned upstream include expander; retain canonical upstream shader sources for other paths. Upstream contracts, patch, composition stages and generated header are pinned. CPU profiles and upstream source are unchanged.
- **Partial subgroups:** even with a fixed reported width, a subgroup can be partial. Active invocations cover additional logical lanes; the full-packed path retains its previous calculation without extra shared scratch. Geometry probing and readback are test-only and add no production CPU readback.
- **Validation and limits:** source-bound selector, cache and dispatch checks and software GPU numerical fixtures cover accepted and rejected cases, including partial geometries at reported widths 32/64. A full-packed width 16 matched the previous subgroup shader bit for bit in the tested cases. Reduction order differs from the upstream baseline; production model quality and target-device speed are not established.
- **No-patch alternative:** retain the existing indexed matvec path or wait for upstream support.
- **Removal:** equivalent upstream support or withdrawal of the optimization; remove patch, preparation, build wiring, tests and provenance together.

## Lane-selected fields in indexed IQ3_S matvec

- **Patch:** `ggml-webgpu-iq3s-field-loads.patch`.
- **Scope:** replace only the existing independent indexed IQ3_S embedded shader. Each half-lane reads its selected qs word, and qh/sign/scale fields use their containing aligned word. Keep the 110-byte block representation, decoder arithmetic and addition order, partial-subgroup main, IDs, bindings, selector, cache key and fallback paths unchanged. Canonical templates and other quantization/ordinary/cooperative shaders are unchanged.
- **Integration:** append a source-bound stage after the original IQ3 stage. Patch only a private accumulator copy and regenerate the dedicated header with the pinned upstream emitter. Preserve the original stage output hash as an intermediate contract; provenance records the actual final compiled header separately. No additional runtime buffers, probes or synchronization are introduced.
- **Validation and limits:** actual factory output for logical widths 16/32/64 is source-bound. Synthetic numerical comparisons covered width16 full and reported-width32 partial execution, input updates, skipped IDs, binding ends and guards. Target-device performance, full hardware width32/64, Mac/Metal and complete-model quality/performance are not established. Source read-expression counts are not physical bandwidth measurements.
- **No-patch alternative:** retain the existing independent indexed matvec or wait for upstream equivalent field loads.
- **Removal:** upstream equivalent support, withdrawal of this optimization, or removal of its parent indexed IQ3 implementation; remove this stage, patch, build dependencies, tests and provenance together.

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
- **Approval and scope:** the owner's continuing performance investigation explicitly authorizes this narrow fusion. Fuse only eligible adjacent F32 SSM_CONV/UNARY-SILU graphs with one use of the convolution result. Preserve canonical scalar/vec4 dot arithmetic, the SiLU expression and existing single-token workgroup specialization. No model graph or CPU behavior changes.
- **Safety and integration:** compose after the other checked WebGPU overlays, for existing WebGPU profiles. Preserve intermediate observers, outputs and callback graph slices through shared fusion checks; validate shapes, u32/grid/device bounds, view roots, buffer ownership and disjoint payload/aligned bindings. The final SiLU may reuse the unused convolution allocation. A dedicated embedded shader keeps canonical sources unchanged; profiling uses only the existing optional profile.
- **No-patch alternative:** retain the existing separate dispatches or wait for upstream fusion. There is no existing runtime setting for this epilogue.
- **Maintenance and validation:** pin the graph, allocator, backend, shader, preprocessor, patch, stage outputs and embedding identities. Source-bound native tests exercise graph admission, actual allocator reuse, runtime bindings, dispatch, cache variants and both profile syntax configurations. These tests do not establish browser/model quality or target-device speed.
- **Removal:** upstream implements equivalent guarded fusion or the owner withdraws it; remove the patch, preparation, build wiring, tests and provenance entry together.


## WebGPU MoE matrix inactive column groups

- **Patch:** `ggml-webgpu-moe-group-skip.patch`.
- **Approval and scope:** the owner selected the constant-loop group guard after
  isolated comparisons. Skip only matrix register-tile calculation for a local
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
- **Validation and maintenance:** pin the patch, input stage, host/gather/shader
  contracts, expanded embedding and outputs. Tests exercise normal preparation,
  actual factory/preprocessor and cache behavior, canonical byte comparisons and
  both profile syntax configurations. Opt-in synthetic GPU checks compare every
  output byte with canonical shaders and verify routing metadata, guards and
  immutable inputs. Software-adapter results do not establish Mac/model speed.
- **Removal:** upstream provides equivalent inactive-group handling, or the owner
  withdraws it; remove the patch, preparation, dedicated embedding, build wiring,
  tests and provenance entry together.

## Fused WebGPU gather normalization tail

- **Patch:** `ggml-webgpu-router-tail.patch`.
- **Approval and scope:** the owner's continuing performance investigation explicitly authorizes this generic F32 GET_ROWS/RESHAPE/SUM_ROWS/CLAMP/DIV/RESHAPE fusion. Preserve the selected IDs, original maximum-workgroup reduction tree, clamp and direct division. No softmax, sorting, quantization or model-specific behavior changes.
- **Aliasing:** disjoint bindings use three storage buffers. Only one-token, one-workgroup execution may merge overlapping probability/output ranges into one storage binding; all selected loads precede a uniform storage barrier and every store. IDs stay disjoint from the full union. Multi-token overlap retains the existing staged path, with no extra model scratch or weight copy.
- **Integration:** compose near the end of the ordinary WebGPU source preparation, before host assembly and the optional profile-only Q8 diagnostic. Emit only dedicated helper/shader headers. CPU profiles and the base bindings/schema remain unchanged.
- **Coupling:** pin the six-node graph and view/use-count contracts, allocator behavior, typed buffers/device limits, four baseline WGSL sources, shader preprocessor/emitter, host/header inputs, patch and outputs. Reject intermediate OUTPUT/side users, unsupported views, callback subgraphs and invalid or overlapping ranges before pipeline creation.
- **No-patch alternative:** retain the four staged dispatches or wait for equivalent upstream graph fusion.
- **Validation and limits:** actual-ggml graph/allocator and extracted dispatcher/factory tests, normal/profile syntax checks and synthetic software-GPU byte comparisons cover the guarded paths. These do not establish target-device speed or full-model quality. See `tests/webgpu-router-tail/README.md`.
- **Removal:** remove the stage when upstream provides equivalent ordered gather normalization and alias handling, or the owner drops the optimization.

## Profile-only Q8 kernel diagnostic

- **Patch:** `ggml-webgpu-q8-diagnostic.patch`.
- **Approval and scope:** the owner's explicitly authorized kernel A/B diagnostic. Only `GGML_WEBGPU_GPU_PROFILE` builds add a versioned typed registry provider, fixed-width bridge, separate generated extension schema and explicit Q8 variants. AUTO preserves real production selection; forced variants never replace normal selection. No model data or host application policy is added.
- **No-patch alternative:** compare separate immutable artifacts or defer the diagnostic; neither supplies explicit same-device/same-buffer selection in one runtime.
- **Safety:** bounded contiguous, non-view Q8_0/F32/F32 single-vector tensors, checked physical/logical bindings and device/grid limits; reject unsupported requests before enqueue. PREPARE creates pipelines and records actual processed-WGSL SHA-256; EXECUTE only looks up prepared keys. Same-context operation guards span pipeline and profile-map suspension. Borrowed tensors are not retained; existing bind-group buffer references follow their generation cleanup contract. Hosts serialize all calls and destroy a poisoned runtime after unexpected rejection rather than freeing pending resources.
- **Integration:** apply a final profile-only stage after the other overlays. Normal final source and common generated ABI remain byte-identical. Pin upstream factory/decoder/submit contracts, patch, stage outputs, bridge contract and the mechanically extracted upstream public-domain SHA-256 reference, including extraction boundaries. Separate extension records use generated native layout queries and a compiled schema fingerprint; absent exports remain optional for older artifacts.
- **Validation and limits:** focused source-bound native tests cover the ABI, cache, shader identity, dispatch, lifetime and guarded fallback; these do not establish real-device speed or model quality. Only the existing profiler supplies GPU timestamps.
- **Removal:** when upstream provides an equivalent optional diagnostic or the owner withdraws it, remove its profile stage, bridge, schema generator, build/provenance wiring and tests together.

## Synchronous host dispatch assembly

- **Patch:** `ggml-webgpu-host-encode-assembly.patch`.
- **Approval and scope:** explicitly authorized by the owner in the current performance-work session. Reduce temporary owning copies in the existing single/multi WebGPU builders. No new ABI, graph rewrite, kernel, selector, profile or host-application policy.
- **No-patch alternative:** keep the current builders and their temporary allocations; the existing exact bind-group cache already avoids warm creation calls. This change does not improve cache hit rate.
- **Contract:** a stack view borrows single-op parameters, entries and pipeline only until synchronous staging/encoding finishes. Cache and queued command ownership remain unchanged. Profile results retain owning pipeline names. Reserve temporary vectors within the existing arena slot bound and retain original parameter padding, binding key, upload/dispatch order and batch boundaries.
- **Caller observation:** the internal single builder takes const references. Moving named vectors into it no longer empties them. Every current caller returns immediately or only records the fused operation count before returning; none reads the vectors afterward. Recheck those call sites on upstream updates.
- **Integration:** append a checked source stage after router-tail and before the optional profile diagnostic. Pin upstream contracts, composed input, patch and output. No shader/header or common ABI changes.
- **Validation and limits:** source-extracted allocation tests use relative baseline comparisons, and the existing cache/queue/free/failure scenarios and complete normal/profile source checks remain required. Native mock allocations are not browser/GPU timings and do not establish model throughput or target-device improvement.
- **Removal condition:** remove the stage, provenance/build wiring and tests when upstream provides equivalent synchronous assembly without the copies, or the owner drops the optimization.

## Profile query capacity with multi-dispatch operations

- **Patch:** `ggml-webgpu-profile-query-budget.patch`.
- **Approval and scope:** the owner's session-wide authorization permits agent-decided upstream patches. This profile-only correctness repair follows a source-bound counterexample showing that a valid multi-dispatch operation could exceed the existing profile QuerySet before the next collection. It is separate from normal submission tuning.
- **No-patch alternative:** leave profiling disabled or restrict profiled graphs to avoid the problematic boundary. Neither makes arbitrary accepted profiled batches safe.
- **Contract:** every timestamped dispatch consumes one arena slot. Reserve queries using that actual slot capacity, including nominal-batch overshoot, and assert during profile initialization that the capacity is at most half the query limit. Keep collection after complete graph batches; do not flush inside the builder or introduce new encoder/name ownership.
- **Integration:** run a checked profile-only stage immediately before the optional Q8 diagnostic. Pin upstream contracts, input, patch and prefix output. Normal prepared source and the shared ABI/schema/shaders remain byte-identical.
- **Validation and limits:** extracted actual graph/build/arena control tests retain the failing baseline at the query boundary, the repaired behavior, capacity and arena rejection, plus normal byte invariance. The Encode callback supplies source-derived multi-dispatch counts; this CPU model is not a GPU or full GGML graph/factory execution. Actual profile runtime validation remains a separate check.
- **Removal condition:** remove the stage and its preparation/build/provenance/test wiring when upstream reserves queries for every accepted batch or supplies an equivalent safe collection boundary.


## F32 small-M prefill register tiles

- **Patch:** `ggml-webgpu-f32-small-m.patch`.
- **Approval and scope:** the owner's session-wide performance authorization permits this agent-selected geometry specialization. Only F32/F32/F32 matrices with K>=128, M1..4 and N>=5, without subgroup matrices or source overlap, use narrower existing register tiles. Existing N<=4 matvec routing, other types and fallback geometries remain unchanged.
- **Contract:** use the pinned upstream register-tile WGSL and TK8 accumulation unchanged. M1 uses tileM1; M2..4 use tileM4. Both use tileN1 and a 1x32 workgroup. Effective geometry is part of pipeline equality/hash before lookup, and profile names distinguish it. Parameter bytes, binding layout and dispatch grid remain unchanged.
- **Integration:** a checked header-only stage follows host assembly and precedes optional profile stages. Normal/profile shader-library identities change; host source, shared bindings/schema and canonical WGSL inputs do not. No extra runtime profile or native API is added.
- **Validation and limits:** source-bound admission/cache/factory/dispatch checks cover boundaries and guarded fallbacks. Private software-GPU geometry comparisons informed the scope; target-device/model speed and model quality are not established by those results. Revalidate arithmetic, layouts and performance before widening the scope.
- **Removal:** remove this stage and its metadata/tests when upstream provides a suitable equivalent or the specialization is no longer useful. Re-review upstream factory, key, geometry decisions, reg-tile loads/accumulation and dispatch contracts on pin changes.


## MoE gather logical lane groups

- **Patch:** `ggml-webgpu-moe-gather-logical32.patch`.
- **Approval and scope:** the owner's session-wide performance authorization permits this agent-selected specialization. Distribute only the existing skipped-slot zero loop among logical groups of at most 32 lanes; the grouping is independent of hardware subgroup features. Preserve expert counting, prefix/tile construction and slot-position atomics.
- **Contract:** shared storage and its expert capacity, existing barriers/atomics, seven parameter words, five bindings, gather grid, cache key and factory label remain unchanged. Partial final groups skip only zeroing; supported input/index/binding contracts are not widened. Matrix and matvec arithmetic are unchanged.
- **Integration:** a checked stage after F32 small-M embeds only the derived gather in a dedicated header and changes its factory source selection. Canonical upstream files and the existing matrix group-skip stage remain intact. Normal/profile library and gather identities are recorded separately from the base ABI/schema.
- **Validation and limits:** source-bound stage, ownership, factory/cache and unchanged-dispatch checks cover skipped/live slots and workgroup boundaries. Private software-GPU checks informed the candidate; benefits are not universal, and target-device/model performance is not established.
- **Removal:** remove the stage and its build/provenance plumbing when upstream supplies an equivalent or it ceases to be useful. Re-review gather ownership, metadata capacity, factory specialization and downstream dispatch ordering on upstream changes.

## Floating-point matmul vec4 layout guards

- **Patches:** `ggml-webgpu-matmul-vec4-layout.patch` and `ggml-webgpu-matmul-vec4-layout-profile.patch`.
- **Approval and scope:** the owner's session-wide authorization includes correctness repairs discovered by the performance tests. Existing F32/F16 ordinary and indexed matrix/vector factories selected vec4 on shape alone, allowing accepted padded or offset views to read rounded-down addresses. This repair uses the existing scalar fallback.
- **Contract:** test actual binding-relative input offsets, including merged source bindings, and active higher-dimensional strides before the factory cache lookup. Matrix stores also require the destination offset to align to four elements. Keep existing shape/type guards, scalar/vector cache dimensions, aligned shader bytes, uniforms, bindings, grids, quantized paths and gather behavior. Extent-one strides are unused.
- **Integration:** append a checked stage after the existing normal/profile prefixes. The profile patch accounts for the optional diagnostic's separate key builder. Preserve every earlier input/output pin; record final compiled identities separately. No vendor files or shader source are modified.
- **No-patch alternative:** callers can materialize aligned contiguous inputs, or avoid the affected layouts. This does not repair accepted operations using existing views.
- **Validation and limits:** source-bound caller/factory checks compare both profile modes, alignment residues, active/unused strides, merged offsets, cache order and unchanged shader/parameter contracts. Full pinned-header syntax and actual runtime checks are distinct gates. Falling back changes the existing reduction partition; incorrectly addressed outputs are not a numerical oracle. This correctness repair makes no speed or model-quality claim.
- **Removal:** retire the stage and associated preparation/build/provenance/tests when upstream checks the same layout conditions or supplies unaligned-safe loads/stores. Re-review binding rebasing and factory/key ownership on upstream updates.

## Tight routing-independent MoE tile capacity

- **Patch:** `ggml-webgpu-moe-tight-tile-bound.patch`.
- **Approval and scope:** the owner's session-wide authorization covers this narrow host-only optimization. Keep the current slot width L=32 and replace only its shared capacity helper after the checked normal/profile final stages. No shader, graph, parameter, key, indirect dispatch or runtime readback is added.
- **Bound:** with nominal slots S, experts E and A=min(E,S), use A+floor((S-A)/L). For a nonempty experts and V valid slots, sum ceil(c_e/L)=a+sum floor((c_e-1)/L)<=a+floor((V-a)/L)<=A+floor((S-A)/L). Assign one slot to A experts and concentrate the remainder to attain this generic bound. Duplicate IDs and skipped slots remain supported; real routing occupancy is not assumed.
- **Allocation and dispatch:** descriptor capacity, aligned binding size and direct matrix workgroups use the same helper. The result is at most S and does not add uint32 arithmetic overflow. Existing tensor-product, size_t, device capacity and grid admission limits remain unchanged. Single-token matvec dispatch is unchanged.
- **Integration:** preserve every earlier stage pin as a historical contract, append a checked stage with separate normal/profile source identities, and record actual final compiled copies. Canonical templates and the upstream checkout remain untouched.
- **No-patch alternative:** retain the conservative existing allocation/dispatch upper bound or wait for an upstream equivalent.
- **Validation and limits:** source-bound integer/extremal routing, real ggml allocator, recorded binding/grid and normal/profile syntax checks cover the shared contract. Reduced empty workgroups or descriptor bytes are not measured speed or peak-memory claims; target-device/model performance remains unestablished.
- **Removal:** remove the stage, patch, preparation/build/provenance and related tests when upstream provides an equivalent bound or the owner drops it. Recheck gather descriptor emission and tensor/device arithmetic contracts on upstream updates.

## Independent IQ3 four-row loop splitting

- **Patch:** `ggml-webgpu-iq3s-row-split.patch`.
- **Approval and scope:** the owner's session-wide performance authorization permits this narrow agent-selected specialization. Split only the existing independent indexed IQ3_S accumulator's row0..3 loop into row0..1 and row2..3, keeping its four accumulators. Do not change the canonical accumulator, ordinary/cooperative shaders, selector or supported subgroup widths.
- **Contract:** the existing factory supplies OUTPUTS_PER_WG=4. Preserve one source-level activation-block load, every row's decoder/FP expression and K order, return layout, main/partial-subgroup completion, reductions, parameters, bindings, geometry, cache and queue behavior. Pin and recheck this factory contract rather than applying the literal two-row split to another row count.
- **Integration:** append a checked dedicated-header stage after tight tile capacity. Copy the field-load stage's pinned derived accumulator, patch only its IQ3 branch and re-embed with the same main/common declarations. Retain every earlier stage identity as history; provenance records the final compiled header separately. Normal/profile host source, other headers and the shared ABI/schema are unchanged.
- **No-patch alternative:** keep the existing single four-row loop or wait for an upstream equivalent. This avoids maintaining another source specialization.
- **Validation and limits:** source-bound inverse/row-order/tail checks and actual factory/cache tests cover fixed widths16/32/64. Private software hot-only comparisons informed the choice; native load/register behavior, streaming/model performance and target-device gains are not established. Reusing the same mathematical source order is not a universal cross-compiler FP-bit guarantee.
- **Removal:** remove this patch and its preparation/build/provenance/tests when upstream provides equivalent code generation or the owner drops the specialization. Re-review the derived field-load input, four-row factory, preprocessor and emitter after upstream changes.


## Four-row F32 matvec scale/sigmoid epilogue

- **Patches:** `ggml-webgpu-f32-matvec-epilogue.patch` and `ggml-webgpu-f32-matvec-epilogue-profile.patch`.
- **Approval and scope:** the owner's session-wide performance authorization permits this narrow agent-selected fusion. Only dense F32/F32/F32 matvecs with K1..16384, M4, N1 and batch1, followed by SCALE, SIGMOID and SCALE, are eligible. No model branch or new public API is added.
- **No-patch alternative:** retain the existing matvec plus the existing three-operation scale/unary/scale fusion, for two dispatches in total.
- **Contract:** retain the original dot arithmetic, row assignment, reduction and workgroup geometry; append the three existing epilogue expressions only at the final store. Require graph use-count/callback visibility checks and all input/output payload and aligned binding ranges to be disjoint. A dead matvec allocation may be reused by the final output. Unsupported graphs, buffers, limits and kinds retain existing fallback.
- **Ownership and cache:** the default-none key and sixteen parameter words retain their original behavior. A distinct typed epilogue key carries no coefficients; four copied coefficient values extend only the fused path to eighty bytes. Check the actual uniform arena slot and legacy 128-byte binding. Buffer facts are local to one admission attempt; no permanent capability cache, runtime probe or timing resource is introduced.
- **Integration:** append a checked normal/profile source stage after the existing final stages, with a dedicated embedded main and canonical accumulator/declarations. Keep historical pins, vendor sources, other dtype shaders and the generated public ABI unchanged. Profile diagnostics keep the default-none path.
- **Validation and limits:** source-bound graph/gallocr, physical range, factory/cache, params and full translation-unit tests cover admission and fallback. Private software comparisons informed this candidate; they do not establish model, target-device or cold-start improvement. Actual runtime correctness and performance remain separate gates.
- **Removal:** remove the stage and its provenance/build/tests when upstream supplies the equivalent fusion or the owner drops the specialization. Re-review graph fusion, allocator reuse, binding rebasing, parameter staging, factory keys and canonical shader arithmetic after upstream updates.
