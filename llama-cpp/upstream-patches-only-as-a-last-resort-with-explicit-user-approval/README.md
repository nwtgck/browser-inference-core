# Accepted upstream exceptions

Read [AGENTS.md](AGENTS.md) before changing an exception. This is an inventory of
existing, deliberately limited deviations, not permission to add another one.
The original reason for carrying source patches was required multimodal
performance, not general feature expansion. The project owner decides which
requirements justify upstream maintenance work.

The two retained patches take the pristine pinned upstream tree as input;
neither depends on a TTS-generation copy. The current reference pin is v0.5.0,
`7fe450e19305b828c199d602c23a8337aaa1f03b`. Re-evaluate compatibility at each pin
update rather than treating that revision or these workarounds as permanent.

## Vision BF16 compatibility/performance

- **Patch:** `mtmd-webgpu-bf16.patch`.
- **Accepted purpose:** avoid the unusably slow vision-projector path identified
  by the project owner as necessary to fix. This is not general permission to
  optimize every model in downstream code.
- **Scope:** vision BF16 (Brain Floating Point 16) weights in the selected WebGPU
  build variants, controlled by `LCB_WEBGPU_BF16_PROJECTOR`. No audio-weight
  expansion and no change to stored model files.
- **Integration:** one `tools/mtmd/clip.cpp` build-tree copy, with the local
  conversion helper in `bridge/mtmd-bf16.h`. Internal loader coupling remains a
  maintenance cost even though the upstream checkout is unchanged.
- **v0.5.0 maintenance:** upstream now owns the graph-allocation failure check.
  Keep it as unchanged context and insert the existing placement diagnostics
  after it, instead of backporting the same guard again. The conversion helper,
  loader changes, profile scope, and independent audio patch are unchanged.
- **No-patch alternatives:** compatible weight formats, the existing CPU path,
  or waiting for suitable upstream support; these do not necessarily satisfy
  the required multimodal performance.
- **Validation and costs:** see [the existing design and tests](../docs/webgpu-bf16-projector.md),
  including increased resident storage and the limits of synthetic checks.
- **Removal condition:** upstream provides the required working, acceptably fast
  path on the target profiles, or the owner drops the requirement. Verify it
  with the affected models before retiring the workaround.

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

## Experimental WebGPU MoE direct selected-slot dispatch

- **Patch:** `ggml-webgpu-moe-direct-slot.patch`.
- **Scope:** locally authorized experiment, not final production approval. The
  user authorized experimental upstream patches on 2026-10-06; adoption/commit
  in their bicore remains their decision.
- **Purpose:** remove expert-count-sized workgroup arrays, barriers and the
  expert scan from the single-token `MUL_MAT_ID` vector shader. Index workgroups
  by selected slot, preserving selected expert IDs and output placement.
- **Integration:** the original patch bytes and upstream embedder generate one
  build-tree shader header. No vendor edits, new public API, model graph changes,
  sampler changes or application policy enter bicore.
- **Reviewed upstream inputs:** `d81235049384534c167caea52b85a694f6103d14` and
  `7fe450e19305b828c199d602c23a8337aaa1f03b`, with separate exact input maps.
  This is source compatibility, not a claim of identical shader expansions or
  completed browser/runtime validation for both revisions.
  Dispatch C++, shader, CMake header lookup and embedder identities are guarded.
  The user clarified on 2026-10-06 that this experiment need not be artificially
  restricted to wasm64 JSPI. All WebGPU profiles share this shader contract.
- **Activation:** standard `scripts/build.py` enables every configured WebGPU
  profile (wasm32 Asyncify/JSPI and wasm64 JSPI), both browser/test variants.
  CPU profiles pass OFF; direct CMake defaults OFF. Existing CI and source-bound publication are reused.
- **No-patch alternative:** retain upstream shader and current artifact, or wait
  for upstream adoption. This is an optional performance experiment.
- **Validation/costs:** see [the integration design](../docs/moe-direct-slot.md).
  Updating upstream requires semantic review; compiling or CI smoke tests alone
  cannot certify target-device model speed or quality.
- **Removal condition:** upstream supplies this behavior, the experiment is
  rejected, or its maintenance cost outweighs the verified improvement. Remove
  this patch, preparation/hook, build flag, tests and provenance entry together.
