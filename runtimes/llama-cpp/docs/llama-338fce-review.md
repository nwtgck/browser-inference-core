# llama.cpp 338fce integration review

## Pin and scope

This update moves the runtime from `620d2225cc436ab0b001e3200e590e3b286ef9ab`
to `338fce6511c744893cc0d3d958b275f40552c7a4`. The pinned Emscripten and Dawn
versions are unchanged. The upstream source checkout remains pristine; accepted
overlays continue to operate on checked build-tree copies.

The new upstream includes LFM3 text, vision and audio architectures. Their
presence in upstream and successful compilation do not establish trained-model
quality, complete multimodal application support, or target-device performance.
No new model-specific browser pipeline is introduced.

## Retained contracts

| Area | Upstream change | Integration decision |
| --- | --- | --- |
| WebGPU and shader generation | Q2_0 support changes shared templates and operation declarations. | Retain upstream additions and regenerate checked composition identities and processed-shader fixtures. Unrelated queue, arithmetic and eligibility contracts remain checked. |
| Indexed matrix multiplication | Expert ID -1 skips a slot. The gather stage gains destination offset/width and an output binding, and zeros skipped outputs. | Preserve the new gather ABI and zero initialization. Teach the independent IQ3 shader the same skip semantics while keeping all lanes that participate in barriers active. |
| QSA | Selection no longer accepts a separate base attention mask; upstream chooses F16/F32 and builds a contiguous two-dimensional result before the K/V stream reshape. | Share only within the same graph-local pool input and mask type. Preserve key-cache writes, the exact all-pools admission guard, the ranked TOP_K_MASK path and its unfused fallback. |
| File-backed PLE | Model definitions and private fixture helpers changed. | Preserve coherent `models.h` ownership across every relocated model translation unit. Preserve unity exclusions for both relocated core translation units and verify all semantic inputs during reconfiguration. |
| Vision BF16 | Upstream captures the embedding output before graph allocation, since scheduling can reorder graph nodes. | Keep that capture and the upstream allocation-failure guard. Insert the existing placement summary after successful allocation; retain bounded conversion and cancellation behavior. |
| Audio preprocessing | LFM3 introduces another caller of the shared mel preprocessing helper. | Retain the existing non-pthread Wasm helper behavior; native and pthread behavior remains upstream. No additional audio preprocessing implementation is copied. |

The source/output and semantic-input checks remain fail-closed. New hashes are
records of reviewed source, not substitutes for numerical and runtime tests.
The Q8 diagnostic stays profile-only and retains its separate schema and export
lists. All ordinary profiles continue to use the common generated API.

## Public API changes

Bindings are regenerated from the pinned public headers. The low-level schema
now includes upstream audio-stream and generation-state APIs, the new precision
and TOP_K_MASK operations, and `LLAMA_DRY_MAX_SEQ_BREAKERS`.

Upstream still implements `mtmd_helper_support_video` and
`mtmd_helper_model_can_chat` but marks them deprecated. Their existing direct
exports are retained alongside `mtmd_helper_get_capabilities`. Other deprecated
or variadic declarations keep their exclusions. No compatibility implementation
is substituted for an upstream function.

Upstream audio generation has an intentional ABI change: `mtmd_gen_audio_process`
requires a generation state, and several input/output record fields changed.
Hosts must use the new state/default helpers and the generated record layout
queries. Offsets or call signatures from an older artifact must not be reused.
The schema fingerprint still binds the module to its matching generated API.

The chat bridge exposes upstream content-part names, extra message metadata,
deferred-tool flags and ASR language metadata. The zero-argument ASR method is
retained; explicitly typed language-aware ASR and TTS prompt methods call the
corresponding upstream helpers. This does not reintroduce a downstream audio
generation pipeline or automatic-language workaround.

## Validation boundaries

The checked source/native tests, native QSA and file-backed model comparisons,
generated API checks, Node/Wasm runtime tests and software-WebGPU fixtures cover
different contracts. None should be described as actual browser testing.

Native QSA comparisons retain finite, nonconstant and byte-identical reference
and optimized logits for fused/unfused masks, F16/Q3_K/Q4_K fixtures, and explicit
F16 masks. Indexed GPU fixtures cover mixed and all-skipped slots, guards,
unchanged inputs and WebGPU validation errors. Negative controls corrupt the
skip guard or zero initialization and must be rejected.

A self-generated LFM3 fixture covers full attention, recurrent short convolution
and sliding-window attention. Its native save/reload check and Node/Wasm decode
checks are compatibility evidence, not trained-model accuracy or speed claims.
State replay checks distinguish upstream slot compaction from incorrect state:
SWA search heads are not serialized, so restoration after eviction can change
the physical key order and floating-point reduction order. Exact equality is
checked between continuations restored from the same snapshot, and before SWA
eviction. Rolling-layout versus restored-layout differences remain visible.

Host/publication gates and the configured Wasm profile builds are required
before publishing. Published assets remain append-only and source-bound.
