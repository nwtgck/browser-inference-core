# llama.cpp v0.8.0 integration review

The llama runtime pin advances from `338fce6511c744893cc0d3d958b275f40552c7a4`
to `8642541b255bfb5dc39cd351a178be09013a55d6`. Shared compiler/Dawn pins and
the independent stable-diffusion runtime pins remain unchanged.

## Retained overlays

The upstream WebGPU source, shader templates, allocator, backend interfaces and
`llama.h` are byte-identical. Every retained patch applies without modification.
The complete normal 22-file and profile 24-file WebGPU compositions retain
their previous bytes, including all historical stage and final output pins.

Reviewed semantic-input hashes change for the LFM decision/context updates and
the model-header declarations. They preserve the existing sampler, optional
snapshot, HC, router, MoE sum, QSA and file-backed-row behavior. The graph delta
changes non-causal attention masks, while the separate context change controls
selected outputs for decoders without memory. Neither changes the checked
Qwen graph builders, indexed-matmul ABI, shader math or queue/copy lifecycle.
File-backed models retain coherent ownership of the complete copied source
tree. Vision/audio overlays preserve the upstream LFM3 video additions.

## Generated API and chat surface

The 978 existing generated C functions retain their names and signatures.
Upstream adds `mtmd_bitmap_set_is_video`, the by-value `mtmd_get_video_config`,
its five-field record, three video-template constants, and a final
`no_soundtrack` field in `mtmd_helper_video_init_params`. Existing record and
field IDs stay unchanged. The generated layout queries and fingerprint bind
these changed layouts to the matching artifact; wire ABI version remains 1.
Return-record storage remains caller-owned. Subprocess video helpers remain
excluded, and the existing deprecated compatibility exports remain present.

Upstream changes `common_chat_templates_get_caps` from a boolean map to
`common_json`, adding a `reasoning_effort_levels` array. The bridge selects that
signature explicitly and returns the owned JSON object without dropping new
information. The existing `bool_map` registration stays available. See
[chat and multimodal bindings](chat-and-multimodal.md) for ownership and usage.

## Validation boundaries

Checked preparation, generated-schema comparison, source/native tests and
Node/Wasm surface checks cover distinct contracts. Hosted CI uses its pinned
Node version; a local run is not a claim about a remote CI result. Real browser,
GPU/model inference, new-model quality and target-device performance require
separate validation. Publication remains append-only and source-bound, with
the existing workflow and package contract unchanged.
