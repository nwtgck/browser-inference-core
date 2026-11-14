# llama.cpp v0.8.0 compatibility notes

These notes describe the upstream transition from
`338fce6511c744893cc0d3d958b275f40552c7a4` to v0.8.0
`8642541b255bfb5dc39cd351a178be09013a55d6`. They explain API and source decisions
needed when reviewing later pins; they do not select the current pin or certify
an artifact built from it. The source gitlink is authoritative.

## Graph and overlay contracts

This transition preserves the WebGPU source, shader templates, allocator,
backend interfaces and `llama.h` used by the retained overlays. Source preparation
still applies individual reviewable patches to private copies. A matching old
output digest is an identity comparison rather than a requirement to freeze all
future upstream source bytes.

The graph changes non-causal attention masks; a separate context change controls
selected outputs for decoders without memory. The Qwen graph, indexed-matmul ABI,
shader arithmetic and queue/copy lifecycle still require their own checks.
File-backed rows require coherent model declarations across the complete private
source tree. Vision/audio preparation must preserve upstream LFM3 video support.

## C API and record layouts

Upstream adds `mtmd_bitmap_set_is_video`, the by-value `mtmd_get_video_config`,
its five-field record, three video-template constants, and a `no_soundtrack`
field in `mtmd_helper_video_init_params`. Return-record storage remains
caller-owned. Subprocess video helpers remain excluded; the explicitly retained
deprecated compatibility functions remain direct upstream exports.

Regenerate the API from the selected public headers. The current binding contract
is ABI 3 with a target schema catalog and module-reported fingerprint. Do not
reuse a historical record/field ID, offset or wire convention merely because an
upstream function name survived this transition. See
[the current API contract](core-contract.md).

## Chat capability metadata

Upstream changes `common_chat_templates_get_caps` from a boolean map to
`common_json`, adding a `reasoning_effort_levels` array. The bridge selects that
signature explicitly and returns the owned JSON value without dropping new
information. The `bool_map` registration remains a separate type. See
[chat and multimodal bindings](chat-and-multimodal.md) for ownership and usage.

## Verification boundary

Preparation, native contracts, generated declarations, actual Wasm adapters and
browser/model execution establish different properties. Later upstream support
or a successful compilation does not certify new-model quality or target-device
performance. The [verification guide](../../../docs/verification.md) identifies
the current entry points. Published artifacts remain immutable and source-bound.
