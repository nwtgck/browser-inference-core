# Retired WebGPU MoE direct selected-slot overlay

As of llama.cpp v0.7.0 (`620d2225cc436ab0b001e3200e590e3b286ef9ab`), the
upstream `mul_mat_id_vec.wgsl` implements the selected-slot behavior previously
carried by `ggml-webgpu-moe-direct-slot.patch`. The downstream patch is removed,
not rewritten to restore the old expert scan or applied a second time.

## Reviewed correspondence

The upstream shader linearizes the two-dimensional workgroup grid, checks the
rectangular tail before reading IDs or entering reduction barriers, obtains the
expert from `ids[offset_ids + slot]`, reads src1 through `slot % b_ne1`, and writes
output in selected-slot order. Partial output rows retain both upstream reduction
paths' bounds checks. The host dispatch still allocates
`ceil(output rows / outputs per workgroup) * selected experts` workgroups and uses
the matching unsigned 32-bit uniforms. Its unrelated multi-token MoE tile-gather
changes remain upstream; this retirement does not revert or optimize them.

The CMake option, preparation script, patch, include-precedence hook and old
patch-specific tests have been removed together. All WebGPU profiles now use the
upstream-generated `ggml-wgsl-shaders.hpp`. The independently prepared shader-library
header for single-token SSM convolution remains; it does not contain embedded WGSL.
Reconfiguration tests poison a stale `moe-direct-slot-overlay` header to ensure it
is not selected. Build commands no longer emit `LCB_WEBGPU_MOE_DIRECT_SLOT`.

`upstream_provenance.py` records retirement separately from active `sourceOverlays`
and rejects build records containing the retired option. No old artifact manifest
or published identity is rewritten. This is a source-bound report change, not a
new runtime-manifest schema or a claim to have rebuilt all profiles.

## Regression evidence and limits

`tests/test_upstream_moe_direct_slot.py` checks the actual pinned shader/dispatch
expressions and independently enumerates address mappings for selected experts,
broadcasts, nonzero offsets, partial rows and rectangular workgroup tails. These
are source and host-arithmetic tests, **not GPU execution or performance results**.
`tests/test_webgpu_source_overlay.py` retains upstream target/header-generation
checks, including the stale-header negative control.

Historical downstream experiments were performed before this retirement; they do
not certify v0.7.0. Real browser, hardware GPU and trained-model validation still
belongs to the ordinary build and consumer-validation pipeline. See
[the v0.7.0 review](llama-v0.7.0-review.md).
