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

All WebGPU profiles use the upstream-generated `ggml-wgsl-shaders.hpp` for this
selected-slot path. The retired patch and its activation/include hook are absent
from the production recipe. Other independently prepared shader-library and
specialized shader inputs remain selected by their own contracts; they must not
restore the obsolete expert scan or shadow this upstream implementation.

Existing published artifacts remain immutable. The absence of an active patch
in a new preparation receipt does not rewrite older manifests.

## Validation boundary

The complete-source native provider records the current factory/dispatch choices
and emitted shaders, with actual ggml types and SDK declarations. It does not
execute the GPU shader. Source address reasoning, native recorded calls,
software-GPU numerical comparisons and trained-model hardware measurements
establish different properties. Use the current entries in
[verification](../../../docs/verification.md); old downstream experiments do not
certify a later upstream artifact. See [v0.7.0 compatibility](llama-v0.7.0-review.md).
