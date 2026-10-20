# Ordered WebGPU expert output sums

The checked v0.7.0 `llama-graph.cpp` emits slot views followed by a left-associated
ADD chain. The supported F32 chain is executed as one vec4 dispatch. It retains
slot and arithmetic order, not a parallel/tree reduction. No model, selection,
weights, quantization, profile count or frontend protocol changes.

Admission requires the complete 3..32-expert chain, contiguous root and outputs,
width divisible by four, no external intermediate user/output, representable
indices, same backend buffer type/device, disjoint **aligned** binding ranges,
and supported device binding/dispatch limits. Failed admission retains the
upstream ADD path. Source and output can share a buffer only in disjoint ranges.
The actual graph allocator, production factory, shader and encoder must all be
checked; standalone GPU validation is not compiled-runtime/model validation.

Preparation composes after HC and Q8-short, without editing upstream or shadowing
the canonical generated shader header. Contracts include graph construction,
fusion use counts, allocator, backend scheduling, binary ADD, and shader generation.
Unknown inputs/patch/output identities fail closed. All WebGPU profiles use the
same conditional optimization. Publication reports the patch and generated files;
CPU-only profiles do not claim compiled WebGPU copies.

No-patch alternative: retain upstream staged ADDs. This overlay adds maintenance
coupling to a narrow graph pattern and the existing WebGPU encoder. Remove it
when upstream supplies the same optimization or the owner drops the requirement.
Scope: the owner's continuing Qwen acceleration work and accepted expert-sum
candidate; not permission for unrelated graph/model changes.

Validation entry point: `tests/webgpu-moe-sum/run_lab.py`. It uses the actual
prepared factory with only pipeline creation recorded, upstream ADD WGSL, real
ggml allocation/admission, a separate F32 sequential oracle, guard/input checks,
and an intentionally corrupted negative control. Requires the supplied offline
Dawn lab and local upstream; installs nothing. Hardware/model speed is unmeasured.
