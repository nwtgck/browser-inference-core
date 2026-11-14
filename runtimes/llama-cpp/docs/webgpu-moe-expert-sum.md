# Ordered WebGPU expert output sums

The selected upstream `llama-graph.cpp` emits slot views followed by a left-associated
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
Required patch conflicts fail preparation; receipts record actual input, patch and output identities. All WebGPU profiles use the
same conditional optimization. Publication reports the patch and generated files;
CPU-only profiles do not claim compiled WebGPU copies.

No-patch alternative: retain upstream staged ADDs. This overlay adds maintenance
coupling to a narrow graph pattern and the existing WebGPU encoder. Remove it
when upstream supplies the same optimization or the owner drops the requirement.
The scope is this ordered chain, without unrelated graph or model changes.

## Optional supplied-shader comparison

The standalone MJS runner consumes `SRC_OVERLAP.wgsl`, `INPLACE.wgsl` and
`moe-3.wgsl`, `moe-4.wgsl`, `moe-8.wgsl`, `moe-10.wgsl`, `moe-16.wgsl`,
`moe-32.wgsl`. Inputs must expose `main` and the storage/uniform layouts expected
by the harness. It compares the supplied fused shaders with a supplied binary
ADD chain and an independent sequential F32 arithmetic reference. It does not
prove the shaders came from the current production factory; retain and review
the input identities separately.

With local Vulkan-capable Dawn, shader-f16 and compatible inputs, run from the
repository root:

```sh
LCB_TEST_WEBGPU_MODULE=/absolute/path/to/webgpu/index.js \
MOE_SUM_BENCHMARKS=1 \
node --expose-gc tests/runtimes/llama_cpp/webgpu-moe-sum/check_gpu.mjs \
  /absolute/path/to/shaders /absolute/path/to/moe-sum-report.json
```

Garbage-collection lifetime stress is required. Omit `MOE_SUM_BENCHMARKS` for
numerical-only checks; timestamp mode requires `timestamp-query`. Numerical
cases cover widths, expert counts, token counts, pooled buffers, input/guard
preservation and exact results. `MOE_SUM_NEGATIVE=1` corrupts a fused result and
must fail. Timing uses the same buffers and one compute pass for each alternative,
with alternating order, four warmup pairs and twelve measured pairs. The report
records adapter, cases, errors and timing pairs without certifying input
provenance, compiled-runtime inference, model quality or chat throughput.
