# Anima image performance changes

This change uses the pinned upstream patch inventory; it does not update vendor
submodule commits or the llama-cpp runtime.

* `0020` selects a single-image causal convolution only with no frame history,
  compatible temporal shape/stride/dilation, supported operations, and no temporal
  wrap induced by both circular-padding flags. It packs the selected weight slice
  and retains the existing fallback.
* `0021` separates adapter preparation from cross-attention padding. Structural
  zero positions retain their normalization mass through explicit softmax/flash
  sinks. Arbitrary weight adapters keep physical padding.
* Sampling-local condition identifiers enable RunnerCache only for immutable
  conditions. Enabled generation extensions disable those identifiers for the
  remainder of the sample. Cache writes commit only after successful computation;
  allocation failure disables caching for that sample and retries ordinary
  computation once. Other errors are not retried as allocation failures.

The existing `model_args` accepts `anima_context_cache=false` and
`anima_compact_context=false` independently. No public C ABI field is added.
Normal fixed-settings retention is at most three 1024-by-512 F32 conditions
(6 MiB of raw values). Direct callers that change attention settings within one
sampling lifetime can retain separate variants, conservatively up to 24 MiB.
Graph-local copies and allocator reservation are additional.

Native tests use deterministic synthetic weights, a real small AnimaRunner,
ModelManager and RunnerCache. They do not establish trained-model output quality,
GPU timing, WebGPU device-loss behavior, or WASM build success. The accompanying
handoff records the actual pass/fail results.

Unconditional CONT removal, per-layer Anima KV caching and preview callback
changes remain outside this implementation unit. The subsequent WebGPU transport
and bounded Qwen convolution/attention changes are described in
`webgpu-image-performance.md`.

Cache hits copy the small saved condition into a graph-local contiguous output.
This deliberately preserves a memory boundary: graph-cut naming and scheduler
bookkeeping must not modify a persistent tensor's metadata. The native runner
regression checks that the saved tensor keeps its cache name across repeated calls.
