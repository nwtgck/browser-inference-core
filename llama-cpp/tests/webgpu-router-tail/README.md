# WebGPU gather normalization tail checks

The ordinary source stage fuses six raw nodes (GET_ROWS, RESHAPE, SUM_ROWS,
CLAMP, DIV, RESHAPE) into one compute dispatch. It keeps the original reduction
tree, clamp and direct division. The one-token variant can merge overlapping
probability/output bindings after all selected loads and a full-lane storage
barrier. Multi-token overlap retains the original staged path.

All storage bindings intentionally use `read_write`, matching ordinary ggml
shaders, although probabilities and IDs are only loaded. WebGPU usage scopes
cover the whole GPUBuffer: mixing read-only storage with writable storage is
invalid even for disjoint ranges in the same allocator buffer. Full aligned
binding disjointness remains required. The GPU fixture exercises both fused
variants with probabilities, IDs and output in one physical buffer as well as
separate ID buffers.

`test_webgpu_router_tail.py` prepares the normal and GPU-profile sources, checks
all stage identities and configure dependencies, executes actual ggml graph/use
count/view/allocator tests with extracted dispatcher code, and compares processed
factory WGSL at workgroup sizes 256/512/1024. Set
`LCB_TEST_GGML_BASE_LIBRARY` to a matching pinned-upstream `libggml-base.a` to run
the native graph/allocator fixture. No core build or GPU execution is implicit.

The opt-in GPU harness uses an existing local Node WebGPU module. It compiles
only a small native factory emitter. Baseline and candidate own independent,
identical union arenas, so the baseline's final write cannot alter candidate
input. It compares exact output bits, classifies special values separately and
checks every arena byte outside the allowed output range, IDs, uniforms and
intermediate guards. All fixtures are synthetic; no model or fixture binaries
are distributed.

```sh
python3 tests/webgpu-router-tail/check_gpu.py \
  --llama-source vendor/llama.cpp --work /tmp/router-tail-check \
  --webgpu-module /path/to/local/webgpu/index.js \
  --node-command-json '["node"]' --backend vulkan
```

Use `--prepare-only` to generate source-bound shaders/manifest without a GPU.
`--profile` uses the actual GPU-profile preparation including its optional Q8
diagnostic stage; the router and baseline shader bytes must remain identical.
The harness records actual adapter properties and device limits and does not
override subgroup information. Software GPU correctness or timing does not
establish target-device performance or full-model quality. Actual Wasm graph
eligibility and model regression checks remain separate validation.
