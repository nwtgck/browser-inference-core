# RMS normalization and scale fusion checks

Run the host checks from `llama-cpp/`:

```sh
python3 -m unittest discover -s tests -p test_webgpu_rms_scale.py
```

The suite prepares the existing overlays and then the RMS/scale candidate. It
uses actual ggml graphs and gallocr, extracts the prepared production dispatch
helpers and pipeline factory, and checks the complete translation unit against
pinned Dawn headers. A temporary native build compiles only `ggml-base`.
`LCB_TEST_LLAMA_SOURCE`, `LCB_TEST_DAWN_PACKAGE`, and
`LCB_TEST_GGML_BASE_LIBRARY` can select existing local prerequisites.

The candidate accepts F32 RMS_NORM/SCALE chains with identical positive 4D
shapes, dense non-view intermediate/output tensors, and contiguous source rows.
Ordered source strides can preserve padding between heads, tokens and sequences.
Runtime checks require a matching WebGPU device and buffer implementation,
valid byte ranges, supported limits, and either identical physical element
mapping or disjoint logical and aligned binding ranges. The existing three-op
SCALE/UNARY/SCALE planner takes precedence so this fusion cannot consume its
first scale and increase the dispatch count. That preference is conservative
when the later fusion's runtime bindings would fall back.

The preparation module pins the upstream arithmetic, ggml graph and allocation
contracts, existing scale/unary planner, patch inputs/outputs, and embedding
tool. It copies `row_norm.wgsl` only into its isolated patch workspace, preserves
the `v * v` reduction and normalization, and embeds the affine epilogue under a
new symbol. Canonical upstream shader embedding is unchanged. Tests compare the
reduction source exactly and exercise both cache orders and in-place variants.

An optional standalone Node WebGPU check uses an already installed module:

```sh
python3 tests/webgpu-rms-scale/check_gpu.py \
  --source vendor/llama.cpp \
  --work /tmp/rms-scale-gpu-check \
  --webgpu-module /absolute/path/to/webgpu/index.js \
  --reference-scale-wg-size 1024 \
  --benchmarks
```

The reference scale workgroup size must equal the adapter's
`maxComputeInvocationsPerWorkgroup`, as in the production SCALE factory. The
runner requests that limit explicitly and records it. Both references use the
untouched upstream shaders. RMS runs at WG128; SCALE runs in place on the RMS
output allocation. The candidate runs the actual prepared factory's WG128
shader. There are 114 finite cases with a Float64 CPU reference, staged/fused
bit comparisons, source preservation, unwritten-output and buffer-guard checks.
Fourteen additional overflow/nonfinite/zero-epsilon probes require observed
staged/fused classification parity; their CPU classifications are recorded
without promising portable nonfinite behavior. Cases include row tails, 4D
padding, nonzero binding prefixes, same-buffer disjoint ranges, exact in-place
mapping, and GDN-shaped source views.

`--benchmarks` adds paired AB/BA timestamp measurements for width 128, 16 heads,
and 1, 17, or 128 tokens. Each sample contains 32 chains and numerical/range
checks run after timing. Reports include individual pairs and shader/source
identities. These standalone checks do not prove Wasm graph integration,
end-to-end model performance, or behavior on another GPU/compiler.
