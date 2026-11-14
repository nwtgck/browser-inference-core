# Optional expert-sum Dawn comparison

This developer experiment consumes supplied WGSL. It compares expert-sum variants
with a supplied binary ADD chain, using synthetic buffers and guard regions.
The caller supplies those inputs and their provenance. Results cover the
standalone kernels and buffers, rather than a compiled inference runtime.

Supply these files in one directory, compatible with `check_gpu.mjs`:

- `SRC_OVERLAP.wgsl` and `INPLACE.wgsl`: F32 ADD, workgroup size 256, entry `main`.
- `moe-3.wgsl`, `moe-4.wgsl`, `moe-8.wgsl`, `moe-10.wgsl`, `moe-16.wgsl`, and
  `moe-32.wgsl`: the corresponding expert-sum variants.

`emit_reference.cpp` can preprocess a supplied upstream `binary.wgsl` into the two
ADD inputs using the matching upstream `pre_wgsl.hpp`. It does not produce the
expert-sum inputs or verify their relation to a production build.

With Node and a compatible Dawn Node module already installed locally:

```sh
LCB_TEST_WEBGPU_MODULE=/absolute/path/to/webgpu/index.js \
MOE_SUM_BENCHMARKS=1 \
node --expose-gc tests/runtimes/llama_cpp/webgpu-moe-sum/check_gpu.mjs \
  /absolute/path/to/shaders /absolute/path/to/report.json
```

The runner requests a Vulkan adapter with `shader-f16`. Optional timing also
requires `timestamp-query`; omit `MOE_SUM_BENCHMARKS=1` for numerical checks only.
`--expose-gc` is required for its owner-lifetime stress. The report records the
adapter, numerical/guard checks, and optional GPU timestamp pairs. These results
measure supplied kernels on that adapter, not model throughput or performance on
another device. No dependencies, weights, or source files are downloaded.
