# Optional Q8 Dawn comparison

Build the repository Node tooling first with `npm run tooling:build` from the
repository root; the script uses its emitted Node helpers.

This developer experiment consumes supplied WGSL and compares a 256-lane baseline
with the short-workgroup choices used by its synthetic cases. The caller supplies
the shader files and their provenance; the report covers the supplied kernels.

Supply a directory containing `q8-32.wgsl`, `q8-64.wgsl`, `q8-128.wgsl`, and
`q8-256.wgsl`, with the entry point and buffer/parameter layout used by
`check_gpu.mjs`. The current native provider fixture emits these filenames when
run with a persistent shader output directory; shader provenance remains the
caller's responsibility. Incompatible inputs fail shader compilation or the
numerical/boundary checks.

With Node and a compatible Dawn Node module already installed locally:

```sh
LCB_TEST_WEBGPU_MODULE=/absolute/path/to/webgpu/index.js \
LCB_Q8_BENCHMARKS=1 \
node tests/runtimes/llama_cpp/webgpu-q8-short/check_gpu.mjs \
  /absolute/path/to/shaders /absolute/path/to/report.json
```

The adapter needs `subgroups` and `shader-f16`; optional timing also needs
`timestamp-query`. Omit `LCB_Q8_BENCHMARKS=1` for numerical checks only. To stress
Dawn owner lifetime, set `LCB_Q8_FORCE_GC=1` and add `--expose-gc` to Node.

Reports include adapter information, numerical/input/guard checks, and optional
GPU timestamp pairs with four warmups, twelve measured pairs, and alternating
AB/BA order over the same buffers. They measure the supplied kernels on that
adapter, not model throughput, browser inference, or target-device performance
on another machine. No dependencies or model data are downloaded.
