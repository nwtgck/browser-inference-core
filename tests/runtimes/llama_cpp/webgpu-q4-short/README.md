# Optional Q4 Dawn comparison and CPU reference

The Dawn experiment consumes supplied ordinary/indexed Q4_K WGSL, case choices,
and CPU reference files. It compares output values and buffer guards and can
record GPU timestamps. The caller supplies input provenance; the script does not
establish production-factory selection or model throughput.

The shader directory must contain `ordinary-{32,64,128,256}.wgsl`,
`indexed-{32,64,128,256}.wgsl`, and `cases.json`. Each case supplies `kind`
(ordinary/indexed), positive K divisible by 256, positive M, `mode` 0/1/2,
`sharedInput` and `broadcast` booleans, and a `workgroups` map from reported
subgroup widths to a selected 32/64/128/256 pipeline. Shaders must match the
uniform and binding layouts implemented in `fixtures.mjs` and `check_gpu.mjs`.

`fixtures.mjs SHADERS OUTPUT_DIRECTORY` writes indexed binary fixture inputs.
`reference-main.cpp` consumes one fixture on stdin and writes Float64 pairs
`[dot, sumAbs]` for every output. It calls the complete current CPU quantization
unit; the test-support `native_quantization.build_reference` compiles that unit
and this ordinary C++ harness. The independent known-block regression is
`source/test_q4_reference.py`.

Supply CPU files named `0.f64`, `1.f64`, and so on in the same case order:

```sh
LCB_TEST_WEBGPU_MODULE=/absolute/path/to/webgpu/index.js \
LCB_Q4_CPU_REFERENCE_DIR=/absolute/path/to/cpu-reference \
LCB_Q4_BENCHMARKS=1 \
node tests/runtimes/llama_cpp/webgpu-q4-short/check_gpu.mjs \
  /absolute/path/to/shaders /absolute/path/to/report.json
```

The local Dawn adapter needs subgroups and shader-f16; optional timing also needs
timestamp-query. Omit `LCB_Q4_BENCHMARKS=1` for numerical checks only. Timing cases
must be present in the supplied case list. Results concern the supplied kernels
and adapter, rather than a compiled inference runtime or another device. The
runner installs no dependencies and downloads no model data.
