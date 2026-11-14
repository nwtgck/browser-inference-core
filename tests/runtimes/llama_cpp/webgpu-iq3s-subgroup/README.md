# Optional IQ3_S numerical experiment and CPU fixtures

The independent fixture generator compiles the complete current CPU quantization
unit and calls its real dequantizer. It produces deterministic weights, inputs,
IDs, uniforms, and Float64 `[dot, sumAbs]` references, including skipped expert
slots and stored-payload boundary cases:

```sh
python3 tests/runtimes/llama_cpp/webgpu-iq3s-subgroup/make_fixtures.py \
  --llama-source /absolute/path/to/llama.cpp --work /absolute/path/outside/source-trees/iq3-fixtures
```

`source/test_native_quantization.py` and `source/test_iq3_fixtures.py` validate the
complete CPU unit, known values, fixture references, and invalid payload ranges.
These are CPU checks, not WGSL execution.

The optional Dawn experiment consumes supplied compatible WGSL and case choices:

```sh
LCB_TEST_WEBGPU_MODULE=/absolute/path/to/webgpu/index.js \
node tests/runtimes/llama_cpp/webgpu-iq3s-subgroup/check_gpu.mjs \
  /absolute/path/to/shaders /absolute/path/to/iq3-fixtures/fixtures \
  /absolute/path/to/report.json 16
```

The final argument is a required fixed reported subgroup width: 16, 32, or 64.
The adapter needs subgroups and shader-f16. The shader directory contains
`iq3s-baseline.wgsl`, the selected WGSL files, and `case-decisions.json`. Each
decision supplies a fixture `name`, `selectedFile`, `wgSize`, and
`outputsPerWg`; names must cover the fixture manifest. The shaders must match the
indexed-matvec buffer and uniform layout used by the fixtures and runner.

A `source.json` beside the shader directory supplies `fixtureManifestSha256` and
`shaderSha256` (filename to digest) for input integrity. These observed input
hashes are not fixed upstream admission values. The caller supplies provenance
and selections; the report's case decisions record those inputs and do not
establish that a production factory chose them.

Every output is compared with the supplied CPU references and baseline, including
skipped-slot zeros and untouched guards. Reports include adapter and actual
subgroup geometry. This entry point records no inference performance and makes
no model-quality claim. It installs no dependencies or model data.
