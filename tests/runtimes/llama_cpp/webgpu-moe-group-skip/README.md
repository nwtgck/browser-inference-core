# MoE gather and matrix WGSL comparison

Build the repository Node tooling first with `npm run tooling:build` from the
repository root; the script uses its emitted Node helpers.

This optional Dawn experiment executes supplied gather, baseline and candidate
WGSL over caller-supplied synthetic fixtures. It checks routing, skipped-slot
zeros, decoded-weight Float64 dot references, exact baseline/candidate output
bytes, guards and input/metadata preservation. It does not prove production
factory provenance or model throughput.

```sh
LCB_TEST_WEBGPU_MODULE=file:///absolute/path/to/webgpu/index.js \
node tests/runtimes/llama_cpp/webgpu-moe-group-skip/check_gpu.mjs \
  /absolute/path/to/inputs --output /tmp/moe-group-report.json
```

The input directory contains `source.json` with `fixtureManifestSha256` and
`manifest.json`, an array of case objects. Each case supplies `name`, `type`,
`k`, `m`, `experts`, `used`, `tokens`, `args` (the address/stride layout),
`gatherParams`, `mainParams`, `grid`, `maxTiles`, `hashes` for `weights`, `input`, `ids`, `dequantized`, and
`shaderHashes` for `gather-baseline`, `gather`, `baseline`, `candidate`. Supply the matching
`shaders/<name>-{gather-baseline,gather,baseline,candidate}.wgsl` and
`fixtures/<name>/{weights,input,ids,dequantized}.bin`. Decoded weights are F32;
the caller is responsible for their independent correctness. Optional case
flags and shapes must match the layouts consumed by the script.

`--large` selects `source-large.json`/`manifest-large.json`. `--filter`,
`--skip` and `--limit` select cases. Set `LCB_TEST_WEBGPU_BACKEND` to choose a
Dawn backend. `--benchmark` adds paired kernel measurements with bounded
`--dispatches` and `--rounds`. GPU timing/speedup is reported only when
`timestamp-query` is available; CPU encoding and submission wait intervals
remain separate measurements.

Input hashes establish transfer integrity only. The caller supplies complete
inputs and an installed Dawn module. There is no repository shader/fixture
producer for this experiment; the former body-extraction generator is retired.
