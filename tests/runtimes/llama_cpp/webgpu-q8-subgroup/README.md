# Optional Q8 subgroup Dawn experiment

Build the repository Node tooling first with `npm run tooling:build` from the
repository root; the script uses its emitted Node helpers.

This experiment consumes supplied WGSL and a manifest. It compares every output
with an independent Float64 Q8_0 reference and the chosen baseline, checks buffer
guards and input mutation, and optionally records GPU timestamp pairs. Input
variants and their provenance are supplied by the caller, not inferred from a
production build.

```sh
LCB_TEST_WEBGPU_MODULE=/absolute/path/to/webgpu/index.js \
node tests/runtimes/llama_cpp/webgpu-q8-subgroup/check_gpu.mjs \
  /absolute/path/to/shaders /absolute/path/to/report.json /absolute/path/to/manifest.json
```

The manifest requires a nonempty `variants` array. Each variant has a unique
`name`, a WGSL `file`, positive `wgSize`, and positive `outputsPerWg`. `baseline`
selects a variant name. Optional `cases` provide positive K divisible by 32,
positive M, `mode` 0/1/2, and `batched`; an external `weightsPath` is relative to
the manifest directory. The built-in cases use deterministic synthetic values.
Shaders must match the Q8 buffer and parameter layouts in `check_gpu.mjs`.

Set `benchmarks` to an object to enable timing; it accepts `warmupPairs`,
`measuredPairs`, `dispatchesPerPass`, and `caseNames`. The device needs the
variant's `requiredFeatures` (subgroups and shader-f16 by default) and
`timestamp-query` for timing. A variant can require fixed advertised subgroup
limits through `experiment.fixedSubgroupSizeRequired`; the runner then also
records actual subgroup geometry. `LCB_Q8_FORCE_GC=1` requires Node's
`--expose-gc` option.

Reports retain numerical error bounds, adapter details, input hashes, and timing
pairs. The arithmetic bound is a sanity check, not proof of model quality or
performance on another device. No runtime build or dependency installation is
performed by this entry point.
