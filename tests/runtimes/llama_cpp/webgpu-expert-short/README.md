# Optional expert-matvec Dawn comparison

This developer experiment consumes supplied Q5_1/IQ4_NL WGSL and experiment cases.
It compares a 256-lane baseline with each case's selected pipeline. The caller
supplies the shader files, case choices, and their provenance; the report covers
those experiment inputs.

The current `webgpu-provider` native fixture emits the four shaders and
`cases.json` into its configured shader output directory. Its small cases record
the current factory's 128/256 choices and warm cache reuse. The Dawn script still
validates whichever directory the caller supplies.

Supply these files in one input directory:

- `Q5_1-128.wgsl` and `Q5_1-256.wgsl`.
- `IQ4_NL-128.wgsl` and `IQ4_NL-256.wgsl`.
- `cases.json`, a nonempty array of objects with `type`, `k`, `m`, `chosen`,
  `storedExperts`, `mode`, and `sharedInput`.

The shaders must use the entry point and indexed-matvec parameter/buffer layout
in `check_gpu.mjs`. Cases accept `type` Q5_1/IQ4_NL, positive K divisible by 32,
positive M, `chosen` 128/256, `storedExperts` 11/512, `mode` 0/1/2, and a boolean
`sharedInput`. These are experiment inputs, not an inferred production policy.
For example, a timing input can contain:

```json
[
  {"type":"Q5_1","k":640,"m":2560,"chosen":128,"storedExperts":11,"mode":2,"sharedInput":false},
  {"type":"Q5_1","k":1056,"m":256,"chosen":256,"storedExperts":11,"mode":2,"sharedInput":false},
  {"type":"IQ4_NL","k":640,"m":2560,"chosen":128,"storedExperts":11,"mode":2,"sharedInput":false},
  {"type":"IQ4_NL","k":1056,"m":256,"chosen":256,"storedExperts":11,"mode":2,"sharedInput":false}
]
```

With Node and a compatible Dawn Node module already installed locally:

```sh
LCB_TEST_WEBGPU_MODULE=/absolute/path/to/webgpu/index.js \
LCB_EXPERT_BENCHMARKS=1 \
node tests/runtimes/llama_cpp/webgpu-expert-short/check_gpu.mjs \
  /absolute/path/to/shaders /absolute/path/to/report.json
```

The adapter needs `subgroups` and `shader-f16`; timing also needs
`timestamp-query`. Omit `LCB_EXPERT_BENCHMARKS=1` for numerical checks only. Timing
requires both `(k=640,m=2560)` and `(k=1056,m=256)` for each type in `cases.json`.
Set `LCB_EXPERT_FORCE_GC=1` and add `--expose-gc` to Node for owner-lifetime stress.

Synthetic cases compare output bits, untouched inputs, and output guards.
Optional timing reports every GPU timestamp pair after four warmups and twelve
measured pairs in alternating AB/BA order with the same buffers. Results concern
the supplied kernels and adapter; they are not model throughput or a prediction
for another device. The native Dawn owner stays alive through teardown and report
writing. No dependency installation or source preparation runs implicitly.
