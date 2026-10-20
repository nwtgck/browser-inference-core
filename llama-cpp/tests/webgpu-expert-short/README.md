# Opt-in production expert-matvec comparison

This is a standalone numerical and microbenchmark harness, not a model or browser
benchmark. Nothing is downloaded. The normal inference runtime never invokes it.

## Required local inputs

Use the pinned llama.cpp checkout, a C++ compiler (`g++`), Node.js, and a compatible
Dawn `webgpu` Node package. The adapter needs subgroups and shader-f16. Timing also
needs timestamp-query. Keep reports from software adapters distinct from real
hardware performance.

From the repository root, with tools already present locally:

```sh
python3 llama-cpp/tests/webgpu-expert-short/check_gpu.py \
  --llama-source /absolute/path/to/llama.cpp \
  --work /absolute/path/outside/source-trees/expert-check \
  --webgpu-module /absolute/path/to/node_modules/webgpu/index.js \
  --force-gc --benchmarks
```

`--node-command-json` accepts a JSON array, for example `["node"]` or a loader,
library path and Node binary for an offline runtime. `--force-gc` adds
`--expose-gc` and fails if garbage collection is unavailable. The Dawn owner is
kept strongly reachable through device teardown and report writing.

## What is compared

The C++ harness extracts the actual prepared factory, uses upstream's WGSL
preprocessor and emitter, and emits both the selected shader and the original
upstream-factory shader. Case admission is recorded by the real selector, rather
than reimplemented in JavaScript. The actual C++ dispatcher is also executed with
a recording device to check parameters and workgroup dimensions; that recording
check is not GPU execution.

The GPU cases cover Q5_1 and IQ4_NL, supported and rejected widths, output tails,
strided rows and experts, nonzero offsets, shared/per-slot inputs, repeated expert
IDs and two-dimensional dispatch tails. They compare output bits, untouched
inputs and output guards. The shader declares 512 experts. Most fixtures allocate
only the low expert IDs actually selected; additional fixtures allocate all 512
and select the last expert. None is a full-model memory-residency experiment.

Optional timing uses the same input buffers for both paths, four warmup pairs,
twelve measured pairs and alternating AB/BA order. An unsupported-width case uses
the same 256-lane pipeline twice as a noise control. Report every trial, not just
the best result. A software-adapter speed ratio is not a predicted Mac speedup.

`source.json` records source and generated shader hashes; `report.json` records
adapter, cases, numerical failures and optional per-pair timings. The process
returns nonzero for any numerical/guard/input mismatch or validation failure.
The original quantization and reduction WGSL is unchanged, but arbitrary GPU
compilers can still choose different subgroup/reduction implementations. Model
and target-device correctness/quality tests remain necessary.
