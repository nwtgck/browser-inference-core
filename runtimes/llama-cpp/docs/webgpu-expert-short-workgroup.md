# Short single-token expert matvec workgroups

This isolated overlay uses upstream `WG_SIZE` rather than replacing quantization
or WGSL. For Q5_1 and IQ4_NL, the reviewed upstream decoder assigns four lanes to
each 32-weight block. With K <= 1024, at most 128 lanes have input work; shrinking
256 lanes to 128 preserves the block and input multiplication order of each
useful lane. The upstream subgroup reduction itself is unchanged. Floating-point
reduction ordering is implementation-dependent, so software-adapter bit parity
is not a universal device or model-quality guarantee.

Admission requires single-token Q5_1 or IQ4_NL weights with F32 input/output,
I32 selection IDs, K divisible by 32 in [32, 1024], at least 256 output rows,
consistent selected-slot dimensions, one or one-per-slot input vector, and no
fourth-dimension batching. Actual device subgroup limits must be valid powers
of two in [4, 128], and at least 128 workgroup invocations must be supported.
Everything else uses the existing 256-lane path. Prefill and other quantizations
are not changed. Shape/limit rejection is an optimization fallback, not a new
validation API for malformed models.

The vector dispatcher forwards subgroup limits to the actual shader factory.
The selected size is part of the pipeline key and the shader decisions, and is
visible as `_wg128` in existing GPU profile labels. No new profile, binary,
copy, GPU wait, timer or model-name switch is added. The shared matrix key keeps
its zero-initialized size; the multi-token factory is otherwise unchanged.

## Checked build and publication

The typed runtime recipe composes the expert-workgroup patch with the other
selected WebGPU changes in the final private source tree. The unchanged upstream
emitter uses the selected quantization templates and shader inputs. The upstream
checkout remains unchanged. Preparation records the actual patch, input and final
output identities; actual compiler inputs establish which files were selected.

Remove this overlay, preparation hook, patch-policy entry, provenance entry and
associated tests when upstream provides equivalent guarded selection. Review
lane mapping and reductions when the upstream changes; a recorded digest is not
a numerical check.

## Validation and limits

The normal build compiles the actual factory against the selected upstream and
SDK declarations. Current complete-source native contracts observe selected
factory/cache/dispatch decisions through recorded API calls; see
[verification](../../../docs/verification.md) for their scope. Compilation and
recorded pipeline creation do not execute shader arithmetic.

The supplied-shader runner below checks numerical bits, unchanged inputs and
buffer guards independently of production factory selection. Adapter identity,
case results and optional timestamp pairs remain separate from browser/model
inference and quality measurements.

## Optional supplied-shader comparison

The standalone runner reads supplied WGSL from a directory containing
`Q5_1-128.wgsl`, `Q5_1-256.wgsl`, `IQ4_NL-128.wgsl`, `IQ4_NL-256.wgsl` and
`cases.json`. Each shader must expose `main` with the harness's bindings and
uniform layout. The case file is a nonempty array of objects with `type`
(`Q5_1` or `IQ4_NL`), positive 32-aligned `k`, positive `m`, `chosen` (128 or 256),
`storedExperts` (11 or 512), `mode` (0, 1 or 2) and boolean `sharedInput`.
Benchmark mode requires both `(k, m) = (640, 2560)` and `(1056, 256)` for each type.
The supplied `chosen` value is an input, not proof of the current production
selector. Keep input identities and any factory comparison as separate evidence.

From the repository root, with local Dawn and compatible inputs:

```sh
npm run tooling:build
LCB_TEST_WEBGPU_MODULE=/absolute/path/to/webgpu/index.js \
LCB_EXPERT_BENCHMARKS=1 \
node tests/runtimes/llama_cpp/webgpu-expert-short/check_gpu.mjs \
  /absolute/path/to/shaders /absolute/path/to/expert-report.json
```

Omit `LCB_EXPERT_BENCHMARKS` for numerical-only checks. Benchmark mode also needs
`timestamp-query`; all runs need subgroups and shader-f16. Set
`LCB_EXPERT_FORCE_GC=1` and use `node --expose-gc` to stress native ownership.
The report records adapter, cases, failures and optional timing pairs; it does not
establish the supplied shaders' source or agreement with a compiled artifact.

The published model still needs measurements in the ordinary profile. Profiling
partitioning and readback waits are not an ordinary wall-time breakdown, and
software-adapter timing is not target-device throughput.
