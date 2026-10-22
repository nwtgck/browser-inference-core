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

`prepare_webgpu_expert_short.py` composes after the existing HC, short-Q8 and
ordered-expert-sum overlays. It checks the exact input and output C++ identities
and the upstream quantization templates, selected-slot shader, embedded-shader
emitter and preprocessor. The upstream checkout stays unchanged. The final
source/header are recorded in source-overlay provenance only when WebGPU is
compiled. All contract files and the patch are CMake reconfiguration inputs.

Remove this overlay, preparation hook, patch-policy entry, provenance entry and
associated tests when upstream provides equivalent guarded selection. Do not
refresh hashes without reviewing the lane mapping and reductions.

## Validation and limits

Host tests execute the actual selector, cache and factory with recorded pipeline
creation. Their baseline is the unchanged upstream factory, not a hand-written
shader. They cover unsupported formats, shapes, limits, cache collisions,
full-subgroup ranges and per-lane input-block correspondence. The GPU checker
runs the emitted shader pair and checks bits, untouched inputs and buffer guards.
Its reports identify the adapter and all tests; it is opt-in, needs local tools,
and makes no network requests.

Run the host checks from the repository root:

```sh
python3 -m unittest discover -s llama-cpp/tests -p 'test_webgpu_expert_short_overlay.py' -v
```

Follow `tests/webgpu-expert-short/README.md` for the GPU experiment. Standalone software-GPU timing is not Mac performance.
The published model must still be measured with the ordinary profile; the GPU
profile's partitioning and readback waits are not an ordinary wall-time breakdown.
