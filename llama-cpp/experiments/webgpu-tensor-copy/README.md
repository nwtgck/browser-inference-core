# Experimental same-device WebGPU tensor copy

This is an isolated proposal, not an adopted runtime change. The owner authorized experimental implementation and patch delivery on 2026-10-08; adoption remains the owner's decision. No consumer flag, default, published artifact, vendor checkout, or existing patch is changed.

## Source identity and scope

- Submission target: `nwtgck/browser-inference-core`.
- Bicore base commit: `4e04bb9b833b97d9367569fcdd225156039dfb9b`.
- The repository patch only adds this experiment directory; it does not activate the upstream patch in any build.

- Upstream: `d81235049384534c167caea52b85a694f6103d14`.
- Exact input SHA-256: see `inputs.json`.
- One translation unit: `ggml/src/ggml-webgpu/ggml-webgpu.cpp`.
- The patch implements the existing optional destination-buffer `cpy_tensor` hook; there is no public API, model, quantization, shader, or state-format change.
- This candidate is deliberately outside the accepted exception registry and build integration. Applying it to another upstream revision requires semantic review, not just a successful patch application.

The installed source has a null copy hook. Its generic tensor-copy fallback allocates a host buffer, reads the source tensor, and writes the destination tensor. ON_DEVICE checkpoints therefore still perform host transfers. This patch makes eligible copies stay on the GPU. It does not change existing host checkpoint behavior or enable ON_DEVICE checkpointing.

## Design and maintenance assumptions

1. Identify the source buffer as WebGPU before casting its context. Resolve view source buffers as the generic backend does.
2. Require the identical global context, which owns the device and queue, plus matching layouts.
3. Use the existing tensor offset helper. Reject wrapped view offsets and out-of-range byte intervals with subtraction-based checks. Never round the copied byte count or touch padding.
4. Accept empty copies and exact same-resource/same-offset copies as no-ops. Conservatively decline every other same-resource copy, including non-overlapping ranges, for older WebGPU implementations.
5. Require four-byte aligned source offset, destination offset and size, and COPY_SRC/COPY_DST usage. Unsupported cases return false for the existing generic fallback. Returning false does not validate malformed tensors or make invalid caller ranges safe.
6. Encode and submit immediately through the common queue. No staging allocation, map, CPU wait or explicit per-copy synchronization is introduced.

The current graph-compute implementation submits its final encoder before returning. Existing set, get and compute operations use this same queue. Checkpoint entry points synchronize before save/restore. This establishes ordering for serialized use of this backend; it does not add thread safety for concurrent calls or permit invoking copies against unsubmitted work in another context.

WebGPU permits destruction after submission: already submitted operations retain their resources until finished. Encoding without submission would not be sufficient. The copy returns after submission, matching this backend's queued write behavior; it does not assert physical completion on return. A subsequent backend synchronize or readback observes completion.

References: [WebGPU buffer copy](https://www.w3.org/TR/webgpu/#dom-gpucommandencoder-copybuffertobuffer), [buffer destruction](https://www.w3.org/TR/webgpu/#dom-gpubuffer-destroy).

## Verification

Run from the repository root with an available pristine d812 checkout:

```sh
python llama-cpp/experiments/webgpu-tensor-copy/test_candidate.py /path/to/upstream-d812
```

The runner verifies the input source hash, applies the patch only to a temporary copy, extracts the actual offset helper and copy hook plus the actual generic backend fallback, and compiles them with `c++ -std=c++20 -Wall -Wextra -Werror -O0`. No compiler or WebGPU packages are downloaded. `--prepare-only` performs the source and patch checks without compiling.

The instrumented CPU model tests byte correctness and untouched adjacent bytes, delayed execution, write-copy-write ordering, snapshot/mutate/restore ordering, immediate source destruction after submission, source and destination views, distinct contexts, foreign backend rejection before casting, absent source buffers, layout mismatch, empty copies, exact self-copy, same-buffer overlap and disjoint fallback, misaligned offsets/lengths, exact boundary ranges, range and view overflow rejection, missing buffer usages, actual generic read/write fallback, and fast-path avoidance of host get/set.

These tests passed in the available ordinary host compiler environment. They are deterministic CPU mock tests of the actual hook, not WebGPU conformance, a full translation-unit build, Wasm compilation, browser execution, GPU correctness, real-model inference, or performance measurements. The separately reported full native/WebGPU build environment issue is not bypassed by this test. No GPU speedup is claimed.

## Before adoption

- Validate the complete target build against its actual Dawn/Emscripten headers and WebGPU implementation.
- Run GPU copy, lifetime and fallback cases plus model checkpoint save/restore equivalence on the target browser/device.
- Recheck asynchronous completion assumptions across all call sites and future upstream queue changes.
- Treat ON_DEVICE allocation failure handling and checkpoint memory reservation as separate blockers. This patch alone does not make ON_DEVICE a safe default.
- If retained, add the approved exception/build integration/provenance without modifying immutable previously published patches or artifacts.

No-patch alternatives remain retaining host snapshots, a separately justified consumer checkpoint-frequency change, or waiting for upstream support. Remove this candidate when upstream supplies equivalent support or the owner declines it.
