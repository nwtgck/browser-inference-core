# llama.cpp b11804 integration review

## Identities and boundaries

This review covers the move from `620d2225cc436ab0b001e3200e590e3b286ef9ab`
to b11804 `c7fc5094179bf8a9c73c18593db714f8da41a402`. The source base is
browser-inference-core PR #8 at `8814fdcea6628e5040c749784c7ab2683e7b1021`,
which already updates the submodule and toolchain pin. This repair does not
change that pin, edit the vendor checkout, publish artifacts, or change
consumer integrity metadata.

Reviewed-input and composed-output checks remain fail-closed. Updating an
expected digest is not by itself evidence of numerical correctness. The
source overlays, their composition, native regressions and distribution
provenance have separate checks. Old version review documents describe their
own pins and are not approvals for this one.

## Upstream differences and integration decisions

| Area | Upstream change | Decision |
| --- | --- | --- |
| Qwen Sparse Attention (QSA) | A shared `new_pool_flat` view and device-probed `TOP_K_MASK` replace parts of the selection graph. | Rebase the all-pools optimization; retain the shared view, every per-layer key-cache write, ranked fused mask and ranked unfused fallback. |
| QSA dependencies | Mask dispatch now depends on `llama-context.cpp`, `llama-graph.cpp`, `llama-cparams.h` and `ggml.h`. | Pin these inputs and register them in `CMAKE_CONFIGURE_DEPENDS`, including when file-backed rows is enabled. |
| WebGPU reductions | `SUM` now requires contiguously allocated input; `SUM_ROWS` still requires contiguous rows. | Retain upstream's distinct support checks. Existing expert-sum fusion is unchanged and remains separately shape/stride-gated. |
| Backend splitting | `ARGMAX` is added to the row-splitting predicate. | Retain upstream behavior; same-device tensor copy still checks type, shape, range, alignment and overlap and falls back when not admissible. |
| Operation definitions and CPU kernels | `TOP_K_MASK` is added; CPU `SUM` handles non-unit inner strides. | Regenerate bindings from the pinned headers. Retain existing HC (hyper-connection) pre/post kernels and admission checks; no new op is silently mapped to an old numeric op ID. |
| Model loader | Metadata display is bounded and a `SET_ROWS` support probe is added. | Keep both changes. The existing bounded upload overlay applies to the new loader and its final composed output is independently pinned. |
| Graph/model definitions | Flint and other model classes and MoE scaling are added. | Preserve all upstream changes. File-backed per-layer embeddings remain restricted to Qwen4Exp and the copied model source set uses one coherent `models.h`. |
| Unity compilation | The relocated core `llama.cpp` lost its per-path unity exclusion. | Preserve `SKIP_UNITY_BUILD_INCLUSION` on both copied core translation units. Model unity batches must not be shifted by a relocated core source. |
| Vision/audio | Existing patches still apply. | Retain them; intentional patch failures printed by negative tests are not evidence of a compatibility failure. |

The WebGPU patch stack itself is unchanged: tensor copy, parameter uploads,
model uploads, state-space convolution, HC, Q8 short workgroups, expert sum and
expert short workgroups. The reviewed upstream and intermediate/final composed
identities are updated together. Loader and shader-library identities are
checked independently of `ggml-webgpu.cpp`.

## QSA correctness contract

The shortcut is admitted only when all padded pools are selected and the
existing F32 dump-row arithmetic remains exact (`n_kv + n_slots <= 2^24`).
Raw and pooled key-cache writes remain before any return. Selection reuse is
local to the graph and requires the same base mask tensor. Visibility,
sentinels, sequence separation and incomplete tails retain the scatter path.

When ranking is required, upstream's device-probed fused mask is preserved.
If the device does not support it, the original gather/scatter fallback is
preserved. The optimization does not add a WebGPU `TOP_K_MASK` kernel or force
a fusion on an unsupported backend.

The native regression links separately built reference and optimized libraries.
The census recognizes both the unfused `ADD` mask and the new `TOP_K_MASK` mask,
while counting the latter separately. The default native CPU comparison requires
actual fused-mask execution in the reference and ranked optimized phases. A
second run explicitly disables this mask through the pinned context internals
inside the test executable, before either context's first decode. It sets both
`fused_top_k_mask` and `auto_ftkm` false; it does not add a runtime switch or
modify the vendor source. The census rejects any fused execution in this mode.
Both runs require all-pools bypass/mask sharing in short phases, unchanged key
writes and the same graph observations after state restoration. Numerical checks
still require finite, nonconstant and byte-identical logits, without increased
tolerances. The runner also compares fused and unfused output identities within
each independently built implementation.

## Required validation paths

From `llama-cpp`, with the pinned upstream checked out:

```sh
npm test
python3 -m unittest discover -s tests -p 'test_*.py'
python3 -m unittest discover -s ../tests -p 'test_*.py'
python3 tests/qsa-selection/run.py --source vendor/llama.cpp --work build/qsa-regression --jobs 2
```

The native runner writes command exit codes, separate library identities,
fixture identities and comparison logs. Its latest `qsa-results/status.json`,
not an older log file or a successful source preflight, determines success.
The normal workflow also builds/runs `tests/mtmd-bf16` and tests all configured
WebAssembly profiles/variants before packaging and publishing. Those gates
must not be weakened or replaced by a synthetic success marker.

The incremental-build regression changes each of the four newly reviewed fusion
dependencies in a temporary source copy. In both header ownership modes, the
normal build entry point must reconfigure and refuse the changed source, then
recover after restoration. This tests actual Ninja regeneration, not only a
recorded CMake property.

Local native tests do not establish Emscripten/Dawn compilation or browser
inference. The toolchain installation, WebAssembly link/runtime checks and
publication need their actual CI environment. In particular, this source
repair is not a claim that Flint is fast or fully GPU-resident: its `POOL_1D`
and `LOG` paths still need real-device scheduling and correctness checks.
