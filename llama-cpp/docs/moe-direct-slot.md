# Experimental WebGPU MoE direct selected-slot integration

This is the previously tested direct-selected-slot WGSL patch, integrated into
bicore's isolated build-tree overlay convention. `scripts/build.py` enables it
for every configured WebGPU profile: `webgpu-wasm32-asyncify`,
`webgpu-wasm32-jspi` and `webgpu-wasm64-jspi`, in browser/test variants. CPU and
stable-diffusion behavior are unchanged. Direct CMake defaults OFF.
Wasm64 JSPI remains the investigation focus, not an implementation restriction.

The only semantic upstream change is `mul_mat_id_vec.wgsl`. The selected slot
identifies an expert via `ids[offset_ids + selected_slot]`; src1 still uses
`selected_slot % b_ne1`, and output is stored by selected slot. Trailing
rectangular dispatch workgroups return uniformly before the kernel reduction.
Multi-token MoE, dense kernels, routing, quantization and public APIs are unchanged.

All three WebGPU profiles compile the same `ggml_webgpu_mul_mat_id_vec` dispatch
and `mul_mat_id_vec.wgsl`. Uniform offsets, counts and strides are `uint32_t` in
host dispatch and `u32` in WGSL in both Wasm memory modes. The selected-slot
arithmetic does not use Wasm pointers. Upstream shader selection depends on
input types, dimensions and device capabilities, not JSPI/Asyncify or memory64.
Those profile settings affect host ABI/suspension; they do not select a different
MoE vector kernel. This supports the same source-level patch scope, not a claim
that every profile has been executed or measured.

`prepare_moe_direct_slot.py` checks an exact reviewed upstream revision and its
revision-specific input hashes: `7fe450e19305b828c199d602c23a8337aaa1f03b` or
`d81235049384534c167caea52b85a694f6103d14`. It verifies the upstream Git root
is the source directory (never an enclosing repository), then checks dispatch C++,
shader, shader-library include header, upstream CMake file, embedder and exact patch digest. It copies only the
shader directory, applies the patch outside vendor, and runs upstream's embedder.
`MoeDirectSlotOverlay.cmake` prepends the generated header directory to the actual
`ggml-webgpu` target. The unchanged original header-generation target still runs;
the compiler resolves the checked overlay header first. CMake tracks all shader
inputs. No-op preparation preserves output mtimes and mismatches fail configure.
The patch payload retains its original bytes, including final empty diff context;
`.gitattributes` allows that context-line whitespace for llama patch files only.

The integration tests run the upstream CMake target definition and inspect its
include search order, generate both complete headers with upstream's embedder,
and compare every shader except the intended vector kernel. These are source,
configuration and embedding checks, not a complete Wasm link or GPU execution.
Earlier independent experiments cover the kernel and the wasm64 JSPI tiny-model
software-GPU path. The wasm32 Asyncify/JSPI activation matrix and provenance are
source/configuration-tested only in this revision; their final Wasm build and
runtime correctness/performance remain unverified here. None of those tests
establishes Apple-GPU throughput or full-size model quality.

Run the existing `.github/workflows/build.yml` on a new committed source branch.
It builds every existing profile and both runtimes, packages and validates them.
Trusted non-Dependabot source pushes, manual branch runs and same-repository PRs
can publish an append-only `artifacts` commit and `consumer-update-N` metadata.
Fork PRs do not publish. Existing llama WebGPU CI smoke checks mocked-adapter
suspension, not shader correctness on a GPU.
There is no new workflow input. Runtime manifest formats and API schema remain
unchanged. Provenance lists the original/patched shader, generated header digest,
patch identity, reviewed inputs and exactly which profile/variants enabled it.

Return the real artifact commit and `consumer-update.yaml` to prepare Naidan's
separate dependency/hash adapter update. Do not substitute a source commit for an
artifact commit or invent generated-runtime hashes before CI has produced them.

## Feature-branch revision compatibility

The supplied feature source `9023d457115b26480d517e730a32366acf274767`
pins upstream `7fe450e19305b828c199d602c23a8337aaa1f03b`. The previous
experiment accepted only `d812350...`, causing configure/test failure before
shader compilation. The patch target `mul_mat_id_vec.wgsl`, dispatch function,
embedder and WebGPU CMake file match. Shader-library F32/F16 definitions and
BF16 support differ, so separate whole-file input maps preserve that distinction.
Included shader templates are not claimed to be identical. The feature's
upstream pin is unchanged; provenance identifies the selected reviewed revision.

Unit tests of map selection and known d812 results do not establish old-revision
build, runtime correctness or performance. Validate with a complete pinned
upstream checkout and the ordinary CI commands, without source overrides.
