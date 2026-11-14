# Source preparation and its evidence

## Production composition

Each runtime owns its typed recipe and preparation functions:

- `runtimes/llama-cpp/tools/recipe.ts` and `prepare.ts` select model, multimodal
  and ordinary/diagnostic WebGPU sources.
- `runtimes/stable-diffusion-cpp/tools/recipe.ts` and `prepare.ts` select its
  independent image and GGML upstreams and patches.
- `tools/source/` supplies Git input capture, file-scoped patch application and
  invocation of the unchanged upstream shader generator.

The executing checkout supplies the tool implementation and static configuration.
Compiler inputs are captured separately into private trees. Preparation applies
reviewable patches to those trees and copies ordinary owned native sources;
it leaves checked-out upstreams unchanged. Patch order and target paths are
explicit in each recipe. The patch runner rejects undeclared paths and failed
applications instead of silently changing or dropping a patch.

The llama WebGPU recipe keeps ordinary shaders, dedicated variants and the
profile-only diagnostic source distinct. The upstream `embed_wgsl.py` runs on
the final selected shader inputs; owned tools do not generate C++ implementation
bodies. Python is needed for that upstream generator. Dedicated variants retain
separate input sets so one specialization cannot replace an ordinary shader by
accident. The Q8 diagnostic's public-domain SHA-256 implementation is an owned
header under `native/diagnostics/`, with its attribution retained.

Prepared sources and their receipt are made available together. A failed
preparation leaves the previous complete output intact. Unchanged output bytes
preserve timestamps for warm builds. The output helper provides rollback for
ordinary filesystem errors; concurrent external modification remains outside
that guarantee.

## Identity and semantic checks

`source-preparation.json` records selected input and owned-source identities,
patches and their order, generated shader inputs, and actual output identities.
Build provenance also records the compiler's real translation units and target
sources. These observations let publication compare the selected Git inputs,
prepared sources and actual compiled inputs.

An observed digest identifies bytes. It does not prove API, layout, arithmetic,
ownership or model behavior. A harmless edit outside a patch hunk does not need
a new source-acceptance constant. Changed hunks, undeclared edits and generator
failures remain preparation errors. Experimental behavior with an explicit
reviewed-input gate must retain its documented rationale and equivalent
ownership/lifetime coverage before that gate can be removed.

## Build dependencies and coherent source sets

CMake receives separate checkout, emitted-tooling, compiler-input and prepared
source roots. The build records its actual compile commands and selected target
sources. Model preparation keeps matching headers and translation units in one
coherent tree. QSA and file-backed model sources retain their source-specific
include boundaries and unity-build exclusions.

The preparation and build tests check recipe order, template isolation, rollback,
unchanged timestamps, selected compile definitions and captured compiler inputs.
Current native validation compiles the prepared implementation with complete
fixtures and the real required headers. Generated Wasm, browser ABI behavior and
model inference remain separate validation layers.

## Historical fixtures

Historical source composers and fixed before/after comparisons belong under
`tests/support/` and named `tests/fixtures/` trees. Their identities describe
those frozen comparisons. They are not production inputs, alternate current
implementations or a compatibility test for arbitrary new upstream versions.
See [validation tiers](validation-tiers.md) for the evidence boundaries.
