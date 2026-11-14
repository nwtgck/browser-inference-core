# WebGPU parameter upload batching

All WebGPU profiles batch parameter uploads in ordinary builds. Push the source
branch to use the existing build, validation and source-bound publication workflow.
No additional workflow input or build option is required. CPU profiles are unchanged.

The arena keeps an aligned CPU mirror and uploads it at the existing graph
submission boundaries. Queue order remains upload A / submit A / upload B / submit B.
Waits, readbacks, command cadence, bindings and slot resets are preserved. The
mirror holds 74 aligned slots (18,944 bytes at 256-byte alignment); padding increases
transferred bytes. Fewer API calls alone do not establish a measured speedup.

`runtimes/llama-cpp/tools/recipe.ts` selects the batching and same-device copy
patches in the final private WebGPU source tree. The upstream checkout remains
unchanged. Preparation receipts identify the patches and final outputs; actual
compiler-input records establish which prepared translation units were selected.
The updater preflight fails on incompatible required patches rather than omitting
either change.

## Validation and maintenance

The current native provider compiles the complete prepared backend with real SDK
declarations. Its recorder observes parameter bytes, alignment, uploads, submit
boundaries, cache ownership and empty/tail behavior. The typed preparation tests
check source composition. See [verification](../../../docs/verification.md) for
those entries and the separate browser boundary. Recorded queue calls do not
execute shaders or establish model throughput.

Upstream updates require review of parameter consumers and submit/reset boundaries.
Unrelated commits with identical reviewed inputs remain compatible.
Textual patch application alone is not semantic validation. Retaining upstream
behavior or waiting for upstream support are no-patch alternatives. Remove this
patch and its integration when upstream supplies suitable support or the benefit
no longer justifies maintenance.
