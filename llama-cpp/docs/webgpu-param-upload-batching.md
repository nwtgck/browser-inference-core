# WebGPU parameter upload batching

All WebGPU profiles batch parameter uploads in ordinary builds. Push the source
branch to use the existing build, validation and source-bound publication workflow.
No additional workflow input or build option is required. CPU profiles are unchanged.

The arena keeps an aligned CPU mirror and uploads it at the existing graph
submission boundaries. Queue order remains upload A / submit A / upload B / submit B.
Waits, readbacks, command cadence, bindings and slot resets are preserved. The
mirror holds 74 aligned slots (18,944 bytes at 256-byte alignment); padding increases
transferred bytes. Fewer API calls alone do not establish a measured speedup.

`prepare_webgpu_source.py` applies this patch and same-device tensor copy to one
build-tree translation unit. Upstream d81235049384534c167caea52b85a694f6103d14,
reviewed source inputs, patch bytes and pinned Dawn headers are checked before use.
The vendor checkout is unchanged. CMake replaces the target source once and keeps
the MoE header include priority. Provenance records both patch identities, the
combined compiled-source identity, compile definition and enabled profile/variants.
Updater preflight fails if the required overlay is incompatible; it does not
silently omit either patch.

## Validation and maintenance

`python3 tests/webgpu-param-upload-batching/check_uploads.py /path/to/upstream`
checks actual arena and graph code against a CPU queue model. Add
`--dawn-package /path/to/emdawnwebgpu_pkg` for pinned-header syntax checks.
`test_webgpu_source_overlay.py` checks source preparation and CMake integration.
These host checks do not certify browser/GPU correctness, numerical equivalence,
or device performance; those require the built runtime and target models/devices.

Upstream updates require review of parameter consumers and submit/reset boundaries.
Unrelated commits with identical reviewed inputs remain compatible.
Textual patch application alone is not semantic validation. Retaining upstream
behavior or waiting for upstream support are no-patch alternatives. Remove this
patch and its integration when upstream supplies suitable support or the benefit
no longer justifies maintenance.
