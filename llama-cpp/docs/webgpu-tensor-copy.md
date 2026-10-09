# Same-device WebGPU tensor copy

Ordinary WebGPU builds use the existing optional backend copy hook for eligible
same-device tensor copies. No extra workflow input or build option is required.
CPU profiles are unchanged; unsupported copies retain the generic fallback.

The hook checks device and layout compatibility, byte ranges, resource identity,
alignment and buffer usage before queuing a copy on the existing GPU queue.
It does not change model/checkpoint policy or add a public capability API.

`prepare_webgpu_source.py` composes this patch with parameter upload batching in
one build-tree translation unit. The vendor checkout is never edited. Preparation
checks reviewed upstream d81235049384534c167caea52b85a694f6103d14 inputs, patch bytes,
combined output identity and pinned Dawn headers. Provenance records the inputs,
patch and compiled source per profile/variant. Existing build and publication
validation remain in use; incompatible required patches fail updater preflight.

## Validation and maintenance

`python3 tests/webgpu-tensor-copy/check_hook.py /path/to/upstream` exercises the
actual hook with mock buffers and a queue. Add
`--dawn-package /path/to/emdawnwebgpu_pkg` for actual-header syntax checks.
The overlay unit tests check guards, source preservation, CMake source replacement
and coexistence with other overlays. These checks do not certify target-browser
GPU correctness, copy lifetime, numerical equivalence, performance or ON_DEVICE
checkpoint capacity. Validate those separately with the built runtime.

Unrelated upstream commits with identical reviewed inputs remain compatible.
Review upstream changes to backend fallback, queue order and buffer lifetime.
Remove the patch and preparation/provenance integration when upstream provides
an equivalent copy hook or the requirement is dropped. Keeping generic host copies
or waiting for upstream support remain no-patch alternatives.
