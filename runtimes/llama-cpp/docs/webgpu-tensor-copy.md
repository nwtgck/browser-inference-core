# Same-device WebGPU tensor copy

Ordinary WebGPU builds use the existing optional backend copy hook for eligible
same-device tensor copies. No extra workflow input or build option is required.
CPU profiles are unchanged; unsupported copies retain the generic fallback.

The hook checks device and layout compatibility, byte ranges, resource identity,
alignment and buffer usage before queuing a copy on the existing GPU queue.
It does not change model/checkpoint policy or add a public capability API.

`runtimes/llama-cpp/tools/recipe.ts` composes this patch with parameter upload
batching in one private coherent source tree. Preparation records selected input,
patch and output identities; publication compares those facts with the actual
compiled inputs. The vendor checkout remains unchanged, and required patch
conflicts fail the updater preflight.

## Validation and maintenance

The complete-source native provider exercises the actual copy hook with a
recorded WebGPU C API, real ggml layout decisions and deferred queue operations.
It checks same-device admission, aliases, buffer ranges and cross-device fallback.
These observations do not establish target-browser GPU correctness, shader
execution, model performance or ON_DEVICE checkpoint capacity. Validate those
separately using the built runtime. See [verification](../../../docs/verification.md).

Unrelated upstream commits with identical reviewed inputs remain compatible.
Review upstream changes to backend fallback, queue order and buffer lifetime.
Remove the patch and preparation/provenance integration when upstream provides
an equivalent copy hook or the requirement is dropped. Keeping generic host copies
or waiting for upstream support remain no-patch alternatives.
