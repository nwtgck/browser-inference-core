# Single-thread WebAssembly audio preprocessing

## Why this overlay exists

The pinned llama.cpp (`7fe450e19305b828c199d602c23a8337aaa1f03b`) requests
four threads inside Qwen3-TTS speaker-reference mel preprocessing. This is separate
from `mtmd_context_params.n_threads`. Parakeet also has an independent fixed
four-worker dispatch. Every current lcore browser profile has `pthreads: false`.
Setting the context thread count to one does not fix those internal calls.

`upstream-patches-only-as-a-last-resort-with-explicit-user-approval/mtmd-audio-single-thread.patch` runs the shared mel worker and Parakeet's
separate worker once with index 0 and stride 1, but only when `__EMSCRIPTEN__` is
defined and `__EMSCRIPTEN_PTHREADS__` is not. All frames are computed; this is not
merely dropping the extra workers while retaining a stride of four. Native and
pthread-enabled builds retain the upstream worker loops, numerical algorithms,
and public APIs. No Qwen-specific binding or WebGPU kernel is introduced.

## Integration and provenance

`cmake/MtmdAudioOverlay.cmake` replaces exactly one mtmd translation unit after
upstream defines the target. It is independent of the optional vision BF16
(Brain Floating Point 16) overlay. Both use isolated build-tree copies, never a
modified vendor checkout. The preparation script fails closed if upstream
changes make patch application ambiguous. The automatic upstream updater checks
both overlays, and artifact provenance hashes both original and compiled files.
The build-time native check compiles the same overlay with its guard inactive.

A consumer must rebuild/publish lcore and then update to that actual artifact
commit. Applying this source patch cannot change already published Wasm bytes.
Do not manufacture an artifact commit or bypass the consumer's manifest/hash
checks. This overlay does not establish real-model TTS quality, WebGPU operator
coverage, latency, cancellation during a synchronous native call, or browser
memory feasibility. Those still require a trained-model browser smoke test.

## Tests

Run `python -m unittest discover -s tests` with the pinned submodule initialized.
`LCB_TEST_LLAMA_SOURCE` can point to a separate local llama.cpp checkout.
The audio tests build the real upstream translation unit, compare its Qwen
speaker and Parakeet mel output before/after the overlay, simulate the Emscripten
single-thread guard with a thread constructor that aborts if used, and exercise
the pthread-enabled guard. Host simulation is not an Emscripten runtime test.
Tests also cover idempotent preparation, unchanged input, overlay conflicts,
provenance, and the automatic updater's audio preflight.

Primary sources:
- https://github.com/ggml-org/llama.cpp/blob/7fe450e19305b828c199d602c23a8337aaa1f03b/tools/mtmd/mtmd-audio.cpp
- https://github.com/nwtgck/browser-inference-core/blob/develop/llama-cpp/config/profiles.json
