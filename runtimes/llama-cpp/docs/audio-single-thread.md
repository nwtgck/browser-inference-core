# Non-pthread mtmd compatibility

Upstream now computes audio log-mel preprocessing through scheduler graphs. The
former `mtmd-audio-single-thread.patch` worker-loop overlay is retired; its old
FFT bitwise checks are not a reference for the new DFT/F32 implementation.

`mtmd-non-pthread-threadpool.patch` preserves the existing non-pthread Wasm
profile by normalizing the private shared pool request to one at
`mtmd_threadpool_init`, using the existing no-pool branch. Native and pthread
requests, upstream audio arithmetic, model selection and public parameter
layouts are unchanged. Both CPU and WebGPU builds apply it through the complete
checked mtmd subtree in `cmake/MtmdOverlay.cmake`.

Controlled tests of the actual helper distinguish pool creation, requested
counts, ownership and failures across the compile guards. Such tests are not
successful model loading, audio quality, actual browser or pthread execution.
See the [exception register](../upstream-patches-only-as-a-last-resort-with-explicit-user-approval/README.md)
for scope, alternatives and removal conditions.
