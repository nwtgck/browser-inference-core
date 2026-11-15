# speech.cpp browser preparation

The checked-out speech.cpp and ggml submodules are never edited in place. Source
preparation exports their selected commits, applies the ordered recipe to those
private trees, and records the original commits and patch identities in the
receipt used by the build and package tools.

`0001-external-ggml-source.patch` gives upstream's ggml source directory a CMake
cache setting. Its default retains upstream's bundled directory. This runtime
passes its independently pinned, prepared ggml tree so updates cannot bind it to
another runtime's ggml. The patch does not alter inference operations or model
formats. The owned CMake target depends on `speech-static`; upstream CLI, server,
shared-library and check targets are excluded from the default target graph.

`0002-webgpu-matmul-vec4-layout.patch` checks actual binding offsets and active
tensor strides before selecting vec4 matrix kernels, including indexed matrix
products. Nonaligned views use the existing scalar kernels. A tensor's logical
row width alone does not establish the alignment of rows in a larger cache.
The predicate lives in a private backend header so native checks can exercise
the same selection rule without a GPU. This patch is owned by this runtime and
does not change its independent upstream pin.

`0003-webgpu-buffer-address-range.patch` splits backend buffers without
overflowing `size_t` and limits each WebGPU buffer to its representable pointer
range, including the backend's synthetic base address. Device limits remain
64-bit until that range check. This keeps GPU weights in bounded buffers on
Wasm32 without narrowing model-file offsets or allocating their full size in
Wasm memory.

The initial source pair is speech.cpp `2f51f66164a906b322a48303bfbcdd6936833302`
and its ggml pin `e1a9ac12edbb57402618fd6936598603f5ee6266`. Current identities
remain the submodule gitlinks and source-bound preparation receipts. Updating a
pin requires rechecking its public API, source configuration, and runtime
contracts; a historical digest is not a behavioral admission rule.
