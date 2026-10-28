# F32 four-row matvec epilogue regression

`python3 tests/test_webgpu_f32_matvec_epilogue.py` exercises the checked source stage,
actual graph/use-count planner and gallocr, complete physical binding guard,
matvec caller and normal/profile factories. Only native GPU objects and final
encoding are recorded; graph construction, allocator metadata, shader embedding,
preprocessing, parameter assembly and cache decisions use the prepared sources.

The native cases cover K boundaries, scalar/vec4 and subgroup/workgroup reduction,
allocator reuse, output visibility, internal/final multiuse, sliced graphs,
unaligned dense input views, all live binding pairs and actual uniform slot limits.
An optional `LCB_TEST_GGML_BASE_LIBRARY` can reuse a matching native `ggml-base`
library; otherwise the test builds that target with one job.

Processed shader identities and an inverse patch pin the original accumulator and
reduction bytes. These CPU tests do not execute WebGPU, prove floating-point bit
identity across compilers or establish throughput. Actual runtime and target-device
performance require separate validation. No model data is used.
