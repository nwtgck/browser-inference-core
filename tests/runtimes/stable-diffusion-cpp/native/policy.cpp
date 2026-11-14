#ifdef SDCB_TEST_BROWSER_API
#define __EMSCRIPTEN__ 1
#endif
#include "browser-api.hpp"
#include "webgpu-policy-fixture.h"

// Compile the complete production TU. The admission predicate uses real ggml
// types and the real backend device context. GPU arithmetic is tested separately.
#include "ggml-webgpu.cpp"

bool webgpu_test_supports(const ggml_tensor * op, const WebgpuTestLimits & limits) {
    auto global = std::make_shared<webgpu_global_context_struct>();
    global->capabilities.limits.maxStorageBufferBindingSize = limits.storage;
    global->capabilities.limits.minStorageBufferOffsetAlignment = 256;
    global->capabilities.limits.maxComputeWorkgroupStorageSize = limits.shared;
    global->capabilities.limits.maxComputeWorkgroupsPerDimension = 65535;
    global->capabilities.limits.maxComputeInvocationsPerWorkgroup = 256;
    global->capabilities.supports_subgroups = limits.subgroups;
    global->capabilities.supports_subgroup_matrix = limits.subgroup_matrix;
    global->capabilities.sg_mat_m = 8;
    global->capabilities.sg_mat_n = 8;
    global->capabilities.sg_mat_k = 8;
    global->capabilities.max_subgroup_size = 32;
    ggml_backend_webgpu_device_context context{global, "contract", "external GPU boundary"};
    ggml_backend_device device{};
    device.context = &context;
    return ggml_backend_webgpu_device_supports_op(&device, op);
}
