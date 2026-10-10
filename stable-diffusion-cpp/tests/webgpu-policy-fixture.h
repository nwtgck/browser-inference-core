#pragma once
#include "ggml-backend-impl.h"
#include <cstddef>
struct WebgpuTestLimits {
    size_t storage = 128 * 1024 * 1024;
    size_t shared = 32768;
    bool subgroups = true;
    bool subgroup_matrix = false;
};
bool webgpu_test_supports(const ggml_tensor* op, const WebgpuTestLimits& limits = {});
