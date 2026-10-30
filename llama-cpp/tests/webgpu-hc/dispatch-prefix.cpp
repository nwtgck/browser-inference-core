// Source-extracted dispatcher test. WGPU objects are recorders, not a GPU.
#include "ggml-webgpu-hc.hpp"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
namespace wgpu {
struct BindGroupEntry {
    uint32_t binding;
    const ggml_tensor * tensor;
};
} // namespace wgpu
struct webgpu_pipeline {
    bool pre;
};
struct webgpu_encoded_op {
    webgpu_pipeline pipeline;
    std::vector<uint32_t> params;
    std::vector<wgpu::BindGroupEntry> entries;
    uint32_t x, y;
};
struct MockLib {
    webgpu_pipeline get_hc_pipeline(bool pre) {
        return {pre};
    }
};
struct MockLimits {
    uint32_t maxComputeWorkgroupsPerDimension = 65535;
};
struct MockCapabilities {
    MockLimits limits;
};
struct MockGlobal {
    MockCapabilities capabilities;
};
struct MockContext {
    MockGlobal * global_ctx;
    MockLib * shader_lib;
};
using webgpu_context = std::shared_ptr<MockContext>;
extern "C" int64_t ggml_nelements(const ggml_tensor * t) {
    return t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3];
}
static float ggml_get_op_params_f32(const ggml_tensor * t, int i) {
    float v;
    std::memcpy(&v, reinterpret_cast<const char *>(t->op_params) + i * 4, 4);
    return v;
}
static size_t ggml_webgpu_tensor_misalignment(webgpu_context &, const ggml_tensor * t) {
    return t->view_offs % 256;
}
static wgpu::BindGroupEntry ggml_webgpu_make_tensor_bind_group_entry(webgpu_context &, uint32_t b, ggml_tensor * t) {
    return {b, t};
}
static webgpu_encoded_op ggml_backend_webgpu_build(webgpu_context &, webgpu_pipeline p, std::vector<uint32_t> params,
                                                   std::vector<wgpu::BindGroupEntry> entries, uint32_t x, uint32_t y) {
    return {p, params, entries, x, y};
}
