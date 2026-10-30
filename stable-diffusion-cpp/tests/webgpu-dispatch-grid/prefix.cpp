// Only the external API surface is doubled. Dispatch builders come from the
// prepared runtime, and index guards come from the prepared WGSL entry points.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#define GGML_ASSERT(x)                                                                                                 \
    do {                                                                                                               \
        if (!(x))                                                                                                      \
            throw std::runtime_error("assert: " #x);                                                                   \
    } while (0)
#define GGML_ABORT(msg) throw std::runtime_error(msg)
struct U3 {
    uint32_t x = 0, y = 0, z = 0;
};
enum ggml_type { GGML_TYPE_F32, GGML_TYPE_F16 };
struct ggml_tensor {
    ggml_type type = GGML_TYPE_F32;
    int64_t ne[4] = {4, 1, 1, 1};
    size_t nb[4] = {4, 16, 16, 16};
    int32_t op_params[16] = {};
};
static size_t ggml_type_size(ggml_type type) {
    return type == GGML_TYPE_F16 ? 2 : 4;
}
static int64_t ggml_nrows(const ggml_tensor * t) {
    return t->ne[1] * t->ne[2] * t->ne[3];
}
static int64_t ggml_nelements(const ggml_tensor * t) {
    return t->ne[0] * ggml_nrows(t);
}
static float ggml_get_op_params_f32(const ggml_tensor * t, int i) {
    float out;
    std::memcpy(&out, &t->op_params[i], 4);
    return out;
}
namespace wgpu {
struct BindGroupEntry {
    uint32_t binding;
};
} // namespace wgpu
struct webgpu_encoded_op {
    std::vector<uint32_t> params;
    std::vector<wgpu::BindGroupEntry> entries;
    uint32_t x, y;
};
struct ggml_webgpu_generic_shader_decisions {
    uint32_t wg_size = 256;
    bool inplace = false;
};
struct ggml_webgpu_rms_norm_mul_shader_decisions {
    uint32_t wg_size = 256;
    bool inplace = false, overlap = false, src_overlap = false;
};
struct ggml_webgpu_glu_shader_decisions {
    uint32_t wg_size = 256;
    bool src_overlap = false;
};
struct webgpu_pipeline {
    std::shared_ptr<void> context;
};
struct ggml_webgpu_shader_lib_context {
    ggml_tensor *src0 = nullptr, *src1 = nullptr, *src2 = nullptr, *dst = nullptr;
    uint32_t max_wg_size = 256;
};
struct ShaderLib {
    bool inplace = false, overlap = false, src_overlap = false;
    webgpu_pipeline get_row_norm_pipeline(const ggml_webgpu_shader_lib_context & c) {
        return {std::make_shared<ggml_webgpu_generic_shader_decisions>(
            ggml_webgpu_generic_shader_decisions{c.max_wg_size, inplace})};
    }
    webgpu_pipeline get_soft_max_pipeline(const ggml_webgpu_shader_lib_context & c) {
        return get_row_norm_pipeline(c);
    }
    webgpu_pipeline get_rms_norm_mul_pipeline(const ggml_webgpu_shader_lib_context & c, bool) {
        return {std::make_shared<ggml_webgpu_rms_norm_mul_shader_decisions>(
            ggml_webgpu_rms_norm_mul_shader_decisions{c.max_wg_size, inplace, overlap, src_overlap})};
    }
    webgpu_pipeline get_glu_pipeline(const ggml_webgpu_shader_lib_context & c, bool) {
        return {std::make_shared<ggml_webgpu_glu_shader_decisions>(
            ggml_webgpu_glu_shader_decisions{c.max_wg_size, src_overlap})};
    }
};
struct Global {
    struct {
        struct {
            uint32_t maxComputeWorkgroupsPerDimension = 65535, maxComputeInvocationsPerWorkgroup = 256;
        } limits;
    } capabilities;
};
struct Context {
    std::shared_ptr<Global> global_ctx = std::make_shared<Global>();
    std::shared_ptr<ShaderLib> shader_lib = std::make_shared<ShaderLib>();
};
using webgpu_context = std::shared_ptr<Context>;
static bool ggml_webgpu_tensor_equal(ggml_tensor * a, ggml_tensor * b) {
    return a == b;
}
static bool ggml_webgpu_tensor_binding_overlap(std::shared_ptr<Global>, ggml_tensor *, ggml_tensor *) {
    return false;
}
static size_t ggml_webgpu_tensor_misalignment(webgpu_context &, ggml_tensor *) {
    return 16;
}
static size_t ggml_webgpu_tensor_align_offset(webgpu_context &, ggml_tensor *) {
    return 0;
}
static size_t ggml_webgpu_tensor_binding_size(webgpu_context &, ggml_tensor * t) {
    return ggml_nelements(t) * ggml_type_size(t->type);
}
static int ggml_webgpu_tensor_buf(ggml_tensor *) {
    return 1;
}
struct ggml_webgpu_merged_binding_range {
    size_t offset = 0, size = 1024;
};
static ggml_webgpu_merged_binding_range ggml_webgpu_tensor_merged_binding_range(webgpu_context &,
                                                                                std::initializer_list<ggml_tensor *>) {
    return {};
}
static uint32_t ggml_webgpu_tensor_merged_element_offset(ggml_tensor *, ggml_webgpu_merged_binding_range) {
    return 4;
}
static wgpu::BindGroupEntry ggml_webgpu_make_tensor_bind_group_entry(webgpu_context &, uint32_t i, ggml_tensor *) {
    return {i};
}
static wgpu::BindGroupEntry ggml_webgpu_make_bind_group_entry(uint32_t i, int, size_t, size_t) {
    return {i};
}
static webgpu_encoded_op ggml_backend_webgpu_build(webgpu_context &, webgpu_pipeline, const std::vector<uint32_t> & p,
                                                   const std::vector<wgpu::BindGroupEntry> & e, uint32_t x,
                                                   uint32_t y = 1) {
    return {p, e, x, y};
}
