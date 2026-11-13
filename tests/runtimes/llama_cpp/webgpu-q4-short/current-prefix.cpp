#include "ggml-webgpu-f32-matvec-epilogue.hpp"
// Actual complete shader libraries. Only device objects and buffer addresses are mocked.
static unsigned buffer_size_queries = 0;
static uint64_t physical_override = 0;
struct WGPUBufferImpl {
    unsigned refs = 1;
    uintptr_t id;
    uint64_t bytes;
};
extern "C" {
void wgpuTextureViewAddRef(WGPUTextureView) {
    throw std::runtime_error("Unexpected texture binding");
}
void wgpuTextureViewRelease(WGPUTextureView) {
    throw std::runtime_error("Unexpected texture binding");
}
void wgpuSamplerAddRef(WGPUSampler) {
    throw std::runtime_error("Unexpected sampler binding");
}
void wgpuSamplerRelease(WGPUSampler) {
    throw std::runtime_error("Unexpected sampler binding");
}
void wgpuBufferAddRef(WGPUBuffer value) {
    ++value->refs;
}
void wgpuBufferRelease(WGPUBuffer value) {
    if (!--value->refs)
        delete value;
}
uint64_t wgpuBufferGetSize(WGPUBuffer value) {
    ++buffer_size_queries;
    return physical_override ? physical_override : value->bytes;
}
}
struct RecordingLibrary : ggml_webgpu_shader_lib {
    RecordingLibrary()
        : ggml_webgpu_shader_lib(wgpu::Device::Acquire(new WGPUDeviceImpl),
                                 wgpu::Instance::Acquire(new WGPUInstanceImpl),
                                 std::make_shared<ggml_webgpu_optional::DeviceGeneration>()) {}
};
struct ReferenceLibrary {
    reference::ggml_webgpu_shader_lib lib{wgpu::Device::Acquire(new WGPUDeviceImpl)};
    static reference::ggml_webgpu_shader_lib_context context(const ggml_webgpu_shader_lib_context & c) {
        reference::ggml_webgpu_shader_lib_context r{};
        r.src0 = c.src0;
        r.src1 = c.src1;
        r.src2 = c.src2;
        r.dst = c.dst;
        r.supports_subgroups = c.supports_subgroups;
        r.max_wg_size = c.max_wg_size;
        r.supports_dot_product = c.supports_dot_product;
        r.supports_subgroup_matrix = c.supports_subgroup_matrix;
        r.min_subgroup_size = c.min_subgroup_size;
        r.max_subgroup_size = c.max_subgroup_size;
        r.vendor = c.vendor;
        r.sg_mat_m = c.sg_mat_m;
        r.sg_mat_n = c.sg_mat_n;
        r.sg_mat_k = c.sg_mat_k;
        return r;
    }
    static webgpu_pipeline convert(const reference::webgpu_pipeline & p) {
        // Distinct C++ types in the two complete headers: copy named fields,
        // never alias the reference context as the candidate's decision type.
        const auto & d = *static_cast<const reference::ggml_webgpu_mul_mat_vec_shader_decisions *>(p.context.get());
        auto c = std::make_shared<ggml_webgpu_mul_mat_vec_shader_decisions>();
        c->wg_size = d.wg_size;
        c->outputs_per_wg = d.outputs_per_wg;
        c->vec_size = d.vec_size;
        c->src_overlap = d.src_overlap;
        webgpu_pipeline result = {};
        result.pipeline = p.pipeline;
        result.name = p.name;
        result.context = c;
        return result;
    }
    webgpu_pipeline get_mul_mat_id_vec_pipeline(const ggml_webgpu_shader_lib_context & c, bool) {
        return convert(lib.get_mul_mat_id_vec_pipeline(context(c)));
    }
    webgpu_pipeline get_mul_mat_vec_pipeline(const ggml_webgpu_shader_lib_context & c, bool mmvq, bool) {
        return convert(lib.get_mul_mat_vec_pipeline(context(c), mmvq));
    }
};
template <class Pipeline> static unsigned pipeline_id(const Pipeline & p) {
    return p.pipeline.Get()->id;
}
template <class Pipeline> static const std::string & pipeline_code(const Pipeline & p) {
    return p.pipeline.Get()->code;
}
struct RecordedLimits {
    uint32_t maxComputeInvocationsPerWorkgroup = 256, maxComputeWorkgroupsPerDimension = 65535;
    uint64_t maxStorageBufferBindingSize = UINT32_MAX;
};
struct RecordedCapabilities {
    bool supports_subgroups = true, supports_dot_product = false, supports_subgroup_matrix = false;
    uint32_t min_subgroup_size = 32, max_subgroup_size = 32, sg_mat_m = 0, sg_mat_n = 0, sg_mat_k = 0;
    bool subgroup_size_control_enabled = false;
    RecordedLimits limits;
};
struct RecordedGlobal {
    RecordedCapabilities capabilities;
    std::string vendor;
};
struct RecordedContext {
    std::shared_ptr<RecordedGlobal> global_ctx = std::make_shared<RecordedGlobal>();
    std::shared_ptr<RecordingLibrary> shader_lib = std::make_shared<RecordingLibrary>();
};
using webgpu_context = std::shared_ptr<RecordedContext>;
struct webgpu_dispatch_desc {
    webgpu_pipeline pipeline;
    std::vector<uint32_t> params;
    std::vector<wgpu::BindGroupEntry> bind_group_entries;
    std::vector<uint32_t> workgroups;
};
struct webgpu_encoded_op {
    webgpu_pipeline pipeline;
    std::vector<uint32_t> params;
    std::vector<wgpu::BindGroupEntry> entries;
    uint32_t x, y;
};
extern "C" size_t ggml_type_size(ggml_type type) {
    switch (type) {
    case GGML_TYPE_Q4_K:
        return 144;
    case GGML_TYPE_F32:
    case GGML_TYPE_I32:
        return 4;
    default:
        throw std::runtime_error("unexpected dispatch type");
    }
}
static size_t ggml_webgpu_tensor_misalignment(webgpu_context &, const ggml_tensor * t) {
    return t->view_offs % 256;
}
static size_t ggml_webgpu_tensor_align_offset(webgpu_context &, const ggml_tensor * t) {
    return t->view_offs / 256 * 256;
}
static wgpu::Buffer ggml_webgpu_tensor_buf(const ggml_tensor * t) {
    return wgpu::Buffer::Acquire(new WGPUBufferImpl{1, reinterpret_cast<uintptr_t>(t), UINT32_MAX});
}
static size_t ggml_webgpu_tensor_binding_size(webgpu_context &, const ggml_tensor *) {
    return 4096;
}
static wgpu::BindGroupEntry ggml_webgpu_make_bind_group_entry(uint32_t binding, wgpu::Buffer buffer, size_t offset,
                                                              size_t bytes) {
    wgpu::BindGroupEntry entry{};
    entry.binding = binding;
    entry.buffer = buffer;
    entry.offset = offset;
    entry.size = bytes;
    return entry;
}
static wgpu::BindGroupEntry ggml_webgpu_make_tensor_bind_group_entry(webgpu_context & ctx, uint32_t binding,
                                                                     const ggml_tensor * t) {
    return ggml_webgpu_make_bind_group_entry(binding, ggml_webgpu_tensor_buf(t),
                                             ggml_webgpu_tensor_align_offset(ctx, t),
                                             ggml_webgpu_tensor_binding_size(ctx, t));
}
static webgpu_encoded_op ggml_backend_webgpu_build(webgpu_context &, webgpu_pipeline pipeline,
                                                   std::vector<uint32_t> params,
                                                   std::vector<wgpu::BindGroupEntry> entries, uint32_t x, uint32_t y) {
    return {pipeline, params, entries, x, y};
}
static webgpu_encoded_op ggml_backend_webgpu_build_multi(webgpu_context &,
                                                         const std::vector<webgpu_dispatch_desc> & ds) {
    if (ds.size() != 1)
        throw std::runtime_error("unexpected multi dispatch");
    const auto & d = ds[0];
    return {d.pipeline, d.params, d.bind_group_entries, d.workgroups[0], d.workgroups[1]};
}
static bool ggml_webgpu_tensor_binding_overlap(const std::shared_ptr<RecordedGlobal> &, ggml_tensor *, ggml_tensor *) {
    return false;
}
struct ggml_webgpu_merged_binding_range {
    size_t offset = 0, size = 0;
};
static ggml_webgpu_merged_binding_range ggml_webgpu_tensor_merged_binding_range(webgpu_context &,
                                                                                std::initializer_list<ggml_tensor *>) {
    throw std::runtime_error("unexpected overlap");
}
static uint32_t ggml_webgpu_tensor_merged_element_offset(ggml_tensor *, const ggml_webgpu_merged_binding_range &) {
    throw std::runtime_error("unexpected overlap");
}
static void ggml_webgpu_quantize_q8_dispatch(webgpu_context &, ggml_tensor *, ggml_tensor *, ggml_tensor *,
                                             std::vector<webgpu_dispatch_desc> &) {
    throw std::runtime_error("unexpected MMVQ");
}
