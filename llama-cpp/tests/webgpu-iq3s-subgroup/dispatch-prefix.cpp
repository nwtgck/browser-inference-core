// Record only device objects and addresses; execute the complete real dispatcher.
#define CEIL_DIV(x, y) (((x) + (y)-1) / (y))
static unsigned buffer_size_queries = 0;
static uint64_t physical_override = 0;
struct RecordedBuffer {
    uintptr_t id;
    uint64_t bytes;
    uint64_t GetSize() const {
        ++buffer_size_queries;
        return physical_override ? physical_override : bytes;
    }
};
namespace wgpu {
struct BindGroupEntry {
    uint32_t binding;
    RecordedBuffer buffer;
    uint64_t offset, size;
};
} // namespace wgpu
struct RecordedLimits {
    uint32_t maxComputeInvocationsPerWorkgroup = 256, maxComputeWorkgroupsPerDimension = 65535;
    uint64_t maxStorageBufferBindingSize = 1ull << 30;
};
struct RecordedCapabilities {
    bool supports_subgroups = true;
    uint32_t min_subgroup_size = 16, max_subgroup_size = 16;
    RecordedLimits limits;
};
struct RecordedGlobal {
    RecordedCapabilities capabilities;
};
struct RecordedContext {
    std::shared_ptr<RecordedGlobal> global_ctx = std::make_shared<RecordedGlobal>();
    std::shared_ptr<RecordingLibrary> shader_lib = std::make_shared<RecordingLibrary>();
};
using webgpu_context = std::shared_ptr<RecordedContext>;
struct webgpu_encoded_op {
    webgpu_pipeline pipeline;
    std::vector<uint32_t> params;
    std::vector<wgpu::BindGroupEntry> entries;
    uint32_t x, y;
};
extern "C" size_t ggml_type_size(ggml_type type) {
    switch (type) {
    case GGML_TYPE_IQ3_S:
        return 110;
    case GGML_TYPE_Q5_1:
        return 24;
    case GGML_TYPE_IQ4_NL:
        return 18;
    case GGML_TYPE_F16:
        return 2;
    case GGML_TYPE_F32:
    case GGML_TYPE_I32:
        return 4;
    default:
        throw std::runtime_error("Unexpected dispatch type");
    }
}
static size_t tensor_span(const ggml_tensor * t) {
    const size_t block = t->type == GGML_TYPE_IQ3_S                                   ? 256
                         : (t->type == GGML_TYPE_Q5_1 || t->type == GGML_TYPE_IQ4_NL) ? 32
                                                                                      : 1;
    size_t bytes = static_cast<size_t>(t->ne[0]) / block * ggml_type_size(t->type);
    for (unsigned i = 1; i < 4; ++i)
        bytes += (static_cast<size_t>(t->ne[i]) - 1) * t->nb[i];
    return bytes;
}
static size_t ggml_webgpu_tensor_misalignment(webgpu_context &, const ggml_tensor * t) {
    return t->view_offs % 256;
}
static size_t ggml_webgpu_tensor_align_offset(webgpu_context &, const ggml_tensor * t) {
    return t->view_offs / 256 * 256;
}
static size_t ggml_webgpu_tensor_binding_size(webgpu_context & ctx, const ggml_tensor * t) {
    return (tensor_span(t) + ggml_webgpu_tensor_misalignment(ctx, t) + 3) / 4 * 4;
}
static RecordedBuffer ggml_webgpu_tensor_buf(const ggml_tensor * t) {
    return {reinterpret_cast<uintptr_t>(t), 1ull << 30};
}
static wgpu::BindGroupEntry ggml_webgpu_make_bind_group_entry(uint32_t binding, RecordedBuffer buffer, size_t offset,
                                                              size_t bytes) {
    return {binding, buffer, offset, bytes};
}
static webgpu_encoded_op ggml_backend_webgpu_build(webgpu_context &, webgpu_pipeline pipeline,
                                                   std::vector<uint32_t> params,
                                                   std::vector<wgpu::BindGroupEntry> entries, uint32_t x, uint32_t y) {
    return {pipeline, params, entries, x, y};
}
