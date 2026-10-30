struct Limits {
    uint32_t minStorageBufferOffsetAlignment = 256, maxComputeInvocationsPerWorkgroup = 256,
             maxComputeWorkgroupSizeX = 256;
    uint32_t maxStorageBuffersPerShaderStage = 3, maxBindingsPerBindGroup = 4, maxUniformBuffersPerShaderStage = 1;
    uint32_t maxComputeWorkgroupsPerDimension = 65535;
    uint64_t maxStorageBufferBindingSize = 1ull << 30, maxUniformBufferBindingSize = 128;
};
struct Capabilities {
    Limits limits;
    bool supports_subgroups = true, supports_dot_product = false, supports_subgroup_matrix = false;
    uint32_t min_subgroup_size = 16, max_subgroup_size = 16, sg_mat_m = 8, sg_mat_n = 8, sg_mat_k = 8;
};
struct Global {
    Capabilities capabilities;
    std::string vendor = "Recorded";
};
using webgpu_global_context = std::shared_ptr<Global>;
struct Context {
    webgpu_global_context global_ctx = std::make_shared<Global>();
    std::shared_ptr<RecordingLibrary> shader_lib = std::make_shared<RecordingLibrary>();
    struct {
        uint64_t slot_size = 128;
    } param_arena;
};
using webgpu_context = std::shared_ptr<Context>;
struct ggml_backend_webgpu_buffer_context {
    wgpu::Buffer buffer;
    webgpu_global_context global_ctx;
};
static webgpu_encoded_op ggml_backend_webgpu_build_multi(webgpu_context &,
                                                         const std::vector<webgpu_dispatch_desc> & ds) {
    ++encoded_calls;
    return {ds.size(), ds};
}
static void ggml_webgpu_quantize_q8_dispatch(webgpu_context &, ggml_tensor *, ggml_tensor *, ggml_tensor *,
                                             std::vector<webgpu_dispatch_desc> &) {
    throw std::runtime_error("F32 post must not quantize input");
}
