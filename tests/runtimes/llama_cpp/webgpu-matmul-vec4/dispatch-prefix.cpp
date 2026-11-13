struct Limits {
    uint32_t maxComputeInvocationsPerWorkgroup = 256, maxComputeWorkgroupsPerDimension = 65535;
    uint64_t maxStorageBufferBindingSize = 1ull << 30;
    size_t minStorageBufferOffsetAlignment = 256;
};
struct Capabilities {
    bool supports_subgroups = true, supports_dot_product = false, supports_subgroup_matrix = false;
    uint32_t min_subgroup_size = 16, max_subgroup_size = 16, sg_mat_m = 8, sg_mat_n = 8, sg_mat_k = 8;
    Limits limits;
};
struct Global {
    Capabilities capabilities;
    std::string vendor = "Recording";
};
using webgpu_global_context = std::shared_ptr<Global>;
struct Context {
    webgpu_global_context global_ctx = std::make_shared<Global>();
    std::shared_ptr<RecordingLibrary> shader_lib = std::make_shared<RecordingLibrary>();
};
using webgpu_context = std::shared_ptr<Context>;
static webgpu_encoded_op ggml_backend_webgpu_build(webgpu_context &, webgpu_pipeline p, std::vector<uint32_t> params,
                                                   std::vector<wgpu::BindGroupEntry> entries, uint32_t x, uint32_t y) {
    return {{p, std::move(params), std::move(entries), {x, y}}};
}
static webgpu_encoded_op ggml_backend_webgpu_build_multi(webgpu_context &,
                                                         const std::vector<webgpu_dispatch_desc> & ds) {
    return ds;
}
static void ggml_webgpu_quantize_q8_dispatch(webgpu_context &, ggml_tensor *, ggml_tensor *, ggml_tensor *,
                                             std::vector<webgpu_dispatch_desc> & ds) {
    auto decisions = std::make_shared<ggml_webgpu_generic_shader_decisions>();
    decisions->wg_size = 256;
    webgpu_pipeline p{0, "recorded-quantize", decisions, "", {}};
    ds.push_back({p, {}, {{0, {1}, 0, 1024}, {1, {1}, 1024, 1024}}, {1, 1}});
}
