struct global_context {
    wgpu::Device device;
    wgpu::Queue queue;
    uint32_t command_submit_batch_size = 64, max_inflight_batches = 2;
};
struct context {
    std::shared_ptr<global_context> global_ctx;
    webgpu_param_arena param_arena;
    wgpu::CommandEncoder active_command_encoder;
    wgpu::ComputePassEncoder active_compute_pass;
    bool batch_compute_passes = false;
};
using webgpu_context = std::shared_ptr<context>;
struct ggml_backend_webgpu_context { webgpu_context webgpu_ctx; };
struct backend { void* context; }; using ggml_backend_t = backend*;
struct ggml_tensor { int op, kernels; uint32_t seed; };
struct ggml_cgraph { int n_nodes; ggml_tensor** nodes; };
enum ggml_status { GGML_STATUS_SUCCESS };
#define GGML_OP_SET_ROWS 999
struct webgpu_encoded_op { uint32_t num_kernels = 0; };
struct webgpu_pipeline { wgpu::Pipeline pipeline; std::string name; };
struct webgpu_dispatch_desc {
    webgpu_pipeline pipeline;
    std::vector<uint32_t> params;
    std::vector<wgpu::BindGroupEntry> bind_group_entries;
    std::pair<uint32_t, uint32_t> workgroups = {1, 1};
};
static wgpu::BindGroupEntry ggml_webgpu_make_bind_group_entry(uint32_t, wgpu::Buffer buffer,
                                                             size_t offset, size_t size) {
    return {buffer, offset, size};
}
static void ggml_backend_webgpu_wait_queue(std::shared_ptr<global_context>& ctx) {
    ++ctx->queue.waits;
    ctx->queue.drain();
}
static int checks = 0;
static void ggml_backend_webgpu_check_set_rows(webgpu_context&, uint32_t&) { ++checks; }
