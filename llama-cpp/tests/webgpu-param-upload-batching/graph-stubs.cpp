
#include <optional>
#define WEBGPU_LOG_DEBUG(x)
#define WEBGPU_CPU_PROFILE_TOTAL_START(x)
#define WEBGPU_CPU_PROFILE_TOTAL_END(x,y)
namespace wgpu {
using Expected=std::vector<std::pair<size_t,std::vector<uint8_t>>>;
struct CommandBuffer { Expected expected; };
struct ComputePassEncoder {
 bool active=false;
 explicit operator bool() const { return active; }
 void End() {active=false;}
 ComputePassEncoder & operator=(std::nullptr_t) {active=false;return *this;}
};
struct CommandEncoder {
 Expected expected;
 ComputePassEncoder BeginComputePass() {return {true};}
 CommandBuffer Finish() {return {expected};}
 CommandEncoder & operator=(std::nullptr_t) {expected.clear();return *this;}
};
CommandEncoder Device::CreateCommandEncoder() {return {};}
}
struct global_context {wgpu::Device device; wgpu::Queue queue; uint32_t command_submit_batch_size=64;};
struct context { std::shared_ptr<global_context> global_ctx; webgpu_param_arena param_arena; wgpu::CommandEncoder active_command_encoder; wgpu::ComputePassEncoder active_compute_pass; bool batch_compute_passes=false; };
using webgpu_context=std::shared_ptr<context>;
struct ggml_backend_webgpu_context {webgpu_context webgpu_ctx;};
struct backend {void *context;}; using ggml_backend_t=backend*;
struct ggml_tensor {int op;int kernels;uint32_t seed;};
struct ggml_cgraph {int n_nodes;ggml_tensor **nodes;};
enum ggml_status {GGML_STATUS_SUCCESS};
#define GGML_OP_SET_ROWS 999
struct webgpu_encoded_op {uint32_t num_kernels;};
static std::optional<webgpu_encoded_op> ggml_webgpu_encode(webgpu_context &ctx,ggml_cgraph *g,int idx,int &) {
 auto node=g->nodes[idx]; if(!node->kernels)return {};
 for(int j=0;j<node->kernels;j++) {
  size_t size=4*((node->seed+j)%33),offset=ctx->param_arena.alloc_slot(size);
  std::vector<uint8_t> params(size,uint8_t((node->seed+j)%251+1));
#ifdef GGML_WEBGPU_BATCH_PARAM_UPLOADS
  ctx->param_arena.stage(offset,params.data(),size);
#else
  ctx->global_ctx->queue.WriteBuffer(ctx->param_arena.buffer,offset,params.data(),size);
#endif
  ctx->active_command_encoder.expected.push_back({offset,params});
 }
 return webgpu_encoded_op{uint32_t(node->kernels)};
}
static void ggml_backend_webgpu_submit_commands(webgpu_context &ctx,const wgpu::CommandBuffer commands,uint32_t &num) {
 ctx->global_ctx->queue.Submit(ctx->param_arena.buffer,commands.expected,ctx->param_arena.slot_stride);num++;
}
static int checks=0;
static void ggml_backend_webgpu_check_set_rows(webgpu_context &,uint32_t &) {checks++;}
