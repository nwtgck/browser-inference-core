// Record only device/resource boundaries; execute the actual prepared dispatchers.
#define CEIL_DIV(x,y) (((x)+(y)-1)/(y))
namespace wgpu { struct BindGroupEntry {uint32_t binding;uintptr_t buffer;size_t offset,size;}; }
struct RecordedLimits {uint32_t maxComputeInvocationsPerWorkgroup=256,maxComputeWorkgroupsPerDimension=65535;};
struct RecordedCapabilities {
    bool supports_subgroups=true,supports_dot_product=false,supports_subgroup_matrix=false;
    uint32_t min_subgroup_size=32,max_subgroup_size=32,sg_mat_m=0,sg_mat_n=0,sg_mat_k=0;
    RecordedLimits limits;
};
struct RecordedGlobal {RecordedCapabilities capabilities;std::string vendor;};
struct RecordedContext {std::shared_ptr<RecordedGlobal> global_ctx=std::make_shared<RecordedGlobal>();std::shared_ptr<RecordingLibrary> shader_lib=std::make_shared<RecordingLibrary>();};
using webgpu_context=std::shared_ptr<RecordedContext>;
struct webgpu_dispatch_desc {webgpu_pipeline pipeline;std::vector<uint32_t> params;std::vector<wgpu::BindGroupEntry> bind_group_entries;std::vector<uint32_t> workgroups;};
struct webgpu_encoded_op {webgpu_pipeline pipeline;std::vector<uint32_t> params;std::vector<wgpu::BindGroupEntry> entries;uint32_t x,y;};
struct ggml_webgpu_mul_mat_shader_decisions {bool use_subgroup_matrix=false;uint32_t subgroup_m=1,subgroup_n=1,subgroup_matrix_m=1,subgroup_matrix_n=1,tile_m=1,tile_n=1,wg_size_m=1,wg_size_n=1;};
extern "C" size_t ggml_type_size(ggml_type type) {
    switch(type) {case GGML_TYPE_Q4_K:return 144;case GGML_TYPE_F32:case GGML_TYPE_I32:return 4;default:throw std::runtime_error("unexpected dispatch type");}
}
static size_t ggml_webgpu_tensor_misalignment(webgpu_context &,const ggml_tensor * t){return t->view_offs%256;}
static size_t ggml_webgpu_tensor_align_offset(webgpu_context &,const ggml_tensor * t){return t->view_offs/256*256;}
static uintptr_t ggml_webgpu_tensor_buf(const ggml_tensor * t){return reinterpret_cast<uintptr_t>(t);}
static size_t ggml_webgpu_tensor_binding_size(webgpu_context &,const ggml_tensor *){return 4096;}
static wgpu::BindGroupEntry ggml_webgpu_make_bind_group_entry(uint32_t binding,uintptr_t buffer,size_t offset,size_t bytes){return {binding,buffer,offset,bytes};}
static wgpu::BindGroupEntry ggml_webgpu_make_tensor_bind_group_entry(webgpu_context & ctx,uint32_t binding,const ggml_tensor * t){return {binding,ggml_webgpu_tensor_buf(t),ggml_webgpu_tensor_align_offset(ctx,t),ggml_webgpu_tensor_binding_size(ctx,t)};}
static webgpu_encoded_op ggml_backend_webgpu_build(webgpu_context &,webgpu_pipeline pipeline,std::vector<uint32_t> params,std::vector<wgpu::BindGroupEntry> entries,uint32_t x,uint32_t y){return {pipeline,params,entries,x,y};}
static webgpu_encoded_op ggml_backend_webgpu_build_multi(webgpu_context &,const std::vector<webgpu_dispatch_desc> & ds){if(ds.size()!=1)throw std::runtime_error("unexpected multi dispatch");const auto & d=ds[0];return {d.pipeline,d.params,d.bind_group_entries,d.workgroups[0],d.workgroups[1]};}
static bool ggml_webgpu_tensor_binding_overlap(const std::shared_ptr<RecordedGlobal> &,ggml_tensor *,ggml_tensor *){return false;}
struct ggml_webgpu_merged_binding_range {size_t offset=0,size=0;};
static ggml_webgpu_merged_binding_range ggml_webgpu_tensor_merged_binding_range(webgpu_context &,std::initializer_list<ggml_tensor *>){throw std::runtime_error("unexpected overlap");}
static uint32_t ggml_webgpu_tensor_merged_element_offset(ggml_tensor *,const ggml_webgpu_merged_binding_range &){throw std::runtime_error("unexpected overlap");}
static void ggml_webgpu_quantize_q8_dispatch(webgpu_context &,ggml_tensor *,ggml_tensor *,ggml_tensor *,std::vector<webgpu_dispatch_desc> &){throw std::runtime_error("unexpected MMVQ");}
