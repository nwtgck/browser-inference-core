// Only the device boundary and tensor-address helpers are recorded. The entire
// source-bound expert dispatcher and its pipeline factory are executed below.
#define CEIL_DIV(x,y) (((x)+(y)-1)/(y))
namespace wgpu { struct BindGroupEntry {uint32_t binding;uintptr_t buffer;size_t offset,size;}; }
struct RecordedLimits {uint32_t maxComputeInvocationsPerWorkgroup=256,maxComputeWorkgroupsPerDimension=65535;};
struct RecordedCapabilities {bool supports_subgroups=true;uint32_t min_subgroup_size=32,max_subgroup_size=32;RecordedLimits limits;};
struct RecordedGlobal {RecordedCapabilities capabilities;};
struct RecordedContext {std::shared_ptr<RecordedGlobal> global_ctx=std::make_shared<RecordedGlobal>();std::shared_ptr<RecordingLibrary> shader_lib=std::make_shared<RecordingLibrary>();};
using webgpu_context=std::shared_ptr<RecordedContext>;
struct webgpu_encoded_op {webgpu_pipeline pipeline;std::vector<uint32_t> params;std::vector<wgpu::BindGroupEntry> entries;uint32_t x,y;};
extern "C" size_t ggml_type_size(ggml_type type) {
    switch(type) {case GGML_TYPE_Q5_1:return 24;case GGML_TYPE_IQ4_NL:return 18;
        case GGML_TYPE_F32:case GGML_TYPE_I32:return 4;default:throw std::runtime_error("unexpected dispatch type");}
}
static size_t ggml_webgpu_tensor_misalignment(webgpu_context &,const ggml_tensor * t){return t->view_offs%256;}
static size_t ggml_webgpu_tensor_align_offset(webgpu_context &,const ggml_tensor * t){return t->view_offs/256*256;}
static uintptr_t ggml_webgpu_tensor_buf(const ggml_tensor * t){return reinterpret_cast<uintptr_t>(t);}
static size_t ggml_webgpu_tensor_binding_size(webgpu_context &,const ggml_tensor *){return 4096;}
static wgpu::BindGroupEntry ggml_webgpu_make_bind_group_entry(uint32_t binding,uintptr_t buffer,size_t offset,size_t bytes){return {binding,buffer,offset,bytes};}
static webgpu_encoded_op ggml_backend_webgpu_build(webgpu_context &,webgpu_pipeline pipeline,std::vector<uint32_t> params,std::vector<wgpu::BindGroupEntry> entries,uint32_t x,uint32_t y){return {pipeline,params,entries,x,y};}
