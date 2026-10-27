#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-webgpu-q8-diagnostic.h"
#include <atomic>
#include <cstring>
#include <functional>
namespace wgpu {
enum class BufferUsage { None=0, Storage=1 };
static BufferUsage operator&(BufferUsage a,BufferUsage b){return BufferUsage(int(a)&int(b));}
enum class FeatureName { ShaderF16 };
enum class MapMode { Read };
struct Buffer {
    uintptr_t id=0; uint64_t bytes=1ull<<30; BufferUsage usage=BufferUsage::Storage;
    explicit operator bool()const{return id!=0;}
    void * Get()const{return reinterpret_cast<void *>(id);}
    uint64_t GetSize()const{return bytes;}
    BufferUsage GetUsage()const{return usage;}
    const void * GetConstMappedRange(uint64_t,uint64_t)const;
    void Unmap()const{}
};
struct BindGroupEntry { uint32_t binding=0;Buffer buffer;uint64_t offset=0,size=0; };
using BindGroup=unsigned;
struct PassTimestampWrites {unsigned querySet=0;uint32_t beginningOfPassWriteIndex=0,endOfPassWriteIndex=0;};
struct ComputePassDescriptor {PassTimestampWrites * timestampWrites=nullptr;};
static unsigned dispatches=0,submits=0,maps=0;
static uint64_t timestamp_values[4096]{};
static std::vector<std::pair<uint32_t,uint32_t>> grids;
const void * Buffer::GetConstMappedRange(uint64_t,uint64_t)const{return timestamp_values;}
struct ComputePassEncoder {
    bool active=false;
    explicit operator bool()const{return active;}
    ComputePassEncoder & operator=(std::nullptr_t){active=false;return *this;}
    void SetPipeline(unsigned){}
    void SetBindGroup(unsigned,BindGroup){}
    void DispatchWorkgroups(uint32_t x,uint32_t y,uint32_t){++dispatches;grids.emplace_back(x,y);}
    void End(){active=false;}
};
struct CommandBuffer {};
struct CommandEncoder {
    bool active=true;
    explicit operator bool()const{return active;}
    CommandEncoder & operator=(std::nullptr_t){active=false;return *this;}
    ComputePassEncoder BeginComputePass(ComputePassDescriptor * p=nullptr){if(p&&p->timestampWrites){auto * w=p->timestampWrites;timestamp_values[w->beginningOfPassWriteIndex]=1000;timestamp_values[w->endOfPassWriteIndex]=2000;}return {true};}
    CommandBuffer Finish(){active=false;return {};}
    void ResolveQuerySet(unsigned,unsigned,unsigned,Buffer,unsigned){}
    void CopyBufferToBuffer(Buffer,unsigned,Buffer,unsigned,size_t){}
};
struct Device {bool f16=true;bool HasFeature(FeatureName)const{return f16;}CommandEncoder CreateCommandEncoder(){return {};}};
struct Queue {};
}
struct Limits {
 uint32_t minStorageBufferOffsetAlignment=256,maxComputeInvocationsPerWorkgroup=1024,maxComputeWorkgroupSizeX=1024;
 uint32_t maxComputeWorkgroupsPerDimension=65535,maxComputeWorkgroupStorageSize=32768,maxStorageBuffersPerShaderStage=8,maxBindingsPerBindGroup=1000,maxUniformBuffersPerShaderStage=12;
 uint64_t maxStorageBufferBindingSize=1ull<<30,maxUniformBufferBindingSize=65536;
};
struct Global {
 struct {Limits limits;bool supports_subgroups=true,supports_subgroup_matrix=false,supports_dot_product=false;uint32_t min_subgroup_size=16,max_subgroup_size=16,sg_mat_m=0,sg_mat_n=0,sg_mat_k=0;} capabilities;
 std::string vendor="apple";wgpu::Device device;wgpu::Queue queue;std::atomic<uint64_t> buffer_generation{1};uint32_t command_submit_batch_size=4;
};
static std::vector<std::vector<uint32_t>> actual_params;
static std::vector<wgpu::BindGroupEntry> actual_bindings;
struct Arena {
 wgpu::Buffer buffer{100};size_t slot_size=256,cursor=0;unsigned flushes=0,resets=0;
 size_t alloc_slot(size_t n){if(n>128)throw std::runtime_error("param overflow");const auto at=cursor;cursor+=256;return at;}
 void stage(size_t,const void * p,size_t n){const auto * v=static_cast<const uint32_t *>(p);actual_params.emplace_back(v,v+n/4);}
 void flush(wgpu::Queue &){++flushes;}
 void reset(){++resets;cursor=0;}
};
struct Cache {
 unsigned begins=0,ends=0;
 void begin_graph(uint64_t){++begins;}
 void end_graph(){++ends;}
 wgpu::BindGroup get(wgpu::Device &,unsigned,unsigned,const std::vector<wgpu::BindGroupEntry> & e,const char *,uint64_t){actual_bindings=e;return 1;}
};
struct Context {
 std::shared_ptr<Global> global_ctx=std::make_shared<Global>();std::unique_ptr<RecordingLibrary> shader_lib=std::make_unique<RecordingLibrary>();
 bool operation_in_progress=false,batch_compute_passes=false;wgpu::CommandEncoder active_command_encoder{false};wgpu::ComputePassEncoder active_compute_pass;
 Arena param_arena;Cache bind_group_cache;unsigned profile_timestamp_query_set=1;uint32_t profile_timestamp_query_count=0;
 wgpu::Buffer profile_timestamp_dev_buf{101},profile_timestamp_host_buf{102};std::unordered_map<std::string,double> shader_gpu_time_ms;
};
using webgpu_context=std::shared_ptr<Context>;
using webgpu_global_context=std::shared_ptr<Global>;
struct ggml_backend_webgpu_context {webgpu_context webgpu_ctx;};
struct ggml_backend_webgpu_buffer_context {wgpu::Buffer buffer;webgpu_global_context global_ctx;};
using ggml_webgpu_shader_lib=RecordingLibrary;
static void * const webgpu_ptr_base=reinterpret_cast<void *>(uintptr_t(0x1000));
#define WEBGPU_PARAMS_BUF_SIZE_BYTES 128
#define WEBGPU_STORAGE_BUF_BINDING_MULT 4
#define WEBGPU_MAX_PROFILE_QUERY_COUNT 4096
#define WEBGPU_LOG_DEBUG(...) ((void)0)
#define WEBGPU_CPU_PROFILE_TOTAL_START(...) ((void)0)
#define WEBGPU_CPU_PROFILE_TOTAL_END(...) ((void)0)
#ifndef CEIL_DIV
#define CEIL_DIV(a,b) (((a)+(b)-1)/(b))
#endif
#ifndef ROUNDUP_POW2
#define ROUNDUP_POW2(a,b) (((a)+(b)-1)&~((b)-1))
#endif
static std::function<void()> suspend_hook;
static void ggml_backend_webgpu_submit_commands(webgpu_context &,wgpu::CommandBuffer,uint32_t & n){++wgpu::submits;++n;}
static void ggml_backend_webgpu_map_buffer(webgpu_global_context &,wgpu::Buffer,wgpu::MapMode,uint64_t,size_t){++wgpu::maps;if(suspend_hook){auto hook=std::move(suspend_hook);suspend_hook={};hook();}}
static void ggml_backend_webgpu_check_set_rows(webgpu_context &,uint32_t &){throw std::runtime_error("unexpected SET_ROWS");}
