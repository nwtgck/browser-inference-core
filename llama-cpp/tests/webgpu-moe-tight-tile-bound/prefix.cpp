#include "ggml.h"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#define ROUNDUP_POW2(x,p) (((x)+(p)-1)&~((p)-1))
#define CEIL_DIV(x,p) (((x)+(p)-1)/(p))
#define WEBGPU_STORAGE_BUF_BINDING_MULT 4
static size_t checks=0;
static void check(bool ok,const char *why){++checks;if(!ok)throw std::runtime_error(std::to_string(checks)+": "+why);}
namespace wgpu {
struct Data {size_t size;std::vector<uint8_t> bytes;explicit Data(size_t n):size(n),bytes(n+128,0xa7) {}};
struct Buffer {std::shared_ptr<Data> data;uint64_t GetSize()const{return data->size;}Data *Get()const{return data.get();}};
struct BindGroupEntry {uint32_t binding;Buffer buffer;uint64_t offset,size;};
}
struct Limits {uint32_t maxComputeInvocationsPerWorkgroup=256,maxComputeWorkgroupsPerDimension=32,maxComputeWorkgroupStorageSize=16384;size_t minStorageBufferOffsetAlignment=256;uint64_t maxStorageBufferBindingSize=1ull<<26;};
struct Global {struct {Limits limits;bool supports_dot_product=false;uint32_t min_subgroup_size=16;} capabilities;std::string vendor="Recorded";};
using webgpu_global_context=std::shared_ptr<Global>;
struct ggml_backend_webgpu_device_context {webgpu_global_context webgpu_global_ctx;};
struct ggml_backend_webgpu_buffer_context {wgpu::Buffer buffer;webgpu_global_context global_ctx;};
static void *const webgpu_ptr_base=reinterpret_cast<void*>(uintptr_t(0x1000));
static void *ggml_backend_webgpu_buffer_get_base(ggml_backend_buffer_t){return webgpu_ptr_base;}
struct ggml_webgpu_shader_lib_context {ggml_tensor *src0,*src1,*src2,*dst;uint32_t max_wg_size;};
struct ggml_webgpu_mul_mat_shader_decisions {uint32_t tile_m=4,wg_size_m=8,tile_n=4,wg_size_n=8;};
struct webgpu_pipeline {unsigned id;std::shared_ptr<void> context;};
struct Library {
 std::shared_ptr<ggml_webgpu_mul_mat_shader_decisions> decisions=std::make_shared<ggml_webgpu_mul_mat_shader_decisions>();
 webgpu_pipeline get_mul_mat_id_gather_pipeline(const ggml_webgpu_shader_lib_context&){return {1,{}};}
 webgpu_pipeline get_mul_mat_id_pipeline(const ggml_webgpu_shader_lib_context&,bool){return {2,decisions};}
};
struct Context {webgpu_global_context global_ctx;std::shared_ptr<Library> shader_lib=std::make_shared<Library>();};
using webgpu_context=std::shared_ptr<Context>;
struct webgpu_dispatch_desc {webgpu_pipeline pipeline;std::vector<uint32_t> params;std::vector<wgpu::BindGroupEntry> bind_group_entries;std::vector<uint32_t> workgroups;};
using webgpu_encoded_op=std::vector<webgpu_dispatch_desc>;
static webgpu_encoded_op ggml_backend_webgpu_build_multi(webgpu_context&,const webgpu_encoded_op& d){return d;}
static webgpu_encoded_op ggml_webgpu_mul_mat_id_vec(webgpu_context&,ggml_tensor*,ggml_tensor*,ggml_tensor*,ggml_tensor*){return {};}
// Non-MoE allocator branches are compiled unchanged but must not execute here.
template<class...T>static bool ggml_webgpu_flash_attn_use_vec_path(T&&...){throw std::runtime_error("unexpected flash allocator path");}
template<class...T>static bool ggml_webgpu_flash_attn_k_direct(T&&...){throw std::runtime_error("unexpected flash key path");}
template<class...T>static bool ggml_webgpu_flash_attn_v_direct(T&&...){throw std::runtime_error("unexpected flash value path");}
template<class...T>static uint32_t ggml_webgpu_flash_attn_get_vec_kv_tile(T&&...){throw std::runtime_error("unexpected flash tile path");}
template<class...T>static uint32_t ggml_webgpu_flash_attn_vec_nwg(T&&...){throw std::runtime_error("unexpected flash grid path");}
template<class...T>static bool ggml_webgpu_can_use_mmvq(T&&...){throw std::runtime_error("unexpected MMVQ allocator path");}
