#include "fixture.hpp"
#include "ggml-backend-impl.h"
#include <iostream>
#include <memory>
#include <optional>
#include <vector>
namespace wgpu {
enum class BufferUsage { None = 0, Storage = 1 };
static BufferUsage operator&(BufferUsage a, BufferUsage b) { return BufferUsage(int(a) & int(b)); }
struct Buffer {
    uintptr_t id = 0; uint64_t bytes = 1ull << 30; BufferUsage usage = BufferUsage::Storage;
    explicit operator bool() const { return id != 0; }
    void * Get() const { return reinterpret_cast<void *>(id); }
    uint64_t GetSize() const { return bytes; }
    BufferUsage GetUsage() const { return usage; }
};
struct BindGroupEntry { uint32_t binding=0; Buffer buffer; uint64_t offset=0,size=0; };
}
struct limits {
    uint32_t minStorageBufferOffsetAlignment=256, maxComputeInvocationsPerWorkgroup=256;
    uint32_t maxComputeWorkgroupSizeX=256, maxComputeWorkgroupSizeY=256, maxComputeWorkgroupsPerDimension=65535;
    uint64_t maxUniformBufferBindingSize=65536, maxStorageBufferBindingSize=1ull<<30;
    uint32_t maxStorageBuffersPerShaderStage=8, maxUniformBuffersPerShaderStage=12, maxBindingsPerBindGroup=1000;
};
struct Global { struct { ::limits limits; } capabilities; };
struct ggml_webgpu_shader_lib_context { ggml_tensor * src0=nullptr, * src1=nullptr, * dst=nullptr; uint32_t max_wg_size=0; };
struct ggml_webgpu_ssm_conv_shader_decisions { uint32_t block_size,tokens_per_wg; };
struct webgpu_pipeline { unsigned id; std::shared_ptr<void> context; };
struct Library {
    unsigned calls=0;
    webgpu_pipeline get_ssm_conv_pipeline(const ggml_webgpu_shader_lib_context & c, bool silu=false) {
        ++calls; return {unsigned(silu),std::make_shared<ggml_webgpu_ssm_conv_shader_decisions>(ggml_webgpu_ssm_conv_shader_decisions{32,c.dst->ne[1]==1?1u:8u})};
    }
};
struct Context { std::shared_ptr<Global> global_ctx=std::make_shared<Global>(); std::shared_ptr<Library> shader_lib=std::make_shared<Library>(); };
using webgpu_context=std::shared_ptr<Context>;
struct ggml_backend_webgpu_buffer_context { wgpu::Buffer buffer; std::shared_ptr<Global> global_ctx; };
struct webgpu_encoded_op { webgpu_pipeline pipeline; std::vector<uint32_t> params; std::vector<wgpu::BindGroupEntry> entries; uint32_t x,y; };
static unsigned encoded=0;
static webgpu_encoded_op ggml_backend_webgpu_build(webgpu_context &,webgpu_pipeline p,std::vector<uint32_t> params,std::vector<wgpu::BindGroupEntry> entries,uint32_t x,uint32_t y) {
    ++encoded; return {p,std::move(params),std::move(entries),x,y};
}
static void * const webgpu_ptr_base=reinterpret_cast<void *>(uintptr_t(0x1000));
#define WEBGPU_PARAMS_BUF_SIZE_BYTES 128
#define WEBGPU_STORAGE_BUF_BINDING_MULT 4
#ifndef CEIL_DIV
#define CEIL_DIV(x,y) (((x)+(y)-1)/(y))
#endif
#ifndef ROUNDUP_POW2
#define ROUNDUP_POW2(x,y) (((x)+(y)-1)&~((y)-1))
#endif
// @ACTUAL_FUNCTIONS@
