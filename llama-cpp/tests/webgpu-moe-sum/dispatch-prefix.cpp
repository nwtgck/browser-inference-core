#include "ggml-webgpu-moe-sum.hpp"
#include "ggml-backend-impl.h"
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <cstring>
#define ROUNDUP_POW2(x, pow2) (((x) + ((pow2) - 1)) & ~((pow2) - 1))
#define WEBGPU_STORAGE_BUF_BINDING_MULT 16
static void * const webgpu_ptr_base = (void *) (uintptr_t) 0x1000;
namespace wgpu {
struct Buffer { uintptr_t id=1; uint64_t size=1ull<<30; uintptr_t Get() const {return id;} uint64_t GetSize() const {return size;} };
struct BindGroupEntry { uint32_t binding; const ggml_tensor * tensor; };
}
struct Limits { uint32_t maxComputeInvocationsPerWorkgroup=256, maxComputeWorkgroupSizeX=256,
    maxStorageBuffersPerShaderStage=8, maxComputeWorkgroupsPerDimension=65535,
    minStorageBufferOffsetAlignment=256; uint64_t maxStorageBufferBindingSize=1ull<<30; };
struct Global { struct { Limits limits; } capabilities; };
using webgpu_global_context=std::shared_ptr<Global>;
struct Library { unsigned calls=0; uint32_t get_moe_sum_pipeline(uint32_t e) {++calls;return e;} };
struct Context { webgpu_global_context global_ctx=std::make_shared<Global>(); std::shared_ptr<Library> shader_lib=std::make_shared<Library>(); };
using webgpu_context=std::shared_ptr<Context>;
struct ggml_backend_webgpu_buffer_context { wgpu::Buffer buffer; webgpu_global_context global_ctx; };
struct webgpu_encoded_op { uint32_t pipeline; std::vector<uint32_t> params; std::vector<wgpu::BindGroupEntry> entries; uint32_t x,y; };
static unsigned encoded=0;
static webgpu_encoded_op ggml_backend_webgpu_build(webgpu_context &, uint32_t pipe, const std::vector<uint32_t> & params,
    const std::vector<wgpu::BindGroupEntry> & entries,uint32_t x,uint32_t y) {++encoded;return {pipe,params,entries,x,y};}
static wgpu::BindGroupEntry ggml_webgpu_make_tensor_bind_group_entry(webgpu_context &, uint32_t index, ggml_tensor * tensor) { return {index,tensor}; }

static const char * ggml_backend_webgpu_buffer_type_get_name(ggml_backend_buffer_type_t) { return "recording-webgpu"; }
