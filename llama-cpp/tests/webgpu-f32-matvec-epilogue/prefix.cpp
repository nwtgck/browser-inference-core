#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend-impl.h"
#include "ggml-webgpu-f32-matvec-epilogue.hpp"
#include "ggml-wgsl-shaders.hpp"
#include "ggml-webgpu-f32-matvec-epilogue-shaders.hpp"
#include "ggml-webgpu-q8-subgroup-shaders.hpp"
#ifdef TEST_PROFILE
#include "ggml-webgpu-q8-diagnostic-sha256.hpp"
#endif
#include "pre_wgsl.hpp"
#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#define CEIL_DIV(a,b) (((a)+(b)-1)/(b))
#define ROUNDUP_POW2(a,b) (((a)+(b)-1)&~((b)-1))
#define WEBGPU_STORAGE_BUF_BINDING_MULT 4
#define WEBGPU_PARAMS_BUF_SIZE_BYTES 128
#undef GGML_ASSERT
#define GGML_ASSERT(x) do {if(!(x)) throw std::runtime_error("Production assertion");}while(0)
#undef GGML_ABORT
#define GGML_ABORT(...) throw std::runtime_error("Unsupported factory type")
static void * const webgpu_ptr_base = reinterpret_cast<void *>(uintptr_t(0x1000));
static unsigned created=0, size_queries=0, usage_queries=0, encoded_calls=0;
struct webgpu_pipeline {unsigned id=0;std::string name;std::shared_ptr<void> context;std::string wgsl;std::array<uint8_t,32> diagnostic_sha256{};};
static webgpu_pipeline ggml_webgpu_create_pipeline(int,const std::string& code,const std::string& name) {return {++created,name,nullptr,code,{}};}
namespace wgpu {
enum class BufferUsage {None=0,Storage=1};
static BufferUsage operator&(BufferUsage a,BufferUsage b) {return BufferUsage(int(a)&int(b));}
struct Buffer {
    uintptr_t id=0;uint64_t bytes=0;BufferUsage usage=BufferUsage::Storage;
    explicit operator bool()const{return id!=0;}
    void * Get()const{return reinterpret_cast<void*>(id);}
    uint64_t GetSize()const{++size_queries;return bytes;}
    BufferUsage GetUsage()const{++usage_queries;return usage;}
};
struct BindGroupEntry {uint32_t binding=0;Buffer buffer;uint64_t offset=0,size=0;};
}
struct webgpu_dispatch_desc {webgpu_pipeline pipeline;std::vector<uint32_t> params;std::vector<wgpu::BindGroupEntry> bind_group_entries;std::vector<uint32_t> workgroups;};
struct webgpu_encoded_op {size_t num_kernels=0;std::vector<webgpu_dispatch_desc> dispatches;};
