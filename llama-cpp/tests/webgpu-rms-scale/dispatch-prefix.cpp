#include "ggml-webgpu-rms-scale.hpp"
#include "ggml-webgpu-scale-unary.hpp"
#include "ggml-backend-impl.h"
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>
namespace wgpu {
enum class BufferUsage { None = 0, Storage = 1 };
static BufferUsage operator&(BufferUsage a, BufferUsage b) {
    return BufferUsage(int(a) & int(b));
}
struct Buffer {
    uintptr_t id = 0;
    uint64_t bytes = 1ull << 30;
    BufferUsage usage = BufferUsage::Storage;
    explicit operator bool() const {
        return id != 0;
    }
    void * Get() const {
        return reinterpret_cast<void *>(id);
    }
    uint64_t GetSize() const {
        return bytes;
    }
    BufferUsage GetUsage() const {
        return usage;
    }
};
struct BindGroupEntry {
    uint32_t binding = 0;
    Buffer buffer;
    uint64_t offset = 0, size = 0;
};
} // namespace wgpu
struct limits {
    uint32_t minStorageBufferOffsetAlignment = 256;
    uint32_t maxComputeInvocationsPerWorkgroup = 256, maxComputeWorkgroupSizeX = 256;
    uint32_t maxComputeWorkgroupsPerDimension = 65535;
    uint32_t maxComputeWorkgroupStorageSize = 32768;
    uint64_t maxUniformBufferBindingSize = 65536;
    uint32_t maxStorageBuffersPerShaderStage = 8, maxUniformBuffersPerShaderStage = 12, maxBindingsPerBindGroup = 1000;
    uint64_t maxStorageBufferBindingSize = 1ull << 30;
};
struct Global {
    struct {
        ::limits limits;
    } capabilities;
};
struct Library {
    unsigned calls = 0;
    unsigned get_rms_scale_pipeline(bool inplace) {
        ++calls;
        return unsigned(inplace);
    }
};
struct Context {
    std::shared_ptr<Global> global_ctx = std::make_shared<Global>();
    std::shared_ptr<Library> shader_lib = std::make_shared<Library>();
};
using webgpu_context = std::shared_ptr<Context>;
struct ggml_backend_webgpu_buffer_context {
    wgpu::Buffer buffer;
    std::shared_ptr<Global> global_ctx;
};
struct webgpu_encoded_op {
    unsigned pipeline;
    std::vector<uint32_t> params;
    std::vector<wgpu::BindGroupEntry> entries;
    uint32_t x, y;
};
static unsigned encoded = 0;
static webgpu_encoded_op ggml_backend_webgpu_build(webgpu_context &, unsigned pipeline, std::vector<uint32_t> params,
                                                   std::vector<wgpu::BindGroupEntry> entries, uint32_t x, uint32_t y) {
    ++encoded;
    return {pipeline, std::move(params), std::move(entries), x, y};
}
static void * const webgpu_ptr_base = reinterpret_cast<void *>(uintptr_t(0x1000));
#define WEBGPU_PARAMS_BUF_SIZE_BYTES 128
// @ACTUAL_FUNCTIONS@

static size_t checks = 0;
static void check(bool condition) {
    ++checks;
    if (!condition)
        throw std::runtime_error("RMS dispatch check " + std::to_string(checks));
}
