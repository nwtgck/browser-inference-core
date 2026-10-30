#include "ggml.h"
#if __has_include("ggml-webgpu-f32-matvec-epilogue-shaders.hpp")
#include "ggml-webgpu-f32-matvec-epilogue-shaders.hpp"
#include "ggml-webgpu-f32-matvec-epilogue.hpp"
#endif
#include "pre_wgsl.hpp"
#include "ggml-wgsl-shaders.hpp"
#include "ggml-webgpu-q8-subgroup-shaders.hpp"
#include "ggml-webgpu-iq3s-id-subgroup-shaders.hpp"
#include "ggml-webgpu-mul-mat-id-group-shaders.hpp"
#include "ggml-webgpu-mul-mat-id-gather-logical32-shaders.hpp"
#ifdef TEST_PROFILE
#include "ggml-webgpu-q8-diagnostic-sha256.hpp"
#endif
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
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#define CEIL_DIV(a, b) (((a) + (b)-1) / (b))
#define ROUNDUP_POW2(a, b) (((a) + (b)-1) & ~((b)-1))
#define WEBGPU_STORAGE_BUF_BINDING_MULT 4
#undef GGML_ABORT
#define GGML_ABORT(...) throw std::runtime_error("Unsupported factory type")
#undef GGML_ASSERT
#define GGML_ASSERT(x)                                                                                                 \
    do {                                                                                                               \
        if (!(x))                                                                                                      \
            throw std::runtime_error("Production assertion");                                                          \
    } while (0)
static void * const webgpu_ptr_base = (void *)(uintptr_t)0x1000;
static unsigned created = 0;
struct webgpu_pipeline {
    unsigned id;
    std::string name;
    std::shared_ptr<void> context;
    std::string wgsl;
    std::array<uint8_t, 32> diagnostic_sha256{};
};
static webgpu_pipeline ggml_webgpu_create_pipeline(int, const std::string & code, const std::string & name) {
    return {++created, name, nullptr, code, {}};
}
extern "C" size_t ggml_type_size(ggml_type type) {
    switch (type) {
    case GGML_TYPE_F16:
    case GGML_TYPE_BF16:
        return 2;
    case GGML_TYPE_Q8_0:
        return 34;
    case GGML_TYPE_Q4_K:
        return 144;
    case GGML_TYPE_IQ3_S:
        return 110;
    default:
        return 4;
    }
}
extern "C" int64_t ggml_blck_size(ggml_type type) {
    return type == GGML_TYPE_Q8_0 ? 32 : (type == GGML_TYPE_Q4_K || type == GGML_TYPE_IQ3_S) ? 256 : 1;
}
extern "C" size_t ggml_nbytes(const ggml_tensor * t) {
    size_t n = t->ne[0] / ggml_blck_size(t->type) * ggml_type_size(t->type);
    for (int i = 1; i < 4; ++i)
        n += (t->ne[i] - 1) * t->nb[i];
    return n;
}
extern "C" bool ggml_is_quantized(ggml_type type) {
    return type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q4_K || type == GGML_TYPE_IQ3_S;
}
extern "C" const ggml_type_traits * ggml_get_type_traits(ggml_type type) {
    static const auto f32 = [] {
        ggml_type_traits t{};
        t.type_name = "f32";
        return t;
    }();
    static const auto f16 = [] {
        ggml_type_traits t{};
        t.type_name = "f16";
        return t;
    }();
    static const auto q8 = [] {
        ggml_type_traits t{};
        t.type_name = "q8_0";
        return t;
    }();
    static const auto q4 = [] {
        ggml_type_traits t{};
        t.type_name = "q4_K";
        return t;
    }();
    static const auto iq3 = [] {
        ggml_type_traits t{};
        t.type_name = "iq3_s";
        return t;
    }();
    static const auto bf = [] {
        ggml_type_traits t{};
        t.type_name = "bf16";
        return t;
    }();
    switch (type) {
    case GGML_TYPE_F32:
        return &f32;
    case GGML_TYPE_F16:
        return &f16;
    case GGML_TYPE_Q8_0:
        return &q8;
    case GGML_TYPE_Q4_K:
        return &q4;
    case GGML_TYPE_IQ3_S:
        return &iq3;
    case GGML_TYPE_BF16:
        return &bf;
    default:
        throw std::runtime_error("Unexpected traits");
    }
}
namespace wgpu {
struct Buffer {
    uintptr_t id = 0;
    uint64_t GetSize() const {
        return 1ull << 32;
    }
};
struct BindGroupEntry {
    uint32_t binding;
    Buffer buffer;
    uint64_t offset, size;
};
static bool operator==(const BindGroupEntry & a, const BindGroupEntry & b) {
    return a.binding == b.binding && a.buffer.id == b.buffer.id && a.offset == b.offset && a.size == b.size;
}
} // namespace wgpu
struct ggml_backend_buffer {
    void * context;
};
struct ggml_backend_webgpu_buffer_context {
    wgpu::Buffer buffer;
};
struct webgpu_dispatch_desc {
    webgpu_pipeline pipeline;
    std::vector<uint32_t> params;
    std::vector<wgpu::BindGroupEntry> bind_group_entries;
    std::vector<uint32_t> workgroups;
};
using webgpu_encoded_op = std::vector<webgpu_dispatch_desc>;
