#include "ggml.h"
#include "pre_wgsl.hpp"
#include "ggml-wgsl-shaders.hpp"
#include "ggml-webgpu-mul-mat-id-group-shaders.hpp"
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#define CEIL_DIV(a, b) (((a) + (b)-1) / (b))
#undef GGML_ABORT
#define GGML_ABORT(...) throw std::runtime_error("unsupported factory type")
#undef GGML_ASSERT
#define GGML_ASSERT(x)                                                                                                 \
    do {                                                                                                               \
        if (!(x))                                                                                                      \
            throw std::runtime_error("production assert");                                                             \
    } while (0)
struct webgpu_pipeline {
    unsigned id;
    std::string name;
    std::shared_ptr<void> context;
    std::string code;
};
static unsigned created = 0;
static webgpu_pipeline ggml_webgpu_create_pipeline(int, const std::string & code, const std::string & name) {
    return {++created, name, nullptr, code};
}
extern "C" const ggml_type_traits * ggml_get_type_traits(ggml_type type) {
    static const auto f = [] {
        ggml_type_traits t{};
        t.type_name = "f32";
        return t;
    }();
    static const auto q = [] {
        ggml_type_traits t{};
        t.type_name = "q8_0";
        return t;
    }();
    static const auto i = [] {
        ggml_type_traits t{};
        t.type_name = "iq3_s";
        return t;
    }();
    if (type == GGML_TYPE_F32)
        return &f;
    if (type == GGML_TYPE_Q8_0)
        return &q;
    if (type == GGML_TYPE_IQ3_S)
        return &i;
    throw std::runtime_error("traits type");
}
extern "C" bool ggml_is_quantized(ggml_type type) {
    return type == GGML_TYPE_Q8_0 || type == GGML_TYPE_IQ3_S;
}
extern "C" size_t ggml_type_size(ggml_type type) {
    return type == GGML_TYPE_Q8_0 ? 34 : type == GGML_TYPE_IQ3_S ? 110 : type == GGML_TYPE_F16 ? 2 : 4;
}
static size_t ggml_webgpu_tensor_misalignment(int, ggml_tensor * t) {
    return t->view_offs;
}
