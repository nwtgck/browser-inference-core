#include "ggml.h"
#include "ggml-wgsl-shaders.hpp"
#include "pre_wgsl.hpp"
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
#define WEBGPU_MUL_MAT_VEC_WG_SIZE 256
#define WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_PER_WG 4
#define WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG 4
#define WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_PER_WG 4
#undef GGML_ABORT
#define GGML_ABORT(...) throw std::runtime_error("Unsupported type")

// Only pipeline creation is replaced. The selector, cache key, factory,
// shader embedding and preprocessor below are the actual prepared/upstream code.
struct webgpu_pipeline {
    unsigned id;
    std::string name;
    std::shared_ptr<void> context;
    std::string wgsl;
};
static unsigned created = 0;
static webgpu_pipeline ggml_webgpu_create_pipeline(int, const std::string & code, const std::string & name) {
    return {++created, name, nullptr, code};
}
extern "C" const ggml_type_traits * ggml_get_type_traits(ggml_type type) {
    static const auto q5 = [] { ggml_type_traits t{}; t.type_name = "q5_1"; return t; }();
    static const auto iq4 = [] { ggml_type_traits t{}; t.type_name = "iq4_nl"; return t; }();
    static const auto q8 = [] { ggml_type_traits t{}; t.type_name = "q8_0"; return t; }();
    if (type == GGML_TYPE_Q5_1) return &q5;
    if (type == GGML_TYPE_IQ4_NL) return &iq4;
    if (type == GGML_TYPE_Q8_0) return &q8;
    throw std::runtime_error("Unexpected recording type");
}
