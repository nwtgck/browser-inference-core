#include "ggml.h"
#include "pre_wgsl.hpp"
#include "ggml-wgsl-shaders.hpp"
#include "ggml-webgpu-router-tail-shaders.hpp"
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#undef GGML_ABORT
#define GGML_ABORT(...) throw std::runtime_error("unsupported exact factory branch")
struct webgpu_pipeline { unsigned id; std::string name; std::shared_ptr<void> context; std::string code; };
static unsigned created = 0;
static webgpu_pipeline ggml_webgpu_create_pipeline(int, const std::string & code, const std::string & name) {
    return {++created, name, nullptr, code};
}
extern "C" const ggml_type_traits * ggml_get_type_traits(ggml_type type) {
    static const auto f32 = [] { ggml_type_traits result{}; result.type_name = "f32"; return result; }();
    if (type != GGML_TYPE_F32) { throw std::runtime_error("fixture type"); }
    return &f32;
}
extern "C" const char * ggml_op_name(ggml_op op) {
    if (op == GGML_OP_CLAMP) { return "CLAMP"; }
    if (op == GGML_OP_DIV) { return "DIV"; }
    throw std::runtime_error("fixture op");
}
extern "C" const char * ggml_unary_op_name(ggml_unary_op) { throw std::runtime_error("fixture unary"); }
extern "C" ggml_unary_op ggml_get_unary_op(const ggml_tensor *) { throw std::runtime_error("fixture unary"); }
static int32_t ggml_get_op_params_i32(const ggml_tensor * tensor, uint32_t index) {
    int32_t result; memcpy(&result, reinterpret_cast<const char *>(tensor->op_params) + 4 * index, 4); return result;
}
