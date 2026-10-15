#pragma once
// Deterministic regression fixtures. No benchmark loops or trained model files.
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace image_test {
using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using Backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
using Allocator = std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)>;
inline void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}
inline Context context(size_t bytes = 8 * 1024 * 1024) {
    Context result(ggml_init({bytes, nullptr, true}), ggml_free);
    require(result != nullptr, "context allocation failed");
    return result;
}
inline Backend cpu() {
    Backend result(ggml_backend_cpu_init(), ggml_backend_free);
    require(result != nullptr, "CPU initialization failed");
    ggml_backend_cpu_set_n_threads(result.get(), 1);
    return result;
}
inline std::vector<float> values(size_t count, unsigned seed = 1, float divisor = 32.f) {
    std::vector<float> result(count);
    for (size_t i = 0; i < count; ++i) {
        result[i] = (static_cast<int>((i * 13 + seed * 7) % 31) - 15) / divisor;
    }
    return result;
}
inline void write(ggml_tensor* tensor, const std::vector<float>& data) {
    require(static_cast<size_t>(ggml_nelements(tensor)) == data.size(), "tensor size mismatch");
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(tensor, data.data(), 0, data.size() * sizeof(float));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> encoded(data.size());
        for (size_t i = 0; i < data.size(); ++i) encoded[i] = ggml_fp32_to_fp16(data[i]);
        ggml_backend_tensor_set(tensor, encoded.data(), 0, encoded.size() * sizeof(ggml_fp16_t));
    } else {
        throw std::runtime_error("unsupported fixture type");
    }
}
inline std::vector<float> read(ggml_tensor* tensor) {
    require(tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor), "expected contiguous F32 output");
    std::vector<float> data(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, data.data(), 0, data.size() * sizeof(float));
    return data;
}
inline ggml_cgraph* graph(ggml_context* ctx, ggml_tensor* output, size_t nodes = 4096) {
    auto result = ggml_new_graph_custom(ctx, nodes, false);
    ggml_set_output(output);
    ggml_build_forward_expand(result, output);
    return result;
}
inline Allocator allocate(ggml_backend_t backend, ggml_cgraph* graph) {
    Allocator result(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free);
    require(result != nullptr && ggml_gallocr_alloc_graph(result.get(), graph), "graph allocation failed");
    return result;
}
inline void compute(ggml_backend_t backend, ggml_cgraph* graph) {
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "graph compute failed");
}
inline void close(const std::vector<float>& expected, const std::vector<float>& actual,
                  float absolute = 0.f, float relative = 0.f) {
    require(expected.size() == actual.size(), "output length differs");
    for (size_t i = 0; i < expected.size(); ++i) {
        require(std::isfinite(expected[i]) && std::isfinite(actual[i]), "non-finite output");
        bool equal = absolute == 0.f && relative == 0.f
                         ? std::memcmp(&expected[i], &actual[i], sizeof(float)) == 0
                         : std::abs(expected[i] - actual[i]) <= absolute + relative * std::abs(expected[i]);
        require(equal, "output differs at " + std::to_string(i) + ": " +
                       std::to_string(expected[i]) + " vs " + std::to_string(actual[i]));
    }
}
inline int count(ggml_cgraph* graph, ggml_op op) {
    int result = 0;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) result += ggml_graph_node(graph, i)->op == op;
    return result;
}
} // namespace image_test
