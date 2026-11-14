#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

// The requested backend executes the real operators. Expectations use raw BF16
// bit fields and independent scalar formulas, without translating shader source.
namespace bf16_kernel_contract {
inline void require(bool value, const char * message) {
    if (!value)
        throw std::runtime_error(message);
}
using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
inline Context context() {
    return Context(ggml_init({2 * 1024 * 1024, nullptr, true}), ggml_free);
}
inline float reference(uint16_t value) {
    const uint32_t bits = uint32_t(value) << 16;
    float output;
    std::memcpy(&output, &bits, sizeof(output));
    return output;
}
inline uint32_t bits(float value) {
    uint32_t output;
    std::memcpy(&output, &value, sizeof(output));
    return output;
}
inline void compute(ggml_backend_t backend, ggml_context * ctx, ggml_tensor * output) {
    require(ggml_backend_supports_op(backend, output), "requested backend rejected BF16 kernel contract");
    const auto graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, output);
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "BF16 kernel graph failed");
    ggml_backend_synchronize(backend);
}
inline void gather(ggml_backend_t backend, unsigned width, unsigned base, bool all_bits) {
    auto ctx = context();
    require(bool(ctx), "BF16 kernel context");
    const unsigned rows = all_bits ? 1 : 5;
    auto raw = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_BF16, base + width * rows + 2);
    auto weight = ggml_view_2d(ctx.get(), raw, width, rows, width * sizeof(uint16_t), base * sizeof(uint16_t));
    auto indices = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, all_bits ? 1 : 4);
    auto before = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 8);
    auto output = ggml_get_rows(ctx.get(), weight, indices);
    auto after = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 8);
    Buffer buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend), ggml_backend_buffer_free);
    require(bool(buffer), "BF16 gather allocation");
    std::vector<uint16_t> source(base + width * rows + 2, 0x7f81);
    for (unsigned index = 0; index < width * rows; ++index)
        source[base + index] = all_bits ? uint16_t(index) : uint16_t(0x3e00 + (index % 51) * 7);
    const std::array<int32_t, 4> selected{4, 0, 3, 4};
    const int32_t zero = 0;
    ggml_backend_tensor_set(raw, source.data(), 0, source.size() * sizeof(uint16_t));
    ggml_backend_tensor_set(indices, all_bits ? &zero : selected.data(), 0, ggml_nbytes(indices));
    std::array<float, 8> guard;
    guard.fill(-1234.0f);
    ggml_backend_tensor_set(before, guard.data(), 0, sizeof(guard));
    ggml_backend_tensor_set(after, guard.data(), 0, sizeof(guard));
    compute(backend, ctx.get(), output);
    std::vector<float> actual(ggml_nelements(output));
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    for (size_t index = 0; index < actual.size(); ++index) {
        const unsigned row = all_bits ? 0 : unsigned(selected[index / width]);
        require(bits(actual[index]) == uint32_t(source[base + row * width + index % width]) << 16,
                "BF16 gather bit recovery/selected row/offset");
    }
    for (auto tensor : {before, after}) {
        std::array<float, 8> observed;
        ggml_backend_tensor_get(tensor, observed.data(), 0, sizeof(observed));
        require(observed == guard, "BF16 gather destination guard");
    }
}
inline void matrix(ggml_backend_t backend, unsigned k, unsigned base, unsigned columns) {
    constexpr unsigned rows = 5;
    auto ctx = context();
    auto raw = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_BF16, base + k * rows + 2);
    auto weight = ggml_view_2d(ctx.get(), raw, k, rows, k * sizeof(uint16_t), base * sizeof(uint16_t));
    auto input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, columns);
    auto output = ggml_mul_mat(ctx.get(), weight, input);
    Buffer buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend), ggml_backend_buffer_free);
    require(bool(buffer), "BF16 matrix allocation");
    std::vector<uint16_t> source(base + k * rows + 2, 0x7f81);
    for (unsigned index = 0; index < k * rows; ++index)
        source[base + index] = uint16_t((index % 3 ? 0x3e00 : 0xbe00) + (index % 29) * 5);
    std::vector<float> values(k * columns), actual(rows * columns);
    for (size_t index = 0; index < values.size(); ++index)
        values[index] = float(int(index % 7) - 3) / 8;
    ggml_backend_tensor_set(raw, source.data(), 0, source.size() * sizeof(uint16_t));
    ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(float));
    compute(backend, ctx.get(), output);
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    for (unsigned column = 0; column < columns; ++column)
        for (unsigned row = 0; row < rows; ++row) {
            double expected = 0;
            for (unsigned inner = 0; inner < k; ++inner)
                expected += double(reference(source[base + row * k + inner])) * values[column * k + inner];
            require(std::isfinite(actual[column * rows + row]) &&
                        std::abs(actual[column * rows + row] - expected) < 2e-6,
                    "BF16 matrix dot/tail/zero padding");
        }
}
inline void indexed_matrix(ggml_backend_t backend, unsigned k, unsigned base, unsigned tokens) {
    constexpr unsigned rows = 5, experts = 3, used = 2;
    auto ctx = context();
    auto raw = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_BF16, base + k * rows * experts + 2);
    auto weight = ggml_view_3d(ctx.get(), raw, k, rows, experts, k * 2, k * rows * 2, base * 2);
    auto input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, k, used, tokens);
    auto indices = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, used, tokens);
    auto output = ggml_mul_mat_id(ctx.get(), weight, input, indices);
    Buffer buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend), ggml_backend_buffer_free);
    require(bool(buffer), "BF16 indexed matrix allocation");
    std::vector<uint16_t> source(base + k * rows * experts + 2, 0x7f81);
    for (unsigned index = 0; index < k * rows * experts; ++index)
        source[base + index] = uint16_t((index % 3 ? 0x3e00 : 0xbe00) + (index % 29) * 5);
    std::vector<float> values(k * used * tokens), actual(rows * used * tokens);
    std::vector<int32_t> ids(used * tokens);
    for (size_t index = 0; index < values.size(); ++index)
        values[index] = float(int(index % 7) - 3) / 8;
    for (unsigned token = 0; token < tokens; ++token)
        for (unsigned slot = 0; slot < used; ++slot)
            ids[token * used + slot] = int32_t((token + 2 * slot) % experts);
    ggml_backend_tensor_set(raw, source.data(), 0, source.size() * sizeof(uint16_t));
    ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(float));
    ggml_backend_tensor_set(indices, ids.data(), 0, ids.size() * sizeof(int32_t));
    compute(backend, ctx.get(), output);
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    for (unsigned token = 0; token < tokens; ++token)
        for (unsigned slot = 0; slot < used; ++slot)
            for (unsigned row = 0; row < rows; ++row) {
                double expected = 0;
                const auto expert = unsigned(ids[token * used + slot]);
                for (unsigned inner = 0; inner < k; ++inner)
                    expected += double(reference(source[base + (expert * rows + row) * k + inner])) *
                                values[(token * used + slot) * k + inner];
                const float observed = actual[(token * used + slot) * rows + row];
                require(std::isfinite(observed) && std::abs(observed - expected) < 2e-6,
                        "BF16 indexed matrix expert/tail");
            }
}
inline void intermediate_range(ggml_backend_t backend) {
    for (int columns : {2, 5}) {
        auto ctx = context();
        auto weight = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_BF16, 1, 1);
        auto input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 1, columns);
        auto output = ggml_mul_mat(ctx.get(), weight, input);
        Buffer buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend), ggml_backend_buffer_free);
        require(bool(buffer), "BF16 intermediate range allocation");
        const uint16_t value = 0x4780; // 65536: finite BF16/F32, outside finite F16.
        std::vector<float> ones(columns, 1.0f), actual(columns);
        ggml_backend_tensor_set(weight, &value, 0, sizeof(value));
        ggml_backend_tensor_set(input, ones.data(), 0, ones.size() * sizeof(float));
        compute(backend, ctx.get(), output);
        ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
        for (float observed : actual) {
            if (columns > 4 && !ggml_backend_is_cpu(backend))
                require(std::isinf(observed) && observed > 0, "fast matrix F16 intermediate range");
            else
                require(observed == 65536.0f, "BF16 scalar matvec exponent range");
        }
    }
}
inline void run(ggml_backend_t backend) {
    for (unsigned base : {0U, 1U})
        gather(backend, 65536, base, true);
    for (unsigned width : {1U, 3U, 7U, 16U})
        for (unsigned base : {0U, 1U, 127U, 129U})
            gather(backend, width, base, false);
    for (unsigned k : {1U, 3U, 7U, 16U})
        for (unsigned base : {0U, 1U, 127U})
            for (unsigned columns : {2U, 5U}) {
                matrix(backend, k, base, columns);
                indexed_matrix(backend, k, base, columns == 2 ? 1 : 2);
            }
    intermediate_range(backend);
}
} // namespace bf16_kernel_contract
