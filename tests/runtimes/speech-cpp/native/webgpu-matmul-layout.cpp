#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-webgpu-mul-mat-layout.hpp"
#include "ggml.h"
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
void check(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void cache_view_contract(int64_t capacity, int64_t total) {
    constexpr int64_t channels = 256, kv_heads = 4, query_heads = 16;
    const std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({2 * 1024 * 1024, nullptr, true}),
                                                                  ggml_free);
    const std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(),
                                                                              ggml_backend_free);
    check(ctx && backend, "cannot create the CPU fixture");
    ggml_backend_cpu_set_n_threads(backend.get(), 1);
    auto * cache = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, capacity, channels, kv_heads);
    auto * values = ggml_view_3d(ctx.get(), cache, total, channels, kv_heads, cache->nb[1], cache->nb[2], 0);
    auto * scores = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, total, 1, query_heads);
    auto * result = ggml_mul_mat(ctx.get(), values, scores);
    ggml_prec_set_acc(result, GGML_PREC_F32);
    auto * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, result);
    const std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()), ggml_backend_buffer_free);
    check(bool(buffer), "cannot allocate the CPU fixture");
    std::vector<float> data(static_cast<size_t>(capacity * channels * kv_heads));
    for (int64_t h = 0; h < kv_heads; ++h) {
        for (int64_t c = 0; c < channels; ++c) {
            for (int64_t p = 0; p < capacity; ++p) {
                data[static_cast<size_t>((h * channels + c) * capacity + p)] =
                    static_cast<float>(10000 * h + 10 * c + p);
            }
        }
    }
    std::vector<float> attention(static_cast<size_t>(total * query_heads));
    for (int64_t h = 0; h < query_heads; ++h) {
        for (int64_t p = 0; p < total; ++p) {
            attention[static_cast<size_t>(h * total + p)] = static_cast<float>(p + 1);
        }
    }
    ggml_backend_tensor_set(cache, data.data(), 0, ggml_nbytes(cache));
    ggml_backend_tensor_set(scores, attention.data(), 0, ggml_nbytes(scores));
    check(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS, "CPU matrix product failed");
    std::vector<float> cpu(static_cast<size_t>(channels * query_heads));
    ggml_backend_tensor_get(result, cpu.data(), 0, ggml_nbytes(result));

    // The real patched backend predicate must preserve every row address when
    // its selected scalar/vec4 representation is evaluated. This simulates
    // addressing only; the native test does not execute a WebGPU shader.
    const bool aligned = ggml_webgpu_mul_mat_vec4_aligned(values, 0);
    check(aligned == (capacity % 4 == 0), "V cache row stride was not considered");
    const bool vectorized = total % 4 == 0 && aligned;
    for (int64_t h = 0; h < query_heads; ++h) {
        for (int64_t c = 0; c < channels; ++c) {
            const int64_t kv = h / (query_heads / kv_heads);
            const int64_t row = (kv * channels + c) * capacity;
            const int64_t addressed = vectorized ? row / 4 * 4 : row;
            float sum = 0;
            for (int64_t p = 0; p < total; ++p) {
                sum += data[static_cast<size_t>(addressed + p)] * attention[static_cast<size_t>(h * total + p)];
            }
            check(sum == cpu[static_cast<size_t>(h * channels + c)], "selected matrix representation changed a V row");
        }
    }
}

void offset_and_type_contract() {
    const std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({65536, nullptr, true}), ggml_free);
    check(bool(ctx), "cannot create tensor descriptors");
    auto * half = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, 8, 2);
    check(ggml_webgpu_mul_mat_vec4_aligned(half, 4), "aligned half rows lost vec4 eligibility");
    check(!ggml_webgpu_mul_mat_vec4_aligned(half, 1), "an unaligned wire offset was accepted");
    auto * half_view = ggml_view_2d(ctx.get(), half, 4, 2, 10, 0);
    check(!ggml_webgpu_mul_mat_vec4_aligned(half_view, 0), "an unaligned half row stride was accepted");
    auto * singleton = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 8, 1, 2, 1);
    singleton->nb[1] = 3;
    check(ggml_webgpu_mul_mat_vec4_aligned(singleton, 0), "an unused singleton stride rejected vec4");
    auto * quant = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q4_K, 256, 2);
    check(!ggml_webgpu_mul_mat_vec4_aligned(quant, 0), "a quantized tensor was treated as four scalar floats");
}
} // namespace

int main() {
    try {
        cache_view_contract(13, 4);
        cache_view_contract(16, 4);
        cache_view_contract(13, 5);
        offset_and_type_contract();
        std::cout << "WebGPU matrix layout selection matches CPU row addressing\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
