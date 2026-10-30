#include "ggml-webgpu-rms-scale.hpp"
#include "ggml-webgpu-scale-unary.hpp"
#include "ggml-alloc.h"
#include "ggml-backend-impl.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
#include "fixture.cpp"
static size_t checks = 0;
static void check(bool condition) {
    ++checks;
    if (!condition)
        throw std::runtime_error("RMS admission check " + std::to_string(checks));
}
static void free_memory(ggml_backend_buffer_t b) {
    std::free(b->context);
}
static void * base_memory(ggml_backend_buffer_t b) {
    return b->context;
}
static ggml_backend_buffer_t alloc_memory(ggml_backend_buffer_type_t type, size_t bytes) {
    ggml_backend_buffer_i iface{};
    iface.free_buffer = free_memory;
    iface.get_base = base_memory;
    void * memory = std::aligned_alloc(256, (bytes + 255) / 256 * 256);
    if (!memory)
        throw std::bad_alloc();
    return ggml_backend_buffer_init(type, iface, memory, bytes);
}
static const char * memory_name(ggml_backend_buffer_type_t) {
    return "rms-scale-test-memory";
}
static size_t memory_alignment(ggml_backend_buffer_type_t) {
    return 256;
}
static ggml_backend_buffer_type memory_type = [] {
    ggml_backend_buffer_type t{};
    t.iface.get_name = memory_name;
    t.iface.get_alignment = memory_alignment;
    t.iface.alloc_buffer = alloc_memory;
    return t;
}();
static uint8_t * row_address(ggml_tensor * t, int r) {
    const int i1 = r % t->ne[1], i2 = (r / t->ne[1]) % t->ne[2], i3 = r / (t->ne[1] * t->ne[2]);
    return static_cast<uint8_t *>(t->data) + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
}
int main() {
    try {
        for (int width : {1, 7, 127, 128, 129, 257})
            for (int heads : {1, 3, 16})
                for (int mode : {0, 1, 2}) {
                    fixture f(width, heads, 3, 2, mode);
                    rms_scale_plan plan;
                    auto admitted = [&] { return plan_rms_scale(f.graph, f.index, plan); };
                    auto reject = [&] { check(!admitted()); };
                    check(admitted());
                    check(plan.source == f.source && plan.rms == f.rms && plan.output == f.output);
                    check(plan.elements == uint32_t(width * heads * 3 * 2) && plan.rows == uint32_t(heads * 3 * 2));
                    check(plan.source_bytes == ggml_nbytes(f.source) && plan.output_bytes == ggml_nbytes(f.output));
                    const auto flags = f.rms->flags;
                    ggml_set_output(f.rms);
                    reject();
                    f.rms->flags = flags;
                    f.rms->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
                    reject();
                    f.rms->flags = flags;
                    const auto outflags = f.output->flags;
                    f.output->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
                    reject();
                    f.output->flags = outflags;
                    f.rms->view_src = f.source;
                    reject();
                    f.rms->view_src = nullptr;
                    f.output->view_src = f.rms;
                    reject();
                    f.output->view_src = nullptr;
                    const auto type = f.source->type;
                    f.source->type = GGML_TYPE_F16;
                    reject();
                    f.source->type = type;
                    auto nb = f.source->nb[0];
                    f.source->nb[0] = 8;
                    reject();
                    f.source->nb[0] = nb;
                    nb = f.source->nb[1];
                    f.source->nb[1] = size_t(width * 4 - 1);
                    reject();
                    f.source->nb[1] = nb;
                    nb = f.source->nb[2];
                    f.source->nb[2] = 0;
                    reject();
                    f.source->nb[2] = nb;
                    nb = f.output->nb[2];
                    f.output->nb[2] += 4;
                    reject();
                    f.output->nb[2] = nb;
                    f.rms->src[1] = f.source;
                    reject();
                    f.rms->src[1] = nullptr;
                    f.output->src[2] = f.source;
                    reject();
                    f.output->src[2] = nullptr;
                    f.output->src[0] = f.source;
                    reject();
                    f.output->src[0] = f.rms;
                    check(!plan_rms_scale(f.graph, -1, plan));
                    check(!plan_rms_scale(f.graph, f.graph->n_nodes - 1, plan));
                    auto * side = ggml_scale(f.context, f.rms, 2);
                    ggml_build_forward_expand(f.graph, side);
                    reject();
                }
        {
            fixture f(128, 16, 1, 1, 2);
            rms_scale_plan plan;
            auto * inplace = ggml_scale_inplace(f.context, f.rms, 0.5f);
            auto * graph = ggml_new_graph_custom(f.context, 64, false);
            ggml_build_forward_expand(graph, inplace);
            int index = -1;
            for (int i = 0; i < graph->n_nodes; i++)
                if (graph->nodes[i] == f.rms)
                    index = i;
            check(index >= 0 && !plan_rms_scale(graph, index, plan));
            ggml_set_output(f.source);
            check(plan_rms_scale(f.graph, f.index, plan));
        }
        {
            fixture f(128, 2, 3, 1);
            rms_scale_plan plan;
            f.output->ne[1] = 3;
            f.output->ne[2] = 2;
            f.output->nb[2] = 128 * 3 * 4;
            f.output->nb[3] = 128 * 3 * 2 * 4;
            check(ggml_nelements(f.output) == ggml_nelements(f.rms));
            check(ggml_nrows(f.output) == ggml_nrows(f.rms));
            check(!plan_rms_scale(f.graph, f.index, plan));
        }
        for (bool sigmoid : {false, true}) {
            fixture f(128, 16, 1, 1, 2);
            rms_scale_plan rms_plan;
            scale_unary_plan longer;
            f.output->flags &= ~GGML_TENSOR_FLAG_OUTPUT;
            auto * unary = sigmoid ? ggml_sigmoid(f.context, f.output) : ggml_silu(f.context, f.output);
            auto * final = ggml_scale_bias(f.context, unary, 0.7f, -0.2f);
            ggml_set_output(final);
            ggml_build_forward_expand(f.graph, final);
            check(plan_rms_scale(f.graph, f.index, rms_plan));
            check(plan_scale_unary(f.graph, f.index + 1, 3, longer) && longer.output == final);
            ggml_set_output(unary);
            check(!plan_scale_unary(f.graph, f.index + 1, 3, longer));
            unary->flags &= ~GGML_TENSOR_FLAG_OUTPUT;
            auto * side = ggml_scale(f.context, unary, 2);
            ggml_build_forward_expand(f.graph, side);
            check(!plan_scale_unary(f.graph, f.index + 1, 3, longer));
        }
        // Exact scalar limits and span arithmetic are checked without huge allocations.
        {
            ggml_tensor t{};
            t.type = GGML_TYPE_F32;
            for (int i = 0; i < 4; i++)
                t.ne[i] = 1;
            t.ne[0] = UINT32_MAX - 127u;
            t.nb[0] = 4;
            for (int i = 1; i < 4; i++)
                t.nb[i] = size_t(t.ne[0]) * 4;
            uint64_t span;
            uint32_t elements, rows;
            check(rms_scale_layout(&t, true, span, elements, rows) && elements == UINT32_MAX - 127u && rows == 1);
            t.ne[0]++;
            check(!rms_scale_layout(&t, false, span, elements, rows));
            t.ne[0] = 128;
            t.ne[1] = 2;
            t.nb[1] = uint64_t(UINT32_MAX) * 4;
            check(!rms_scale_layout(&t, false, span, elements, rows));
            t.nb[1] = 512;
            t.nb[2] = t.nb[3] = 1024;
            t.ne[2] = UINT32_MAX;
            check(!rms_scale_layout(&t, false, span, elements, rows));
            t.ne[2] = 2;
            t.nb[2] = UINT64_MAX - 3;
            check(!rms_scale_layout(&t, false, span, elements, rows));
        }
        int aliases = 0, disjoint = 0, views = 0;
        for (int mode : {0, 1, 2})
            for (bool produced : {false, true})
                for (bool preserve : {false, true})
                    for (int width : {7, 128, 129}) {
                        fixture f(width, 3, 2, 2, mode, produced, preserve);
                        auto * allocator = ggml_gallocr_new(&memory_type);
                        check(ggml_gallocr_alloc_graph(allocator, f.graph));
                        rms_scale_plan plan;
                        check(plan_rms_scale(f.graph, f.index, plan));
                        const auto a = reinterpret_cast<uintptr_t>(f.source->data),
                                   b = reinterpret_cast<uintptr_t>(f.output->data);
                        const bool inplace = a == b && rms_scale_same_mapping(f.source, f.output);
                        check(inplace || rms_scale_disjoint(a, plan.source_bytes, b, plan.output_bytes));
                        if (inplace)
                            aliases++;
                        else
                            disjoint++;
                        if (mode)
                            views++;
                        if (preserve)
                            check(!inplace);
                        std::vector<float> input(plan.elements), expected(plan.elements);
                        for (uint32_t i = 0; i < plan.elements; i++)
                            input[i] = float(int(i % 37) - 18) * 0.125f;
                        for (uint32_t r = 0; r < plan.rows; r++)
                            std::memcpy(row_address(f.source, r), input.data() + r * width, width * 4);
                        for (uint32_t r = 0; r < plan.rows; r++) {
                            const auto * src = reinterpret_cast<const float *>(row_address(f.source, r));
                            float sum = 0;
                            for (int c = 0; c < width; c++)
                                sum += src[c] * src[c];
                            const float scale = 1.0f / std::sqrt(sum / width + 1e-5f / float(width));
                            auto * dst = reinterpret_cast<float *>(row_address(f.rms, r));
                            for (int c = 0; c < width; c++)
                                dst[c] = scale * (src[c] - 0.0f);
                        }
                        auto * out = static_cast<float *>(f.output->data);
                        const auto * middle = static_cast<const float *>(f.rms->data);
                        for (uint32_t i = 0; i < plan.elements; i++)
                            out[i] = middle[i] * -0.3f + 0.07f;
                        std::memcpy(expected.data(), out, plan.output_bytes);
                        for (uint32_t r = 0; r < plan.rows; r++)
                            std::memcpy(row_address(f.source, r), input.data() + r * width, width * 4);
                        for (uint32_t r = 0; r < plan.rows; r++) {
                            const auto * src = reinterpret_cast<const float *>(row_address(f.source, r));
                            float sum = 0;
                            for (int c = 0; c < width; c++)
                                sum += src[c] * src[c];
                            const float scale = 1.0f / std::sqrt(sum / width + 1e-5f / float(width));
                            for (int c = 0; c < width; c++) {
                                const float normalized = scale * (src[c] - 0.0f);
                                out[r * width + c] = normalized * -0.3f + 0.07f;
                            }
                        }
                        check(std::memcmp(expected.data(), out, plan.output_bytes) == 0);
                        ggml_gallocr_free(allocator);
                    }
        check(aliases > 0 && disjoint > 0 && views > 0);
        for (uint32_t limit : {1u, 2u, 3u, 257u, 65535u})
            for (uint64_t rows : {1ull, 2ull, 3ull, 257ull, 65536ull, 0xffffffffull}) {
                uint32_t x, y;
                const bool ok = rms_scale_grid(rows, limit, x, y);
                if (ok)
                    check(x <= limit && y <= limit && uint64_t(x) * y >= rows && uint64_t(x) * y <= UINT32_MAX);
                else
                    check(x == 0 && y == 0);
            }
        uint32_t x, y;
        check(!rms_scale_grid(0, 1, x, y));
        check(!rms_scale_grid(1, 0, x, y));
        std::cout << checks << " actual RMS/scale graph/layout/allocator checks passed; aliases=" << aliases
                  << " disjoint=" << disjoint << " views=" << views << '\n';
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
