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

static int checks = 0;
static void check(bool condition) {
    ++checks;
    if (!condition)
        throw std::runtime_error("admission check " + std::to_string(checks));
}
struct fixture {
    ggml_context * context = ggml_init({1 << 20, nullptr, true});
    ggml_cgraph * graph;
    ggml_tensor *source, *first, *unary, *output;
    int index;
    fixture(int width, int tokens, bool sigmoid, bool final, bool produced = false, bool preserve_source = false) {
        if (!context)
            throw std::bad_alloc();
        graph = ggml_new_graph_custom(context, 128, false);
        source = ggml_new_tensor_2d(context, GGML_TYPE_F32, width, tokens);
        ggml_set_input(source);
        if (produced)
            source = ggml_scale(context, source, 1.125f);
        if (preserve_source)
            ggml_set_output(source);
        first = ggml_scale_bias(context, source, 0.25f, 0.125f);
        unary = sigmoid ? ggml_sigmoid(context, first) : ggml_silu(context, first);
        output = final ? ggml_scale_bias(context, unary, 2.0f, -0.125f) : unary;
        ggml_set_output(output);
        ggml_build_forward_expand(graph, output);
        index = -1;
        for (int i = 0; i < graph->n_nodes; ++i)
            if (graph->nodes[i] == first)
                index = i;
        check(index >= 0);
    }
    ~fixture() {
        ggml_free(context);
    }
};
static void free_memory(ggml_backend_buffer_t buffer) {
    std::free(buffer->context);
}
static void * base_memory(ggml_backend_buffer_t buffer) {
    return buffer->context;
}
static ggml_backend_buffer_t alloc_memory(ggml_backend_buffer_type_t type, size_t size) {
    ggml_backend_buffer_i iface{};
    iface.free_buffer = free_memory;
    iface.get_base = base_memory;
    void * memory = std::aligned_alloc(256, ((size + 255) / 256) * 256);
    if (!memory)
        throw std::bad_alloc();
    return ggml_backend_buffer_init(type, iface, memory, size);
}
static const char * memory_name(ggml_backend_buffer_type_t) {
    return "scale-unary-test-memory";
}
static size_t memory_alignment(ggml_backend_buffer_type_t) {
    return 256;
}
static ggml_backend_buffer_type memory_type = [] {
    ggml_backend_buffer_type type{};
    type.iface.get_name = memory_name;
    type.iface.get_alignment = memory_alignment;
    type.iface.alloc_buffer = alloc_memory;
    return type;
}();
static float activate(float value, bool sigmoid) {
    return sigmoid ? 1.0f / (1.0f + std::exp(-value)) : value / (1.0f + std::exp(-value));
}
static bool same_float(float a, float b) {
    if (std::isnan(a) || std::isnan(b))
        return std::isnan(a) && std::isnan(b);
    uint32_t x, y;
    std::memcpy(&x, &a, 4);
    std::memcpy(&y, &b, 4);
    return x == y;
}
int main() {
    for (int width : {1, 7, 127, 128, 129, 257})
        for (int tokens : {1, 3})
            for (bool sigmoid : {false, true})
                for (bool final : {false, true}) {
                    fixture f(width, tokens, sigmoid, final);
                    scale_unary_plan plan;
                    check(plan_scale_unary(f.graph, f.index, final ? 3 : 2, plan));
                    check(plan.source == f.source && plan.output == f.output &&
                          plan.elements == uint32_t(width * tokens));
                    check(plan.operations == (final ? 3 : 2));
                    auto fail = [&] { check(!plan_scale_unary(f.graph, f.index, final ? 3 : 2, plan)); };
                    const auto first_flags = f.first->flags;
                    ggml_set_output(f.first);
                    fail();
                    f.first->flags = first_flags;
                    f.first->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
                    fail();
                    f.first->flags = first_flags;
                    const auto unary_flags = f.unary->flags;
                    f.unary->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
                    fail();
                    f.unary->flags = unary_flags;
                    const auto stride = f.source->nb[1];
                    f.source->nb[1] += 4;
                    fail();
                    f.source->nb[1] = stride;
                    const auto type = f.source->type;
                    f.source->type = GGML_TYPE_F16;
                    fail();
                    f.source->type = type;
                    f.source->view_src = f.first;
                    fail();
                    f.source->view_src = nullptr;
                    f.output->view_src = f.source;
                    fail();
                    f.output->view_src = nullptr;
                    f.first->src[1] = f.source;
                    fail();
                    f.first->src[1] = nullptr;
                    f.unary->src[0] = f.source;
                    f.unary->src[1] = f.first;
                    fail();
                    f.unary->src[0] = f.first;
                    f.unary->src[1] = nullptr;
                    int32_t operation;
                    std::memcpy(&operation, f.unary->op_params, 4);
                    int32_t unsupported = GGML_UNARY_OP_TANH;
                    std::memcpy(f.unary->op_params, &unsupported, 4);
                    fail();
                    std::memcpy(f.unary->op_params, &operation, 4);
                    check(!plan_scale_unary(f.graph, -1, 2, plan));
                    check(!plan_scale_unary(f.graph, f.graph->n_nodes, 2, plan));
                    check(!plan_scale_unary(f.graph, f.index, 1, plan));
                    check(!plan_scale_unary(f.graph, f.index, 4, plan));
                    auto * side = ggml_scale(f.context, f.first, 2);
                    ggml_build_forward_expand(f.graph, side);
                    fail();
                }
    // An observed unary output may terminate the two-op fusion, but cannot be skipped by a three-op fusion.
    {
        fixture f(129, 1, true, true);
        scale_unary_plan plan;
        ggml_set_output(f.unary);
        check(!plan_scale_unary(f.graph, f.index, 3, plan));
        check(plan_scale_unary(f.graph, f.index, 2, plan) && plan.output == f.unary);
    }
    {
        fixture f(129, 1, true, true);
        scale_unary_plan plan;
        auto * side = ggml_scale(f.context, f.unary, 3);
        ggml_build_forward_expand(f.graph, side);
        check(!plan_scale_unary(f.graph, f.index, 3, plan));
        check(plan_scale_unary(f.graph, f.index, 2, plan));
    }
    // Explicit inplace API nodes are views, and deliberately remain on the ordinary path.
    {
        fixture f(128, 1, false, false);
        scale_unary_plan plan;
        auto * inplace = ggml_silu_inplace(f.context, f.first);
        auto * graph = ggml_new_graph_custom(f.context, 128, false);
        ggml_build_forward_expand(graph, inplace);
        check(!plan_scale_unary(graph, 0, 2, plan));
    }
    // Real dynamic allocator: first-SCALE input remains readable at fusion start,
    // including produced inputs and allocator-inplace outputs. There is no intervening op.
    int aliases = 0, disjoint = 0;
    for (bool produced : {false, true})
        for (bool preserve : {false, true})
            for (int width : {1, 7, 129, 257})
                for (bool sigmoid : {false, true})
                    for (bool final : {false, true}) {
                        fixture f(width, 3, sigmoid, final, produced, preserve);
                        scale_unary_plan plan;
                        auto * allocator = ggml_gallocr_new(&memory_type);
                        check(ggml_gallocr_alloc_graph(allocator, f.graph));
                        check(plan_scale_unary(f.graph, f.index, final ? 3 : 2, plan));
                        const auto a = reinterpret_cast<uintptr_t>(plan.source->data),
                                   b = reinterpret_cast<uintptr_t>(plan.output->data);
                        const auto bytes = ggml_nbytes(plan.source);
                        check(a == b || scale_unary_disjoint(a, bytes, b, bytes));
                        if (a == b)
                            ++aliases;
                        else
                            ++disjoint;
                        std::vector<float> input(plan.elements), expected(plan.elements);
                        for (size_t i = 0; i < input.size(); ++i)
                            input[i] = float(int(i % 37) - 18) * 0.125f;
                        std::memcpy(plan.source->data, input.data(), bytes);
                        auto * src = static_cast<float *>(plan.source->data);
                        auto * first = static_cast<float *>(f.first->data);
                        auto * unary = static_cast<float *>(f.unary->data);
                        auto * output = static_cast<float *>(plan.output->data);
                        for (uint32_t i = 0; i < plan.elements; ++i)
                            first[i] = src[i] * 0.25f + 0.125f;
                        for (uint32_t i = 0; i < plan.elements; ++i)
                            unary[i] = activate(first[i], sigmoid);
                        if (final)
                            for (uint32_t i = 0; i < plan.elements; ++i)
                                output[i] = unary[i] * 2.0f - 0.125f;
                        std::memcpy(expected.data(), output, bytes);
                        std::memcpy(plan.source->data, input.data(), bytes);
                        for (uint32_t i = 0; i < plan.elements; ++i) {
                            const float scaled = src[i] * 0.25f + 0.125f;
                            const float activated = activate(scaled, sigmoid);
                            output[i] = final ? activated * 2.0f - 0.125f : activated;
                        }
                        for (uint32_t i = 0; i < plan.elements; ++i)
                            check(same_float(output[i], expected[i]));
                        ggml_gallocr_free(allocator);
                    }
    check(aliases > 0 && disjoint > 0);
    for (uint32_t limit : {1u, 2u, 3u, 7u, 65535u})
        for (uint64_t count : {1ull, 127ull, 128ull, 129ull, 257ull, 4096ull, 0xffffffffull}) {
            uint32_t x, y;
            if (scale_unary_grid(count, limit, x, y)) {
                check(x <= limit && y <= limit && uint64_t(x) * y * 128 >= count &&
                      uint64_t(x) * y * 128 <= UINT32_MAX);
            } else
                check(x == 0 && y == 0);
        }
    uint32_t x, y;
    check(!scale_unary_grid(0, 1, x, y));
    check(!scale_unary_grid(1, 0, x, y));
    check(scale_unary_disjoint(UINT64_MAX - 128, 128, 0, 128));
    check(!scale_unary_disjoint(0, 129, 128, 128));
    check(!scale_unary_disjoint(0, 0, 0, 0));
    std::cout << checks << " actual graph/admission/allocator checks passed; aliases=" << aliases
              << " disjoint=" << disjoint << '\n';
}
