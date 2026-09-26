// Synthetic arithmetic/placement regression, NOT trained-model image inference.
// This translation unit is linked only into test variants/native checks.
#include "model/diffusion/qwen_image.hpp"
#include "core/compute_workspace.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

extern "C" int sdc_test_qwen_timestep(const char* backend_name) {
    using Backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
    using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
    using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
    Backend backend(ggml_backend_init_by_name(backend_name, nullptr), ggml_backend_free);
    if (!backend) return -1;
    if (ggml_backend_is_cpu(backend.get())) ggml_backend_cpu_set_n_threads(backend.get(), 1);
    Context params(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
    Context graph_ctx(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
    if (!params || !graph_ctx) return -3;

    constexpr int width = 32;
    // Qwen Image 2.1's timestep block has no bias. The first linear falls back
    // on the pinned WebGPU backend; the SiLU is supported on WebGPU.
    Qwen::TimestepEmbedding model(width, width, 0, 0, false);
    String2TensorStorage types;
    types["probe.linear_1.weight"].type = GGML_TYPE_BF16;
    types["probe.linear_2.weight"].type = GGML_TYPE_F16;
    model.init(params.get(), types, "probe");
    std::map<std::string, ggml_tensor*> weights;
    model.get_param_tensors(weights, "probe");
    Buffer buffer(ggml_backend_alloc_ctx_tensors(params.get(), backend.get()), ggml_backend_buffer_free);
    if (!buffer || weights.size() != 2) return -4;
    ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<ggml_bf16_t> first(width * width);
    std::vector<ggml_fp16_t> second(width * width);
    for (int i = 0; i < width; ++i) {
        first[i * width + i] = ggml_fp32_to_bf16(1.f);
        second[i * width + i] = ggml_fp32_to_fp16(1.f);
    }
    ggml_backend_tensor_set(weights.at("probe.linear_1.weight"), first.data(), 0, first.size() * sizeof(first[0]));
    ggml_backend_tensor_set(weights.at("probe.linear_2.weight"), second.data(), 0, second.size() * sizeof(second[0]));

    GGMLRunnerContext runner;
    runner.ggml_ctx = graph_ctx.get(); runner.backend = backend.get();
    auto input = ggml_new_tensor_2d(graph_ctx.get(), GGML_TYPE_F32, width, 2);
    ggml_set_input(input);
    auto output = model.forward(&runner, input);
    ggml_set_output(output);
    auto graph = ggml_new_graph_custom(graph_ctx.get(), 64, false);
    ggml_build_forward_expand(graph, output);
    ggml_tensor* activation = nullptr;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto node = ggml_graph_node(graph, i);
        if (node->op == GGML_OP_UNARY && ggml_get_unary_op(node) == GGML_UNARY_OP_SILU) activation = node;
    }
    // Fails on the original source even on machines without a GPU. A standalone
    // SiLU output is essential: copying its src[0] does not relocate view_src.
    if (!activation || activation->view_src != nullptr) return -5;
    auto first_linear = activation->src[0];

    sd::ComputeWorkspace workspace(backend.get());
    // Mirror the runner's preferred-backend assignments, including the CPU
    // fallback caused by the original BF16 first-linear weights.
    if (!workspace.allocate(graph, [&](ggml_backend_sched_t scheduler, ggml_cgraph* current) {
            for (int i = 0; i < ggml_graph_n_nodes(current); ++i) {
                auto node = ggml_graph_node(current, i);
                if (ggml_backend_supports_op(backend.get(), node)) ggml_backend_sched_set_tensor_backend(scheduler, node, backend.get());
            }
        })) return -7;
    auto scheduler = workspace.scheduler();
    if (workspace.cpu_backend() != nullptr) ggml_backend_cpu_set_n_threads(workspace.cpu_backend(), 1);
    if (!ggml_backend_is_cpu(backend.get()) &&
        (scheduler == nullptr || ggml_backend_sched_get_tensor_backend(scheduler, first_linear) != workspace.cpu_backend() ||
         ggml_backend_sched_get_tensor_backend(scheduler, activation) != backend.get())) return -8;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto node = ggml_graph_node(graph, i);
        auto assigned = scheduler != nullptr ? ggml_backend_sched_get_tensor_backend(scheduler, node) : backend.get();
        if (!assigned || !node->buffer || !ggml_backend_supports_buft(assigned, ggml_backend_buffer_get_type(node->buffer))) return -9;
    }
    std::vector<float> values(width * 2), actual(values.size());
    for (size_t i = 0; i < values.size(); ++i) values[i] = (int(i) - width) / 16.f;
    ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(float));
    auto status = scheduler != nullptr ? ggml_backend_sched_graph_compute(scheduler, graph) : ggml_backend_graph_compute(backend.get(), graph);
    workspace.synchronize();
    if (status != GGML_STATUS_SUCCESS) return -10;
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    for (size_t i = 0; i < values.size(); ++i) {
        float expected = values[i] / (1.f + std::exp(-values[i]));
        if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected) > 0.002f) return -11;
    }
    return 1;
}
