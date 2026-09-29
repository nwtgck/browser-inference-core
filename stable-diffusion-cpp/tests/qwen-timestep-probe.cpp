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

namespace {
    // Exercise the real measurement clone and support callbacks without reading
    // model data. A second CPU backend forces the scheduler measurement path on
    // CPU-only CI as well as the optional WebGPU run of this existing probe.
    bool measurement_preserves_addresses(ggml_backend_t backend) {
        using Backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
        using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
        using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
        Backend peer(ggml_backend_cpu_init(), ggml_backend_free);
        if (!peer) return false;
        for (auto type : {GGML_TYPE_F16, GGML_TYPE_F32}) {
            std::vector<size_t> resident_sizes;
            for (bool resident : {true, false}) {
                Context params(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
                Context computation(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
                if (!params || !computation) return false;
                // Place the real weight after another allocation, and use a row
                // view as well, so retaining only an aligned base is insufficient.
                ggml_new_tensor_1d(params.get(), GGML_TYPE_F32, 16);
                auto weight = ggml_new_tensor_2d(params.get(), type, 16, 1024);
                Buffer buffer(resident ? ggml_backend_alloc_ctx_tensors(params.get(), backend) : nullptr,
                              ggml_backend_buffer_free);
                if (resident && !buffer) return false;
                if (buffer) ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                auto view = ggml_view_2d(computation.get(), weight, 16, 2, weight->nb[1], weight->nb[1]);
                auto ids = ggml_new_tensor_1d(computation.get(), GGML_TYPE_I32, 2);
                auto direct_rows = ggml_get_rows(computation.get(), weight, ids);
                auto view_rows = ggml_get_rows(computation.get(), view, ids);
                auto result = ggml_add(computation.get(), direct_rows, view_rows);
                auto graph = ggml_new_graph_custom(computation.get(), 32, false);
                ggml_build_forward_expand(graph, result);
                const auto original_data = weight->data;
                const auto original_buffer = weight->buffer;
                const auto original_view_data = view->data;
                const auto original_view_offset = view->view_offs;
                const bool direct_supported = ggml_backend_supports_op(backend, direct_rows);
                const bool view_supported = ggml_backend_supports_op(backend, view_rows);
                if (!direct_supported || !view_supported) return false;

                sd::ComputeWorkspace workspace(backend);
                workspace.set_extra_backends({peer.get()});
                bool valid = true;
                int checked = 0;
                const auto measurement = workspace.measure(graph, 0,
                    [&](const ggml_tensor* source) { return source == weight ? backend : nullptr; },
                    [&](ggml_backend_sched_t scheduler, ggml_cgraph* copy) {
                        for (int i = 0; i < ggml_graph_n_nodes(copy); ++i) {
                            auto node = ggml_graph_node(copy, i);
                            if (node->op == GGML_OP_GET_ROWS) {
                                auto source = node->src[0];
                                const bool is_view = source->view_src != nullptr;
                                auto root = is_view ? source->view_src : source;
                                valid = valid && root != weight && root->data == original_data &&
                                    root->buffer != nullptr &&
                                    ggml_backend_buffer_get_size(root->buffer) == 0 &&
                                    ggml_backend_buffer_get_usage(root->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                                    source->type == type && source->ne[0] == 16 &&
                                    source->nb[1] == weight->nb[1] &&
                                    (!is_view || (source->view_offs == original_view_offset && source->data == original_view_data));
                                // Keep a regression against the old pointer=1
                                // sentinel bounded: record failure before repairing
                                // only this test clone for its remaining cleanup.
                                if (root->data != original_data) root->data = original_data;
                                const bool expected = is_view ? view_supported : direct_supported;
                                valid = valid && ggml_backend_supports_op(backend, node) == expected;
                                ++checked;
                            }
                            if (ggml_backend_supports_op(backend, node)) {
                                ggml_backend_sched_set_tensor_backend(scheduler, node, backend);
                            }
                        }
                    });
                if (!valid || checked != 2 || !measurement.scheduler || measurement.buffers.empty() ||
                    weight->data != original_data || weight->buffer != original_buffer ||
                    view->data != original_view_data || view->view_src != weight || view->view_offs != original_view_offset ||
                    direct_rows->src[0] != weight || view_rows->src[0] != view) return false;
                std::vector<size_t> sizes;
                for (const auto& entry : measurement.buffers) sizes.push_back(entry.bytes);
                if (resident) resident_sizes = sizes;
                // Null data still denotes an external weight through its dummy
                // buffer: it must not add the weight to the compute reservation.
                else if (sizes != resident_sizes) return false;
            }
        }
        return true;
    }
}

extern "C" int sdc_test_qwen_timestep(const char* backend_name) {
    using Backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
    using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
    using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
    Backend backend(ggml_backend_init_by_name(backend_name, nullptr), ggml_backend_free);
    if (!backend) return -1;
    if (ggml_backend_is_cpu(backend.get())) ggml_backend_cpu_set_n_threads(backend.get(), 1);
    if (!measurement_preserves_addresses(backend.get())) return -12;
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
