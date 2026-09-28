// Synthetic arithmetic/placement regression, NOT trained-model image inference.
// Linked only into test variants/native checks.
#include "core/ggml_extend.h"
#include "core/compute_workspace.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cmath>
#include <memory>
#include <vector>

namespace {
using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;

int check_convolution(ggml_backend_t backend, bool direct, bool force_prec_f32, bool with_bias) {
    Context params(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
    Context graph_ctx(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
    if (!params || !graph_ctx) return -2;

    constexpr int width = 4, height = 3, depth = 2, channels = 2, outputs = 2, kernel = 3;
    constexpr int volume = width * height * depth;
    constexpr int kernel_volume = kernel * kernel * kernel;
    auto weight = ggml_new_tensor_4d(params.get(), GGML_TYPE_F16, kernel, kernel, kernel, channels * outputs);
    auto bias = ggml_new_tensor_1d(params.get(), GGML_TYPE_F32, outputs);
    Buffer buffer(ggml_backend_alloc_ctx_tensors(params.get(), backend), ggml_backend_buffer_free);
    if (!buffer) return -3;
    ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<float> weight_values(kernel_volume * channels * outputs);
    std::vector<ggml_fp16_t> weights(weight_values.size());
    for (size_t i = 0; i < weights.size(); ++i) {
        weight_values[i] = (int(i % 7) - 3) / 8.f;
        weights[i] = ggml_fp32_to_fp16(weight_values[i]);
    }
    const float biases[outputs] = {-0.25f, 0.5f};
    ggml_backend_tensor_set(weight, weights.data(), 0, weights.size() * sizeof(weights[0]));
    ggml_backend_tensor_set(bias, biases, 0, sizeof(biases));

    auto input = ggml_new_tensor_4d(graph_ctx.get(), GGML_TYPE_F32, width, height, depth, channels);
    ggml_set_input(input);
    auto output = ggml_ext_conv_3d(graph_ctx.get(), backend, input, weight, with_bias ? bias : nullptr,
                                 channels, 1, 1, 1, 1, 1, 1, 1, 1, 1, force_prec_f32, direct);
    ggml_set_output(output);
    // The original in-place bias fails this assertion even in CPU-only CI.
    // A scheduler copy of src[0] cannot relocate the output's CPU view_src.
    if (with_bias && (output->op != GGML_OP_ADD || output->view_src != nullptr)) return -4;
    if (output->type != GGML_TYPE_F32 || output->ne[0] != width || output->ne[1] != height ||
        output->ne[2] != depth || output->ne[3] != outputs) return -5;
    auto graph = ggml_new_graph_custom(graph_ctx.get(), 64, false);
    ggml_build_forward_expand(graph, output);
    ggml_tensor* convolution = nullptr;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto node = ggml_graph_node(graph, i);
        if (node->op == GGML_OP_CONV_3D || node->op == GGML_OP_IM2COL_3D) convolution = node;
    }
    // CPU auto uses im2col; pinned WebGPU auto must exercise direct fallback.
    const bool expect_direct = direct || (!ggml_backend_is_cpu(backend) && !force_prec_f32);
    if (!convolution || convolution->op != (expect_direct ? GGML_OP_CONV_3D : GGML_OP_IM2COL_3D)) return -6;

    sd::ComputeWorkspace workspace(backend);
    // Mirror the runner's supported-node assignments, leaving views with their
    // source rather than pinning metadata-only operations to the GPU.
    if (!workspace.allocate(graph, [&](ggml_backend_sched_t scheduler, ggml_cgraph* current) {
            for (int i = 0; i < ggml_graph_n_nodes(current); ++i) {
                auto node = ggml_graph_node(current, i);
                if (node->op == GGML_OP_NONE || node->op == GGML_OP_VIEW || node->op == GGML_OP_RESHAPE ||
                    node->op == GGML_OP_PERMUTE || node->op == GGML_OP_TRANSPOSE) continue;
                if (ggml_backend_supports_op(backend, node)) ggml_backend_sched_set_tensor_backend(scheduler, node, backend);
            }
        })) return -7;
    auto scheduler = workspace.scheduler();
    if (workspace.cpu_backend() != nullptr) ggml_backend_cpu_set_n_threads(workspace.cpu_backend(), 1);
    if (!ggml_backend_is_cpu(backend) &&
        (scheduler == nullptr || ggml_backend_sched_get_tensor_backend(scheduler, convolution) != workspace.cpu_backend() ||
         (with_bias && ggml_backend_sched_get_tensor_backend(scheduler, output) != backend))) return -8;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto node = ggml_graph_node(graph, i);
        auto assigned = scheduler != nullptr ? ggml_backend_sched_get_tensor_backend(scheduler, node) : backend;
        if (!assigned || !node->buffer || !ggml_backend_supports_buft(assigned, ggml_backend_buffer_get_type(node->buffer))) return -9;
    }

    std::vector<float> values(volume * channels), actual(volume * outputs);
    for (size_t i = 0; i < values.size(); ++i) values[i] = (int(i % 17) - 8) / 16.f;
    ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(float));
    auto status = scheduler != nullptr ? ggml_backend_sched_graph_compute(scheduler, graph) : ggml_backend_graph_compute(backend, graph);
    workspace.synchronize();
    if (status != GGML_STATUS_SUCCESS) return -10;
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    // Independent zero-padded convolution reference. Inputs and weights are exact
    // binary fractions, so F16 conversion does not hide indexing/rounding errors.
    for (int oc = 0; oc < outputs; ++oc) {
        for (int z = 0; z < depth; ++z) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    float expected = with_bias ? biases[oc] : 0.f;
                    for (int ic = 0; ic < channels; ++ic) {
                        for (int kz = 0; kz < kernel; ++kz) {
                            for (int ky = 0; ky < kernel; ++ky) {
                                for (int kx = 0; kx < kernel; ++kx) {
                                    const int sx = x + kx - 1, sy = y + ky - 1, sz = z + kz - 1;
                                    if (sx < 0 || sx >= width || sy < 0 || sy >= height || sz < 0 || sz >= depth) continue;
                                    const int input_index = ((ic * depth + sz) * height + sy) * width + sx;
                                    const int weight_index = (((oc * channels + ic) * kernel + kz) * kernel + ky) * kernel + kx;
                                    expected += values[input_index] * weight_values[weight_index];
                                }
                            }
                        }
                    }
                    const float value = actual[((oc * depth + z) * height + y) * width + x];
                    if (!std::isfinite(value) || std::abs(value - expected) > 0.002f) return -11;
                }
            }
        }
    }
    return 1;
}
}  // namespace

extern "C" int sdc_test_conv3d_bias(const char* backend_name) {
    using Backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
    Backend backend(ggml_backend_init_by_name(backend_name, nullptr), ggml_backend_free);
    if (!backend) return -1;
    if (ggml_backend_is_cpu(backend.get())) ggml_backend_cpu_set_n_threads(backend.get(), 1);
    // Direct, backend-selected, and forced-F32 paths all share the bias boundary.
    // The no-bias controls also exercise the original convolution arithmetic.
    for (int path = 0; path < 3; ++path) {
        for (bool with_bias : {false, true}) {
            const int result = check_convolution(backend.get(), path == 0, path == 2, with_bias);
            if (result != 1) return result - 100 * (2 * path + int(with_bias));
        }
    }
    return 1;
}
