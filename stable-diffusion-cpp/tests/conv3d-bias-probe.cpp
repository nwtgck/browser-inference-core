// Synthetic arithmetic/placement regression, NOT trained-model image inference.
// Linked only into test variants/native checks.
#include "core/ggml_extend.h"
#include "core/compute_workspace.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include <cmath>
#include <memory>
#include <vector>

namespace {
using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;

struct ConvolutionCase {
    int depth = 2;
    int stride = 1;
    int padding = 1;
    int dilation = 1;
    int batches = 1;
    ggml_type weight_type = GGML_TYPE_F16;
    bool emulate_fallback = false;
    bool reject_2d = false;
    bool expect_2d = false;
};

// Only used while building the graph. CPU CI can select the browser fallback
// branch without pretending to execute on a GPU. Allocation/compute always use
// the real backend. WebGPU positive cases use its real capability checks.
struct CapabilityFilter {
    ggml_backend_t real;
    bool reject_2d;
    ggml_backend_device device = {};
    ggml_backend backend;

    CapabilityFilter(ggml_backend_t source, bool reject) : real(source), reject_2d(reject), backend(*source) {
        device.context = this;
        device.iface.supports_op = [](ggml_backend_dev_t dev, const ggml_tensor* op) {
            const auto* filter = static_cast<CapabilityFilter*>(dev->context);
            if (op->op == GGML_OP_IM2COL_3D || (filter->reject_2d && op->op == GGML_OP_CONV_2D)) return false;
            return ggml_backend_supports_op(filter->real, op);
        };
        backend.device = &device;
    }
};

int check_convolution(ggml_backend_t backend, bool direct, bool force_prec_f32, bool with_bias,
                      ConvolutionCase test = {}) {
    Context params(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
    Context graph_ctx(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
    if (!params || !graph_ctx) return -2;

    constexpr int width = 4, height = 3, channels = 2, outputs = 3, kernel = 3;
    const int depth = test.depth;
    const int output_depth = (depth + 2 * test.padding - test.dilation * (kernel - 1) - 1) / test.stride + 1;
    const int volume = width * height * depth;
    const int output_volume = width * height * output_depth;
    constexpr int kernel_volume = kernel * kernel * kernel;
    auto weight = ggml_new_tensor_4d(params.get(), test.weight_type, kernel, kernel, kernel, channels * outputs);
    auto bias = ggml_new_tensor_1d(params.get(), GGML_TYPE_F32, outputs);
    Buffer buffer(ggml_backend_alloc_ctx_tensors(params.get(), backend), ggml_backend_buffer_free);
    if (!buffer) return -3;
    ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<float> weight_values(kernel_volume * channels * outputs);
    std::vector<ggml_fp16_t> weights(weight_values.size());
    for (size_t i = 0; i < weights.size(); ++i) {
        weight_values[i] = (int(i % 7) - 3) / 8.f;
        if (test.weight_type == GGML_TYPE_F32) weight_values[i] += 0.0000371f;
        weights[i] = ggml_fp32_to_fp16(weight_values[i]);
    }
    const float biases[outputs] = {-0.25f, 0.5f, 0.125f};
    if (test.weight_type == GGML_TYPE_F16) {
        ggml_backend_tensor_set(weight, weights.data(), 0, weights.size() * sizeof(weights[0]));
    } else {
        ggml_backend_tensor_set(weight, weight_values.data(), 0, weight_values.size() * sizeof(float));
    }
    ggml_backend_tensor_set(bias, biases, 0, sizeof(biases));

    auto input = ggml_new_tensor_4d(graph_ctx.get(), GGML_TYPE_F32, width, height, depth, channels * test.batches);
    ggml_set_input(input);
    CapabilityFilter filter(backend, test.reject_2d);
    const bool filtered = test.emulate_fallback && (ggml_backend_is_cpu(backend) || test.reject_2d);
    auto output = ggml_ext_conv_3d(graph_ctx.get(), filtered ? &filter.backend : backend, input, weight, with_bias ? bias : nullptr,
                                 channels, 1, 1, test.stride, 1, 1, test.padding, 1, 1, test.dilation, force_prec_f32, direct);
    ggml_set_output(output);
    // The original in-place bias fails this assertion even in CPU-only CI.
    // A scheduler copy of src[0] cannot relocate the output's CPU view_src.
    if (with_bias && (output->op != GGML_OP_ADD || output->view_src != nullptr)) return -4;
    if (output->type != GGML_TYPE_F32 || output->ne[0] != width || output->ne[1] != height ||
        output->ne[2] != output_depth || output->ne[3] != outputs * test.batches) return -5;
    auto graph = ggml_new_graph_custom(graph_ctx.get(), 64, false);
    ggml_build_forward_expand(graph, output);
    ggml_tensor* convolution = nullptr;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto node = ggml_graph_node(graph, i);
        if (node->op == GGML_OP_CONV_3D || node->op == GGML_OP_IM2COL_3D || node->op == GGML_OP_CONV_2D) {
            if (convolution != nullptr) return -6;
            convolution = node;
        }
        if (node->op == GGML_OP_CPY && node->src[0]->type != node->type) return -12;
    }
    // CPU auto normally uses im2col. Filtered CPU and WebGPU must use either
    // the guarded 2D path or the original 3D fallback, never silently skip it.
    const bool expect_direct = direct || ((!ggml_backend_is_cpu(backend) || filtered) && !force_prec_f32);
    const auto expected_op = test.expect_2d ? GGML_OP_CONV_2D : (expect_direct ? GGML_OP_CONV_3D : GGML_OP_IM2COL_3D);
    if (!convolution || convolution->op != expected_op || convolution->src[0]->type != test.weight_type) return -6;
    if (test.expect_2d && (convolution->src[0]->view_src != weight || convolution->src[1]->view_src != input)) return -12;

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
        (scheduler == nullptr || ggml_backend_sched_get_tensor_backend(scheduler, convolution) !=
             (test.expect_2d ? backend : workspace.cpu_backend()) ||
         (with_bias && ggml_backend_sched_get_tensor_backend(scheduler, output) != backend))) return -8;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto node = ggml_graph_node(graph, i);
        auto assigned = scheduler != nullptr ? ggml_backend_sched_get_tensor_backend(scheduler, node) : backend;
        if (!assigned || !node->buffer || !ggml_backend_supports_buft(assigned, ggml_backend_buffer_get_type(node->buffer))) return -9;
    }

    std::vector<float> values(volume * channels * test.batches), actual(output_volume * outputs * test.batches);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = (int(i % 17) - 8) / 16.f;
        if (test.emulate_fallback) values[i] += 0.00003f;
    }
    ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(float));
    auto status = scheduler != nullptr ? ggml_backend_sched_graph_compute(scheduler, graph) : ggml_backend_graph_compute(backend, graph);
    workspace.synchronize();
    if (status != GGML_STATUS_SUCCESS) return -10;
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    // Independent 3D reference, including batches and temporal padding/stride.
    // New cases include non-F16 input values. CPU direct rounds input patches to
    // the weight type; GPU direct keeps F32 input, so parity is not bitwise.
    for (int batch_channel = 0; batch_channel < outputs * test.batches; ++batch_channel) {
        const int batch = batch_channel / outputs, oc = batch_channel % outputs;
        for (int z = 0; z < output_depth; ++z) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    float expected = with_bias ? biases[oc] : 0.f;
                    for (int ic = 0; ic < channels; ++ic) {
                        for (int kz = 0; kz < kernel; ++kz) {
                            for (int ky = 0; ky < kernel; ++ky) {
                                for (int kx = 0; kx < kernel; ++kx) {
                                    const int sx = x + kx - 1, sy = y + ky - 1;
                                    const int sz = z * test.stride + kz * test.dilation - test.padding;
                                    if (sx < 0 || sx >= width || sy < 0 || sy >= height || sz < 0 || sz >= depth) continue;
                                    const int input_index = (((batch * channels + ic) * depth + sz) * height + sy) * width + sx;
                                    const int weight_index = (((oc * channels + ic) * kernel + kz) * kernel + ky) * kernel + kx;
                                    expected += values[input_index] * weight_values[weight_index];
                                }
                            }
                        }
                    }
                    const float value = actual[((batch_channel * output_depth + z) * height + y) * width + x];
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
    ConvolutionCase single_depth;
    single_depth.depth = 3;
    single_depth.padding = 0;
    single_depth.emulate_fallback = true;
    single_depth.expect_2d = true;
    for (auto type : {GGML_TYPE_F16, GGML_TYPE_F32}) {
        single_depth.weight_type = type;
        for (bool with_bias : {false, true}) {
            const int result = check_convolution(backend.get(), false, false, with_bias, single_depth);
            if (result != 1) return result - 1000 - 100 * (2 * int(type == GGML_TYPE_F32) + int(with_bias));
        }
    }
    single_depth.weight_type = GGML_TYPE_F16;
    single_depth.expect_2d = false;
    // Each excluded case must still construct and evaluate its original 3D
    // operation. In particular, one output frame alone is not sufficient.
    for (int excluded = 0; excluded < 8; ++excluded) {
        auto test = single_depth;
        if (excluded == 0) test.depth = 4;
        if (excluded == 1) test.padding = 1;
        if (excluded == 2) { test.depth = 5; test.dilation = 2; }
        if (excluded == 3) test.stride = 2;
        if (excluded == 4) test.batches = 2;
        if (excluded == 5) test.reject_2d = true;
        const int result = check_convolution(backend.get(), excluded == 6, excluded == 7, true, test);
        if (result != 1) return result - 2000 - 100 * excluded;
    }
    return 1;
}
