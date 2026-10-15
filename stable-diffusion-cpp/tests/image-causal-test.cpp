// Actual patched WAN::CausalConv3d. The capability filter only controls graph
// selection; every arithmetic operation is evaluated by the real native CPU.
#include "image-performance-test-utils.h"
#include "model/vae/wan_vae.hpp"
#include "ggml-backend-impl.h"
#include <cstdio>
#include <map>

namespace {
using namespace image_test;
class ReferenceConv : public WAN::CausalConv3d {
public:
    using WAN::CausalConv3d::CausalConv3d;
    void weight_type(ggml_context* ctx, ggml_type type) {
        auto old = params.at("weight");
        params["weight"] = ggml_new_tensor(ctx, type, 4, old->ne);
    }
    // Frozen pre-optimization path; never invokes the new causal fast path.
    ggml_tensor* legacy(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* history) {
        auto w = params.at("weight");
        auto b = bias ? params.at("bias") : nullptr;
        int left = 2 * std::get<0>(padding);
        if (history != nullptr && left > 0) {
            x = ggml_concat(ctx->ggml_ctx, history, x, 2);
            left -= static_cast<int>(history->ne[2]);
        }
        x = ggml_ext_pad_ext(ctx->ggml_ctx, ctx->backend, x,
                              std::get<2>(padding), std::get<2>(padding),
                              std::get<1>(padding), std::get<1>(padding),
                              left, 0, 0, 0, ctx->circular_x_enabled, ctx->circular_y_enabled);
        if (w->ne[2] == 1 && x->ne[2] == 1 && x->ne[3] == in_channels) {
            x = ggml_ext_cont(ctx->ggml_ctx, x);
            auto x2 = ggml_reshape_4d(ctx->ggml_ctx, x, x->ne[0], x->ne[1], in_channels, 1);
            auto w2 = ggml_reshape_4d(ctx->ggml_ctx, w, w->ne[0], w->ne[1], in_channels, out_channels);
            auto y = ggml_ext_conv_2d(ctx->ggml_ctx, x2, w2, b,
                                       std::get<2>(stride), std::get<1>(stride), 0, 0,
                                       std::get<2>(dilation), std::get<1>(dilation), ctx->conv2d_direct_enabled);
            return ggml_reshape_4d(ctx->ggml_ctx, y, y->ne[0], y->ne[1], 1, out_channels);
        }
        return ggml_ext_conv_3d(ctx->ggml_ctx, ctx->backend, x, w, b, in_channels,
                                 std::get<2>(stride), std::get<1>(stride), std::get<0>(stride),
                                 0, 0, 0, std::get<2>(dilation), std::get<1>(dilation), std::get<0>(dilation),
                                 false, ctx->conv3d_direct_enabled);
    }
};
struct Filter {
    ggml_backend_t cpu;
    ggml_op rejected;
    ggml_backend_device device = {};
    ggml_backend backend;
    Filter(ggml_backend_t source, ggml_op reject) : cpu(source), rejected(reject), backend(*source) {
        device.context = this;
        device.iface.get_name = [](ggml_backend_dev_t) { return "CPU-capability-filter"; };
        device.iface.supports_op = [](ggml_backend_dev_t device, const ggml_tensor* node) {
            const auto& self = *static_cast<Filter*>(device->context);
            if (node->op == GGML_OP_IM2COL_3D || (self.rejected != GGML_OP_NONE && node->op == self.rejected)) return false;
            return ggml_backend_supports_op(self.cpu, node);
        };
        backend.device = &device;
    }
};
struct Case {
    ggml_type type = GGML_TYPE_F16;
    int frames = 1, batches = 1, temporal_kernel = 3, temporal_stride = 1, temporal_dilation = 1;
    int spatial_stride = 1, spatial_dilation = 1;
    bool history = false, direct = false, circular_x = false, circular_y = false, bias = true, strided = false;
    ggml_op rejected = GGML_OP_NONE;
    bool fast = true;
};
void run_case(ggml_backend_t cpu, const Case& test) {
    constexpr int width = 7, height = 6, channels = 3, outputs = 5;
    auto params = context(), reference_ctx = context(), actual_ctx = context();
    ReferenceConv layer(channels, outputs, {test.temporal_kernel, 3, 3},
                        {test.temporal_stride, test.spatial_stride, test.spatial_stride},
                        {test.temporal_kernel == 3 ? 1 : 0, 1, 1},
                        {test.temporal_dilation, test.spatial_dilation, test.spatial_dilation}, test.bias);
    layer.init(params.get());
    layer.weight_type(params.get(), test.type);
    std::map<std::string, ggml_tensor*> weights;
    layer.get_param_tensors(weights);
    auto storage = ggml_new_tensor_4d(params.get(), GGML_TYPE_F32, width + (test.strided ? 2 : 0),
                                     height, test.frames, channels * test.batches);
    auto x = test.strided ? ggml_view_4d(params.get(), storage, width, height, test.frames, channels * test.batches,
                                       storage->nb[1], storage->nb[2], storage->nb[3], sizeof(float)) : storage;
    auto history = test.history ? ggml_new_tensor_4d(params.get(), GGML_TYPE_F32, width, height, 2, channels * test.batches) : nullptr;
    Buffer buffer(ggml_backend_alloc_ctx_tensors(params.get(), cpu), ggml_backend_buffer_free);
    require(buffer != nullptr, "parameter allocation failed");
    for (const auto& [name, weight] : weights) write(weight, values(ggml_nelements(weight), name == "weight" ? 4 : 8));
    if (history) write(history, values(ggml_nelements(history), 15));
    Filter selection(cpu, test.rejected);
    GGMLRunnerContext ctx;
    ctx.backend = &selection.backend;
    ctx.circular_x_enabled = test.circular_x;
    ctx.circular_y_enabled = test.circular_y;
    ctx.conv3d_direct_enabled = test.direct;
    ctx.ggml_ctx = reference_ctx.get();
    auto expected = layer.legacy(&ctx, x, history);
    ctx.ggml_ctx = actual_ctx.get();
    auto actual = layer.forward(&ctx, x, history);
    for (int axis = 0; axis < 4; ++axis) require(expected->ne[axis] == actual->ne[axis], "shape changed");
    auto reference_graph = graph(reference_ctx.get(), expected), actual_graph = graph(actual_ctx.get(), actual);
    bool selected = false;
    for (int i = 0; i < ggml_graph_n_nodes(actual_graph); ++i) {
        auto node = ggml_graph_node(actual_graph, i);
        if (node->op == GGML_OP_CONV_2D && node->src[0]->ne[2] == channels && test.temporal_kernel == 3) {
            selected = true;
            require(ggml_is_contiguous(node->src[0]), "selected kernel was not packed");
        }
    }
    require(selected == test.fast, "wrong causal path selected");
    if (test.fast && test.bias) require(actual->op == GGML_OP_ADD && actual->view_src == nullptr, "bias must own its output");
    auto reference_alloc = allocate(cpu, reference_graph), actual_alloc = allocate(cpu, actual_graph);
    for (unsigned input = 1; input <= 2; ++input) {
        write(storage, values(ggml_nelements(storage), input));
        compute(cpu, reference_graph); compute(cpu, actual_graph);
        close(read(expected), read(actual), test.fast ? 2e-5f : 0.f, test.fast ? 2e-5f : 0.f);
    }
}
} // namespace
int main() {
    int cases = 0;
    try {
        auto backend = image_test::cpu();
        for (auto type : {GGML_TYPE_F16, GGML_TYPE_F32}) {
            for (bool bias : {false, true}) for (bool strided : {false, true}) for (int circular = 0; circular < 4; ++circular) {
                Case test;
                test.type = type; test.bias = bias; test.strided = strided;
                test.circular_x = (circular & 1) != 0; test.circular_y = (circular & 2) != 0; test.fast = circular != 3;
                run_case(backend.get(), test); ++cases;
            }
            for (int mode = 0; mode < 11; ++mode) {
                Case test; test.type = type; test.fast = false;
                switch (mode) {
                    case 0: test.frames = 3; break;
                    case 1: test.batches = 2; break;
                    case 2: test.history = true; break;
                    case 3: test.temporal_kernel = 1; break;
                    case 4: test.direct = true; break;
                    case 5: test.temporal_stride = 2; test.frames = 3; break;
                    case 6: test.temporal_dilation = 2; test.frames = 3; break;
                    case 7: test.rejected = GGML_OP_CONV_2D; break;
                    case 8: test.rejected = GGML_OP_CONT; break;
                    case 9: test.rejected = GGML_OP_ADD; break;
                    case 10:
                        // The existing pad helper retains its right-pad/roll
                        // fallback. Only newly selected operations are gated.
                        test.rejected = GGML_OP_PAD; test.fast = true; break;
                }
                run_case(backend.get(), test); ++cases;
            }
            for (int stride : {1, 2}) for (int dilation : {1, 2}) {
                Case test; test.type = type; test.spatial_stride = stride; test.spatial_dilation = dilation;
                run_case(backend.get(), test); ++cases;
            }
        }
        std::printf("%d causal cases passed (two inputs each, synthetic CPU)\n", cases);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "causal case %d failed: %s\n", cases, error.what()); return 1;
    }
}
