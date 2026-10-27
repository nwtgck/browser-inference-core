#pragma once
#include "ggml-webgpu-ssm-conv-silu.hpp"
#include <stdexcept>
struct fixture {
    ggml_context * context = ggml_init({1 << 20, nullptr, true});
    ggml_cgraph * graph;
    ggml_tensor * history_root, * history, * weights_root, * weights, * conv, * output;
    int index = -1;
    fixture(int nc, int channels, int tokens, int sequences, bool padded = false) {
        if (!context) { throw std::bad_alloc(); }
        graph = ggml_new_graph_custom(context, 64, false);
        const int width = nc - 1 + tokens;
        if (padded) {
            const size_t nb1 = (width + 3) * 4, nb2 = nb1 * channels + 20;
            const size_t bytes = 4 + (sequences - 1) * nb2 + (channels - 1) * nb1 + width * 4 + 32;
            history_root = ggml_new_tensor_1d(context, GGML_TYPE_F32, bytes / 4);
            history = ggml_view_3d(context, history_root, width, channels, sequences, nb1, nb2, 4);
            weights_root = ggml_new_tensor_1d(context, GGML_TYPE_F32, (channels * (nc + 2) + 8));
            weights = ggml_view_2d(context, weights_root, nc, channels, (nc + 2) * 4, 4);
        } else {
            history = history_root = ggml_new_tensor_3d(context, GGML_TYPE_F32, width, channels, sequences);
            weights = weights_root = ggml_new_tensor_2d(context, GGML_TYPE_F32, nc, channels);
        }
        ggml_set_input(history_root); ggml_set_input(weights_root);
        conv = ggml_ssm_conv(context, history, weights);
        output = ggml_silu(context, conv); ggml_set_output(output);
        ggml_build_forward_expand(graph, output);
        for (int i = 0; i < graph->n_nodes; ++i) { if (graph->nodes[i] == conv) { index = i; } }
        if (index < 0) { throw std::runtime_error("convolution node missing"); }
    }
    ~fixture() { ggml_free(context); }
};
