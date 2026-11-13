struct fixture {
    ggml_context * context = ggml_init({1 << 20, nullptr, true});
    ggml_cgraph * graph;
    ggml_tensor *source_root, *source, *rms, *output;
    int index = -1;
    fixture(int width, int heads, int tokens, int sequences, int mode = 0, bool produced = false,
            bool preserve = false) {
        if (!context)
            throw std::bad_alloc();
        graph = ggml_new_graph_custom(context, 64, false);
        if (mode == 0) {
            source_root = ggml_new_tensor_4d(context, GGML_TYPE_F32, width, heads, tokens, sequences);
            ggml_set_input(source_root);
            source = produced ? ggml_scale(context, source_root, 1.125f) : source_root;
            if (produced)
                source_root = source;
        } else {
            const size_t nb1 = (width + (mode == 1 ? 1 : 0)) * 4;
            const size_t nb2 = nb1 * heads + (mode == 1 ? 12 : size_t(width * heads * 8));
            const size_t nb3 = nb2 * tokens + 20;
            const size_t offset = mode == 1 ? 4 : size_t(width * heads * 4);
            const size_t bytes =
                offset + (sequences - 1) * nb3 + (tokens - 1) * nb2 + (heads - 1) * nb1 + width * 4 + 32;
            source_root = ggml_new_tensor_1d(context, GGML_TYPE_F32, bytes / 4);
            ggml_set_input(source_root);
            if (produced)
                source_root = ggml_scale(context, source_root, 1.125f);
            source = ggml_view_4d(context, source_root, width, heads, tokens, sequences, nb1, nb2, nb3, offset);
        }
        if (preserve)
            ggml_set_output(source);
        rms = ggml_rms_norm(context, source, 1e-5f / float(width));
        output = ggml_scale_bias(context, rms, -0.3f, 0.07f);
        ggml_set_output(output);
        ggml_build_forward_expand(graph, output);
        for (int i = 0; i < graph->n_nodes; i++)
            if (graph->nodes[i] == rms)
                index = i;
        if (index < 0)
            throw std::runtime_error("missing RMS node");
    }
    ~fixture() {
        ggml_free(context);
    }
};
