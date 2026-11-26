#pragma once

namespace raw_view_fusion_cases {
enum class Family { rms_mul, rms_mul_inplace, rms_scale, rms_scale_unary, scale_unary, matvec_epilogue, moe_sum };

struct DispatchCounts {
    unsigned prefix;
    unsigned reader;
    unsigned whole;
};

static DispatchCounts run(Family family, int raw_output) {
    transfer_cases::Backend gpu(transfer_cases::BackendKind::webgpu);
    gpu.type.iface.get_name = ggml_backend_webgpu_buffer_type_get_name;
    const auto memory = transfer_cases::Memory(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
    auto * ctx = memory.get();
    BICORE_CHECK(ctx);
    auto * graph = ggml_new_graph_custom(ctx, 64, false);
    std::vector<ggml_tensor *> chain;
    if (family == Family::moe_sum) {
        auto * experts = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 4, 1);
        ggml_set_input(experts);
        ggml_tensor * slots[4];
        for (unsigned i = 0; i < 4; ++i) {
            slots[i] = ggml_view_2d(ctx, experts, 64, 1, experts->nb[2], i * experts->nb[1]);
            ggml_build_forward_expand(graph, slots[i]);
        }
        chain.push_back(ggml_add(ctx, slots[0], slots[1]));
        for (unsigned i = 2; i < 4; ++i)
            chain.push_back(ggml_add(ctx, chain.back(), slots[i]));
    } else {
        auto * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, family == Family::matvec_epilogue ? 1 : 2);
        ggml_set_input(input);
        if (family == Family::matvec_epilogue) {
            auto * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 4);
            ggml_set_input(weights);
            chain.push_back(ggml_mul_mat(ctx, weights, input));
        } else if (family != Family::scale_unary) {
            chain.push_back(ggml_rms_norm(ctx, input, 1e-5f));
        }
        if (family == Family::rms_mul || family == Family::rms_mul_inplace) {
            auto * weights = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
            ggml_set_input(weights);
            auto * rms = chain.back();
            chain.push_back(family == Family::rms_mul_inplace ? ggml_mul_inplace(ctx, rms, weights)
                                                              : ggml_mul(ctx, rms, weights));
        } else {
            chain.push_back(ggml_scale(ctx, chain.empty() ? input : chain.back(), 0.5f));
            if (family != Family::rms_scale) {
                chain.push_back(ggml_sigmoid(ctx, chain.back()));
                chain.push_back(ggml_scale(ctx, chain.back(), 0.25f));
            }
        }
    }
    auto * output = chain.back();
    // A requested in-place output also marks its root as an explicit output.
    // Exercise an internal MUL consumed by the raw reader instead.
    if (family != Family::rms_mul_inplace)
        ggml_set_output(output);
    ggml_build_forward_expand(graph, output);
    const int stop = graph->n_nodes;
    if (raw_output >= 0) {
        auto * source = chain.at(raw_output);
        auto * view = ggml_view_tensor(ctx, source);
        auto * backing = source->view_src ? source->view_src : source;
        BICORE_CHECK(view->view_src == backing && std::all_of(std::begin(view->src), std::end(view->src),
                                                              [](const auto * tensor) { return tensor == nullptr; }));
        auto * reader = ggml_dup(ctx, view);
        ggml_set_output(reader);
        ggml_build_forward_expand(graph, reader);
        BICORE_CHECK(ggml_hash_contains(&graph->visited_hash_set, view));
    }
    const auto storage =
        transfer_cases::Buffer(ggml_backend_alloc_ctx_tensors_from_buft(ctx, &gpu.type), ggml_backend_buffer_free);
    BICORE_CHECK(storage);
    if (family == Family::rms_mul_inplace) {
        BICORE_CHECK(!(chain.front()->flags & GGML_TENSOR_FLAG_OUTPUT) && output->view_src == chain.front() &&
                     ggml_are_same_layout(output, chain.front()) && ggml_webgpu_tensor_equal(output, chain.front()));
    }
    const auto compute = [&](ggml_cgraph * slice) {
        commands = {};
        readback = {};
        BICORE_CHECK(ggml_backend_graph_compute_async(&gpu.backend, slice) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(&gpu.backend);
        return unsigned(commands.dispatches.size());
    };
    if (raw_output < 0)
        return {0, 0, compute(graph)};
    // Both scheduler slices share the original graph's visit set. The producer
    // cannot see the later DUP as a node of its own slice.
    auto prefix = ggml_graph_view(graph, 0, stop);
    auto reader = ggml_graph_view(graph, stop, graph->n_nodes);
    BICORE_CHECK(prefix.visited_hash_set.keys == reader.visited_hash_set.keys);
    return {compute(&prefix), compute(&reader), compute(graph)};
}

static void all() {
    struct Case {
        Family family;
        int raw_output;
        unsigned expected;
    };
    const Case cases[] = {
        {Family::rms_mul, -1, 1},         {Family::rms_mul, 0, 2},         {Family::rms_mul, 1, 1},
        {Family::rms_mul_inplace, -1, 1}, {Family::rms_mul_inplace, 0, 1}, {Family::rms_mul_inplace, 1, 1},
        {Family::rms_scale, -1, 1},       {Family::rms_scale, 0, 2},       {Family::rms_scale, 1, 1},
        {Family::rms_scale_unary, -1, 2}, {Family::rms_scale_unary, 1, 3}, {Family::rms_scale_unary, 2, 3},
        {Family::rms_scale_unary, 3, 2},  {Family::scale_unary, -1, 1},    {Family::scale_unary, 0, 3},
        {Family::scale_unary, 1, 2},      {Family::scale_unary, 2, 1},     {Family::matvec_epilogue, -1, 1},
        {Family::matvec_epilogue, 0, 2},  {Family::matvec_epilogue, 1, 4}, {Family::matvec_epilogue, 2, 3},
        {Family::matvec_epilogue, 3, 1},  {Family::moe_sum, -1, 1},        {Family::moe_sum, 0, 3},
        {Family::moe_sum, 1, 3},          {Family::moe_sum, 2, 1},
    };
    for (const auto & test : cases) {
        const auto counts = run(test.family, test.raw_output);
        if (test.raw_output >= 0)
            BICORE_CHECK(counts.prefix == test.expected && counts.reader == 1);
        BICORE_CHECK(counts.whole == test.expected + unsigned(test.raw_output >= 0));
    }
}
} // namespace raw_view_fusion_cases
