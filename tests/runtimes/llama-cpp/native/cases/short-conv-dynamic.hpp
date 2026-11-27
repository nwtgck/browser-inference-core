#pragma once

namespace short_conv_dynamic_cases {
enum class Reader { none, staged_result, output };

static void graph(int64_t channels, int64_t heads, int64_t tokens, bool reuse, Reader reader) {
    transfer_cases::Backend gpu(transfer_cases::BackendKind::webgpu);
    const auto memory = transfer_cases::Memory(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
    BICORE_CHECK(memory);
    auto * ctx = memory.get();
    auto * values = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, channels, tokens);
    auto * tap_values = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4 * heads, tokens);
    auto * previous = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, channels, 1);
    auto * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, channels);
    for (auto * leaf : {values, tap_values, previous, weights})
        ggml_set_input(leaf);

    auto * current = ggml_dup(ctx, values);
    auto * history =
        ggml_concat(ctx, previous, ggml_transpose(ctx, ggml_reshape_3d(ctx, current, channels, tokens, 1)), 0);
    auto * static_conv = ggml_ssm_conv(ctx, history, weights);
    auto * accum = ggml_reshape_2d(ctx, static_conv, channels, tokens);
    auto * taps = ggml_reshape_3d(ctx, ggml_dup(ctx, tap_values), 4, heads, tokens);
    ggml_tensor * intermediate = nullptr;
    ggml_tensor * start = nullptr;
    for (size_t lag = 0; lag < 4; ++lag) {
        auto * tap = ggml_view_3d(ctx, taps, 1, heads, tokens, taps->nb[1], taps->nb[2], lag * sizeof(float));
        auto * window =
            ggml_view_3d(ctx, history, tokens, channels, 1, history->nb[1], history->nb[2], (3 - lag) * sizeof(float));
        window = ggml_cont(ctx, ggml_transpose(ctx, window));
        window = ggml_reshape_3d(ctx, window, channels / heads, heads, tokens);
        auto * product = ggml_mul(ctx, window, tap);
        if (lag == 0)
            start = product;
        accum = ggml_add(ctx, accum, ggml_reshape_2d(ctx, product, channels, tokens));
        if (lag == 1)
            intermediate = accum;
    }
    auto * output = ggml_scale(ctx, ggml_add(ctx, current, accum), 1.0f / 3.0f);
    ggml_set_output(output);
    auto * full = ggml_new_graph_custom(ctx, 128, false);
    ggml_build_forward_expand(full, output);
    const int stop = full->n_nodes;
    if (reader != Reader::none) {
        auto * raw = ggml_view_tensor(ctx, reader == Reader::output ? output : intermediate);
        auto * observed = ggml_dup(ctx, raw);
        ggml_set_output(observed);
        ggml_build_forward_expand(full, observed);
    }
    const auto allocator = std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)>(
        reuse ? ggml_gallocr_new(&gpu.type) : nullptr, ggml_gallocr_free);
    const auto storage = transfer_cases::Buffer(
        reuse ? nullptr : ggml_backend_alloc_ctx_tensors_from_buft(ctx, &gpu.type), ggml_backend_buffer_free);
    BICORE_CHECK(reuse ? allocator && ggml_gallocr_alloc_graph(allocator.get(), full) : bool(storage));
    const auto it = std::find(full->nodes, full->nodes + stop, start);
    BICORE_CHECK(it != full->nodes + stop);
    const int index = static_cast<int>(it - full->nodes);
    const bool want = tokens == 1 && reader != Reader::staged_result;
    auto prefix = ggml_graph_view(full, 0, stop);
    webgpu_short_conv_dynamic::plan plan{};
    webgpu_fusion_view_sources views(prefix);
    BICORE_CHECK(webgpu_short_conv_dynamic::match(prefix, index, views, plan) == want);
    unsigned count = 0;
    for (int i = 0; i < stop; ++i) {
        auto op = full->nodes[i]->op;
        if (op != GGML_OP_VIEW && op != GGML_OP_RESHAPE && op != GGML_OP_TRANSPOSE)
            count++;
    }
    commands = {};
    BICORE_CHECK(ggml_backend_graph_compute_async(&gpu.backend, &prefix) == GGML_STATUS_SUCCESS);
    BICORE_CHECK(commands.dispatches.size() == count - (want ? 12u : 0u));
    ggml_backend_synchronize(&gpu.backend);
    if (want) {
        const auto inplace =
            ggml_webgpu_tensor_equal(output, static_conv) ? 1u : (ggml_webgpu_tensor_equal(output, current) ? 2u : 0u);
        const auto pipeline = gpu.context->shader_lib->get_short_conv_dynamic_pipeline(inplace);
        BICORE_CHECK(commands.dispatches.back().pipeline == pipeline.pipeline.Get()->id);
        BICORE_CHECK(commands.dispatches.back().x == (channels + 127) / 128);
        const auto old_data = output->data;
        BICORE_CHECK(output->buffer == history->buffer);
        output->data = reinterpret_cast<void *>(uintptr_t(history->data) + sizeof(float));
        int consumed = 1;
        commands = {};
        const auto rejected = ggml_webgpu_short_conv_dynamic(gpu.context, prefix, index, views, consumed);
        output->data = old_data;
        BICORE_CHECK(!rejected && consumed == 1 && commands.dispatches.empty());
        if (channels == 1024 && !reuse && reader == Reader::none) {
            // Exercise both exact aliases even when the graph allocator picks
            // only one of them. Each aliased row is dead after the suffix.
            const std::array alias_sources{static_conv, current};
            for (size_t i = 0; i < alias_sources.size(); ++i) {
                BICORE_CHECK(output->buffer == alias_sources[i]->buffer);
                output->data = alias_sources[i]->data;
                commands = {};
                BICORE_CHECK(ggml_backend_graph_compute_async(&gpu.backend, &prefix) == GGML_STATUS_SUCCESS);
                ggml_backend_synchronize(&gpu.backend);
                output->data = old_data;
                const auto alias_pipeline = gpu.context->shader_lib->get_short_conv_dynamic_pipeline(uint32_t(i + 1));
                BICORE_CHECK(commands.dispatches.size() == count - 12 &&
                             commands.dispatches.back().pipeline == alias_pipeline.pipeline.Get()->id);
            }
        }
    }
}

static void all() {
    for (const auto [channels, heads] : {std::pair{1024, 8}, std::pair{2304, 8}, std::pair{231, 7}})
        for (bool reuse : {false, true})
            for (auto reader : {Reader::none, Reader::staged_result, Reader::output})
                graph(channels, heads, 1, reuse, reader);
    graph(1024, 8, 2, true, Reader::none);
}
} // namespace short_conv_dynamic_cases
