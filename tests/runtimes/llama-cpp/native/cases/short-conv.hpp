#pragma once

namespace short_conv_cases {
enum class Guard { ordinary, observed_convolution, dynamic_filter, observed_state_source };

static void graph(int64_t channels, bool natural, Guard guard) {
    transfer_cases::Backend gpu(transfer_cases::BackendKind::webgpu);
    const auto memory = transfer_cases::Memory(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
    BICORE_CHECK(memory);
    auto * ctx = memory.get();
    auto * state = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3 * channels, 4);
    auto * slot = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    auto * activation = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, channels, 1);
    auto * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, channels);
    for (auto * tensor : {state, slot, activation, weights})
        ggml_set_input(tensor);
    auto * snapshot = ggml_reshape_3d(ctx, ggml_get_rows(ctx, state, slot), 3, channels, 1);
    auto * current = ggml_dup(ctx, activation);
    auto * history = ggml_concat(ctx, snapshot, ggml_transpose(ctx, ggml_reshape_3d(ctx, current, channels, 1, 1)), 0);
    auto * recent = ggml_view_3d(ctx, history, 3, channels, 1, history->nb[1], history->nb[2], sizeof(float));
    auto * cell = ggml_view_2d(ctx, state, 3 * channels, 1, state->nb[1], state->nb[1]);
    auto * graph = ggml_new_graph_custom(ctx, 64, false);
    auto * copy = ggml_cpy(ctx, recent, cell);
    ggml_build_forward_expand(graph, copy);
    auto * convolution = ggml_ssm_conv(ctx, history, weights);
    if (guard == Guard::observed_convolution)
        ggml_set_output(convolution);
    if (guard == Guard::observed_state_source)
        ggml_set_output(recent);
    auto * flat = ggml_reshape_2d(ctx, convolution, channels, 1);
    if (guard == Guard::dynamic_filter) {
        auto * tap = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, channels, 1);
        flat = ggml_add(ctx, flat, ggml_mul(ctx, current, tap));
    }
    auto * output = ggml_scale(ctx, ggml_add(ctx, current, flat), 0.4472136f);
    auto * retained = ggml_dup(ctx, output);
    ggml_set_output(retained);
    ggml_build_forward_expand(graph, retained);
    const auto allocator = std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)>(
        natural ? ggml_gallocr_new(&gpu.type) : nullptr, ggml_gallocr_free);
    const auto storage =
        transfer_cases::Buffer(natural ? nullptr : ggml_backend_alloc_ctx_tensors_from_buft(memory.get(), &gpu.type),
                               ggml_backend_buffer_free);
    BICORE_CHECK(natural ? allocator && ggml_gallocr_alloc_graph(allocator.get(), graph) : bool(storage));
    commands = {};
    readback = {};
    BICORE_CHECK(ggml_backend_graph_compute_async(&gpu.backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(&gpu.backend);

    const bool epilogue = guard == Guard::ordinary || guard == Guard::observed_state_source;
    const bool state_store = guard != Guard::observed_state_source;
    const unsigned ordinary = guard == Guard::dynamic_filter ? 10u : 8u;
    BICORE_CHECK(commands.dispatches.size() == ordinary - (epilogue ? 2 : 0) - (state_store ? 1 : 0));
    const auto selected = [&](const webgpu_pipeline & pipeline) {
        return std::count_if(commands.dispatches.begin(), commands.dispatches.end(), [&](const auto & dispatch) {
            return dispatch.pipeline == pipeline.pipeline.Get()->id && dispatch.x == uint32_t(channels / 128) &&
                   dispatch.y == 1 && dispatch.z == 1;
        });
    };
    const auto state_pipeline = gpu.context->shader_lib->get_short_conv_state_pipeline();
    BICORE_CHECK(selected(state_pipeline) == (state_store ? 1 : 0));
    const auto residual_inplace = gpu.context->shader_lib->get_short_conv_residual_pipeline(true);
    const auto residual_disjoint = gpu.context->shader_lib->get_short_conv_residual_pipeline(false);
    BICORE_CHECK(selected(residual_inplace) == (epilogue && natural ? 1 : 0));
    BICORE_CHECK(selected(residual_disjoint) == (epilogue && !natural ? 1 : 0));

    if (guard == Guard::ordinary && natural) {
        const auto index = [&](const ggml_tensor * tensor) {
            const auto * found = std::find(graph->nodes, graph->nodes + graph->n_nodes, tensor);
            BICORE_CHECK(found < graph->nodes + graph->n_nodes);
            return static_cast<int>(found - graph->nodes);
        };
        BICORE_CHECK(output->buffer == current->buffer && state->buffer == history->buffer);
        commands = {};
        int consumed = 1;
        const auto old_output = output->data;
        output->data = reinterpret_cast<void *>(uintptr_t(current->data) + sizeof(float));
        const auto partial = ggml_webgpu_short_conv_residual(gpu.context, graph, index(convolution), consumed);
        output->data = old_output;
        BICORE_CHECK(!partial && consumed == 1 && commands.dispatches.empty());

        const auto old_state = state->data;
        const auto old_copy = copy->data;
        state->data = reinterpret_cast<void *>(uintptr_t(history->data) - copy->view_offs);
        copy->data = history->data;
        const auto overlap = ggml_webgpu_short_conv_state(gpu.context, graph, index(history), consumed);
        copy->data = old_copy;
        state->data = old_state;
        BICORE_CHECK(!overlap && consumed == 1 && commands.dispatches.empty());
    }
}

static void all() {
    for (int64_t channels : {1024, 2304})
        for (bool natural : {false, true})
            for (auto guard :
                 {Guard::ordinary, Guard::observed_convolution, Guard::dynamic_filter, Guard::observed_state_source})
                graph(channels, natural, guard);
}
} // namespace short_conv_cases
