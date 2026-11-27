#pragma once

namespace ssm_conv_gate_cases {
enum class Guard { ordinary, observed_convolution, raw_convolution, raw_output, reversed };

static void graph(int64_t taps, int64_t channels, int64_t tokens, int64_t sequences, bool natural, Guard guard) {
    transfer_cases::Backend gpu(transfer_cases::BackendKind::webgpu);
    const auto memory = transfer_cases::Memory(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
    BICORE_CHECK(memory);
    auto * ctx = memory.get();
    auto * history = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, taps - 1 + tokens, channels, sequences);
    auto * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, taps, channels);
    auto * packed = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3 * channels, tokens, sequences);
    for (auto * tensor : {history, weights, packed})
        ggml_set_input(tensor);
    auto * gate =
        ggml_view_3d(ctx, packed, channels, tokens, sequences, packed->nb[1], packed->nb[2], channels * sizeof(float));
    auto * convolution = ggml_ssm_conv(ctx, history, weights);
    if (guard == Guard::observed_convolution)
        ggml_set_output(convolution);
    auto * output = guard == Guard::reversed ? ggml_mul(ctx, convolution, gate) : ggml_mul(ctx, gate, convolution);
    auto * retained = ggml_dup(ctx, output);
    ggml_set_output(retained);
    auto * graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, retained);
    const int stop = graph->n_nodes;
    const bool raw_reader = guard == Guard::raw_convolution || guard == Guard::raw_output;
    if (raw_reader) {
        auto * root = guard == Guard::raw_convolution ? convolution : output;
        auto * view = ggml_view_tensor(ctx, root);
        BICORE_CHECK(view->view_src == root && !view->src[0]);
        auto * reader = ggml_dup(ctx, view);
        ggml_set_output(reader);
        ggml_build_forward_expand(graph, reader);
    }
    const auto allocator = std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)>(
        natural ? ggml_gallocr_new(&gpu.type) : nullptr, ggml_gallocr_free);
    const auto storage =
        transfer_cases::Buffer(natural ? nullptr : ggml_backend_alloc_ctx_tensors_from_buft(memory.get(), &gpu.type),
                               ggml_backend_buffer_free);
    BICORE_CHECK(natural ? allocator && ggml_gallocr_alloc_graph(allocator.get(), graph) : bool(storage));
    const bool fused = guard == Guard::ordinary || guard == Guard::raw_output;
    const auto compute = [&](ggml_cgraph * slice, unsigned expected) {
        commands = {};
        readback = {};
        BICORE_CHECK(ggml_backend_graph_compute_async(&gpu.backend, slice) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(&gpu.backend);
        BICORE_CHECK(commands.dispatches.size() == expected);
    };
    if (raw_reader) {
        auto slice = ggml_graph_view(graph, 0, stop);
        compute(&slice, fused ? 2u : 3u);
    }
    compute(graph, (fused ? 2u : 3u) + unsigned(raw_reader));

    ggml_webgpu_shader_lib_context shader = {};
    shader.src0 = history;
    shader.src1 = weights;
    shader.dst = output;
    shader.max_wg_size = gpu.context->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
    const auto selected = gpu.context->shader_lib->get_ssm_conv_pipeline(shader, webgpu_ssm_conv_epilogue::gate);
    const uint32_t token_width = tokens == 1 ? 1 : 8;
    const auto selected_count =
        std::count_if(commands.dispatches.begin(), commands.dispatches.end(), [&](const auto & d) {
            return d.pipeline == selected.pipeline.Get()->id && d.x == uint32_t((channels + 31) / 32) &&
                   d.y == uint32_t((tokens + token_width - 1) / token_width * sequences) && d.z == 1;
        });
    BICORE_CHECK(selected_count == (fused ? 1 : 0));
    if (fused) {
        const auto dispatch = std::find_if(commands.dispatches.begin(), commands.dispatches.end(),
                                           [&](const auto & d) { return d.pipeline == selected.pipeline.Get()->id; });
        BICORE_CHECK(dispatch != commands.dispatches.end());
        const auto & slots = gpu.context->bind_group_cache.slots;
        const auto slot = std::find_if(slots.begin(), slots.end(), [&](const auto & entry) {
            return entry.group && entry.group.Get()->id == dispatch->group;
        });
        BICORE_CHECK(slot != slots.end() && slot->bindings.size() == 5);
        const ggml_tensor * tensors[] = {history, weights, output, gate};
        std::array<uint32_t, 4> offsets{};
        for (size_t i = 0; i < std::size(tensors); ++i) {
            const auto & entry = slot->bindings[i];
            const auto * tensor = tensors[i];
            BICORE_CHECK(entry.binding_index == i && entry.buffer.Get() == ggml_webgpu_tensor_buf(tensor).Get() &&
                         entry.offset == ggml_webgpu_tensor_align_offset(gpu.context, tensor) &&
                         entry.size == ggml_webgpu_tensor_binding_size(gpu.context, tensor));
            offsets[i] = ggml_webgpu_tensor_misalignment(gpu.context, tensor) / sizeof(float);
        }
        const auto & uniform = slot->bindings.back();
        std::array<uint32_t, 17> params{};
        std::memcpy(params.data(), uniform.buffer.Get()->bytes.data() + uniform.offset, sizeof(params));
        const std::array<uint32_t, 17> expected = {
            offsets[0],
            offsets[1],
            offsets[2],
            uint32_t(taps - 1 + tokens),
            uint32_t((taps - 1 + tokens) * channels),
            uint32_t(taps),
            1,
            uint32_t(channels),
            uint32_t(channels * tokens),
            uint32_t(taps),
            uint32_t(channels),
            uint32_t(tokens),
            uint32_t(sequences),
            uint32_t((tokens + token_width - 1) / token_width),
            offsets[3],
            uint32_t(3 * channels),
            uint32_t(3 * channels * tokens),
        };
        BICORE_CHECK(params == expected);
    }
    BICORE_CHECK(
        gpu.context->shader_lib->get_ssm_conv_pipeline(shader, webgpu_ssm_conv_epilogue::gate).pipeline.Get() ==
        selected.pipeline.Get());
    BICORE_CHECK(gpu.context->shader_lib->get_ssm_conv_pipeline(shader).pipeline.Get() != selected.pipeline.Get());
    BICORE_CHECK(
        gpu.context->shader_lib->get_ssm_conv_pipeline(shader, webgpu_ssm_conv_epilogue::silu).pipeline.Get() !=
        selected.pipeline.Get());

    if (guard == Guard::ordinary) {
        const auto * found = std::find(graph->nodes, graph->nodes + graph->n_nodes, convolution);
        BICORE_CHECK(found < graph->nodes + graph->n_nodes);
        const int index = static_cast<int>(found - graph->nodes);
        webgpu_fusion_view_sources views(*graph);
        commands = {};
        int consumed = 1;
        const auto old_output = output->data;
        const auto old_buffer = output->buffer;
        output->data = gate->data;
        output->buffer = gate->buffer;
        const auto overlap = ggml_webgpu_ssm_conv_gate(gpu.context, graph, index, views, consumed);
        output->data = old_output;
        output->buffer = old_buffer;
        BICORE_CHECK(!overlap && consumed == 1 && commands.dispatches.empty());

        const auto old_limit = gpu.context->global_ctx->capabilities.limits.maxStorageBuffersPerShaderStage;
        gpu.context->global_ctx->capabilities.limits.maxStorageBuffersPerShaderStage = 3;
        const auto insufficient = ggml_webgpu_ssm_conv_gate(gpu.context, graph, index, views, consumed);
        gpu.context->global_ctx->capabilities.limits.maxStorageBuffersPerShaderStage = old_limit;
        BICORE_CHECK(!insufficient && consumed == 1 && commands.dispatches.empty());

        auto & limits = gpu.context->global_ctx->capabilities.limits;
        const auto old_y = limits.maxComputeWorkgroupSizeY;
        limits.maxComputeWorkgroupSizeY = token_width - 1;
        const auto short_workgroup = ggml_webgpu_ssm_conv_gate(gpu.context, graph, index, views, consumed);
        limits.maxComputeWorkgroupSizeY = old_y;
        BICORE_CHECK(!short_workgroup && consumed == 1 && commands.dispatches.empty());
        const auto old_grid = limits.maxComputeWorkgroupsPerDimension;
        limits.maxComputeWorkgroupsPerDimension = uint32_t((channels + 31) / 32 - 1);
        const auto short_grid = ggml_webgpu_ssm_conv_gate(gpu.context, graph, index, views, consumed);
        limits.maxComputeWorkgroupsPerDimension = old_grid;
        BICORE_CHECK(!short_grid && consumed == 1 && commands.dispatches.empty());
    }
}

static void all() {
    for (int64_t taps : {3, 4, 7})
        for (int64_t tokens : {1, 17})
            for (int64_t sequences : {1, 2})
                for (bool natural : {false, true})
                    graph(taps, 129, tokens, sequences, natural, Guard::ordinary);
    for (auto guard : {Guard::observed_convolution, Guard::raw_convolution, Guard::raw_output, Guard::reversed})
        for (bool natural : {false, true})
            graph(7, 2048, 1, 1, natural, guard);
    for (int64_t taps : {2, 5, 8}) {
        graph(taps, 33, 1, 1, false, Guard::ordinary);
        graph(taps, 129, 17, 2, true, Guard::ordinary);
    }
}
} // namespace ssm_conv_gate_cases
