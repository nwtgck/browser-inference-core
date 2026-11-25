#pragma once

namespace situ_cases {

enum class Guard { ordinary, intermediate_view, final_view, overlapping_storage, prefill, prefill_intermediate_view };

static void graph(Guard guard) {
    transfer_cases::Backend gpu(transfer_cases::BackendKind::webgpu);
    const auto memory = transfer_cases::Memory(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
    BICORE_CHECK(memory);
    auto * ctx = memory.get();
    constexpr int64_t width = 1152;
    const bool prefill = guard == Guard::prefill || guard == Guard::prefill_intermediate_view;
    const int64_t tokens = prefill ? 128 : 1;
    auto & limits = gpu.context->global_ctx->capabilities.limits;
    limits.maxStorageBufferBindingSize = limits.maxBufferSize = 64 * 1024 * 1024;
    auto * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2 * width, 8, tokens);
    ggml_set_input(input);
    auto * packed = ggml_scale(ctx, input, 1.0f);
    auto * gate = ggml_view_3d(ctx, packed, width, 8, tokens, packed->nb[1], packed->nb[2], 0);
    auto * up = ggml_view_3d(ctx, packed, width, 8, tokens, packed->nb[1], packed->nb[2], width * sizeof(float));
    auto * gate_act = ggml_scale(ctx, ggml_tanh(ctx, ggml_scale(ctx, gate, 1.0f / 2.25f)), 2.25f);
    auto * sigmoid = ggml_sigmoid(ctx, gate);
    auto * up_act = ggml_scale(ctx, ggml_tanh(ctx, ggml_scale(ctx, up, 1.0f / 1.625f)), 1.625f);
    auto * output = ggml_mul(ctx, ggml_mul(ctx, gate_act, sigmoid), up_act);
    ggml_set_output(output);
    auto * layout = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(layout, output);
    const int activation_end = ggml_graph_n_nodes(layout);

    const bool view_reader = guard == Guard::intermediate_view || guard == Guard::prefill_intermediate_view || guard == Guard::final_view;
    if (view_reader) {
        auto * source = guard == Guard::final_view ? output : sigmoid;
        auto * alias = ggml_view_tensor(ctx, source);
        BICORE_CHECK(alias->view_src == source && !alias->src[0]);
        auto * reader = ggml_dup(ctx, alias);
        ggml_set_output(reader);
        ggml_build_forward_expand(layout, reader);
    }
    const auto allocator =
        std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)>(ggml_gallocr_new(&gpu.type), ggml_gallocr_free);
    BICORE_CHECK(allocator && ggml_gallocr_alloc_graph(allocator.get(), layout));
    auto * const destination = output->data;
    if (guard == Guard::overlapping_storage) {
        BICORE_CHECK(packed->buffer == output->buffer);
        output->data = packed->data;
    }

    const bool should_fuse = guard == Guard::ordinary || guard == Guard::final_view || guard == Guard::prefill;
    const unsigned dispatches = should_fuse ? 2u : 10u;
    const auto run = [&](ggml_cgraph * graph, unsigned expected) {
        commands = {};
        BICORE_CHECK(ggml_backend_graph_compute_async(&gpu.backend, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(&gpu.backend);
        BICORE_CHECK(commands.executed == expected && commands.dispatches.size() == expected);
        if (should_fuse) {
            const auto pipeline = gpu.context->shader_lib->get_situ_pipeline();
            const auto & dispatch = commands.dispatches[1];
            BICORE_CHECK(pipeline.pipeline && dispatch.pipeline == pipeline.pipeline.Get()->id && dispatch.x == 72 * uint32_t(tokens) &&
                         dispatch.y == 1 && dispatch.z == 1);
        }
    };
    if (view_reader) {
        // The aliased reader belongs to a later scheduler slice.
        auto slice = ggml_graph_view(layout, 0, activation_end);
        run(&slice, dispatches);
    }
    run(layout, dispatches + unsigned(view_reader));
    output->data = destination;
}

static void all() {
    graph(Guard::ordinary);
    graph(Guard::intermediate_view);
    graph(Guard::final_view);
    graph(Guard::overlapping_storage);
    graph(Guard::prefill);
    graph(Guard::prefill_intermediate_view);
}

} // namespace situ_cases
