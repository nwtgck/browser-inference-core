#pragma once

namespace depth_mix_cases {

enum class Guard { ordinary, graph_output, external_consumer };

struct Mix {
    ggml_tensor * normalized;
    ggml_tensor * probabilities;
    ggml_tensor * output;
};

static ggml_tensor * input(ggml_context * ctx, int rows) {
    auto * tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2304, rows);
    ggml_set_input(tensor);
    return tensor;
}

static Mix mix(ggml_context * ctx, ggml_tensor * current, int depth, int queries) {
    auto * stack = ggml_reshape_3d(ctx, current, 2304, 1, 1);
    for (int i = 1; i < depth; ++i) {
        auto * earlier = ggml_reshape_3d(ctx, input(ctx, 1), 2304, 1, 1);
        stack = ggml_concat(ctx, earlier, stack, 1);
    }
    auto * normalized = ggml_rms_norm(ctx, stack, 1e-6f);
    auto * query = input(ctx, queries);
    auto * expanded = ggml_repeat(ctx, query, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2304, queries, 1));
    auto * scores = ggml_mul_mat(ctx, normalized, expanded);
    auto * probabilities = ggml_soft_max_ext(ctx, scores, nullptr, 1.0f / std::sqrt(2304.0f), 0.0f);
    auto * output = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, normalized)), probabilities);
    ggml_set_output(output);
    return {normalized, probabilities, output};
}

static void check_grid(const RecordedDispatch & dispatch, const webgpu_pipeline & pipeline, uint32_t x, uint32_t y) {
    BICORE_CHECK(pipeline.pipeline && dispatch.pipeline == pipeline.pipeline.Get()->id && dispatch.x == x &&
                 dispatch.y == y && dispatch.z == 1);
}

static void graph(uint32_t subgroup, Guard guard) {
    transfer_cases::Backend gpu(transfer_cases::BackendKind::webgpu);
    auto & capabilities = gpu.context->global_ctx->capabilities;
    capabilities.supports_subgroups = subgroup != 0;
    capabilities.min_subgroup_size = subgroup;
    capabilities.max_subgroup_size = subgroup;
    const auto memory = transfer_cases::Memory(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
    BICORE_CHECK(memory);
    auto * ctx = memory.get();
    auto * layout = ggml_new_graph_custom(ctx, 64, false);
    const auto first = mix(ctx, input(ctx, 1), 2, 1);
    const auto second = mix(ctx, first.output, 4, 3);
    if (guard == Guard::graph_output) {
        ggml_set_output(second.normalized);
    }
    ggml_build_forward_expand(layout, second.output);
    if (guard == Guard::external_consumer) {
        auto * consumer = ggml_sqr(ctx, second.normalized);
        ggml_set_output(consumer);
        ggml_build_forward_expand(layout, consumer);
    }
    const auto allocator =
        std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)>(ggml_gallocr_new(&gpu.type), ggml_gallocr_free);
    BICORE_CHECK(allocator && ggml_gallocr_alloc_graph(allocator.get(), layout));
    const auto run = [&] {
        commands = {};
        readback = {};
        BICORE_CHECK(ggml_backend_graph_compute_async(&gpu.backend, layout) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(&gpu.backend);
        BICORE_CHECK(commands.dispatches.size() == commands.executed);
    };
    run();
    const unsigned expected = guard == Guard::ordinary ? 6u : 12u + unsigned(guard == Guard::external_consumer);
    BICORE_CHECK(commands.executed == expected);
    const auto & first_pipelines = gpu.context->depth_mix_pipelines[0];
    const auto & second_pipelines = gpu.context->depth_mix_pipelines[2];
    const uint32_t blend_groups = subgroup ? 18 : 576;
    check_grid(commands.dispatches[0], first_pipelines[0], 2, 1);
    check_grid(commands.dispatches[1], first_pipelines[1], 1, 1);
    check_grid(commands.dispatches[2], first_pipelines[2], blend_groups, 1);
    if (guard != Guard::ordinary) {
        BICORE_CHECK(!second_pipelines[0].pipeline);
        return;
    }

#ifndef GGML_WEBGPU_GPU_PROFILE
    BICORE_CHECK(commands.passes == 1 && commands.ends == 1 && commands.submissions == 1);
#endif
    check_grid(commands.dispatches[3], second_pipelines[0], 4, 1);
    check_grid(commands.dispatches[4], second_pipelines[1], 3, 1);
    check_grid(commands.dispatches[5], second_pipelines[2], blend_groups, 3);
    const auto values = gpu.context->depth_mix_values.Get();
    const auto probabilities = gpu.context->depth_mix_probabilities.Get();
    BICORE_CHECK(values && probabilities && values != probabilities);
    const auto pipelines_created = ::created;
    const auto groups_created = resources.groups_created;
    run();
    BICORE_CHECK(commands.executed == 6 && ::created == pipelines_created &&
                 resources.groups_created == groups_created);
    BICORE_CHECK(values == gpu.context->depth_mix_values.Get() &&
                 probabilities == gpu.context->depth_mix_probabilities.Get());
}

static void storage_limit(uint32_t subgroup, int depth, uint32_t limit) {
    transfer_cases::Backend gpu(transfer_cases::BackendKind::webgpu);
    auto & capabilities = gpu.context->global_ctx->capabilities;
    capabilities.supports_subgroups = subgroup != 0;
    capabilities.min_subgroup_size = subgroup;
    capabilities.max_subgroup_size = subgroup;
    capabilities.limits.maxComputeWorkgroupStorageSize = limit;
    const auto memory = transfer_cases::Memory(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
    BICORE_CHECK(memory);
    auto * ctx = memory.get();
    auto * layout = ggml_new_graph_custom(ctx, 64, false);
    const auto selected = mix(ctx, input(ctx, 1), depth, 1);
    ggml_build_forward_expand(layout, selected.output);
    const auto allocator =
        std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)>(ggml_gallocr_new(&gpu.type), ggml_gallocr_free);
    BICORE_CHECK(allocator && ggml_gallocr_alloc_graph(allocator.get(), layout));
    const auto run = [&](unsigned expected) {
        commands = {};
        readback = {};
        BICORE_CHECK(ggml_backend_graph_compute_async(&gpu.backend, layout) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(&gpu.backend);
        BICORE_CHECK(commands.dispatches.size() == expected && commands.executed == expected);
    };
    --capabilities.limits.maxComputeWorkgroupStorageSize;
    run(unsigned(depth + 5));
    BICORE_CHECK(!gpu.context->depth_mix_values && !gpu.context->depth_mix_probabilities);
    ++capabilities.limits.maxComputeWorkgroupStorageSize;
    run(3);
}

static void view_reader(bool of_intermediate) {
    transfer_cases::Backend gpu(transfer_cases::BackendKind::webgpu);
    const auto memory = transfer_cases::Memory(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
    BICORE_CHECK(memory);
    auto * ctx = memory.get();
    auto * layout = ggml_new_graph_custom(ctx, 64, false);
    const auto selected = mix(ctx, input(ctx, 1), 2, 1);
    ggml_build_forward_expand(layout, selected.output);
    const int mix_end = ggml_graph_n_nodes(layout);
    auto * source = of_intermediate ? selected.probabilities : selected.output;
    auto * alias = ggml_view_tensor(ctx, source);
    BICORE_CHECK(alias->view_src == source && !alias->src[0]);
    auto * reader = ggml_dup(ctx, alias);
    ggml_set_output(reader);
    ggml_build_forward_expand(layout, reader);
    const auto allocator =
        std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)>(ggml_gallocr_new(&gpu.type), ggml_gallocr_free);
    BICORE_CHECK(allocator && ggml_gallocr_alloc_graph(allocator.get(), layout));
    const auto run = [&](ggml_cgraph * graph, unsigned expected) {
        commands = {};
        BICORE_CHECK(ggml_backend_graph_compute_async(&gpu.backend, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(&gpu.backend);
        BICORE_CHECK(commands.executed == expected && commands.dispatches.size() == expected);
    };
    // A scheduler slice can end before the reader and still shares the parent visit set.
    auto slice = ggml_graph_view(layout, 0, mix_end);
    const unsigned mix_dispatches = of_intermediate ? 7 : 3;
    run(&slice, mix_dispatches);
    run(layout, mix_dispatches + 1);
}

static void all() {
    graph(0, Guard::ordinary);
    graph(16, Guard::ordinary);
    graph(32, Guard::ordinary);
    graph(16, Guard::graph_output);
    graph(16, Guard::external_consumer);
    view_reader(true);
    view_reader(false);
    // The unoptimized blend has more workgroup memory than the score at D2 and D3.
    storage_limit(0, 2, 4096);
    storage_limit(0, 3, 4096);
    storage_limit(16, 2, 2096);
    storage_limit(32, 5, 5184);
}

} // namespace depth_mix_cases
