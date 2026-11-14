#pragma once

#ifdef GGML_WEBGPU_GPU_PROFILE
namespace profile_query_cases {
struct Case {
    const char * name;
    std::vector<size_t> counts;
    bool arena_overflow = false;
};

static void run(const Case & input, bool batched) {
    auto ctx = recording_context();
    ctx->batch_compute_passes = batched;
    ctx->param_arena.cleanup();
    ctx->param_arena.init(ctx->global_ctx->device, 128, 74, 256);
    const size_t query_bytes = WEBGPU_MAX_PROFILE_QUERY_COUNT * sizeof(uint64_t);
    wgpu::BufferDescriptor descriptor{};
    descriptor.size = query_bytes;
    descriptor.usage = wgpu::BufferUsage::QueryResolve | wgpu::BufferUsage::CopySrc;
    ctx->profile_timestamp_dev_buf = ctx->global_ctx->device.CreateBuffer(&descriptor);
    descriptor.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
    ctx->profile_timestamp_host_buf = ctx->global_ctx->device.CreateBuffer(&descriptor);
    ggml_backend_webgpu_context bridge;
    bridge.webgpu_ctx = ctx;
    ggml_backend backend{};
    backend.context = &bridge;
    std::vector<ggml_tensor> tensors(input.counts.size());
    std::vector<ggml_tensor *> nodes;
    std::vector<uint32_t> expected;
    for (size_t index = 0; index < tensors.size(); ++index) {
        nodes.push_back(&tensors[index]);
        for (size_t dispatch = 0; dispatch < input.counts[index]; ++dispatch)
            expected.push_back(uint32_t(17 + index));
    }
    ggml_cgraph graph{};
    graph.n_nodes = int(nodes.size());
    graph.nodes = nodes.data();
    webgpu_pipeline pipeline{wgpu::ComputePipeline::Acquire(new WGPUComputePipelineImpl{1, 43, "", "query"}), "query",
                             nullptr};
    commands = {};
    const auto previous_waits = waits;
    bool rejected = false;
    try {
        // The production template owns batching, submission, arena reset and
        // collection. Only graph-node encoding is a normal typed callback.
        const auto status = ggml_backend_webgpu_graph_compute_impl(
            &backend, &graph, [&](webgpu_context & current, ggml_cgraph * current_graph, int index, int & encoded) {
                BICORE_CHECK(current == ctx && current_graph == &graph && encoded == 1);
                std::vector<webgpu_dispatch_desc> dispatches(input.counts.at(size_t(index)),
                                                             {pipeline, {uint32_t(17 + index)}, {}, {1, 1}});
                return std::optional<webgpu_encoded_op>{ggml_backend_webgpu_build_multi(current, dispatches)};
            });
        BICORE_CHECK(status == GGML_STATUS_SUCCESS);
    } catch (const std::runtime_error & error) {
        rejected = true;
        BICORE_CHECK(input.arena_overflow &&
                     std::string(error.what()).find("parameter arena exhausted") != std::string::npos);
    }
    BICORE_CHECK(rejected == input.arena_overflow);
    drain_queue(ctx->global_ctx->queue.Get());
    if (rejected) {
        BICORE_CHECK(ctx->param_arena.next_slot == ctx->param_arena.slot_count);
        BICORE_CHECK(commands.observed_params.size() <= expected.size() &&
                     std::equal(commands.observed_params.begin(), commands.observed_params.end(), expected.begin()));
    } else {
        BICORE_CHECK(commands.observed_params == expected && commands.executed == expected.size());
        BICORE_CHECK(ctx->param_arena.next_slot == 0);
        BICORE_CHECK(commands.maps == commands.resolved_queries.size() && commands.unmaps == commands.maps);
        BICORE_CHECK(commands.queue_waits == 0 && waits - previous_waits == commands.maps);
        size_t total_queries = 0;
        for (size_t index = 0; index < commands.resolved_queries.size(); ++index) {
            const auto count = commands.resolved_queries[index];
            BICORE_CHECK(count > 0 && count <= WEBGPU_MAX_PROFILE_QUERY_COUNT && count % 2 == 0);
            total_queries += count;
            // Collect before the next largest accepted arena can overflow,
            // but do not introduce an earlier profiling readback per batch.
            if (index + 1 < commands.resolved_queries.size())
                BICORE_CHECK(count + 2 * ctx->param_arena.slot_count >= WEBGPU_MAX_PROFILE_QUERY_COUNT);
        }
        BICORE_CHECK(total_queries == 2 * expected.size());
        BICORE_CHECK(commands.timestamp_pairs.size() == expected.size());
    }
    // Intercepted native abort cleanup only; production traps do not recover.
    ctx->active_compute_pass = nullptr;
    ctx->active_command_encoder = nullptr;
    std::cout << "profile query case " << input.name << " passed\n";
}

static void all() {
    std::vector<Case> cases{{"empty", {}},
                            {"one", {1}},
                            {"boundary64", std::vector<size_t>(64, 1)},
                            {"boundary65", std::vector<size_t>(65, 1)},
                            {"single1983", std::vector<size_t>(1983, 1)},
                            {"single1984", std::vector<size_t>(1984, 1)},
                            {"single2048", std::vector<size_t>(2048, 1)}};
    auto prefix = std::vector<size_t>(185, 11);
    prefix.push_back(8);
    cases.push_back({"prefix186-q4086", prefix});
    for (const auto extra : {5u, 6u, 11u}) {
        auto counts = prefix;
        counts.push_back(extra);
        cases.push_back({extra == 5   ? "exact2048-dispatches"
                         : extra == 6 ? "first2049-dispatches"
                                      : "minimal187nodes-max74slots",
                         std::move(counts)});
    }
    auto slots = std::vector<size_t>(63, 1);
    slots.push_back(11);
    cases.push_back({"exact74-arena-slots", slots});
    slots.back() = 12;
    cases.push_back({"existing75-arena-rejection", std::move(slots), true});
    std::vector<size_t> small;
    for (int batch = 0; batch < 21; ++batch) {
        small.insert(small.end(), 15, 4);
        small.push_back(3);
        small.push_back(4);
    }
    small.insert(small.end(), 9 * 16, 4);
    small.insert(small.end(), 15, 4);
    small.push_back(3);
    small.push_back(4);
    cases.push_back({"small-width518nodes", std::move(small)});
    BICORE_CHECK(cases.size() == 14);
    for (const bool batched : {false, true})
        for (const auto & input : cases)
            run(input, batched);
}
} // namespace profile_query_cases
#endif
