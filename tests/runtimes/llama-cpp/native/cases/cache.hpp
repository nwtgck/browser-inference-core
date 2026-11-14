#pragma once

namespace cache_cases {
static void key_only_ownership() {
    auto ctx = recording_context();
    auto & cache = ctx->bind_group_cache;
    auto buffer = wgpu::Buffer::Acquire(new WGPUBufferImpl);
    auto pipeline = wgpu::ComputePipeline::Acquire(new WGPUComputePipelineImpl{1, 46, "", "owned"});
    auto entries = std::vector{ggml_webgpu_make_bind_group_entry(0, buffer, 0, 16)};
    cache.begin_graph(0);
    cache.get(ctx->global_ctx->device, pipeline, 0, entries, "key ownership", 0);
    cache.end_graph();
    cache.slots[0].group = nullptr;
    const auto raw_buffer = buffer.Get();
    const auto raw_pipeline = pipeline.Get();
    const auto buffers = resources.buffers_released, pipelines = resources.pipelines_released;
    entries.clear();
    buffer = nullptr;
    pipeline = nullptr;
    BICORE_CHECK(raw_buffer->refs == 1 && raw_pipeline->refs == 1);
    BICORE_CHECK(resources.buffers_released == buffers && resources.pipelines_released == pipelines);
    cache.begin_graph(1);
    BICORE_CHECK(cache.slots.empty() && resources.buffers_released == buffers + 1 &&
                 resources.pipelines_released == pipelines + 1);
}
static void bounds_and_free() {
    auto ctx = recording_context();
    auto & global = *ctx->global_ctx;
    auto & cache = ctx->bind_group_cache;
    wgpu::BufferDescriptor descriptor{};
    descriptor.size = 128;
    descriptor.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    auto buffer = global.device.CreateBuffer(&descriptor);
    const auto pipeline = wgpu::ComputePipeline::Acquire(new WGPUComputePipelineImpl{1, 44, "", "cache"});
    auto entries = std::vector{ggml_webgpu_make_bind_group_entry(0, buffer, 0, 16)};
    const auto get = [&] {
        return cache.get(global.device, pipeline, 0, entries, "bounded", global.buffer_generation.load());
    };
    const auto initial = resources.groups_created;
    for (unsigned repetition = 0; repetition < 2; ++repetition) {
        cache.begin_graph(0);
        for (size_t index = 0; index < cache.max_slots + 3; ++index)
            BICORE_CHECK(get());
        cache.end_graph();
        BICORE_CHECK(cache.cursor == cache.max_slots && cache.slots.size() == cache.max_slots);
    }
    BICORE_CHECK(resources.groups_created - initial == cache.max_slots + 6);
    cache.begin_graph(0);
    BICORE_CHECK(get());
    cache.end_graph();
    BICORE_CHECK(cache.slots.size() == 1);
    cache.begin_graph(0);
    cache.end_graph();
    BICORE_CHECK(cache.slots.empty());
    cache.begin_graph(0);
    auto retained = get();
    const auto old_id = retained.Get()->id;
    // Call the actual buffer-free hook between two dispatch ordinals. The
    // shared epoch must invalidate entries even without another begin_graph.
    ggml_backend_buffer allocation{};
    allocation.context = new ggml_backend_webgpu_buffer_context(buffer, "retired", ctx->global_ctx);
    bool retired = false;
    resources.before_buffer_destroy = [&](WGPUBuffer value) {
        if (value == buffer.Get()) {
            BICORE_CHECK(global.buffer_generation.load() == 1 && !value->destroyed);
            retired = true;
        }
    };
    ggml_backend_webgpu_buffer_free_buffer(&allocation);
    resources.before_buffer_destroy = {};
    BICORE_CHECK(retired && global.buffer_generation.load() == 1 && buffer.Get()->destroyed);
    buffer = global.device.CreateBuffer(&descriptor);
    entries[0].buffer = buffer;
    auto replacement = get();
    BICORE_CHECK(replacement.Get()->id != old_id && cache.generation == 1 && cache.cursor == 2 &&
                 cache.slots.size() == 2 && !cache.slots[0].group);
    BICORE_CHECK(retained.Get()->buffers[0]->destroyed);
    cache.begin_graph(1);
    cache.end_graph();
    BICORE_CHECK(cache.slots.empty());
}

static void queued_ownership() {
    auto ctx = recording_context();
    auto & global = *ctx->global_ctx;
    webgpu_pipeline pipeline{wgpu::ComputePipeline::Acquire(new WGPUComputePipelineImpl{1, 45, "", "queued"}), "queued",
                             nullptr};
    wgpu::BufferDescriptor descriptor{};
    descriptor.size = 128;
    descriptor.usage = wgpu::BufferUsage::Storage;
    auto buffer = global.device.CreateBuffer(&descriptor);
    commands = {};
    wgpu::BindGroup old_group;
    for (uint32_t seed : {100u, 200u}) {
        ctx->param_arena.reset();
        ctx->bind_group_cache.begin_graph(0);
        ctx->active_command_encoder = global.device.CreateCommandEncoder();
#ifndef GGML_WEBGPU_GPU_PROFILE
        ctx->active_compute_pass = ctx->active_command_encoder.BeginComputePass();
#endif
        const auto offset = seed == 100 ? 0u : 4u;
        std::vector<webgpu_dispatch_desc> dispatches{
            {pipeline, {seed}, {ggml_webgpu_make_bind_group_entry(0, buffer, offset, 16)}, {1, 1}},
            {pipeline, {seed + 1}, {ggml_webgpu_make_bind_group_entry(0, buffer, offset, 16)}, {1, 1}}};
        BICORE_CHECK(ggml_backend_webgpu_build_multi(ctx, dispatches).num_kernels == 2);
        ctx->bind_group_cache.end_graph();
        if (seed == 100)
            old_group = ctx->bind_group_cache.slots[0].group;
        else
            BICORE_CHECK(ctx->bind_group_cache.slots[0].group.Get() != old_group.Get());
#ifndef GGML_WEBGPU_GPU_PROFILE
        ctx->active_compute_pass.End();
#endif
        ctx->active_compute_pass = nullptr;
        ctx->param_arena.flush(global.queue);
        const auto recorded = ctx->active_command_encoder.Finish();
        global.queue.Submit(1, &recorded);
        ctx->active_command_encoder = nullptr;
    }
    BICORE_CHECK(commands.executed == 0 && old_group.Get()->refs == 2);
    // No caller or cache ownership remains, but the pending command still owns
    // this obsolete group and its actual SDK binding/resource references.
    old_group = nullptr;
    const auto alive = resources.groups_alive;
    drain_queue(global.queue.Get());
    BICORE_CHECK((commands.observed_params == std::vector<uint32_t>{100, 101, 200, 201}));
    BICORE_CHECK(resources.groups_alive + 2 == alive && commands.executed == 4 && commands.queue_waits == 0);
}
static void actual_graphs() {
    auto ctx = recording_context();
    auto & global = *ctx->global_ctx;
    global.capabilities.limits.minStorageBufferOffsetAlignment = 256;
    global.capabilities.limits.maxComputeWorkgroupsPerDimension = 65535;
    global.capabilities.limits.maxComputeInvocationsPerWorkgroup = 256;
    global.capabilities.limits.maxStorageBufferBindingSize = 1024 * 1024;
    ctx->param_arena.cleanup();
    ctx->param_arena.init(global.device, 128, 74, 256);
#ifdef GGML_WEBGPU_GPU_PROFILE
    ctx->batch_compute_passes = false;
    wgpu::BufferDescriptor timestamps{};
    timestamps.size = WEBGPU_MAX_PROFILE_QUERY_COUNT * sizeof(uint64_t);
    timestamps.usage = wgpu::BufferUsage::QueryResolve | wgpu::BufferUsage::CopySrc;
    ctx->profile_timestamp_dev_buf = global.device.CreateBuffer(&timestamps);
    timestamps.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
    ctx->profile_timestamp_host_buf = global.device.CreateBuffer(&timestamps);
#endif
    ggml_backend_webgpu_context bridge;
    bridge.webgpu_ctx = ctx;
    ggml_backend backend{};
    backend.context = &bridge;
    backend.iface.graph_compute = ggml_backend_webgpu_graph_compute;
    const auto memory =
        std::unique_ptr<ggml_context, decltype(&ggml_free)>(ggml_init({2 * 1024 * 1024, nullptr, true}), ggml_free);
    BICORE_CHECK(memory);
    auto * graph = ggml_new_graph_custom(memory.get(), 256, false);
    auto * input = ggml_new_tensor_1d(memory.get(), GGML_TYPE_F32, 16);
    ggml_set_input(input);
    wgpu::BufferDescriptor descriptor{};
    descriptor.size = 256 * 132;
    descriptor.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    ggml_backend_webgpu_buffer_context storage(global.device.CreateBuffer(&descriptor), "graph", ctx->global_ctx);
    ggml_backend_buffer buffer{};
    buffer.context = &storage;
    buffer.size = descriptor.size;
    input->buffer = &buffer;
    input->data = webgpu_ptr_base;
    for (size_t index = 0; index < 130; ++index) {
        auto * output = ggml_scale(memory.get(), input, float(index + 1));
        output->buffer = &buffer;
        output->data = reinterpret_cast<void *>(uintptr_t(webgpu_ptr_base) + 256 * (index + 1));
        ggml_build_forward_expand(graph, output);
    }
    BICORE_CHECK(graph->n_nodes == 130);
    commands = {};
    readback = {};
#ifdef GGML_WEBGPU_GPU_PROFILE
    bool reentered = false;
    readback.before_map = [&] {
        reentered = true;
        BICORE_CHECK(ggml_backend_graph_compute_async(&backend, graph) == GGML_STATUS_FAILED);
    };
#endif
    const auto compute = [&] {
        BICORE_CHECK(ggml_backend_graph_compute_async(&backend, graph) == GGML_STATUS_SUCCESS);
    };
    compute();
    const auto created = resources.groups_created;
    BICORE_CHECK(ctx->bind_group_cache.slots.size() == 130);
    const auto batch_size = global.command_submit_batch_size;
    BICORE_CHECK(batch_size < 130);
    BICORE_CHECK(ctx->bind_group_cache.slots[batch_size - 1].bindings.back().offset == 256 * (batch_size - 1));
    BICORE_CHECK(ctx->bind_group_cache.slots[batch_size].bindings.back().offset == 0);
    const auto uploads = commands.writes.size();
    const auto submissions = commands.submissions;
    for (int index = 0; index < graph->n_nodes; ++index) {
        const float coefficient = float(2001 + index);
        std::memcpy(graph->nodes[index]->op_params, &coefficient, sizeof(coefficient));
    }
    compute();
    BICORE_CHECK(resources.groups_created == created);
    const auto batches = (130 + batch_size - 1) / batch_size;
    BICORE_CHECK(commands.writes.size() - uploads == batches);
    unsigned expected_submissions = batches;
#ifdef GGML_WEBGPU_GPU_PROFILE
    ++expected_submissions; // The actual timestamp collector submits its copy.
#endif
    BICORE_CHECK(commands.submissions - submissions == expected_submissions);
    // A warm cache still uploads the changed graph coefficients. Find each
    // value in actual upload bytes without pinning its parameter-field offset.
    for (unsigned index = 0; index < 130; ++index) {
        const float coefficient = float(2001 + index);
        std::array<uint8_t, sizeof(float)> bytes;
        std::memcpy(bytes.data(), &coefficient, bytes.size());
        size_t occurrences = 0;
        for (size_t upload = uploads; upload < commands.writes.size(); ++upload) {
            const auto & data = commands.writes[upload].bytes;
            for (size_t offset = 0; offset + bytes.size() <= data.size(); offset += sizeof(float))
                occurrences += std::memcmp(data.data() + offset, bytes.data(), bytes.size()) == 0;
        }
        BICORE_CHECK(occurrences == 1);
    }
    for (int index = 0; index < graph->n_nodes; ++index)
        graph->nodes[index]->data = reinterpret_cast<void *>(uintptr_t(graph->nodes[index]->data) + 256);
    compute();
    BICORE_CHECK(resources.groups_created == created + 130);
    graph->n_nodes = 2;
    compute();
    BICORE_CHECK(ctx->bind_group_cache.slots.size() == 2);
    graph->n_nodes = 0;
    compute();
    BICORE_CHECK(ctx->bind_group_cache.slots.empty());
#ifdef GGML_WEBGPU_GPU_PROFILE
    readback.before_map = {};
    BICORE_CHECK(reentered && !ctx->operation_in_progress && commands.maps == 4);
#else
    BICORE_CHECK(commands.maps == 0 && commands.executed == 0);
#endif
    drain_queue(global.queue.Get());
    BICORE_CHECK(commands.executed == 392 && commands.queue_waits == 0);
    BICORE_CHECK(resources.groups_alive == 0);
}
static void all() {
    key_only_ownership();
    bounds_and_free();
    queued_ownership();
    actual_graphs();
}
} // namespace cache_cases
