#pragma once

namespace argsort_cases {

static void all(const std::filesystem::path & output) {
    struct Case {
        uint32_t network, elements, workgroup;
        ggml_sort_order order;
        bool top_k = false;
        uint32_t max_x = 1024;
        uint32_t max_invocations = 1024;
        uint32_t max_storage = 32768;
    };
    constexpr Case cases[] = {
        {1024, 128, 128, GGML_SORT_ORDER_DESC},
        {1024, 256, 256, GGML_SORT_ORDER_DESC},
        {1024, 512, 512, GGML_SORT_ORDER_DESC},
        {1024, 128, 128, GGML_SORT_ORDER_ASC},
        {1024, 64, 1024, GGML_SORT_ORDER_DESC},
        {1024, 127, 1024, GGML_SORT_ORDER_DESC},
        {1024, 1024, 1024, GGML_SORT_ORDER_DESC},
        {512, 128, 128, GGML_SORT_ORDER_DESC, false, 512},
        {512, 128, 128, GGML_SORT_ORDER_DESC, false, 600},
        {512, 128, 128, GGML_SORT_ORDER_ASC, false, 1024, 512},
        {512, 256, 256, GGML_SORT_ORDER_DESC, false, 1024, 1024, 2048},
        {256, 128, 128, GGML_SORT_ORDER_DESC, false, 256},
        {128, 128, 128, GGML_SORT_ORDER_DESC, false, 128},
        {1024, 128, 128, GGML_SORT_ORDER_DESC, true},
        {64, 128, 64, GGML_SORT_ORDER_DESC, true, 64},
        {64, 128, 64, GGML_SORT_ORDER_DESC, false, 64},
        {1024, 128, 128, GGML_SORT_ORDER_DESC},
    };
    auto ctx = recording_context();
    auto & limits = ctx->global_ctx->capabilities.limits;
    limits.maxComputeWorkgroupsPerDimension = 65535;
    limits.minStorageBufferOffsetAlignment = 256;
    ggml_backend_webgpu_device_context device_context{};
    device_context.webgpu_global_ctx = ctx->global_ctx;
    ggml_backend_device device{};
    device.context = &device_context;
    ggml_backend_buffer_type buffer_type{};
    buffer_type.device = &device;

    std::map<std::tuple<uint32_t, uint32_t, ggml_sort_order>, unsigned> seen;
    for (const auto & c : cases) {
        limits.maxComputeWorkgroupSizeX = c.max_x;
        limits.maxComputeInvocationsPerWorkgroup = c.max_invocations;
        limits.maxComputeWorkgroupStorageSize = c.max_storage;
        std::unique_ptr<ggml_context, decltype(&ggml_free)> mem(ggml_init({ggml_tensor_overhead() * 3, nullptr, true}),
                                                                &ggml_free);
        auto * src = ggml_new_tensor_2d(mem.get(), GGML_TYPE_F32, c.elements, 3);
        auto * dst = c.top_k ? ggml_top_k(mem.get(), src, 8) : ggml_argsort(mem.get(), src, c.order);
        const auto operands = std::array{src, dst};
        std::vector<ggml_backend_webgpu_buffer_context> backing;
        backing.reserve(operands.size());
        std::array<ggml_backend_buffer, operands.size()> buffers{};
        for (size_t i = 0; i < operands.size(); ++i) {
            backing.emplace_back(wgpu::Buffer::Acquire(new WGPUBufferImpl), "argsort test", ctx->global_ctx);
            buffers[i].context = &backing.back();
            buffers[i].size = ggml_backend_webgpu_buffer_type_get_alloc_size(&buffer_type, operands[i]);
            operands[i]->buffer = &buffers[i];
            operands[i]->data = webgpu_ptr_base;
        }
        ggml_webgpu_shader_lib_context context{};
        context.src0 = src;
        context.dst = dst;
        context.max_wg_size = std::min(c.max_x, c.max_invocations);
        context.wg_mem_limit_bytes = c.max_storage;
        const auto pipeline = ctx->shader_lib->get_argsort_pipeline(context);
        const auto & decisions = *static_cast<const ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());
        BICORE_CHECK(decisions.wg_size == c.workgroup);
        const auto key = std::tuple{c.network, c.workgroup, c.order};
        const auto [position, inserted] = seen.emplace(key, pipeline.pipeline.Get()->id);
        BICORE_CHECK(inserted || position->second == pipeline.pipeline.Get()->id);
        for (const auto & [old_key, id] : seen)
            BICORE_CHECK(old_key == key || id != pipeline.pipeline.Get()->id);

        ctx->active_command_encoder = ctx->global_ctx->device.CreateCommandEncoder();
        commands = {};
#ifndef GGML_WEBGPU_GPU_PROFILE
        ctx->batch_compute_passes = true;
        ctx->active_compute_pass = ctx->active_command_encoder.BeginComputePass();
#endif
        const unsigned pipelines_before = created;
        const auto encoded = ggml_webgpu_argsort(ctx, src, dst);
#ifndef GGML_WEBGPU_GPU_PROFILE
        ctx->active_compute_pass.End();
        ctx->active_compute_pass = nullptr;
#endif
        const unsigned tiles = c.elements > c.workgroup ? 2 : 1;
        BICORE_CHECK(encoded.num_kernels == tiles && commands.dispatches.size() == tiles);
        BICORE_CHECK(commands.dispatches[0].pipeline == pipeline.pipeline.Get()->id);
        BICORE_CHECK(commands.dispatches[0].x == 3 * tiles && commands.dispatches[0].y == 1);
        if (tiles == 1)
            BICORE_CHECK(created == pipelines_before);
        ctx->param_arena.flush(ctx->global_ctx->queue);
        BICORE_CHECK(commands.writes.size() == 1 && commands.writes[0].bytes.size() >= 15 * sizeof(uint32_t));
        std::array<uint32_t, 15> params{};
        std::memcpy(params.data(), commands.writes[0].bytes.data(), sizeof(params));
        BICORE_CHECK(params[8] == c.elements && params[12] == (c.top_k ? 8u : c.workgroup));
        BICORE_CHECK(params[13] == tiles && params[14] == 3);
        const auto finished = ctx->active_command_encoder.Finish();
        ctx->global_ctx->queue.Submit(1, &finished);
        drain_queue(ctx->global_ctx->queue.Get());
        BICORE_CHECK(commands.executed == tiles);
        ctx->param_arena.reset();
        if (inserted) {
            const auto name = "argsort-" + std::to_string(c.network) + "-" + std::to_string(c.elements) + "-" +
                              std::to_string(c.order) + ".wgsl";
            std::ofstream shader(output / name);
            shader << pipeline.pipeline.Get()->code;
            BICORE_CHECK(bool(shader));
        }
    }
}

} // namespace argsort_cases
