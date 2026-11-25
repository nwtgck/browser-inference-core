#pragma once

namespace expert_cases {

static void indexed_wide_m(const std::filesystem::path & output) {
    struct Fixture {
        ggml_type type;
        int64_t k, m, activation_slots;
        bool eligible;
        int64_t experts = 512;
        int64_t selected = 10;
    };
    constexpr Fixture fixtures[] = {
        {GGML_TYPE_IQ3_S, 2560, 640, 1, true},          {GGML_TYPE_Q4_K, 2560, 640, 1, true},
        {GGML_TYPE_IQ4_XS, 2560, 640, 1, true},         {GGML_TYPE_IQ4_NL, 640, 2560, 10, true},
        {GGML_TYPE_IQ2_S, 2560, 640, 1, true},          {GGML_TYPE_IQ3_XXS, 2560, 640, 1, true},
        {GGML_TYPE_Q5_1, 640, 2560, 10, true},          {GGML_TYPE_IQ2_XS, 2560, 640, 1, false},
        {GGML_TYPE_Q5_0, 640, 2560, 10, false},         {GGML_TYPE_IQ4_NL, 2304, 2304, 1, true, 128, 8},
        {GGML_TYPE_NVFP4, 1152, 2304, 8, true, 128, 8},
    };
    for (const auto [type, k, m, activation_slots, eligible, experts, selected] : fixtures) {
        auto ctx = recording_context();
        const bool narrow_k = experts == 128;
        if (narrow_k) {
            ctx->global_ctx->vendor = "apple";
        }
        auto & limits = ctx->global_ctx->capabilities.limits;
        limits.maxBufferSize = 1024 * 1024 * 1024;
        limits.maxStorageBufferBindingSize = limits.maxBufferSize;
        limits.minStorageBufferOffsetAlignment = 256;
        limits.maxStorageBuffersPerShaderStage = 8;
        limits.maxComputeWorkgroupSizeX = limits.maxComputeInvocationsPerWorkgroup = 256;
        limits.maxComputeWorkgroupsPerDimension = 65535;

        std::unique_ptr<ggml_context, decltype(&ggml_free)> graph(
            ggml_init({ggml_tensor_overhead() * 6, nullptr, true}), &ggml_free);
        auto * weights = ggml_new_tensor_3d(graph.get(), type, k, m, experts);
        auto * input = ggml_new_tensor_3d(graph.get(), GGML_TYPE_F32, k, activation_slots, 128);
        auto * ids = ggml_new_tensor_2d(graph.get(), GGML_TYPE_I32, selected, 128);
        auto * dst = ggml_mul_mat_id(graph.get(), weights, input, ids);
        if (type == GGML_TYPE_NVFP4) {
            BICORE_CHECK(ggml_prec_set_acc(dst, GGML_PREC_F32));
        }
        ggml_webgpu_shader_lib_context context{};
        context.src0 = weights;
        context.src1 = input;
        context.src2 = ids;
        context.dst = dst;
        context.max_wg_size = limits.maxComputeInvocationsPerWorkgroup;
        context.prefer_narrow_indexed_k = narrow_k;

        const auto select = [&](uint32_t memory, uint32_t expected_rows) {
            context.wg_mem_limit_bytes = memory;
            const auto pipeline = ctx->shader_lib->get_mul_mat_id_pipeline(context, false);
            const auto & d = *static_cast<const ggml_webgpu_mul_mat_shader_decisions *>(pipeline.context.get());
            BICORE_CHECK(d.tile_m * d.wg_size_m == expected_rows && d.tile_n * d.wg_size_n == 32);
            BICORE_CHECK(d.tile_k == 32 && d.wg_size == 64);
            return pipeline;
        };
        const auto narrow = select(6399, 32);
        const auto chosen = select(6400, eligible ? 64 : 32);
        BICORE_CHECK((chosen.pipeline.Get() != narrow.pipeline.Get()) == eligible);
        BICORE_CHECK(select(32768, eligible ? 64 : 32).pipeline.Get() == chosen.pipeline.Get());
        if (!eligible) {
            continue;
        }
        if (narrow_k) {
            context.prefer_narrow_indexed_k = false;
            const auto pipeline = ctx->shader_lib->get_mul_mat_id_pipeline(context, false);
            const auto & d = *static_cast<const ggml_webgpu_mul_mat_shader_decisions *>(pipeline.context.get());
            BICORE_CHECK(d.tile_k == 64 && d.tile_m * d.wg_size_m == 32);
            context.prefer_narrow_indexed_k = true;
        }
        if (type == GGML_TYPE_NVFP4) {
            BICORE_CHECK(ggml_prec_set_acc(dst, GGML_PREC_UNDEFINED));
            const auto pipeline = select(6400, 32);
            BICORE_CHECK(pipeline.pipeline.Get() != chosen.pipeline.Get());
            BICORE_CHECK(ggml_prec_set_acc(dst, GGML_PREC_F32));
        }

        context.max_wg_size = 63;
        BICORE_CHECK(select(6400, 32).pipeline.Get() == narrow.pipeline.Get());
        context.max_wg_size = 64;
        BICORE_CHECK(select(6400, 64).pipeline.Get() == chosen.pipeline.Get());
        context.max_wg_size = limits.maxComputeInvocationsPerWorkgroup;
        const auto check_extent = [&](int64_t & extent, int64_t rejected) {
            const auto original = extent;
            extent = rejected;
            BICORE_CHECK(select(6400, 32).pipeline.Get() == narrow.pipeline.Get());
            extent = original;
        };
        check_extent(input->ne[1], activation_slots == 1 ? selected : 1);
        check_extent(ids->ne[0], selected - 1);
        check_extent(ids->ne[1], 127);
        check_extent(dst->ne[2], 109);
        check_extent(weights->ne[1], m - 1);
        if (narrow_k) {
            check_extent(ids->ne[2], 2);
        }
        BICORE_CHECK(select(6400, 64).pipeline.Get() == chosen.pipeline.Get());
        BICORE_CHECK(select(6399, 32).pipeline.Get() == narrow.pipeline.Get());

        ggml_backend_webgpu_device_context device_context{};
        device_context.webgpu_global_ctx = ctx->global_ctx;
        ggml_backend_device device{};
        device.context = &device_context;
        ggml_backend_buffer_type buffer_type{};
        buffer_type.device = &device;
        const std::array operands{weights, input, ids, dst};
        std::array<ggml_backend_buffer, operands.size()> buffers{};
        std::vector<ggml_backend_webgpu_buffer_context> backing;
        backing.reserve(operands.size());
        for (size_t i = 0; i < operands.size(); ++i) {
            backing.emplace_back(wgpu::Buffer::Acquire(new WGPUBufferImpl), "metadata-only binding", ctx->global_ctx);
            buffers[i].context = &backing.back();
            buffers[i].size = ggml_backend_webgpu_buffer_type_get_alloc_size(&buffer_type, operands[i]);
            operands[i]->data = webgpu_ptr_base;
            operands[i]->buffer = &buffers[i];
        }
        const auto max_tiles = ggml_webgpu_mul_mat_id_max_tiles(dst);
        BICORE_CHECK(max_tiles == (narrow_k ? 156 : 536));
        std::vector<uint8_t> narrow_parameters;
        for (const auto & pipeline : {narrow, chosen}) {
            const bool wide = pipeline.pipeline.Get() == chosen.pipeline.Get();
            limits.maxComputeWorkgroupStorageSize = wide ? 6400 : 6399;
            commands = {};
            ctx->active_command_encoder = ctx->global_ctx->device.CreateCommandEncoder();
            ctx->bind_group_cache.begin_graph(wide ? 1 : 0);
#ifndef GGML_WEBGPU_GPU_PROFILE
            ctx->batch_compute_passes = true;
            ctx->active_compute_pass = ctx->active_command_encoder.BeginComputePass();
#endif
            const auto encoded = ggml_webgpu_mul_mat_id(ctx, weights, input, ids, dst);
#ifndef GGML_WEBGPU_GPU_PROFILE
            ctx->active_compute_pass.End();
            ctx->active_compute_pass = nullptr;
#endif
            ctx->bind_group_cache.end_graph();
            ctx->param_arena.flush(ctx->global_ctx->queue);
            BICORE_CHECK(encoded.num_kernels == 2 && commands.dispatches.size() == 2 && commands.writes.size() == 1);
            BICORE_CHECK(commands.dispatches.front().x == 1 && commands.dispatches.front().y == 1);
            const auto & main = commands.dispatches.back();
            BICORE_CHECK(main.pipeline == pipeline.pipeline.Get()->id && main.y == 1 && main.z == 1);
            BICORE_CHECK(main.x == max_tiles * m / (wide ? 64 : 32));
            BICORE_CHECK(commands.writes.front().bytes.size() == 512);
            if (wide) {
                BICORE_CHECK(commands.writes.front().bytes == narrow_parameters);
            } else {
                narrow_parameters = commands.writes.front().bytes;
            }
#ifdef GGML_WEBGPU_GPU_PROFILE
            const uint32_t first = wide ? 4 : 0;
            BICORE_CHECK(commands.timestamp_pairs ==
                         (std::vector<std::pair<uint32_t, uint32_t>>{{first, first + 1}, {first + 2, first + 3}}));
#endif
            const auto encoded_commands = ctx->active_command_encoder.Finish();
            ctx->global_ctx->queue.Submit(1, &encoded_commands);
            drain_queue(ctx->global_ctx->queue.Get());
            BICORE_CHECK(commands.executed == 2);
            ctx->param_arena.reset();
            const std::string_view prefix = narrow_k ? "indexed-wide-m-e128-" : "indexed-wide-m-";
            std::ofstream shader(output / (std::string(prefix) + pipeline.name + ".wgsl"));
            shader << pipeline.pipeline.Get()->code;
            BICORE_CHECK(bool(shader));
        }
    }
}

} // namespace expert_cases
