#pragma once

namespace expert_cases {

static void indexed_q4_0_paired(const std::filesystem::path & output) {
    for (const int experts : {48, 128}) {
        auto ctx = recording_context();
        auto & limits = ctx->global_ctx->capabilities.limits;
        limits.maxBufferSize = 1024 * 1024 * 1024;
        limits.maxStorageBufferBindingSize = limits.maxBufferSize;
        limits.minStorageBufferOffsetAlignment = 256;
        limits.maxStorageBuffersPerShaderStage = 8;
        limits.maxComputeWorkgroupSizeX = limits.maxComputeInvocationsPerWorkgroup = 256;
        limits.maxComputeWorkgroupsPerDimension = 65535;

        std::unique_ptr<ggml_context, decltype(&ggml_free)> graph(
            ggml_init({ggml_tensor_overhead() * 5, nullptr, true}), &ggml_free);
        auto * weights = ggml_new_tensor_3d(graph.get(), GGML_TYPE_Q4_0, 2816, 1408, experts);
        auto * input = ggml_new_tensor_3d(graph.get(), GGML_TYPE_F32, 2816, 1, 128);
        auto * ids = ggml_new_tensor_2d(graph.get(), GGML_TYPE_I32, 8, 128);
        auto * dst = ggml_mul_mat_id(graph.get(), weights, input, ids);
        ggml_webgpu_shader_lib_context context{};
        context.src0 = weights;
        context.src1 = input;
        context.src2 = ids;
        context.dst = dst;
        context.indexed_src0_aligned = true;
        context.max_wg_size = 64;

        const auto select = [&](uint32_t storage, bool paired) {
            context.wg_mem_limit_bytes = storage;
            const auto pipeline = ctx->shader_lib->get_mul_mat_id_pipeline(context, false);
            const auto & d = *static_cast<const ggml_webgpu_mul_mat_shader_decisions *>(pipeline.context.get());
            BICORE_CHECK(d.tile_m * d.wg_size_m == (paired ? 64 : 32) && d.tile_k == (paired ? 64 : 32));
            BICORE_CHECK(d.tile_n * d.wg_size_n == 32 && d.wg_size == 64);
            BICORE_CHECK((pipeline.name.find("_pairs") != std::string::npos) == paired);
            BICORE_CHECK((pipeline.pipeline.Get()->code.find("load_q4_aligned_pair(") != std::string::npos) == paired);
            return pipeline;
        };
        const auto stock = select(12543, false);
        const auto paired = select(12544, true);
        BICORE_CHECK(stock.pipeline.Get() != paired.pipeline.Get());
        context.max_wg_size = 63;
        BICORE_CHECK(select(12544, false).pipeline.Get() == stock.pipeline.Get());
        context.max_wg_size = 64;
        context.indexed_src0_aligned = false;
        BICORE_CHECK(select(12544, false).pipeline.Get() == stock.pipeline.Get());
        context.indexed_src0_aligned = true;
        const auto check_rejected = [&](int64_t & value, int64_t alternative) {
            const auto original = value;
            value = alternative;
            BICORE_CHECK(select(12544, false).pipeline.Get() == stock.pipeline.Get());
            value = original;
        };
        check_rejected(input->ne[1], 8);
        check_rejected(input->ne[2], 127);
        check_rejected(ids->ne[0], 7);
        check_rejected(dst->ne[2], 53);
        check_rejected(weights->ne[0], 2784);
        check_rejected(weights->ne[1], 1344);
        ggml_tensor view = *weights;
        view.view_src = weights;
        context.src0 = &view;
        BICORE_CHECK(select(12544, false).pipeline.Get() == stock.pipeline.Get());
        view.view_src = nullptr;
        ++view.nb[1];
        BICORE_CHECK(select(12544, false).pipeline.Get() == stock.pipeline.Get());
        view.type = GGML_TYPE_Q5_0;
        BICORE_CHECK(select(12544, false).name == "mul_mat_id_q5_0_f32");
        context.src0 = weights;
        BICORE_CHECK(select(16384, true).pipeline.Get() == paired.pipeline.Get());

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
        BICORE_CHECK(ggml_backend_webgpu_device_supports_op(&device, dst));
        limits.maxStorageBufferBindingSize = 128 * 1024 * 1024;
        BICORE_CHECK(ggml_backend_webgpu_device_supports_op(&device, dst) == (experts == 48));
        limits.maxStorageBufferBindingSize = limits.maxBufferSize;
        const auto tiles = ggml_webgpu_mul_mat_id_max_tiles(dst);
        BICORE_CHECK(tiles == (experts == 48 ? 78u : 156u));
        std::vector<uint8_t> parameters;
        for (uint32_t arm = 0; arm < 3; ++arm) {
            const bool use_paired = arm == 1;
            limits.maxComputeWorkgroupStorageSize = arm == 0 ? 12543 : 12544;
            const uintptr_t offset = arm == 2 ? 18 : 0;
            weights->data = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(webgpu_ptr_base) + offset);
            commands = {};
            ctx->active_command_encoder = ctx->global_ctx->device.CreateCommandEncoder();
            ctx->bind_group_cache.begin_graph(arm);
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
            BICORE_CHECK(main.pipeline == (use_paired ? paired : stock).pipeline.Get()->id);
            BICORE_CHECK(main.x == tiles * (use_paired ? 22 : 44) && main.y == 1 && main.z == 1);
            BICORE_CHECK(commands.writes.front().bytes.size() == 512);
            if (arm == 0) {
                parameters = commands.writes.front().bytes;
            } else if (use_paired) {
                BICORE_CHECK(commands.writes.front().bytes == parameters);
            } else {
                // The main params begin at byte 256. The canonical path consumes the one-block prefix.
                uint32_t packed_offset = 0;
                memcpy(&packed_offset, commands.writes.front().bytes.data() + 256, sizeof(packed_offset));
                BICORE_CHECK(packed_offset == 1);
            }
#ifdef GGML_WEBGPU_GPU_PROFILE
            const uint32_t first = 4 * arm;
            BICORE_CHECK(commands.timestamp_pairs ==
                         (std::vector<std::pair<uint32_t, uint32_t>>{{first, first + 1}, {first + 2, first + 3}}));
#endif
            const auto encoded_commands = ctx->active_command_encoder.Finish();
            ctx->global_ctx->queue.Submit(1, &encoded_commands);
            drain_queue(ctx->global_ctx->queue.Get());
            BICORE_CHECK(commands.executed == 2);
            ctx->param_arena.reset();
        }
        for (const auto & pipeline : {stock, paired}) {
            std::ofstream shader(output / ("indexed-q4-0-e" + std::to_string(experts) + "-" + pipeline.name + ".wgsl"));
            shader << pipeline.pipeline.Get()->code;
            BICORE_CHECK(bool(shader));
        }
    }
}

} // namespace expert_cases
