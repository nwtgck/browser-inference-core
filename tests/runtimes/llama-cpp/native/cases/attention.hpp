#pragma once

namespace attention_cases {
enum class MaskLayout { standard, per_head, padded_row, absent };

struct Case {
    int64_t width = 128;
    int64_t queries = 128;
    int64_t keys = 1280;
    float max_bias = 0;
    ggml_type kv_type = GGML_TYPE_F16;
    bool overlap = false;
    bool subgroup_matrix = false;
    size_t shared_limit = 32768;
    MaskLayout mask_layout = MaskLayout::standard;
};

static webgpu_pipeline select(const webgpu_context & context, const Case & input) {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> metadata(ggml_init({32768, nullptr, true}), ggml_free);
    BICORE_CHECK(bool(metadata));
    ggml_webgpu_shader_lib_context shader{};
    shader.src0 = ggml_new_tensor_4d(metadata.get(), GGML_TYPE_F32, input.width, input.queries, 32, 1);
    shader.src1 = ggml_new_tensor_4d(metadata.get(), input.kv_type, input.width, input.keys, 2, 1);
    shader.src2 = ggml_new_tensor_4d(metadata.get(), input.kv_type, input.width, input.keys, 2, 1);
    shader.dst = ggml_flash_attn_ext(metadata.get(), shader.src0, shader.src1, shader.src2, nullptr, 1.0f, 0.0f, 0.0f);
    ggml_set_op_params_f32(shader.dst, 1, input.max_bias);
    if (input.mask_layout != MaskLayout::absent) {
        const auto heads = input.mask_layout == MaskLayout::per_head ? 32 : 1;
        auto * mask = ggml_new_tensor_4d(metadata.get(), GGML_TYPE_F16, input.keys, input.queries, heads, 1);
        if (input.mask_layout == MaskLayout::padded_row) {
            mask->nb[1] += sizeof(uint16_t);
        }
        shader.src3 = mask;
    }
    shader.supports_subgroups = true;
    shader.supports_subgroup_matrix = input.subgroup_matrix;
    shader.sg_mat_m = shader.sg_mat_n = shader.sg_mat_k = 8;
    shader.min_subgroup_size = shader.max_subgroup_size = 16;
    shader.max_wg_size = 128;
    shader.wg_mem_limit_bytes = input.shared_limit;
    return context->shader_lib->get_flash_attn_pipeline(shader, input.overlap);
}

static bool skips_masked_tiles(const webgpu_pipeline & pipeline) {
    return pipeline.name.find("_maskskip") != std::string::npos;
}

static uint32_t kv_tile(const webgpu_pipeline & pipeline) {
    return static_cast<const ggml_webgpu_flash_attn_decisions *>(pipeline.context.get())->kv_tile;
}

static void vector_mask_dispatch(const std::filesystem::path & output) {
    auto ctx = recording_context();
    auto & capabilities = ctx->global_ctx->capabilities;
    auto & limits = capabilities.limits;
    capabilities.supports_subgroups = true;
    capabilities.min_subgroup_size = capabilities.max_subgroup_size = 32;
    limits.maxComputeInvocationsPerWorkgroup = limits.maxComputeWorkgroupSizeX = 1024;
    limits.maxComputeWorkgroupsPerDimension = 65535;
    limits.maxComputeWorkgroupStorageSize = 32768;
    limits.minStorageBufferOffsetAlignment = 256;
    limits.maxStorageBufferBindingSize = limits.maxBufferSize = 1024 * 1024 * 1024;
    ctx->param_arena.cleanup();
    ctx->param_arena.init(ctx->global_ctx->device, WEBGPU_PARAMS_BUF_SIZE_BYTES, 8,
                          limits.minStorageBufferOffsetAlignment);

    ggml_webgpu_shader_lib_context block{};
    block.max_wg_size = limits.maxComputeInvocationsPerWorkgroup;
    const auto chosen = ctx->shader_lib->get_flash_attn_blk_pipeline(block, 32);
    BICORE_CHECK(chosen.name == "flash_attn_vec_blk_kvt32_wg32");
    const auto partial = ctx->shader_lib->get_flash_attn_blk_pipeline(block, 19);
    BICORE_CHECK(partial.name == "flash_attn_vec_blk_kvt19_wg32" && partial.pipeline.Get() != chosen.pipeline.Get());
    block.max_wg_size = 16;
    const auto lower = ctx->shader_lib->get_flash_attn_blk_pipeline(block, 32);
    BICORE_CHECK(lower.name == "flash_attn_vec_blk_kvt32_wg16" && lower.pipeline.Get() != chosen.pipeline.Get());
    block.max_wg_size = 256;
    BICORE_CHECK(ctx->shader_lib->get_flash_attn_blk_pipeline(block, 32).pipeline.Get() == chosen.pipeline.Get());

    ggml_backend_webgpu_device_context bridge{};
    bridge.webgpu_global_ctx = ctx->global_ctx;
    ggml_backend_device device{};
    device.context = &bridge;
    ggml_backend_buffer_type buffer_type{};
    buffer_type.device = &device;
    for (const bool masked : {true, false}) {
        std::unique_ptr<ggml_context, decltype(&ggml_free)> metadata(ggml_init({32768, nullptr, true}), ggml_free);
        auto * q = ggml_new_tensor_4d(metadata.get(), GGML_TYPE_F32, 128, 1, 24, 1);
        auto * k = ggml_new_tensor_4d(metadata.get(), GGML_TYPE_F16, 128, 256, 8, 1);
        auto * v = ggml_new_tensor_4d(metadata.get(), GGML_TYPE_F16, 128, 256, 8, 1);
        auto * mask = masked ? ggml_new_tensor_4d(metadata.get(), GGML_TYPE_F16, 256, 1, 24, 1) : nullptr;
        auto * dst = ggml_flash_attn_ext(metadata.get(), q, k, v, mask, 1.0f, 0.0f, 0.0f);
        const auto operands = std::array{q, k, v, mask, dst};
        std::array<ggml_backend_buffer, operands.size()> buffers{};
        std::vector<ggml_backend_webgpu_buffer_context> backing;
        backing.reserve(operands.size());
        for (size_t i = 0; i < operands.size(); ++i) {
            if (!operands[i])
                continue;
            backing.emplace_back(wgpu::Buffer::Acquire(new WGPUBufferImpl), "attention test", ctx->global_ctx);
            buffers[i].context = &backing.back();
            buffers[i].size = ggml_backend_webgpu_buffer_type_get_alloc_size(&buffer_type, operands[i]);
            operands[i]->data = webgpu_ptr_base;
            operands[i]->buffer = &buffers[i];
        }
        ctx->active_command_encoder = ctx->global_ctx->device.CreateCommandEncoder();
        commands = {};
#ifndef GGML_WEBGPU_GPU_PROFILE
        ctx->batch_compute_passes = true;
        ctx->active_compute_pass = ctx->active_command_encoder.BeginComputePass();
#endif
        const auto encoded = ggml_webgpu_flash_attn(ctx, q, k, v, mask, nullptr, dst);
#ifndef GGML_WEBGPU_GPU_PROFILE
        ctx->active_compute_pass.End();
        ctx->active_compute_pass = nullptr;
#endif
        BICORE_CHECK(encoded.num_kernels == (masked ? 3u : 2u) && commands.dispatches.size() == encoded.num_kernels);
        ctx->param_arena.flush(ctx->global_ctx->queue);
        if (masked) {
            const auto & dispatch = commands.dispatches[0];
            BICORE_CHECK(dispatch.pipeline == chosen.pipeline.Get()->id && dispatch.x == 8 && dispatch.y == 24);
            BICORE_CHECK(commands.writes.size() == 1 && commands.writes[0].bytes.size() >= 32);
            std::array<uint32_t, 8> params{};
            std::memcpy(params.data(), commands.writes[0].bytes.data(), sizeof(params));
            BICORE_CHECK((params == std::array<uint32_t, 8>{0, 1, 256, 6144, 256, 24, 8, 1}));
        } else {
            BICORE_CHECK(commands.dispatches[0].pipeline != chosen.pipeline.Get()->id);
        }
        const auto finished = ctx->active_command_encoder.Finish();
        ctx->global_ctx->queue.Submit(1, &finished);
        drain_queue(ctx->global_ctx->queue.Get());
        BICORE_CHECK(commands.executed == encoded.num_kernels);
        ctx->param_arena.reset();
    }
    for (const auto & pipeline : {chosen, partial}) {
        std::ofstream shader(output / (pipeline.name + ".wgsl"));
        shader << pipeline.pipeline.Get()->code;
        BICORE_CHECK(bool(shader));
    }
}

static void all(const std::filesystem::path & output) {
    vector_mask_dispatch(output);
    auto context = recording_context();
    const auto short_kv = select(context, {.keys = 767});
    const auto long_kv = select(context, {.keys = 768});
    BICORE_CHECK(!skips_masked_tiles(short_kv) && skips_masked_tiles(long_kv));
    BICORE_CHECK(select(context, {.queries = 31}).pipeline.Get() == short_kv.pipeline.Get());
    BICORE_CHECK(select(context, {.queries = 32}).pipeline.Get() == long_kv.pipeline.Get());
    BICORE_CHECK(select(context, {.mask_layout = MaskLayout::per_head}).pipeline.Get() == long_kv.pipeline.Get());
    const auto alibi = select(context, {.max_bias = 8.0f});
    BICORE_CHECK(!skips_masked_tiles(alibi) && kv_tile(alibi) == kv_tile(long_kv));
    // max_bias is a uniform. At this geometry only the masked-tile specialization changes the compiled program.
    BICORE_CHECK(long_kv.pipeline.Get()->code != alibi.pipeline.Get()->code);
    BICORE_CHECK(!skips_masked_tiles(select(context, {.kv_type = GGML_TYPE_Q8_0})));
    BICORE_CHECK(!skips_masked_tiles(select(context, {.overlap = true})));
    BICORE_CHECK(!skips_masked_tiles(select(context, {.subgroup_matrix = true})));
    BICORE_CHECK(!skips_masked_tiles(select(context, {.mask_layout = MaskLayout::padded_row})));

    for (const auto [width, base, wider] :
         std::array<std::array<uint32_t, 3>, 3>{{{128, 54, 64}, {256, 23, 55}, {512, 7, 23}}}) {
        const auto masked = select(context, {.width = width});
        const auto unmasked = select(context, {.width = width, .mask_layout = MaskLayout::absent});
        const auto narrow = select(context, {.width = width, .keys = 767, .mask_layout = MaskLayout::absent});
        const auto lower_limit = select(context, {.width = width, .shared_limit = 24576});
        BICORE_CHECK(kv_tile(masked) == wider && kv_tile(unmasked) == wider);
        BICORE_CHECK(kv_tile(narrow) == base && kv_tile(lower_limit) <= base);
        BICORE_CHECK(!skips_masked_tiles(unmasked));
        BICORE_CHECK(unmasked.pipeline.Get() != narrow.pipeline.Get());
        BICORE_CHECK(
            select(context, {.width = width, .queries = 31, .mask_layout = MaskLayout::absent}).pipeline.Get() ==
            narrow.pipeline.Get());
        BICORE_CHECK(
            select(context, {.width = width, .queries = 53, .mask_layout = MaskLayout::absent}).pipeline.Get() ==
            unmasked.pipeline.Get());
    }

    // At width 256, KV57 needs 33,776 bytes after each shader allocation is rounded separately.
    BICORE_CHECK(kv_tile(select(context, {.width = 256, .shared_limit = 33760})) == 56);
    BICORE_CHECK(kv_tile(select(context, {.width = 256, .shared_limit = 33775})) == 56);
    BICORE_CHECK(kv_tile(select(context, {.width = 256, .shared_limit = 33776})) == 57);
    BICORE_CHECK(kv_tile(select(context, {.width = 256, .shared_limit = 33760, .mask_layout = MaskLayout::absent})) ==
                 57);
}
} // namespace attention_cases
