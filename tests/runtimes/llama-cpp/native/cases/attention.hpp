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

static void all() {
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
