#pragma once

namespace expert_cases {
static void nvfp4_accumulator() {
    auto ctx = recording_context();
    ggml_tensor weights{}, input{}, indices{}, output{};
    weights.type = GGML_TYPE_NVFP4;
    input.type = output.type = GGML_TYPE_F32;
    indices.type = GGML_TYPE_I32;
    output.op = GGML_OP_MUL_MAT_ID;
    weights.ne[0] = input.ne[0] = 1152;
    weights.ne[1] = output.ne[0] = 36;
    weights.ne[2] = 17;
    ggml_webgpu_shader_lib_context context{};
    context.src0 = &weights;
    context.src1 = &input;
    context.src2 = &indices;
    context.dst = &output;
    const auto select = [&](ggml_prec precision) {
        BICORE_CHECK(ggml_prec_set_acc(&output, precision));
        return ctx->shader_lib->get_mul_mat_id_pipeline(context, false).pipeline;
    };
    const auto narrow = select(GGML_PREC_UNDEFINED);
    const auto wide = select(GGML_PREC_BF16);
    BICORE_CHECK(narrow.Get() != wide.Get());
    BICORE_CHECK(narrow.Get()->code.find("src0_tile[tm] * src1_val") != std::string::npos);
    BICORE_CHECK(wide.Get()->code.find("array<array<f32,") != std::string::npos);
    BICORE_CHECK(wide.Get()->code.find("f32(src0_tile[tm]) * f32(src1_val)") != std::string::npos);
    BICORE_CHECK(select(GGML_PREC_F32).Get() == wide.Get());
    BICORE_CHECK(select(GGML_PREC_UNDEFINED).Get() == narrow.Get());
}

static void sparse_matrix(const std::filesystem::path & output) {
    auto ctx = recording_context();
    for (const auto type : {GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_NL, GGML_TYPE_Q4_K}) {
        ggml_tensor a{}, x{};
        a.type = type;
        x.type = GGML_TYPE_F32;
        a.ne[0] = x.ne[0] = type == GGML_TYPE_IQ4_NL ? 640 : 2560;
        a.ne[1] = 640;
        a.ne[2] = 512;
        x.ne[1] = x.ne[3] = 1;
        x.ne[2] = 5;
        ggml_webgpu_shader_lib_context context{};
        context.src0 = &a;
        context.src1 = &x;
        const auto pipeline = ctx->shader_lib->get_mul_mat_id_pipeline(context, true);
        const auto & d = *static_cast<const ggml_webgpu_mul_mat_shader_decisions *>(pipeline.context.get());
        BICORE_CHECK(d.wg_size == 64 && d.tile_k == 32 && d.tile_m * d.wg_size_m == 32 && d.tile_n * d.wg_size_n == 32);
        x.ne[2] = 128;
        BICORE_CHECK(ctx->shader_lib->get_mul_mat_id_pipeline(context, true).pipeline.Get() == pipeline.pipeline.Get());
        std::ofstream shader(output / (pipeline.name + ".wgsl"));
        shader << pipeline.pipeline.Get()->code;
        BICORE_CHECK(bool(shader));
    }
}

static void indexed_wide_k(const std::filesystem::path & output) {
    auto ctx = recording_context();
    for (const auto type : {GGML_TYPE_IQ4_NL, GGML_TYPE_NVFP4}) {
        ggml_tensor a{}, x{}, dst{};
        a.type = type;
        x.type = dst.type = GGML_TYPE_F32;
        dst.op = GGML_OP_MUL_MAT_ID;
        a.ne[2] = 128;
        auto c = ggml_webgpu_shader_lib_context{};
        c.src0 = &a;
        c.src1 = &x;
        c.dst = &dst;
        const auto precision = type == GGML_TYPE_NVFP4 ? GGML_PREC_F32 : GGML_PREC_UNDEFINED;
        BICORE_CHECK(ggml_prec_set_acc(&dst, precision));
        const auto select = [&](int64_t k, int64_t m, int64_t tokens, uint32_t shared, uint32_t expected) {
            a.ne[0] = x.ne[0] = k;
            a.ne[1] = dst.ne[0] = m;
            x.ne[2] = dst.ne[2] = tokens;
            c.wg_mem_limit_bytes = shared;
            const auto pipeline = ctx->shader_lib->get_mul_mat_id_pipeline(c, false);
            const auto & d = *static_cast<const ggml_webgpu_mul_mat_shader_decisions *>(pipeline.context.get());
            BICORE_CHECK(d.tile_k == expected && d.wg_size == 64);
            BICORE_CHECK((pipeline.name.find("_tk64") != std::string::npos) == (expected == 64));
            return pipeline;
        };
        const auto narrow = select(2304, 2304, 12, 8447, 32);
        const auto wide = select(2304, 2304, 12, 8448, 64);
        BICORE_CHECK(narrow.pipeline.Get() != wide.pipeline.Get());
        BICORE_CHECK(select(2304, 2304, 11, 32768, 32).pipeline.Get() == narrow.pipeline.Get());
        BICORE_CHECK(select(2304, 2303, 128, 32768, 32).pipeline.Get() == narrow.pipeline.Get());
        BICORE_CHECK(select(2304, 512, 128, 32768, 64).pipeline.Get() == wide.pipeline.Get());
        BICORE_CHECK(select(1152, 2304, 19, 32768, 64).pipeline.Get() == wide.pipeline.Get());
        x.type = GGML_TYPE_F16;
        select(2304, 2304, 128, 32768, 32);
        x.type = GGML_TYPE_F32;
        if (type == GGML_TYPE_NVFP4) {
            BICORE_CHECK(ggml_prec_set_acc(&dst, GGML_PREC_UNDEFINED));
            BICORE_CHECK(select(2304, 2304, 128, 32768, 32).pipeline.Get() != narrow.pipeline.Get());
            BICORE_CHECK(ggml_prec_set_acc(&dst, GGML_PREC_BF16));
            BICORE_CHECK(select(2304, 2304, 128, 32768, 64).pipeline.Get() == wide.pipeline.Get());
        }
        std::ofstream shader(output / (wide.name + ".wgsl"));
        shader << wide.pipeline.Get()->code;
        BICORE_CHECK(bool(shader));
    }
}

static void all(const std::filesystem::path & output) {
    nvfp4_accumulator();
    sparse_matrix(output);
    indexed_wide_k(output);
    auto ctx = recording_context();
    std::ofstream cases(output / "cases.json");
    cases << "[\n";
    bool first = true;
    for (const auto type : {GGML_TYPE_Q5_1, GGML_TYPE_IQ4_NL}) {
        ggml_tensor a{}, x{}, ids{}, dst{};
        a.type = type;
        x.type = dst.type = GGML_TYPE_F32;
        ids.type = GGML_TYPE_I32;
        for (auto * tensor : {&a, &x, &ids, &dst})
            for (auto & extent : tensor->ne)
                extent = 1;
        a.ne[2] = 512;
        x.ne[1] = ids.ne[0] = dst.ne[1] = 10;
        ggml_webgpu_shader_lib_context context{};
        context.src0 = &a;
        context.src1 = &x;
        context.src2 = &ids;
        context.dst = &dst;
        context.min_subgroup_size = context.max_subgroup_size = 32;
        context.supports_subgroups = true;
        context.max_wg_size = 256;
        const auto select = [&](int64_t k, int64_t m) {
            a.ne[0] = x.ne[0] = k;
            a.ne[1] = dst.ne[0] = m;
            return ctx->shader_lib->get_mul_mat_id_vec_pipeline(context, true);
        };
        const auto short_pipeline = select(640, 2560);
        const auto fallback = select(1056, 256);
        BICORE_CHECK(short_pipeline.pipeline.Get() != fallback.pipeline.Get());
        const auto created_before_warm = created;
        const char * name = type == GGML_TYPE_Q5_1 ? "Q5_1" : "IQ4_NL";
        for (const auto [k, m, expected] : {std::tuple{640, 2560, 128u}, {1056, 256, 256u}}) {
            const auto pipeline = select(k, m);
            const auto & decision =
                *static_cast<const ggml_webgpu_mul_mat_vec_shader_decisions *>(pipeline.context.get());
            BICORE_CHECK(ggml_webgpu_expert_short_workgroup(context) == expected && decision.wg_size == expected);
            BICORE_CHECK(pipeline.pipeline.Get() == (expected == 128 ? short_pipeline : fallback).pipeline.Get());
            BICORE_CHECK(created == created_before_warm);
            BICORE_CHECK(pipeline.pipeline.Get()->code.find("@workgroup_size(" + std::to_string(expected) + ")") !=
                         std::string::npos);
            std::ofstream shader(output / (std::string(name) + "-" + std::to_string(expected) + ".wgsl"));
            shader << pipeline.pipeline.Get()->code;
            BICORE_CHECK(bool(shader));
            if (!first)
                cases << ",\n";
            first = false;
            cases << "{\"type\":\"" << name << "\",\"k\":" << k << ",\"m\":" << m
                  << ",\"mode\":0,\"sharedInput\":false,\"storedExperts\":11,\"chosen\":" << expected << "}";
        }
        BICORE_CHECK(select(640, 2560).pipeline.Get() == short_pipeline.pipeline.Get() &&
                     created == created_before_warm);
    }
    cases << "\n]\n";
    BICORE_CHECK(bool(cases));
}
} // namespace expert_cases
