#pragma once

namespace expert_cases {
static void all(const std::filesystem::path & output) {
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
