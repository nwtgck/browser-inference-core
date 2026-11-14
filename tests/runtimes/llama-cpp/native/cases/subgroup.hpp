#pragma once

namespace subgroup_cases {
struct Fixture : q4_current::Fixture {
    explicit Fixture(ggml_type type) : q4_current::Fixture(type, 2048, type == GGML_TYPE_Q8_0 ? 5120 : 4097) {}
    ggml_webgpu_shader_lib_context context(unsigned minimum, unsigned maximum, bool feature) {
        min_sg = minimum;
        max_sg = maximum;
        auto result = q4_current::Fixture::context<ggml_webgpu_shader_lib_context>();
        result.subgroup_size_control_enabled = feature;
        return result;
    }
};
static const ggml_webgpu_mul_mat_vec_shader_decisions & decisions(const webgpu_pipeline & pipeline) {
    BICORE_CHECK(pipeline.context != nullptr);
    return *static_cast<const ggml_webgpu_mul_mat_vec_shader_decisions *>(pipeline.context.get());
}
static std::string without_size_control(std::string output) {
    // Compare emitted shader arithmetic across two current supported modes;
    // this never changes compiler inputs or reconstructs an upstream body.
    for (const std::string declaration : {"enable subgroup_size_control;\n", "@subgroup_size(16)\n"}) {
        const auto at = output.find(declaration);
        BICORE_CHECK(at != std::string::npos && output.find(declaration, at + declaration.size()) == std::string::npos);
        output.erase(at, declaration.size());
    }
    return output;
}
static void emit(const std::filesystem::path & output, const std::string & name, const webgpu_pipeline & pipeline) {
    std::ofstream file(output / name);
    file << pipeline.pipeline.Get()->code;
    BICORE_CHECK(bool(file));
}
static void requests(const std::filesystem::path & output) {
    for (const auto type : {GGML_TYPE_Q4_0, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0}) {
        Fixture f(type);
        if (type == GGML_TYPE_Q8_0)
            f.a.ne[0] = f.x.ne[0] = 6144;
        if (type == GGML_TYPE_Q6_K)
            f.a.ne[1] = f.dst.ne[0] = 2048;
        auto ctx = recording_context();
        auto & library = *ctx->shader_lib;
        for (unsigned width : {16u, 32u}) {
            const auto fixed = library.get_mul_mat_vec_pipeline(f.context(width, width, false), false, false);
            const auto before = std::tuple{created, pushes, waits};
            const auto enabled = library.get_mul_mat_vec_pipeline(f.context(width, width, true), false, false);
            BICORE_CHECK(enabled.pipeline.Get()->code == fixed.pipeline.Get()->code && enabled.name == fixed.name);
            BICORE_CHECK((std::tuple{created, pushes, waits} == before));
            emit(output, std::string(ggml_type_name(type)) + "-fixed-sg" + std::to_string(width) + ".wgsl", enabled);
        }
        auto requested_context = f.context(8, 32, true);
        const auto requested = library.get_mul_mat_vec_pipeline(requested_context, false, false);
        emit(output, std::string(ggml_type_name(type)) + "-requested.wgsl", requested);
        const auto fixed = library.get_mul_mat_vec_pipeline(f.context(16, 16, true), false, false);
        BICORE_CHECK(without_size_control(requested.pipeline.Get()->code) == fixed.pipeline.Get()->code);
        BICORE_CHECK(decisions(requested).logical_subgroup_size == 16 &&
                     decisions(requested).requested_subgroup_size == 16 && decisions(requested).outputs_per_wg == 32);
        const auto key = ggml_webgpu_shader_lib::make_mul_mat_vec_key(requested_context, false, false);
        const auto before = std::tuple{created, pushes, waits};
        BICORE_CHECK(library.get_mul_mat_vec_pipeline(requested_context, false, false).pipeline.Get() ==
                     requested.pipeline.Get());
        BICORE_CHECK((std::tuple{created, pushes, waits} == before));
        auto canonical_context = requested_context;
        canonical_context.subgroup_size_control_enabled = false;
        const auto canonical = library.get_mul_mat_vec_pipeline(canonical_context, false, false);
        BICORE_CHECK((production_dispatch(requested_context, 65535, false) ==
                      std::array<uint32_t, 2>{uint32_t((f.dst.ne[0] + 31) / 32), 1}));
        inject = WGPUErrorType_Validation;
        BICORE_CHECK((production_dispatch(requested_context, 65535, false) ==
                      std::array<uint32_t, 2>{uint32_t((f.dst.ne[0] + 3) / 4), 1}));
        auto rejected_ctx = recording_context();
        auto & rejected = *rejected_ctx->shader_lib;
        inject = WGPUErrorType_Validation;
        const auto fallback = rejected.get_mul_mat_vec_pipeline(requested_context, false, false);
        BICORE_CHECK(fallback.pipeline.Get()->code == canonical.pipeline.Get()->code &&
                     fallback.name == canonical.name);
        BICORE_CHECK(decisions(fallback).logical_subgroup_size == 0 &&
                     decisions(fallback).requested_subgroup_size == 0 && decisions(fallback).outputs_per_wg == 4);
        const auto saved = std::tuple{created, pushes, waits};
        BICORE_CHECK(rejected.get_mul_mat_vec_pipeline(requested_context, false, false).pipeline.Get() ==
                     fallback.pipeline.Get());
        BICORE_CHECK((std::tuple{created, pushes, waits} == saved));
        if (type == GGML_TYPE_Q4_K) {
            f.a.ne[0] = f.x.ne[0] = 1024;
            BICORE_CHECK(ggml_webgpu_shader_lib::make_mul_mat_vec_key(requested_context, false, false) == key);
            const auto smaller = rejected.get_mul_mat_vec_pipeline(requested_context, false, false);
            const auto expected = library.get_mul_mat_vec_pipeline(canonical_context, false, false);
            BICORE_CHECK(decisions(smaller).wg_size == 64 &&
                         smaller.pipeline.Get()->code == expected.pipeline.Get()->code);
            BICORE_CHECK(pushes == std::get<1>(saved) && waits == std::get<2>(saved));
            f.a.ne[0] = f.x.ne[0] = 2048;
            BICORE_CHECK(rejected.get_mul_mat_vec_pipeline(requested_context, false, false).pipeline.Get() ==
                         fallback.pipeline.Get());
            BICORE_CHECK(pushes == std::get<1>(saved) && waits == std::get<2>(saved));
        }
        // Rejection is memoized by the optional request, while canonical
        // geometry must still be rebuilt from the new shape's own key.
        f.a.ne[0] = f.x.ne[0] = type == GGML_TYPE_Q4_0 ? 10240 : type == GGML_TYPE_Q4_K ? 17408 : 2560;
        BICORE_CHECK(ggml_webgpu_shader_lib::make_mul_mat_vec_key(requested_context, false, false) == key);
        const auto longer = rejected.get_mul_mat_vec_pipeline(requested_context, false, false);
        const auto expected = library.get_mul_mat_vec_pipeline(canonical_context, false, false);
        BICORE_CHECK(longer.pipeline.Get()->code == expected.pipeline.Get()->code && longer.name == expected.name);
        BICORE_CHECK(pushes == std::get<1>(saved) && waits == std::get<2>(saved));
        if (type == GGML_TYPE_Q6_K) {
            BICORE_CHECK(decisions(longer).wg_size == 256);
            f.a.ne[0] = f.x.ne[0] = 1536;
            BICORE_CHECK(rejected.get_mul_mat_vec_pipeline(requested_context, false, false).pipeline.Get() ==
                         fallback.pipeline.Get());
            BICORE_CHECK(pushes == std::get<1>(saved) && waits == std::get<2>(saved));
        }
        if (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_K) {
            for (int tensor = 0; tensor < 3; ++tensor) {
                for (int axis = 2; axis < 4; ++axis) {
                    auto & value = (tensor == 0 ? f.a : tensor == 1 ? f.x : f.dst).ne[axis];
                    value = 2;
                    BICORE_CHECK(ggml_webgpu_q4_requested_subgroup_size(requested_context, false, false) == 0);
                    value = 1;
                }
            }
            for (const auto range :
                 {std::pair{0u, 32u}, std::pair{16u, 16u}, std::pair{32u, 64u}, std::pair{4u, 8u}, std::pair{32u, 8u}})
                BICORE_CHECK(ggml_webgpu_q4_requested_subgroup_size(f.context(range.first, range.second, true), false,
                                                                    false) == 0);
            for (unsigned mutation = 0; mutation < 10; ++mutation) {
                Fixture bad(type);
                auto c = bad.context(8, 32, true);
                bool overlap = false, mmvq = false;
                switch (mutation) {
                case 0:
                    bad.a.ne[0] = bad.x.ne[0] = 1023;
                    break;
                case 1:
                    bad.a.ne[0] = bad.x.ne[0] = 4097;
                    break;
                case 2:
                    bad.a.ne[1] = bad.dst.ne[0] = 4095;
                    break;
                case 3:
                    bad.x.ne[1] = bad.dst.ne[1] = 2;
                    break;
                case 4:
                    bad.x.type = GGML_TYPE_F16;
                    break;
                case 5:
                    bad.dst.type = GGML_TYPE_F16;
                    break;
                case 6:
                    c.src2 = &bad.ids;
                    break;
                case 7:
                    c.max_wg_size = 64;
                    break;
                case 8:
                    overlap = true;
                    break;
                case 9:
                    mmvq = true;
                    break;
                }
                BICORE_CHECK(ggml_webgpu_q4_requested_subgroup_size(c, mmvq, overlap) == 0);
            }
        }
        BICORE_CHECK(futures.empty());
    }
}

static void q6(const std::filesystem::path & output) {
    Fixture f(GGML_TYPE_Q6_K);
    f.a.ne[1] = f.dst.ne[0] = 2048;
    auto ctx = recording_context();
    auto & library = *ctx->shader_lib;
    for (const auto width : {16u, 32u}) {
        auto c = f.context(width, width, true);
        const auto first = library.get_mul_mat_vec_pipeline(c, false, false);
        for (int64_t k : {1536, 1792, 2048}) {
            f.a.ne[0] = f.x.ne[0] = k;
            const auto current = library.get_mul_mat_vec_pipeline(c, false, false);
            BICORE_CHECK(current.pipeline.Get() == first.pipeline.Get());
            BICORE_CHECK(decisions(current).logical_subgroup_size == width && decisions(current).wg_size == 128);
            BICORE_CHECK(
                (production_dispatch(c, 65535, false) == std::array<uint32_t, 2>{2048u / (4u * (128u / width)), 1}));
        }
    }
    auto c = f.context(8, 32, true);
    for (int64_t k : {1280, 1535, 1536, 1537, 1792, 2047, 2048, 2049, 2560, 10751, 10752, 11008}) {
        for (int64_t m : {2047, 2048, 2049, 65536}) {
            f.a.ne[0] = f.x.ne[0] = k;
            f.a.ne[1] = f.dst.ne[0] = m;
            const bool eligible = k >= 1536 && k <= 10752 && k % 256 == 0 && m >= 2048;
            BICORE_CHECK(ggml_webgpu_q6_matvec_subgroup_eligible(c, false, false) == eligible);
            BICORE_CHECK(ggml_webgpu_q6_requested_subgroup_size(c, false, false) == (eligible ? 16u : 0u));
        }
    }
    f = Fixture(GGML_TYPE_Q6_K);
    c = f.context(8, 32, true);
    for (unsigned which = 0; which < 20; ++which) {
        Fixture bad(GGML_TYPE_Q6_K);
        auto b = bad.context(16, 16, true);
        bool mmvq = false, overlap = false;
        switch (which) {
        case 0:
            bad.a.type = GGML_TYPE_Q4_K;
            break;
        case 1:
            bad.x.type = GGML_TYPE_F16;
            break;
        case 2:
            bad.dst.type = GGML_TYPE_F16;
            break;
        case 3:
            b.src2 = &bad.ids;
            break;
        case 4:
            bad.x.ne[1] = bad.dst.ne[1] = 2;
            break;
        case 5:
            b.max_wg_size = 127;
            break;
        case 6:
            mmvq = true;
            break;
        case 7:
            overlap = true;
            break;
        case 8:
            bad.a.ne[2] = 0;
            break;
        case 9:
            bad.a.ne[3] = -1;
            break;
        case 10:
            bad.x.ne[2] = bad.dst.ne[2] = 0;
            break;
        case 11:
            bad.x.ne[3] = bad.dst.ne[3] = int64_t(UINT32_MAX) + 1;
            break;
        case 12:
            bad.a.ne[2] = 3;
            bad.x.ne[2] = bad.dst.ne[2] = 4;
            break;
        case 13:
            bad.x.ne[3] = 2;
            break;
        case 14:
            bad.x.ne[2] = bad.dst.ne[2] = UINT32_MAX;
            break;
        case 15:
            bad.a.ne[1] = bad.dst.ne[0] = int64_t(UINT32_MAX) - 126;
            break;
        case 16:
            b.src0 = nullptr;
            break;
        case 17:
            b.src1 = nullptr;
            break;
        case 18:
            b.dst = nullptr;
            break;
        case 19:
            b.matvec_epilogue = decltype(b.matvec_epilogue)(1);
            break;
        }
        BICORE_CHECK(!ggml_webgpu_q6_matvec_subgroup_eligible(b, mmvq, overlap));
    }
    const auto requested_key = ggml_webgpu_shader_lib::make_mul_mat_vec_key(c, false, false);
    for (const auto type : {GGML_TYPE_Q4_K, GGML_TYPE_Q8_0}) {
        Fixture other(type);
        BICORE_CHECK(
            !(ggml_webgpu_shader_lib::make_mul_mat_vec_key(other.context(8, 32, true), false, false) == requested_key));
    }
    for (int axis = 2; axis < 4; ++axis) {
        f.a.ne[axis] = 2;
        f.x.ne[axis] = f.dst.ne[axis] = 4;
        BICORE_CHECK(ggml_webgpu_q6_matvec_subgroup_eligible(c, false, false));
        f.a.ne[axis] = f.x.ne[axis] = f.dst.ne[axis] = 1;
    }
    for (const auto range :
         {std::pair{0u, 32u}, std::pair{16u, 16u}, std::pair{32u, 64u}, std::pair{4u, 8u}, std::pair{32u, 8u}}) {
        BICORE_CHECK(ggml_webgpu_q6_requested_subgroup_size(f.context(range.first, range.second, true), false, false) ==
                     0);
    }
    for (unsigned width : {8u, 64u})
        BICORE_CHECK(ggml_webgpu_q6_matvec_subgroup_size(f.context(width, width, true), false, false) == 0);
    c = f.context(8, 32, false);
    BICORE_CHECK(ggml_webgpu_q6_requested_subgroup_size(c, false, false) == 0);
    c.supports_subgroups = false;
    BICORE_CHECK(ggml_webgpu_q6_matvec_subgroup_size(c, false, false) == 0);
    f.a.ne[0] = f.x.ne[0] = 256;
    f.a.ne[1] = f.dst.ne[0] = 512;
    f.x.ne[1] = f.dst.ne[1] = 32;
    c = f.context(16, 16, true);
    const auto tiled = library.get_mul_mat_fast_pipeline(c, false, true);
    emit(output, "q6_K-c8.wgsl", tiled);
    using D = ggml_webgpu_mul_mat_shader_decisions;
    BICORE_CHECK(static_cast<const D *>(tiled.context.get())->tile_m == 8 &&
                 tiled.name.find("_half_pairs_m8") != std::string::npos);
    f.a.ne[1] = f.dst.ne[0] = 448;
    const auto scalar = library.get_mul_mat_fast_pipeline(c, false, true);
    BICORE_CHECK(static_cast<const D *>(scalar.context.get())->tile_m == 4 &&
                 scalar.name.find("_half_pairs") == std::string::npos);
    f.a.ne[1] = f.dst.ne[0] = 512;
    BICORE_CHECK(library.get_mul_mat_fast_pipeline(c, false, true).pipeline.Get() == tiled.pipeline.Get());
}
static void all(const std::filesystem::path & output) {
    requests(output);
    q6(output);
}
} // namespace subgroup_cases
