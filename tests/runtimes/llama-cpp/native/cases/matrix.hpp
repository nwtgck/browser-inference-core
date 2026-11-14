#pragma once

namespace matrix_cases {
struct Case {
    ggml_type left;
    ggml_type right = GGML_TYPE_F32, output = GGML_TYPE_F32;
    int64_t k = 256, m = 64, n = 32, batch2 = 1, batch3 = 1;
    bool overlap = false, subgroup = false;
};
static unsigned mode(const Case & c) {
    if (c.right != GGML_TYPE_F32 || c.output != GGML_TYPE_F32 || c.k <= 0 || c.n < 5 || c.overlap || c.subgroup)
        return 0;
    const bool q4 = (c.left == GGML_TYPE_Q4_0 && c.k % 32 == 0) || (c.left == GGML_TYPE_Q4_K && c.k % 256 == 0);
    if (c.left == GGML_TYPE_Q6_K && c.k % 256 == 0 && c.m >= 64 && c.m % 64 == 0 && c.n >= 32 && c.batch2 > 0 &&
        c.batch3 > 0) {
        // Independent division-based parallelism oracle: avoid reproducing
        // the production saturated-product implementation for large extents.
        uint64_t remaining = 8;
        for (const int64_t factor : {c.m / 64, c.n / 32 + (c.n % 32 != 0), c.batch2, c.batch3})
            remaining = remaining / uint64_t(factor) + (remaining % uint64_t(factor) != 0);
        return remaining == 1 ? 2 : 0;
    }
    return !q4 ? 0 : c.m >= 64 && c.m % 64 == 0 && c.n >= 32 ? 2 : 1;
}

static void exercise(const std::vector<Case> & inputs, const std::filesystem::path & output) {
    auto ctx = recording_context();
    std::map<std::tuple<ggml_type, ggml_type, bool, bool, bool, unsigned, uint32_t, uint32_t, uint32_t>, unsigned>
        cache;
    for (const auto & input : inputs) {
        ggml_tensor a{}, b{}, dst{};
        a.type = input.left;
        b.type = input.right;
        dst.type = input.output;
        a.ne[0] = b.ne[0] = input.k;
        a.ne[1] = dst.ne[0] = input.m;
        b.ne[1] = dst.ne[1] = input.n;
        a.ne[2] = a.ne[3] = 1;
        b.ne[2] = dst.ne[2] = input.batch2;
        b.ne[3] = dst.ne[3] = input.batch3;
        ggml_webgpu_shader_lib_context c{};
        c.src0 = &a;
        c.src1 = &b;
        c.dst = &dst;
        c.max_wg_size = 256;
        c.supports_subgroup_matrix = input.subgroup;
        c.min_subgroup_size = c.max_subgroup_size = 16;
        c.sg_mat_m = c.sg_mat_n = c.sg_mat_k = 8;
        const auto pipeline = ctx->shader_lib->get_mul_mat_fast_pipeline(c, input.overlap, true);
        BICORE_CHECK(ctx->shader_lib->get_mul_mat_fast_pipeline(c, input.overlap, true).pipeline.Get() ==
                     pipeline.pipeline.Get());
        const auto & d = *static_cast<const ggml_webgpu_mul_mat_shader_decisions *>(pipeline.context.get());
        const auto selected = mode(input);
        BICORE_CHECK((pipeline.name.find("_half_pairs") != std::string::npos) == (selected != 0));
        BICORE_CHECK((pipeline.name.find("_half_pairs_m8") != std::string::npos) == (selected == 2));
        BICORE_CHECK(d.use_subgroup_matrix == input.subgroup);
        if (selected) {
            BICORE_CHECK(d.tile_m == (selected == 2 ? 8u : 4u) && d.tile_n == 4 && d.tile_k == 32 && d.wg_size == 64);
            BICORE_CHECK(pipeline.pipeline.Get()->code.find(
                             "array<vec2<f16>, (TILE_SRC0_SHMEM + TILE_SRC1_SHMEM) / 2u>") != std::string::npos);
        }
        const bool vectorized =
            input.k % 4 == 0 && input.m % 4 == 0 && (input.left == GGML_TYPE_F32 || input.left == GGML_TYPE_F16);
        const auto key = std::tuple{input.left, input.right, vectorized, input.subgroup, input.overlap,
                                    selected,   d.tile_m,    d.tile_n,   d.wg_size};
        const auto [position, inserted] = cache.emplace(key, pipeline.pipeline.Get()->id);
        BICORE_CHECK(inserted || position->second == pipeline.pipeline.Get()->id);
        for (const auto & item : cache)
            BICORE_CHECK(item.first == key || item.second != pipeline.pipeline.Get()->id);
        const bool valid_layout = input.k > 0 && input.k % ggml_blck_size(input.left) == 0 && input.m > 0 &&
                                  input.n >= 5 && input.batch2 > 0 && input.batch3 > 0;
        if (valid_layout && !input.subgroup && input.m < 1000 && input.n < 1000 && input.batch2 < 100 &&
            input.batch3 < 100) {
            const uint64_t groups = ((input.m + d.tile_m * d.wg_size_m - 1) / (d.tile_m * d.wg_size_m)) *
                                    ((input.n + d.tile_n * d.wg_size_n - 1) / (d.tile_n * d.wg_size_n)) * input.batch2 *
                                    input.batch3;
            BICORE_CHECK(groups <= 65535);
            BICORE_CHECK(
                (production_dispatch(c, 65535, input.overlap) == std::array<uint32_t, 2>{uint32_t(groups), 1}));
        }
        if (!output.empty() && inserted) {
            std::ofstream shader(output / (pipeline.name + ".wgsl"));
            shader << pipeline.pipeline.Get()->code;
            BICORE_CHECK(bool(shader));
        }
    }
}

static void supported_type_controls(std::vector<Case> & cases) {
    auto ctx = recording_context();
    ctx->global_ctx->capabilities.limits.maxStorageBufferBindingSize = UINT64_MAX;
    ggml_backend_webgpu_device_context bridge;
    bridge.webgpu_global_ctx = ctx->global_ctx;
    ggml_backend_device device{};
    device.context = &bridge;
    // Query the real provider instead of parsing its source's switch labels.
    for (int index = 0; index < GGML_TYPE_COUNT; ++index) {
        const auto type = static_cast<ggml_type>(index);
        const auto * traits = ggml_get_type_traits(type);
        if (!traits->type_size || !traits->blck_size)
            continue;
        q4_current::Fixture f(type, traits->blck_size * 8, 64);
        f.x.ne[1] = f.dst.ne[1] = 32;
        f.dst.op = GGML_OP_MUL_MAT;
        f.dst.src[0] = &f.a;
        f.dst.src[1] = &f.x;
        for (auto * tensor : {&f.a, &f.x, &f.dst}) {
            tensor->nb[0] = ggml_type_size(tensor->type);
            tensor->nb[1] = ggml_row_size(tensor->type, tensor->ne[0]);
            tensor->nb[2] = tensor->nb[1] * tensor->ne[1];
            tensor->nb[3] = tensor->nb[2];
        }
        if (ggml_backend_webgpu_device_supports_op(&device, &f.dst))
            cases.push_back({type, GGML_TYPE_F32, GGML_TYPE_F32, f.a.ne[0]});
    }
}

static void all(const std::filesystem::path & output) {
    std::vector<Case> cases;
    for (const auto type : {GGML_TYPE_Q4_0, GGML_TYPE_Q4_K}) {
        // Keep independent admission boundaries, then exercise interactions at
        // the cache/mode transitions below rather than a full Cartesian grid.
        for (int64_t k : {0, 31, 32, 33, 96, 255, 256, 257, 512, 2048})
            cases.push_back({type, GGML_TYPE_F32, GGML_TYPE_F32, k, 64, 32});
        for (int64_t m : {0, 1, 31, 32, 33, 63, 64, 65, 128})
            cases.push_back({type, GGML_TYPE_F32, GGML_TYPE_F32, 256, m, 32});
        for (int64_t n : {1, 4, 5, 31, 32, 33, 128})
            cases.push_back({type, GGML_TYPE_F32, GGML_TYPE_F32, 256, 64, n});
        cases.push_back({type, GGML_TYPE_F16});
        cases.push_back({type, GGML_TYPE_F32, GGML_TYPE_F16});
        cases.push_back({type, GGML_TYPE_F32, GGML_TYPE_F32, 256, 128, 33, 1, 1, true});
        cases.push_back({type, GGML_TYPE_F32, GGML_TYPE_F32, 256, 128, 33, 1, 1, false, true});
        for (int64_t batches : {1, 4})
            cases.push_back({type, GGML_TYPE_F32, GGML_TYPE_F32, 256, 33, 5, batches});
    }
    for (int64_t k : {0, 255, 256, 257, 512, 10752})
        cases.push_back({GGML_TYPE_Q6_K, GGML_TYPE_F32, GGML_TYPE_F32, k, 512, 32});
    for (int64_t m : {0, 63, 64, 65, 448, 512, 576})
        cases.push_back({GGML_TYPE_Q6_K, GGML_TYPE_F32, GGML_TYPE_F32, 256, m, 32});
    for (int64_t n : {4, 31, 32, 224, 225, 256, 257})
        cases.push_back({GGML_TYPE_Q6_K, GGML_TYPE_F32, GGML_TYPE_F32, 256, 64, n});
    for (const auto [batch2, batch3] : {std::pair{1, 1},
                                        {1, 7},
                                        {1, 8},
                                        {1, 9},
                                        {2, 3},
                                        {2, 4},
                                        {4, 2},
                                        {7, 1},
                                        {8, 1},
                                        {9, 1},
                                        {0, 8},
                                        {8, 0},
                                        {-1, 8},
                                        {8, -1}})
        cases.push_back({GGML_TYPE_Q6_K, GGML_TYPE_F32, GGML_TYPE_F32, 256, 64, 32, batch2, batch3});
    supported_type_controls(cases);
    for (const auto right : {GGML_TYPE_F32, GGML_TYPE_F16})
        for (const auto result : {GGML_TYPE_F32, GGML_TYPE_F16})
            cases.push_back({GGML_TYPE_Q6_K, right, result, 256, 512, 32});
    cases.push_back({GGML_TYPE_Q6_K, GGML_TYPE_F32, GGML_TYPE_F32, 256, 512, 32, 1, 1, true});
    cases.push_back({GGML_TYPE_Q6_K, GGML_TYPE_F32, GGML_TYPE_F32, 256, 512, 32, 1, 1, false, true});
    for (int64_t batches : {INT64_MAX, int64_t(0), int64_t(-1)})
        cases.push_back(
            {GGML_TYPE_Q6_K, GGML_TYPE_F32, GGML_TYPE_F32, 256, INT64_MAX - 63, INT64_MAX, INT64_MAX, batches});
    for (int64_t m : {1, 4, 64, 65})
        cases.push_back({GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, 256, m, 32});
    for (const auto & transition : std::vector<Case>{{GGML_TYPE_Q4_K, GGML_TYPE_F32, GGML_TYPE_F32, 256, 64, 31},
                                                     {GGML_TYPE_Q4_K, GGML_TYPE_F32, GGML_TYPE_F32, 256, 64, 32},
                                                     {GGML_TYPE_Q4_K, GGML_TYPE_F32, GGML_TYPE_F32, 256, 65, 32},
                                                     {GGML_TYPE_Q4_K, GGML_TYPE_F32, GGML_TYPE_F32, 255, 64, 32},
                                                     {GGML_TYPE_Q4_K, GGML_TYPE_F32, GGML_TYPE_F32, 256, 64, 32}})
        cases.push_back(transition);
    exercise(cases, output);
    std::reverse(cases.begin(), cases.end());
    exercise(cases, {});
    using Key = ggml_webgpu_mul_mat_pipeline_key;
    Key scalar{}, paired{}, wide{};
    paired.shared_mode = Key::quant_shared_mode::half_pairs;
    wide.shared_mode = Key::quant_shared_mode::half_pairs_m8;
    BICORE_CHECK(!(scalar == paired) && !(scalar == wide) && !(paired == wide));
    std::cout << "matrix factory/dispatch cases passed: " << cases.size() << " in each order\n";
}
} // namespace matrix_cases
