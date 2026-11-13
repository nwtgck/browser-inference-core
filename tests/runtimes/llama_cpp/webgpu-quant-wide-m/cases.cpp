#include <map>

static unsigned checks = 0;
static void require(bool value, const char * message) {
    ++checks;
    if (!value)
        throw std::runtime_error(message);
}
struct Case {
    ggml_type left, right = GGML_TYPE_F32, output = GGML_TYPE_F32;
    int64_t k = 256, m = 64, n = 32;
    bool overlap = false, subgroup = false;
    int64_t batches = 1, batches3 = 1;
    bool grid_check = true;
};
static unsigned expected_mode(const Case & c) {
    const bool paired = WEBGPU_MUL_MAT_TILE_M == 4 && WEBGPU_MUL_MAT_TILE_N == 4 && WEBGPU_MUL_MAT_WG_SIZE_M == 8 &&
                        WEBGPU_MUL_MAT_WG_SIZE_N == 8 && WEBGPU_MUL_MAT_REG_TILE_K_QUANT == 32 && c.k > 0 &&
                        ((c.left == GGML_TYPE_Q4_0 && c.k % 32 == 0) || (c.left == GGML_TYPE_Q4_K && c.k % 256 == 0)) &&
                        c.right == GGML_TYPE_F32 && c.output == GGML_TYPE_F32 && c.n >= 5 && !c.overlap && !c.subgroup;
    if (c.left == GGML_TYPE_Q6_K && WEBGPU_MUL_MAT_TILE_M == 4 && WEBGPU_MUL_MAT_TILE_N == 4 &&
        WEBGPU_MUL_MAT_WG_SIZE_M == 8 && WEBGPU_MUL_MAT_WG_SIZE_N == 8 && WEBGPU_MUL_MAT_REG_TILE_K_QUANT == 32 &&
        c.k > 0 && c.k % 256 == 0 && c.right == GGML_TYPE_F32 && c.output == GGML_TYPE_F32 && c.m >= 64 &&
        c.m % 64 == 0 && c.n >= 32 && c.batches > 0 && c.batches3 > 0 && !c.overlap && !c.subgroup) {
        // Division-based independent oracle: no saturating-product implementation.
        uint64_t remaining = 8;
        for (const auto factor : {c.m / 64, c.n / 32 + (c.n % 32 != 0), c.batches, c.batches3})
            remaining = remaining / uint64_t(factor) + (remaining % uint64_t(factor) != 0);
        return remaining == 1 ? 2 : 0;
    }
    return !paired ? 0 : (c.m >= 64 && c.m % 64 == 0 && c.n >= 32 ? 2 : 1);
}
static void exercise(const std::vector<Case> & cases, const std::filesystem::path & output) {
    baseline::Library old;
    candidate::Library actual;
    std::map<std::string, unsigned> cache;
    for (const auto & c : cases) {
        ggml_tensor a{}, b{}, dst{};
        for (int i = 0; i < 4; ++i)
            a.ne[i] = b.ne[i] = dst.ne[i] = 1;
        a.type = c.left;
        b.type = c.right;
        dst.type = c.output;
        a.ne[0] = b.ne[0] = c.k;
        a.ne[1] = dst.ne[0] = c.m;
        b.ne[1] = dst.ne[1] = c.n;
        b.ne[2] = dst.ne[2] = c.batches;
        b.ne[3] = dst.ne[3] = c.batches3;
        ggml_webgpu_shader_lib_context context{};
        context.src0 = &a;
        context.src1 = &b;
        context.dst = &dst;
        context.supports_subgroup_matrix = c.subgroup;
        context.max_wg_size = 256;
        context.min_subgroup_size = context.max_subgroup_size = 16;
        context.sg_mat_m = context.sg_mat_n = context.sg_mat_k = 8;
        const auto before = old.get_mul_mat_fast_pipeline(context, c.overlap, true);
        const auto after = actual.get_mul_mat_fast_pipeline(context, c.overlap, true);
        const auto again = actual.get_mul_mat_fast_pipeline(context, c.overlap, true);
        const auto * x = static_cast<const baseline::Library::Decisions *>(before.context.get());
        const auto * y = static_cast<const candidate::Library::Decisions *>(after.context.get());
        const unsigned mode = expected_mode(c);
        require(again.id == after.id && again.wgsl == after.wgsl, "Repeated key missed");
        require(x->tile_k == y->tile_k && x->tile_n == y->tile_n && x->wg_size_m == y->wg_size_m &&
                    x->wg_size_n == y->wg_size_n && x->wg_size == y->wg_size &&
                    x->use_subgroup_matrix == y->use_subgroup_matrix,
                "Unrelated geometry changed");
        if (mode == 2) {
            require(x->tile_m == 4 && y->tile_m == 8 && y->tile_n == 4 && y->tile_k == 32 && y->wg_size == 64,
                    "Wide decisions disagree");
            require(after.name == before.name + (c.left == GGML_TYPE_Q6_K ? "_half_pairs_m8" : "_m8") &&
                        after.wgsl != before.wgsl,
                    "Wide label or source missing");
            require(after.wgsl.find("src0_pair: array<vec2<f16>, 8u>") != std::string::npos,
                    "Private pair array width changed");
        } else {
            require(x->tile_m == y->tile_m && before.wgsl == after.wgsl && before.name == after.name,
                    "Existing fallback changed");
        }
        const bool vectorized = c.k % 4 == 0 && c.m % 4 == 0 && (c.left == GGML_TYPE_F32 || c.left == GGML_TYPE_F16);
        const auto key = std::to_string(c.left) + ":" + std::to_string(c.right) + ":" + std::to_string(vectorized) +
                         ":" + std::to_string(c.subgroup) + ":" + std::to_string(c.overlap) + ":" +
                         std::to_string(mode) + ":" + std::to_string(y->tile_m) + ":" + std::to_string(y->tile_n) +
                         ":" + std::to_string(y->wg_size);
        const auto inserted = cache.emplace(key, after.id);
        require(inserted.second || inserted.first->second == after.id, "Runtime shape polluted the key");
        for (const auto & row : cache)
            require(row.first == key || row.second != after.id, "Different modes shared one pipeline");
        if (c.grid_check && !c.subgroup && c.m > 0 && c.n > 0 && c.batches > 0 && c.batches3 > 0) {
            const auto grid = production_grid<candidate::Library>(after, &dst);
            const uint32_t groups = uint32_t((c.m + y->tile_m * y->wg_size_m - 1) / (y->tile_m * y->wg_size_m)) *
                                    uint32_t((c.n + y->tile_n * y->wg_size_n - 1) / (y->tile_n * y->wg_size_n)) *
                                    uint32_t(c.batches) * uint32_t(c.batches3);
            const uint32_t rows = std::max(1u, (groups + 16) / 17);
            require(grid[0] == (groups + rows - 1) / rows && grid[1] == rows,
                    "Actual production grid disagrees with the effective tile");
            if (mode != 2)
                require(grid == production_grid<baseline::Library>(before, &dst), "Fallback grid changed");
        }
        if (!output.empty()) {
            std::ofstream(output / (after.name + "-baseline.wgsl")) << before.wgsl;
            std::ofstream(output / (after.name + "-candidate.wgsl")) << after.wgsl;
        }
    }
}
int main(int argc, char ** argv) {
    try {
        std::filesystem::path output;
        if (argc == 2) {
            output = argv[1];
            std::filesystem::create_directories(output);
        }
        std::vector<Case> cases;
        for (auto type : {GGML_TYPE_Q4_0, GGML_TYPE_Q4_K}) {
            for (int64_t k : {0, 31, 32, 33, 96, 255, 256, 257, 512, 2048})
                for (int64_t m : {0, 63, 64, 65, 128})
                    for (int64_t n : {4, 5, 31, 32, 33, 128})
                        cases.push_back({type, GGML_TYPE_F32, GGML_TYPE_F32, k, m, n});
            cases.push_back({type, GGML_TYPE_F16});
            cases.push_back({type, GGML_TYPE_F32, GGML_TYPE_F16});
            cases.push_back({type, GGML_TYPE_F32, GGML_TYPE_F32, 256, 128, 33, true});
            cases.push_back({type, GGML_TYPE_F32, GGML_TYPE_F32, 256, 128, 33, false, true});
            cases.push_back({type, GGML_TYPE_F32, GGML_TYPE_F32, 256, 128, 33, false, false, 4});
        }
        for (int64_t k : {0, 255, 256, 257, 512, 10752})
            for (int64_t m : {0, 63, 64, 65, 448, 512, 576})
                for (int64_t n : {4, 31, 32, 224, 225, 256, 257})
                    cases.push_back({GGML_TYPE_Q6_K, GGML_TYPE_F32, GGML_TYPE_F32, k, m, n});
        for (int64_t batch2 : {-1, 0, 1, 2, 7, 8, 9})
            for (int64_t batch3 : {-1, 0, 1, 2, 4, 8})
                cases.push_back(
                    {GGML_TYPE_Q6_K, GGML_TYPE_F32, GGML_TYPE_F32, 256, 64, 32, false, false, batch2, batch3});
        for (const auto & c :
             std::vector<Case>{{GGML_TYPE_Q6_K, GGML_TYPE_F16, GGML_TYPE_F32, 256, 512, 32},
                               {GGML_TYPE_Q6_K, GGML_TYPE_F32, GGML_TYPE_F16, 256, 512, 32},
                               {GGML_TYPE_Q6_K, GGML_TYPE_F32, GGML_TYPE_F32, 256, 512, 32, true},
                               {GGML_TYPE_Q6_K, GGML_TYPE_F32, GGML_TYPE_F32, 256, 512, 32, false, true}})
            cases.push_back(c);
        const int64_t huge = INT64_MAX;
        for (const auto & c : std::vector<Case>{
                 {GGML_TYPE_Q6_K, GGML_TYPE_F32, GGML_TYPE_F32, 256, huge - 63, huge, false, false, huge, huge, false},
                 {GGML_TYPE_Q6_K, GGML_TYPE_F32, GGML_TYPE_F32, 256, huge - 63, huge, false, false, huge, 0, false},
                 {GGML_TYPE_Q6_K, GGML_TYPE_F32, GGML_TYPE_F32, 256, huge - 63, huge, false, false, huge, -1, false}})
            cases.push_back(c);
        for (auto type : source_nonf32_types)
            if (type != GGML_TYPE_Q4_0 && type != GGML_TYPE_Q4_K && type != GGML_TYPE_Q6_K)
                cases.push_back({type});
        for (int64_t m : {1, 4, 64, 65})
            cases.push_back({GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, 256, m, 32});
        // One cache alternates canonical, four-row and eight-row kernels, then
        // returns to the same shapes. Reversing the list exercises the other order.
        for (int i = 0; i < 3; ++i)
            for (const auto & c : std::vector<Case>{{GGML_TYPE_Q4_K, GGML_TYPE_F32, GGML_TYPE_F32, 256, 64, 31},
                                                    {GGML_TYPE_Q4_K, GGML_TYPE_F32, GGML_TYPE_F32, 256, 64, 32},
                                                    {GGML_TYPE_Q4_K, GGML_TYPE_F32, GGML_TYPE_F32, 256, 65, 32},
                                                    {GGML_TYPE_Q4_K, GGML_TYPE_F32, GGML_TYPE_F32, 255, 64, 32},
                                                    {GGML_TYPE_Q4_K, GGML_TYPE_F32, GGML_TYPE_F32, 256, 64, 32}})
                cases.push_back(c);
        exercise(cases, output);
        std::reverse(cases.begin(), cases.end());
        exercise(cases, {});
        using Key = candidate::ggml_webgpu_mul_mat_pipeline_key;
        Key a{}, b{}, c{};
        b.shared_mode = Key::quant_shared_mode::half_pairs;
        c.shared_mode = Key::quant_shared_mode::half_pairs_m8;
        require(!(a == b) && !(a == c) && !(b == c), "Mode missing from equality");
        const auto hash = candidate::ggml_webgpu_mul_mat_pipeline_key_hash{};
        require(hash(a) != hash(b) && hash(a) != hash(c) && hash(b) != hash(c), "Mode missing from hash");
        std::cout << "{\"status\":\"passed\",\"checks\":" << checks << ",\"casesPerOrder\":" << cases.size()
                  << ",\"orders\":2}\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
