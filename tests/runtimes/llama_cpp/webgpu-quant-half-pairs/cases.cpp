#include <map>

static size_t checks = 0;
static void require(bool value, const char * message) {
    ++checks;
    if (!value)
        throw std::runtime_error(message);
}

struct Case {
    ggml_type left, right = GGML_TYPE_F32, output = GGML_TYPE_F32;
    int64_t k = 256, m = 33, n = 5;
    bool overlap = false, subgroup = false;
    uint32_t batches = 1;
};

static bool eligible(const Case & c) {
    return WEBGPU_MUL_MAT_TILE_M == 4 && WEBGPU_MUL_MAT_TILE_N == 4 && WEBGPU_MUL_MAT_WG_SIZE_M == 8 &&
           WEBGPU_MUL_MAT_WG_SIZE_N == 8 && WEBGPU_MUL_MAT_REG_TILE_K_QUANT == 32 && c.k > 0 &&
           ((c.left == GGML_TYPE_Q4_0 && c.k % 32 == 0) || (c.left == GGML_TYPE_Q4_K && c.k % 256 == 0)) &&
           c.right == GGML_TYPE_F32 && c.output == GGML_TYPE_F32 && c.n >= 5 && !c.overlap && !c.subgroup;
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
        require(again.id == after.id && again.wgsl == after.wgsl, "Repeated cache key missed");
        const auto * x = static_cast<const baseline::Library::Decisions *>(before.context.get());
        const auto * y = static_cast<const candidate::Library::Decisions *>(after.context.get());
        require(x->tile_k == y->tile_k && x->tile_m == y->tile_m && x->tile_n == y->tile_n &&
                    x->wg_size_m == y->wg_size_m && x->wg_size_n == y->wg_size_n && x->wg_size == y->wg_size &&
                    x->use_subgroup_matrix == y->use_subgroup_matrix,
                "Geometry or decisions changed");
        if (!c.subgroup) {
            require(production_grid<baseline::Library>(before, &dst) ==
                        production_grid<candidate::Library>(after, &dst),
                    "Production grid changed");
        }
        if (eligible(c)) {
            require(after.name == before.name + "_half_pairs", "Paired profile label missing");
            require(after.wgsl != before.wgsl && y->tile_k == 32 && y->tile_m == 4 && y->tile_n == 4 &&
                        y->wg_size == 64,
                    "Paired shader or fixed geometry missing");
            require(after.wgsl.find("array<vec2<f16>, (TILE_SRC0_SHMEM + TILE_SRC1_SHMEM) / 2u>") != std::string::npos,
                    "Pair shared storage missing");
        } else {
            require(after.wgsl == before.wgsl && after.name == before.name, "Fallback WGSL or label changed");
        }
        const bool vectorized = c.k % 4 == 0 && c.m % 4 == 0 && (c.left == GGML_TYPE_F32 || c.left == GGML_TYPE_F16);
        const auto key = std::to_string(c.left) + ":" + std::to_string(c.right) + ":" + std::to_string(vectorized) +
                         ":" + std::to_string(c.subgroup) + ":" + std::to_string(c.overlap) + ":" +
                         std::to_string(eligible(c)) + ":" + std::to_string(y->tile_m) + ":" +
                         std::to_string(y->wg_size);
        const auto inserted = cache.emplace(key, after.id);
        require(inserted.second || inserted.first->second == after.id, "Runtime shape entered the key");
        for (const auto & pair : cache)
            require(pair.first == key || pair.second != after.id, "Distinct modes reused a cached pipeline");
        if (!output.empty() && c.left != GGML_TYPE_F32) {
            const auto name = output / after.name;
            std::ofstream(name.string() + "-baseline.wgsl") << before.wgsl;
            std::ofstream(name.string() + "-candidate.wgsl") << after.wgsl;
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
                for (int64_t n : {1, 4, 5, 7, 32, 33, 128})
                    cases.push_back({type, GGML_TYPE_F32, GGML_TYPE_F32, k, 33, n});
            for (int64_t m : {1, 4, 5, 31, 32, 33, 65})
                for (uint32_t batches : {1u, 4u})
                    cases.push_back({type, GGML_TYPE_F32, GGML_TYPE_F32, 256, m, 5, false, false, batches});
            cases.push_back({type, GGML_TYPE_F16});
            cases.push_back({type, GGML_TYPE_F32, GGML_TYPE_F16});
            cases.push_back({type, GGML_TYPE_F32, GGML_TYPE_F32, 256, 33, 5, true});
            cases.push_back({type, GGML_TYPE_F32, GGML_TYPE_F32, 256, 33, 5, false, true});
        }
        for (ggml_type type : source_nonf32_types)
            if (type != GGML_TYPE_Q4_0 && type != GGML_TYPE_Q4_K)
                cases.push_back({type});
        // Unchanged floating point and small-M decisions; repeated old/new
        // eligibility with otherwise identical dtype/shape-independent keys.
        for (int i = 0; i < 3; ++i) {
            cases.push_back({GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, 256, 33, 5});
            cases.push_back({GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, 256, 1, 5});
            for (int64_t n : {5, 4, 5})
                cases.push_back({GGML_TYPE_Q4_K, GGML_TYPE_F32, GGML_TYPE_F32, 256, 33, n});
        }
        exercise(cases, output);
        std::reverse(cases.begin(), cases.end());
        exercise(cases, {});
        candidate::ggml_webgpu_mul_mat_pipeline_key a{}, b{};
        b.shared_mode = candidate::ggml_webgpu_mul_mat_pipeline_key::quant_shared_mode::half_pairs;
        require(!(a == b), "Shared mode missing from equality");
        require(candidate::ggml_webgpu_mul_mat_pipeline_key_hash{}(a) !=
                    candidate::ggml_webgpu_mul_mat_pipeline_key_hash{}(b),
                "Shared mode missing from hash");
        std::cout << "{\"status\":\"passed\",\"checks\":" << checks << ",\"casesPerOrder\":" << cases.size()
                  << ",\"orders\":2}\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
