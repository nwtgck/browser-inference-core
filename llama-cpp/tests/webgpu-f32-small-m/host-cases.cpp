#include <map>

static unsigned checks = 0;
static void require(bool condition, const char * message) {
    ++checks;
    if (!condition)
        throw std::runtime_error(message);
}

struct Case {
    int64_t k, m, n;
    uint32_t expected_tile;
    ggml_type left = GGML_TYPE_F32, right = GGML_TYPE_F32, output = GGML_TYPE_F32;
    bool overlap = false, subgroup_matrix = false;
    uint32_t batches = 1;
};

static ggml_webgpu_shader_lib_context context_for(const Case & c, ggml_tensor & a, ggml_tensor & b, ggml_tensor & dst) {
    for (int axis = 0; axis < 4; ++axis)
        a.ne[axis] = b.ne[axis] = dst.ne[axis] = 1;
    a.type = c.left;
    b.type = c.right;
    dst.type = c.output;
    a.ne[0] = b.ne[0] = c.k;
    a.ne[1] = dst.ne[0] = c.m;
    b.ne[1] = dst.ne[1] = c.n;
    a.ne[2] = c.batches == 4 ? 2 : 1;
    b.ne[2] = dst.ne[2] = c.batches;
    a.ne[3] = b.ne[3] = dst.ne[3] = 3;
    ggml_webgpu_shader_lib_context context{};
    context.src0 = &a;
    context.src1 = &b;
    context.dst = &dst;
    context.supports_subgroup_matrix = c.subgroup_matrix;
    context.supports_subgroups = true;
    context.max_wg_size = 256;
    context.min_subgroup_size = context.max_subgroup_size = 16;
    context.sg_mat_m = context.sg_mat_n = context.sg_mat_k = 8;
    return context;
}

static std::string expected_key(const Case & c) {
    const bool vectorized = c.k % 4 == 0 && c.m % 4 == 0 && (c.left == GGML_TYPE_F32 || c.left == GGML_TYPE_F16);
    return std::to_string(c.left) + ":" + std::to_string(c.right) + ":" + std::to_string(vectorized) + ":" +
           std::to_string(c.subgroup_matrix) + ":" + std::to_string(c.overlap) + ":" + std::to_string(c.expected_tile);
}

static void exercise(const std::vector<Case> & cases, const std::filesystem::path & output) {
    baseline::Library old;
    candidate::Library actual;
    std::map<std::string, unsigned> cached;
    unsigned index = 0;
    for (const auto & c : cases) {
        ggml_tensor a{}, b{}, dst{};
        const auto context = context_for(c, a, b, dst);
        const auto before = old.get_mul_mat_fast_pipeline(context, c.overlap, true);
        const auto after = actual.get_mul_mat_fast_pipeline(context, c.overlap, true);
        const auto repeated = actual.get_mul_mat_fast_pipeline(context, c.overlap, true);
        require(repeated.id == after.id && repeated.wgsl == after.wgsl, "Repeated key missed actual cache");
        const auto key = expected_key(c);
        const auto inserted = cached.emplace(key, after.id);
        require(inserted.second || inserted.first->second == after.id, "Shape-dependent cache identity changed");
        for (const auto & entry : cached) {
            require(entry.first == key || entry.second != after.id, "Different effective geometry reused a pipeline");
        }
        const auto * d = static_cast<const candidate::Library::Decisions *>(after.context.get());
        const auto * previous = static_cast<const baseline::Library::Decisions *>(before.context.get());
        if (c.expected_tile) {
            require(d->tile_k == 8 && d->tile_m == c.expected_tile && d->tile_n == 1 && d->wg_size_m == 1 &&
                        d->wg_size_n == 32 && d->wg_size == 32,
                    "Selected geometry disagrees with decisions");
            require(after.name == before.name + "_small_m" + std::to_string(c.expected_tile),
                    "Missing geometry profile identity");
            require(after.wgsl != before.wgsl, "Admitted geometry kept old WGSL");
            require(after.wgsl.find("acc[tm][tn] += f32(src0_tile[tm]) * f32(src1_val)") != std::string::npos,
                    "Canonical per-output accumulator changed");
        } else {
            require(after.wgsl == before.wgsl && after.name == before.name, "Guarded fallback shader or label changed");
            require(d->tile_k == previous->tile_k && d->tile_m == previous->tile_m && d->tile_n == previous->tile_n &&
                        d->wg_size == previous->wg_size,
                    "Fallback decisions changed");
        }
        require(d->use_subgroup_matrix == previous->use_subgroup_matrix &&
                    d->mul_mat_wg_size == previous->mul_mat_wg_size,
                "Unrelated decisions changed");
        if (!c.subgroup_matrix && c.m > 0) {
            require(production_grid<candidate::Library>(after, &dst) ==
                        production_grid<baseline::Library>(before, &dst),
                    "Production dispatch grid changed");
        }
        if (!output.empty()) {
            const auto stem = output / (std::to_string(index) + "-k" + std::to_string(c.k) + "-m" +
                                        std::to_string(c.m) + "-n" + std::to_string(c.n));
            std::ofstream(stem.string() + "-baseline.wgsl") << before.wgsl;
            std::ofstream(stem.string() + "-candidate.wgsl") << after.wgsl;
        }
        ++index;
    }
}

int main(int argc, char ** argv) {
    try {
        std::filesystem::path output;
        if (argc > 2)
            return 2;
        if (argc == 2) {
            output = argv[1];
            std::filesystem::create_directories(output);
        }
        std::vector<Case> cases;
        for (int64_t k : {127, 128, 129, 255, 256, 257, 260, 10240}) {
            for (int64_t m : {1, 2, 3, 4, 5, 8}) {
                for (int64_t n : {4, 5, 31, 32, 33, 128, 1025}) {
                    const uint32_t tile = k >= 128 && m <= 4 && n >= 5 ? (m == 1 ? 1 : 4) : 0;
                    cases.push_back({k, m, n, tile});
                }
            }
        }
        // Every other left-hand type admitted by the actual production MUL_MAT
        // support switch, including both float types and all quantization families.
        for (ggml_type left : source_nonf32_types) {
            for (int64_t m : {1, 3, 4})
                cases.push_back({256, m, 5, 0, left});
        }
        for (int64_t m : {1, 3, 4}) {
            cases.push_back({256, m, 5, 0, GGML_TYPE_F32, GGML_TYPE_F16});
            cases.push_back({256, m, 5, 0, GGML_TYPE_F16, GGML_TYPE_F16});
            cases.push_back({256, m, 5, 0, GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F16});
            cases.push_back({256, m, 5, 0, GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, true});
            cases.push_back({256, m, 5, 0, GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, false, true});
            cases.push_back(
                {256, m, 5, uint32_t(m == 1 ? 1 : 4), GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, false, false, 4});
        }
        for (int64_t n : {1, 2})
            cases.push_back({256, 4, n, 0});
        cases.push_back({256, 0, 5, 0});
        // Geometry/cutoff/old fallback alternation in a single actual cache.
        for (int i = 0; i < 4; ++i) {
            for (const Case c : std::vector<Case>{{256, 1, 5, 1},
                                                  {256, 4, 5, 4},
                                                  {256, 8, 5, 0},
                                                  {257, 4, 5, 4},
                                                  {257, 3, 33, 4},
                                                  {257, 5, 33, 0},
                                                  {127, 1, 5, 0},
                                                  {256, 1, 5, 1}})
                cases.push_back(c);
        }
        exercise(cases, output);
        std::reverse(cases.begin(), cases.end());
        exercise(cases, {});
        candidate::ggml_webgpu_mul_mat_pipeline_key a{}, b{};
        a.small_m_tile = 1;
        b.small_m_tile = 4;
        require(!(a == b), "Geometry missing from equality");
        require(candidate::ggml_webgpu_mul_mat_pipeline_key_hash{}(a) !=
                    candidate::ggml_webgpu_mul_mat_pipeline_key_hash{}(b),
                "Geometry missing from hash");
        std::cout << "{\"checks\":" << checks << ",\"casesPerOrder\":" << cases.size() << ",\"orders\":2}\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
