static size_t checks = 0;
static void require(bool condition, const char * message) {
    ++checks;
    if (!condition)
        throw std::runtime_error(message);
}
struct Fixture {
    ggml_tensor a{}, x{}, dst{}, ids{};
    ggml_webgpu_shader_lib_context c{};
    Fixture() {
        reset();
    }
    void reset() {
        a = {};
        x = {};
        dst = {};
        ids = {};
        c = {};
        for (int axis = 0; axis < 4; ++axis)
            a.ne[axis] = x.ne[axis] = dst.ne[axis] = 1;
        a.type = GGML_TYPE_Q8_0;
        x.type = dst.type = GGML_TYPE_F32;
        a.ne[0] = x.ne[0] = 2560;
        a.ne[1] = dst.ne[0] = 1024;
        c.src0 = &a;
        c.src1 = &x;
        c.dst = &dst;
        c.supports_subgroups = true;
        c.max_wg_size = 256;
        c.min_subgroup_size = c.max_subgroup_size = 8;
    }
    uint32_t select(bool mmvq = false, bool overlap = false) const {
        return ggml_webgpu_q8_matvec_subgroup_size(c, mmvq, overlap);
    }
};
static const ggml_webgpu_mul_mat_vec_shader_decisions & decisions(const webgpu_pipeline & pipeline) {
    return *static_cast<ggml_webgpu_mul_mat_vec_shader_decisions *>(pipeline.context.get());
}
int main(int argc, char ** argv) {
    try {
        for (const uint32_t sg : {4u, 8u, 16u, 32u, 64u}) {
            Fixture f;
            f.c.min_subgroup_size = f.c.max_subgroup_size = sg;
            for (const int64_t k : {2048, 2080, 2560, 4096, 6144, 8192}) {
                f.a.ne[0] = f.x.ne[0] = k;
                for (const int64_t m : {1024, 1025, 6144, 10240}) {
                    f.a.ne[1] = f.dst.ne[0] = m;
                    require(f.select() == sg, "eligible fixed subgroup rejected");
                }
            }
            // Every subgroup covers all K weights exactly once, using the unchanged
            // four-lane block partition and eight elements within each lane.
            for (const uint32_t k : {2048u, 2560u, 4096u, 6144u, 8192u}) {
                std::vector<unsigned> visits(k);
                for (uint32_t lane = 0; lane < sg; ++lane) {
                    for (uint32_t block = lane / 4; block < k / 32; block += sg / 4) {
                        for (uint32_t j = 0; j < 8; ++j)
                            ++visits[block * 32 + (lane % 4) * 8 + j];
                    }
                }
                for (auto count : visits)
                    require(count == 1, "subgroup input partition");
            }
        }
        for (const int64_t k : {int64_t(-32), int64_t(0), int64_t(2016), int64_t(2047), int64_t(2049), int64_t(8193),
                                int64_t(8224), INT64_MAX}) {
            Fixture f;
            f.a.ne[0] = f.x.ne[0] = k;
            require(f.select() == 0, "K fallback");
        }
        for (const int64_t m : {int64_t(-1), int64_t(0), int64_t(1023), int64_t(UINT32_MAX), INT64_MAX}) {
            Fixture f;
            f.a.ne[1] = f.dst.ne[0] = m;
            require(f.select() == 0, "M fallback");
        }
        for (const auto & range : std::vector<std::pair<uint32_t, uint32_t>>{
                 {0, 0}, {2, 2}, {3, 3}, {4, 8}, {8, 4}, {16, 32}, {64, 128}, {128, 128}}) {
            Fixture f;
            f.c.min_subgroup_size = range.first;
            f.c.max_subgroup_size = range.second;
            require(f.select() == 0, "unknown or variable subgroup fallback");
        }
        {
            Fixture f;
            f.c.supports_subgroups = false;
            require(f.select() == 0, "missing subgroups fallback");
        }
        {
            Fixture f;
            f.c.max_wg_size = 127;
            require(f.select() == 0, "workgroup limit fallback");
        }
        {
            Fixture f;
            f.c.max_wg_size = 128;
            require(f.select() == 8, "exact workgroup limit admitted");
        }
        {
            Fixture f;
            require(f.select(true) == 0 && f.select(false, true) == 0, "MMVQ and overlap fallback");
        }
        for (const auto type : {GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_Q4_0, GGML_TYPE_Q4_K}) {
            Fixture f;
            f.a.type = type;
            require(f.select() == 0, "weight type fallback");
        }
        {
            Fixture f;
            f.x.type = GGML_TYPE_F16;
            require(f.select() == 0, "F16 input fallback");
        }
        {
            Fixture f;
            f.dst.type = GGML_TYPE_F16;
            require(f.select() == 0, "F16 output fallback");
        }
        {
            Fixture f;
            f.c.src2 = &f.ids;
            require(f.select() == 0, "indexed fallback");
        }
        {
            Fixture f;
            f.c.src0 = nullptr;
            require(f.select() == 0, "null matrix fallback");
        }
        {
            Fixture f;
            f.c.src1 = nullptr;
            require(f.select() == 0, "null input fallback");
        }
        {
            Fixture f;
            f.c.dst = nullptr;
            require(f.select() == 0, "null output fallback");
        }
        {
            Fixture f;
            f.x.ne[0]++;
            require(f.select() == 0, "input K mismatch fallback");
        }
        {
            Fixture f;
            f.dst.ne[0]++;
            require(f.select() == 0, "output M mismatch fallback");
        }
        {
            Fixture f;
            f.x.ne[1] = 2;
            require(f.select() == 0, "input columns fallback");
        }
        {
            Fixture f;
            f.dst.ne[1] = 2;
            require(f.select() == 0, "output columns fallback");
        }
        for (int axis = 2; axis < 4; ++axis) {
            for (const int64_t bad : {int64_t(-1), int64_t(0), int64_t(UINT32_MAX) + 1, INT64_MAX}) {
                Fixture f;
                f.a.ne[axis] = bad;
                require(f.select() == 0, "invalid matrix batch fallback");
                Fixture g;
                g.x.ne[axis] = g.dst.ne[axis] = bad;
                require(g.select() == 0, "invalid vector batch fallback");
            }
            {
                Fixture f;
                f.a.ne[axis] = 2;
                f.x.ne[axis] = f.dst.ne[axis] = 3;
                require(f.select() == 0, "indivisible broadcast fallback");
            }
            {
                Fixture f;
                f.x.ne[axis] = 2;
                require(f.select() == 0, "batch output mismatch fallback");
            }
            {
                Fixture f;
                f.x.ne[axis] = f.dst.ne[axis] = UINT32_MAX;
                require(f.select() == 0, "u32 output index overflow fallback");
            }
        }
        {
            Fixture f;
            f.x.ne[2] = f.dst.ne[2] = 65536;
            f.x.ne[3] = f.dst.ne[3] = 65536;
            require(f.select() == 0, "batch product overflow fallback");
        }
        {
            Fixture f;
            f.a.ne[2] = f.a.ne[3] = 2;
            f.x.ne[2] = f.x.ne[3] = f.dst.ne[2] = f.dst.ne[3] = 4;
            require(f.select() == 8, "batch broadcast admission");
        }

        // A subgroup pipeline and the existing short path both use WG128. The
        // added subgroup key must distinguish them and every fixed width.
        for (const bool tuned_first : {false, true}) {
            Fixture f;
            RecordingLibrary lib;
            auto get = [&](int64_t k) {
                f.a.ne[0] = f.x.ne[0] = k;
                return lib.get_mul_mat_vec_pipeline(f.c, false, true);
            };
            const auto first = get(tuned_first ? 2560 : 640), second = get(tuned_first ? 640 : 2560);
            require(first.id != second.id, "same-WG tuned/fallback cache collision");
            require(get(tuned_first ? 2560 : 640).id == first.id, "mixed-order first cache hit");
            require(get(tuned_first ? 640 : 2560).id == second.id, "mixed-order second cache hit");
            require(decisions(first).wg_size == 128 && decisions(second).wg_size == 128, "same-WG cache control");
        }
        Fixture f;
        RecordingLibrary lib;
        std::vector<unsigned> ids;
        if (argc == 2)
            std::filesystem::create_directories(argv[1]);
        for (const uint32_t sg : {4u, 8u, 16u, 32u, 64u}) {
            f.c.min_subgroup_size = f.c.max_subgroup_size = sg;
            const auto pipeline = lib.get_mul_mat_vec_pipeline(f.c, false, true);
            require(std::find(ids.begin(), ids.end(), pipeline.id) == ids.end(), "fixed-width cache distinction");
            ids.push_back(pipeline.id);
            const uint32_t rows = 128 / sg * 4;
            require(decisions(pipeline).wg_size == 128 && decisions(pipeline).outputs_per_wg == rows,
                    "factory dispatch decisions");
            require(pipeline.name == "mul_mat_vec_q8_0_f32_independent_sg" + std::to_string(sg) + "_wg128",
                    "dedicated profile identity");
            require(pipeline.wgsl.find("@workgroup_size(128)") != std::string::npos, "actual workgroup specialization");
            require(pipeline.wgsl.find("workgroupBarrier") == std::string::npos &&
                        pipeline.wgsl.find("var<workgroup>") == std::string::npos,
                    "shared reduction retained");
            require(pipeline.wgsl.find("block += " + std::to_string(sg) + " / 4") != std::string::npos,
                    "accumulator subgroup stride specialization");
            require(pipeline.wgsl.find("accumulate_vec_dot(subgroup_invocation_id,") != std::string::npos,
                    "subgroup lane mapping");
            require(pipeline.wgsl.find("if (num_subgroups == expected_subgroups)") != std::string::npos,
                    "full-packed fast path");
            require(pipeline.wgsl.find("let active_count = subgroupAdd(1u)") != std::string::npos,
                    "partial active width");
            require(pipeline.wgsl.find("virtual_tid += active_count") != std::string::npos,
                    "partial logical lane completion");
            require(pipeline.wgsl.find("if (owns_rows && subgroup_invocation_id == 0u") != std::string::npos,
                    "extra subgroup output guard");
            require(pipeline.wgsl.find("subgroupAdd(acc[0][row])") != std::string::npos,
                    "single subgroup reduction per row");
            require(lib.get_mul_mat_vec_pipeline(f.c, false, true).id == pipeline.id, "repeat tuned cache hit");
            // Execute the production decision-to-dispatch calculation, including
            // tails, broadcast batches and two-dimensional surplus workgroups.
            for (const int64_t m : {1024, 1025, 6144, 10240}) {
                f.a.ne[1] = f.dst.ne[0] = m;
                f.a.ne[2] = f.a.ne[3] = 2;
                f.x.ne[2] = f.x.ne[3] = f.dst.ne[2] = f.dst.ne[3] = 4;
                const auto cached = lib.get_mul_mat_vec_pipeline(f.c, false, true);
                const auto grid = production_dispatch(cached, &f.dst, 257);
                const uint32_t groups = static_cast<uint32_t>((m + rows - 1) / rows) * 16;
                require(grid[0] <= 257 && grid[1] <= 257 && grid[0] * grid[1] >= groups,
                        "production split dispatch coverage");
                for (uint32_t batch = 0; batch < 16; ++batch) {
                    std::vector<unsigned> visits(static_cast<size_t>(m));
                    for (uint32_t tile = 0; tile < (static_cast<uint32_t>(m) + rows - 1) / rows; ++tile)
                        for (uint32_t subgroup = 0; subgroup < 128 / sg; ++subgroup)
                            for (uint32_t row = 0; row < 4; ++row) {
                                const uint32_t output = tile * rows + subgroup * 4 + row;
                                if (output < static_cast<uint32_t>(m))
                                    ++visits[output];
                            }
                    for (auto count : visits)
                        require(count == 1, "output row/tail coverage");
                }
            }
            if (argc == 2)
                std::ofstream(std::filesystem::path(argv[1]) / ("q8-independent-sg" + std::to_string(sg) + ".wgsl"))
                    << pipeline.wgsl;
            f.reset();
        }
        const auto tuned = lib.get_mul_mat_vec_pipeline(f.c, false, true);
        f.a.ne[0] = f.x.ne[0] = 10240;
        const auto baseline = lib.get_mul_mat_vec_pipeline(f.c, false, true);
        require(decisions(baseline).wg_size == 256 && decisions(baseline).outputs_per_wg == 4, "long K fallback");
        require(baseline.name == "mul_mat_vec_q8_0_f32_sg_reduce", "fallback profile identity preserved");
        if (argc == 2)
            std::ofstream(std::filesystem::path(argv[1]) / "q8-baseline.wgsl") << baseline.wgsl;
        f.reset();
        f.c.min_subgroup_size = 4;
        f.c.max_subgroup_size = 8;
        require(lib.get_mul_mat_vec_pipeline(f.c, false, true).id == baseline.id, "variable subgroup fallback cache");
        f.reset();
        f.a.ne[1] = f.dst.ne[0] = 1023;
        require(lib.get_mul_mat_vec_pipeline(f.c, false, true).id == baseline.id, "small M fallback cache");
        f.reset();
        f.dst.type = GGML_TYPE_F16;
        require(lib.get_mul_mat_vec_pipeline(f.c, false, true).id == baseline.id, "output type fallback cache");
        f.reset();
        f.c.src2 = &f.ids;
        require(lib.get_mul_mat_vec_pipeline(f.c, false, true).id == baseline.id, "indexed context exclusion cache");
        f.reset();
        f.x.type = GGML_TYPE_F16;
        require(lib.get_mul_mat_vec_pipeline(f.c, false, true).id != tuned.id, "F16 cache distinction");
        f.reset();
        require(lib.get_mul_mat_vec_pipeline(f.c, true, true).id != tuned.id, "overlap cache distinction");
        f.reset();
        f.c.vendor = "nvidia";
        f.c.supports_dot_product = true;
        const auto mmvq = lib.get_mul_mat_vec_pipeline(f.c, false, true);
        require(decisions(mmvq).wg_size == 256 && mmvq.id != tuned.id, "MMVQ cache exclusion");
        ggml_webgpu_mul_mat_vec_pipeline_key a{}, b{};
        a.workgroup_size = b.workgroup_size = 128;
        b.subgroup_size = 8;
        require(!(a == b), "subgroup key equality");
        std::unordered_map<ggml_webgpu_mul_mat_vec_pipeline_key, int, ggml_webgpu_mul_mat_vec_pipeline_key_hash> cache;
        cache[a] = 1;
        cache[b] = 2;
        require(cache.size() == 2 && cache[a] == 1 && cache[b] == 2, "subgroup key hash map");
        std::cout << checks << " actual subgroup selector/cache/factory/dispatch checks passed; " << created
                  << " recorded pipelines\n";
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
