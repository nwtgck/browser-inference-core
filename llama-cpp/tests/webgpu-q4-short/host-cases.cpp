static unsigned checks = 0;
static void require(bool ok, const char * message) {
    ++checks;
    if (!ok)
        throw std::runtime_error(message);
}
struct Fixture {
    ggml_tensor a{}, x{}, ids{}, dst{};
    ggml_webgpu_shader_lib_context context{};
    bool indexed;
    explicit Fixture(bool id) : indexed(id) {
        a.type = GGML_TYPE_Q4_K;
        x.type = dst.type = GGML_TYPE_F32;
        ids.type = GGML_TYPE_I32;
        for (auto * t : {&a, &x, &ids, &dst})
            for (auto & n : t->ne)
                n = 1;
        a.ne[0] = x.ne[0] = 512;
        a.ne[1] = dst.ne[0] = 2048;
        if (indexed) {
            a.ne[2] = 11;
            ids.ne[0] = dst.ne[1] = 8;
            x.ne[1] = 8;
        }
        context.src0 = &a;
        context.src1 = &x;
        context.src2 = indexed ? &ids : nullptr;
        context.dst = &dst;
        context.min_subgroup_size = context.max_subgroup_size = 32;
        context.supports_subgroups = true;
        context.max_wg_size = 256;
    }
    uint32_t select(bool mmvq = false, bool overlap = false) {
        return ggml_webgpu_q4_short_workgroup(context, indexed, mmvq, overlap);
    }
    template <typename Library> webgpu_pipeline pipeline(Library & lib) {
        return indexed ? lib.get_mul_mat_id_vec_pipeline(context, true)
                       : lib.get_mul_mat_vec_pipeline(context, false, true);
    }
};
static uint32_t size(const webgpu_pipeline & p) {
    return static_cast<ggml_webgpu_mul_mat_vec_shader_decisions *>(p.context.get())->wg_size;
}
int main(int argc, char ** argv) {
    try {
        if (argc != 1 && argc != 2)
            return 2;
        check_dispatch();
        for (bool indexed : {false, true})
            for (ggml_type type :
                 {GGML_TYPE_Q4_K, GGML_TYPE_Q4_0, GGML_TYPE_Q5_K, GGML_TYPE_Q8_0, GGML_TYPE_F16, GGML_TYPE_F32})
                for (uint32_t sg : {4, 8, 16, 32, 64, 128})
                    for (uint32_t wg : {16, 32, 64, 128, 256, 512})
                        for (int k = -256; k <= 4608; k += 128) {
                            Fixture f(indexed);
                            f.a.type = type;
                            f.a.ne[0] = f.x.ne[0] = k;
                            f.context.min_subgroup_size = f.context.max_subgroup_size = sg;
                            f.context.max_wg_size = wg;
                            uint32_t expected = 256;
                            if (type == GGML_TYPE_Q4_K && k > 0 && k <= 4096 && k % 256 == 0) {
                                expected = std::max(32u, sg);
                                while (expected < uint32_t(k / 16))
                                    expected *= 2;
                                if (expected > wg)
                                    expected = 256;
                            }
                            require(f.select() == expected, "selector range/type/subgroup/device limit");
                        }
        for (bool indexed : {false, true}) {
            for (auto range : std::vector<std::pair<uint32_t, uint32_t>>{
                     {0, 0}, {0, 32}, {2, 32}, {32, 0}, {32, 16}, {8, 24}, {24, 32}, {32, 256}}) {
                Fixture f(indexed);
                f.context.min_subgroup_size = range.first;
                f.context.max_subgroup_size = range.second;
                require(f.select() == 256, "invalid subgroup range");
            }
            {
                Fixture f(indexed);
                f.context.min_subgroup_size = 4;
                f.context.max_subgroup_size = 128;
                require(f.select() == 128, "subgroup range maximum");
            }
            {
                Fixture f(indexed);
                f.context.supports_subgroups = false;
                require(f.select() == 256, "subgroups required");
            }
            {
                Fixture f(indexed);
                require(f.select(true, false) == 256 && f.select(false, true) == 256, "MMVQ and overlap excluded");
            }
            for (int which = 0; which < (indexed ? 4 : 3); which++) {
                Fixture f(indexed);
                if (which == 0)
                    f.context.src0 = nullptr;
                if (which == 1)
                    f.context.src1 = nullptr;
                if (which == 2)
                    f.context.dst = nullptr;
                if (which == 3)
                    f.context.src2 = nullptr;
                require(f.select() == 256, "missing tensor");
            }
            for (int which = 0; which < (indexed ? 4 : 3); which++)
                for (int axis = 0; axis < 4; axis++) {
                    Fixture f(indexed);
                    ggml_tensor * ts[] = {&f.a, &f.x, &f.dst, &f.ids};
                    ts[which]->ne[axis] = 0;
                    require(f.select() == 256, "empty dimension");
                }
            for (int which : {0, 1, 2, 3}) {
                if (!indexed && which == 3)
                    continue;
                Fixture f(indexed);
                ggml_tensor * ts[] = {&f.a, &f.x, &f.dst, &f.ids};
                ts[which]->type = GGML_TYPE_F16;
                require(f.select() == 256, "unsupported type");
            }
            {
                Fixture f(indexed);
                f.x.ne[0]++;
                require(f.select() == 256, "K mismatch");
            }
            {
                Fixture f(indexed);
                f.dst.ne[0]++;
                require(f.select() == 256, "M mismatch");
            }
            if (indexed) {
                {
                    Fixture f(true);
                    f.ids.ne[0]++;
                    require(f.select() == 256, "slot count mismatch");
                }
                {
                    Fixture f(true);
                    f.x.ne[1] = 3;
                    require(f.select() == 256, "unsupported input broadcast");
                    f.x.ne[1] = 1;
                    require(f.select() == 32, "shared input");
                }
                {
                    Fixture f(true);
                    f.a.ne[2] = 7;
                    require(f.select() == 256, "slots exceed experts");
                    f.a.ne[2] = int64_t(UINT32_MAX) + 1;
                    require(f.select() == 256, "expert key narrowing");
                }
                for (int which = 0; which < 4; which++) {
                    Fixture f(true);
                    ggml_tensor * ts[] = {&f.a, &f.x, &f.dst, &f.ids};
                    ts[which]->ne[3] = 2;
                    require(f.select() == 256, "higher indexed batch excluded");
                }
                for (int which = 0; which < 4; which++) {
                    Fixture f(true);
                    if (which == 0)
                        f.x.ne[2] = 2;
                    if (which == 1)
                        f.dst.ne[2] = 2;
                    if (which == 2)
                        f.ids.ne[1] = 2;
                    if (which == 3)
                        f.ids.ne[2] = 2;
                    require(f.select() == 256, "multi-token indexed excluded");
                }
            } else {
                {
                    Fixture f(false);
                    f.x.ne[1] = f.dst.ne[1] = 2;
                    require(f.select() == 256, "multi-column excluded");
                }
                for (int axis : {2, 3}) {
                    Fixture f(false);
                    f.a.ne[axis] = 2;
                    require(f.select() == 256, "invalid batch divisor");
                    f.x.ne[axis] = f.dst.ne[axis] = 6;
                    require(f.select() == 32, "ordinary broadcast supported");
                    f.dst.ne[axis] = 3;
                    require(f.select() == 256, "dst batch mismatch");
                }
            }
            {
                Fixture f(indexed);
                RecordingLibrary lib;
                ReferenceLibrary ref;
                const auto small = f.pipeline(lib);
                if (indexed) {
                    f.a.ne[2] = 23;
                    const auto different = f.pipeline(lib);
                    require(different.id != small.id && different.wgsl != small.wgsl, "expert-count cache dimension");
                    f.a.ne[2] = 11;
                    f.a.type = GGML_TYPE_Q5_1;
                    require(size(f.pipeline(lib)) == 128, "existing expert selector retained");
                    f.a.type = GGML_TYPE_Q4_K;
                    f.dst.ne[2] = 2;
                    const auto fallback = f.pipeline(lib), baseline = f.pipeline(ref);
                    require(size(fallback) == 256 && fallback.wgsl == baseline.wgsl,
                            "indexed multi-token vector fallback unchanged");
                    f.dst.ne[2] = 1;
                } else {
                    f.a.type = GGML_TYPE_Q8_0;
                    require(size(f.pipeline(lib)) == 64, "existing Q8 selector retained");
                    f.a.type = GGML_TYPE_Q4_K;
                    f.x.ne[1] = f.dst.ne[1] = 2;
                    const auto multicol = f.pipeline(lib), baseline = f.pipeline(ref);
                    require(size(multicol) == 256 && multicol.wgsl == baseline.wgsl, "multi-column factory unchanged");
                    f.x.ne[1] = f.dst.ne[1] = 1;
                    f.context.supports_dot_product = true;
                    f.context.vendor = "intel";
                    const auto mmvq = f.pipeline(lib), mmvqbase = f.pipeline(ref);
                    require(size(mmvq) == 256 && mmvq.wgsl == mmvqbase.wgsl, "MMVQ factory unchanged");
                    f.context.supports_dot_product = false;
                    f.context.vendor = "";
                    const auto overlap = lib.get_mul_mat_vec_pipeline(f.context, true, true),
                               overlapbase = ref.get_mul_mat_vec_pipeline(f.context, true, true);
                    require(size(overlap) == 256 && overlap.wgsl == overlapbase.wgsl, "overlap factory unchanged");
                }
                f.x.type = GGML_TYPE_F16;
                const auto half = f.pipeline(lib), halfbase = f.pipeline(ref);
                require(size(half) == 256 && half.wgsl == halfbase.wgsl, "F16 vector factory unchanged");
                f.x.type = GGML_TYPE_F32;
                require(f.pipeline(lib).id == small.id, "excluded path never poisons short Q4 cache");
            }
            // All original useful lanes retain the same one-block ownership.
            for (int k = 256; k <= 4096; k += 256) {
                Fixture f(indexed);
                f.a.ne[0] = f.x.ne[0] = k;
                const int chosen = f.select();
                for (int lane = 0; lane < 256; lane++) {
                    std::vector<int> before, after;
                    for (int block = lane / 16; block < k / 256; block += 256 / 16)
                        before.push_back(block);
                    if (lane < chosen)
                        for (int block = lane / 16; block < k / 256; block += chosen / 16)
                            after.push_back(block);
                    require(before == after, "lane/block ownership changed");
                }
            }
            // Exercise every pipeline construction order, not just increasing K.
            std::vector<int> order = {256, 768, 2048, 4352};
            do {
                RecordingLibrary lib;
                ReferenceLibrary ref;
                std::vector<unsigned> ids;
                for (int k : order) {
                    Fixture f(indexed);
                    f.a.ne[0] = f.x.ne[0] = k;
                    auto p = f.pipeline(lib), base = f.pipeline(ref);
                    const auto chosen = f.select();
                    require(size(p) == chosen && size(base) == 256, "factory selected workgroup");
                    require(size(p) == 256 ? p.wgsl == base.wgsl : p.wgsl != base.wgsl,
                            "upstream fallback byte identity");
                    require(p.name == base.name + (chosen == 256 ? "" : "_wg" + std::to_string(chosen)),
                            "pipeline label");
                    require(p.wgsl.find("@workgroup_size(" + std::to_string(chosen) + ")") != std::string::npos,
                            "emitted workgroup");
                    require(f.pipeline(lib).id == p.id, "repeat cache hit");
                    for (auto id : ids)
                        require(id != p.id, "cache bucket collision");
                    ids.push_back(p.id);
                    if (argc == 2) {
                        std::filesystem::create_directories(argv[1]);
                        std::ofstream(std::filesystem::path(argv[1]) /
                                      ((indexed ? std::string("indexed") : std::string("ordinary")) + "-" +
                                       std::to_string(chosen) + ".wgsl"))
                            << p.wgsl;
                    }
                }
                for (int k : order) {
                    Fixture f(indexed);
                    f.a.ne[0] = f.x.ne[0] = k;
                    const auto p = f.pipeline(lib);
                    require(std::find(ids.begin(), ids.end(), p.id) != ids.end(), "mixed cache revisit");
                }
            } while (std::next_permutation(order.begin(), order.end()));
        }
        if (argc == 2) {
            std::ofstream out(std::filesystem::path(argv[1]) / "cases.json");
            out << "[\n";
            bool first = true;
            auto emit = [&](bool indexed, int k, int m, int mode, bool shared, bool broadcast) {
                Fixture f(indexed);
                f.a.ne[0] = f.x.ne[0] = k;
                f.a.ne[1] = f.dst.ne[0] = m;
                if (indexed)
                    f.x.ne[1] = shared ? 1 : 8;
                else if (broadcast) {
                    f.a.ne[2] = 2;
                    f.a.ne[3] = 2;
                    f.x.ne[2] = f.dst.ne[2] = 4;
                    f.x.ne[3] = f.dst.ne[3] = 6;
                }
                if (!first)
                    out << ",\n";
                first = false;
                out << "{\"kind\":\"" << (indexed ? "indexed" : "ordinary") << "\",\"k\":" << k << ",\"m\":" << m
                    << ",\"mode\":" << mode << ",\"sharedInput\":" << (shared ? "true" : "false")
                    << ",\"broadcast\":" << (broadcast ? "true" : "false") << ",\"workgroups\":{";
                bool sgfirst = true;
                for (uint32_t sg : {4, 8, 16, 32, 64, 128}) {
                    f.context.min_subgroup_size = f.context.max_subgroup_size = sg;
                    RecordingLibrary lib;
                    const auto p = f.pipeline(lib);
                    if (!sgfirst)
                        out << ",";
                    sgfirst = false;
                    out << "\"" << sg << "\":" << size(p);
                }
                out << "}}";
            };
            for (bool id : {false, true})
                for (int k : {256, 512, 768, 1024, 1280, 1792, 2048, 2304, 3840, 4096, 4352})
                    for (int mode : {0, 1, 2})
                        emit(id, k, mode ? 5 : 4, mode, id && mode == 1, !id && mode == 1);
            for (bool id : {false, true})
                for (int m : {1, 3, 257})
                    emit(id, 512, m, 2, false, false);
            for (bool id : {false, true}) {
                emit(id, 512, 2048, 1, false, false);
                emit(id, 2048, 512, 2, false, false);
            }
            emit(false, 2048, 2048, 2, false, false);
            emit(false, 2048, 10752, 2, false, false);
            emit(false, 2048, 6144, 2, false, false);
            emit(false, 10752, 2048, 2, false, false);
            out << "\n]\n";
        }
        std::cout << checks << " actual Q4 selector/cache/factory checks passed; " << created
                  << " recorded pipelines\n";
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
