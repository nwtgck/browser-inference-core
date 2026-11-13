static unsigned checks = 0;
static void require(bool ok, const char * message) {
    ++checks;
    if (!ok)
        throw std::runtime_error(message);
}
struct Fixture {
    ggml_tensor a{}, x{}, ids{}, dst{};
    ggml_webgpu_shader_lib_context context{};
    Fixture() {
        a.type = GGML_TYPE_Q5_1;
        x.type = dst.type = GGML_TYPE_F32;
        ids.type = GGML_TYPE_I32;
        for (auto * t : {&a, &x, &ids, &dst})
            for (auto & n : t->ne)
                n = 1;
        a.ne[0] = x.ne[0] = 640;
        a.ne[1] = dst.ne[0] = 2560;
        a.ne[2] = 512;
        x.ne[1] = ids.ne[0] = dst.ne[1] = 10;
        context.src0 = &a;
        context.src1 = &x;
        context.src2 = &ids;
        context.dst = &dst;
        context.min_subgroup_size = context.max_subgroup_size = 32;
        context.supports_subgroups = true;
        context.max_wg_size = 256;
    }
    uint32_t select() {
        return ggml_webgpu_expert_short_workgroup(context);
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
        for (ggml_type type :
             {GGML_TYPE_Q5_1, GGML_TYPE_IQ4_NL, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_F16, GGML_TYPE_F32})
            for (int sg : {4, 8, 16, 32, 64, 128})
                for (int wg : {64, 128, 256, 512})
                    for (int k = -32; k <= 1120; k += 16)
                        for (int m : {1, 255, 256, 257, 2560}) {
                            Fixture f;
                            f.a.type = type;
                            f.a.ne[0] = f.x.ne[0] = k;
                            f.a.ne[1] = f.dst.ne[0] = m;
                            f.context.min_subgroup_size = f.context.max_subgroup_size = sg;
                            f.context.max_wg_size = wg;
                            const bool expected = (type == GGML_TYPE_Q5_1 || type == GGML_TYPE_IQ4_NL) && k > 0 &&
                                                  k <= 1024 && k % 32 == 0 && m >= 256 && wg >= 128;
                            require(f.select() == (expected ? 128u : 256u), "selector range/type limits");
                        }
        for (auto range : std::vector<std::pair<uint32_t, uint32_t>>{
                 {0, 0}, {0, 32}, {2, 32}, {32, 0}, {32, 16}, {8, 24}, {24, 32}, {32, 256}}) {
            Fixture f;
            f.context.min_subgroup_size = range.first;
            f.context.max_subgroup_size = range.second;
            require(f.select() == 256, "invalid subgroup range fallback");
        }
        {
            Fixture f;
            f.context.min_subgroup_size = 4;
            f.context.max_subgroup_size = 128;
            require(f.select() == 128, "range includes full subgroups");
        }
        {
            Fixture f;
            f.context.supports_subgroups = false;
            require(f.select() == 256, "no subgroups fallback");
        }
        for (int which = 0; which < 4; which++) {
            Fixture f;
            if (which == 0)
                f.context.src0 = nullptr;
            if (which == 1)
                f.context.src1 = nullptr;
            if (which == 2)
                f.context.src2 = nullptr;
            if (which == 3)
                f.context.dst = nullptr;
            require(f.select() == 256, "missing tensor fallback");
        }
        // Every dimension guard is checked independently of the others.
        for (int which = 0; which < 4; which++)
            for (int axis = 0; axis < 4; axis++) {
                Fixture f;
                ggml_tensor * tensors[] = {&f.a, &f.x, &f.ids, &f.dst};
                const auto saved = tensors[which]->ne[axis];
                tensors[which]->ne[axis] = 0;
                require(f.select() == 256, "empty dimension fallback");
                tensors[which]->ne[axis] = saved;
                require(f.select() == 128, "restored dimensions");
            }
        for (int which : {1, 2, 3}) {
            Fixture f;
            ggml_tensor * tensors[] = {&f.a, &f.x, &f.ids, &f.dst};
            tensors[which]->type = GGML_TYPE_F16;
            require(f.select() == 256, "input/output type fallback");
        }
        {
            Fixture f;
            f.x.ne[0]++;
            require(f.select() == 256, "input width mismatch");
        }
        {
            Fixture f;
            f.dst.ne[0]++;
            require(f.select() == 256, "output width mismatch");
        }
        {
            Fixture f;
            f.ids.ne[0]++;
            require(f.select() == 256, "selection count mismatch");
        }
        {
            Fixture f;
            f.x.ne[1] = 3;
            require(f.select() == 256, "unsupported input broadcast");
            f.x.ne[1] = 1;
            require(f.select() == 128, "shared input");
        }
        {
            Fixture f;
            f.a.ne[2] = 9;
            require(f.select() == 256, "slots exceed experts");
            f.a.ne[2] = int64_t(UINT32_MAX) + 1;
            require(f.select() == 256, "expert key narrowing");
        }
        for (int which = 0; which < 4; which++) {
            Fixture f;
            ggml_tensor * tensors[] = {&f.a, &f.x, &f.ids, &f.dst};
            tensors[which]->ne[3] = 2;
            require(f.select() == 256, "higher batch rejected");
        }
        {
            Fixture f;
            f.x.ne[2] = f.dst.ne[2] = f.ids.ne[1] = 2;
            require(f.select() == 256, "multiple tokens rejected");
        }
        // Each useful lane keeps exactly the same quant block under WG128.
        for (int k = 32; k <= 1024; k += 32)
            for (int lane = 0; lane < 256; lane++) {
                std::vector<int> old_blocks, new_blocks;
                for (int block = lane / 4; block < k / 32; block += 256 / 4)
                    old_blocks.push_back(block);
                if (lane < 128)
                    for (int block = lane / 4; block < k / 32; block += 128 / 4)
                        new_blocks.push_back(block);
                require(old_blocks == new_blocks, "lane block ownership changed");
            }
        for (ggml_type type : {GGML_TYPE_Q5_1, GGML_TYPE_IQ4_NL}) {
            Fixture f;
            f.a.type = type;
            RecordingLibrary lib;
            ReferenceLibrary reference;
            const auto small = lib.get_mul_mat_id_vec_pipeline(f.context, true);
            const auto baseline = reference.get_mul_mat_id_vec_pipeline(f.context, true);
            require(size(small) == 128 && size(baseline) == 256, "actual factory size");
            require(small.name == baseline.name + "_wg128", "profile label");
            require(small.wgsl.find("@workgroup_size(128)") != std::string::npos, "actual WGSL size");
            require(lib.get_mul_mat_id_vec_pipeline(f.context, true).id == small.id, "repeat cache hit");
            f.a.ne[0] = f.x.ne[0] = 1024;
            require(lib.get_mul_mat_id_vec_pipeline(f.context, true).id == small.id, "same selected bucket hit");
            f.a.ne[0] = f.x.ne[0] = 1056;
            const auto large = lib.get_mul_mat_id_vec_pipeline(f.context, true);
            require(large.id != small.id && size(large) == 256, "workgroup-specific cache separation");
            require(large.name == baseline.name && large.wgsl == baseline.wgsl,
                    "fallback is byte-identical to upstream factory");
            f.a.ne[0] = f.x.ne[0] = 640;
            f.a.ne[1] = f.dst.ne[0] = 255;
            require(lib.get_mul_mat_id_vec_pipeline(f.context, true).id == large.id, "small row fallback cache");
            f.a.ne[1] = f.dst.ne[0] = 2560;
            f.a.ne[2] = 11;
            const auto other = lib.get_mul_mat_id_vec_pipeline(f.context, true);
            require(other.id != small.id && other.wgsl != small.wgsl, "expert count cache separation");
            f.a.ne[2] = 512;
            f.a.type = GGML_TYPE_Q8_0;
            require(size(lib.get_mul_mat_id_vec_pipeline(f.context, true)) == 256, "other quant unchanged");
            f.a.type = type;
            require(lib.get_mul_mat_id_vec_pipeline(f.context, true).id == small.id,
                    "fallback never poisons selected cache");
            if (argc == 2) {
                std::filesystem::create_directories(argv[1]);
                const auto name = type == GGML_TYPE_Q5_1 ? "Q5_1" : "IQ4_NL";
                std::ofstream(std::filesystem::path(argv[1]) / (std::string(name) + "-128.wgsl")) << small.wgsl;
                std::ofstream(std::filesystem::path(argv[1]) / (std::string(name) + "-256.wgsl")) << baseline.wgsl;
            }
        }
        if (argc == 2) {
            std::ofstream out(std::filesystem::path(argv[1]) / "cases.json");
            out << "[\n";
            bool first = true;
            auto emit = [&](ggml_type type, int k, int m, int mode, bool shared, int stored) {
                Fixture f;
                f.a.type = type;
                f.a.ne[0] = f.x.ne[0] = k;
                f.a.ne[1] = f.dst.ne[0] = m;
                f.x.ne[1] = shared ? 1 : 10;
                RecordingLibrary lib;
                const auto pipeline = lib.get_mul_mat_id_vec_pipeline(f.context, true);
                if (!first)
                    out << ",\n";
                first = false;
                out << "{\"type\":\"" << (type == GGML_TYPE_Q5_1 ? "Q5_1" : "IQ4_NL") << "\",\"k\":" << k
                    << ",\"m\":" << m << ",\"mode\":" << mode << ",\"sharedInput\":" << (shared ? "true" : "false")
                    << ",\"storedExperts\":" << stored << ",\"chosen\":" << size(pipeline) << ",\"label\":\""
                    << pipeline.name << "\"}";
            };
            for (ggml_type type : {GGML_TYPE_Q5_1, GGML_TYPE_IQ4_NL}) {
                for (int k : {32, 64, 96, 256, 320, 640, 992, 1024, 1056, 1088})
                    for (int mode : {0, 1, 2})
                        for (bool shared : {false, true})
                            emit(type, k, mode ? 257 : 256, mode, shared, 11);
                for (int m : {1, 255, 2560})
                    emit(type, 640, m, 2, false, 11);
                for (bool shared : {false, true})
                    emit(type, 640, 257, 1, shared, 512);
            }
            out << "\n]\n";
        }
        std::cout << checks << " actual expert selector/cache/factory checks passed; " << created
                  << " recorded pipelines\n";
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
