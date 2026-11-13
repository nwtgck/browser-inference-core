static void check(bool condition) {
    if (!condition) {
        throw std::runtime_error("Q8 diagnostic factory check failed");
    }
}
int main(int argc, char ** argv) {
    try {
        check(argc == 2);
        for (bool reverse : {false, true})
            for (uint32_t sg : {4u, 8u, 16u, 32u, 64u}) {
                RecordingLibrary library;
                ReferenceLibrary reference;
                ggml_tensor a{}, x{}, dst{};
                a.type = GGML_TYPE_Q8_0;
                a.ne[0] = 2560;
                a.ne[1] = 10240;
                a.ne[2] = a.ne[3] = 1;
                x.type = dst.type = GGML_TYPE_F32;
                x.ne[0] = 2560;
                dst.ne[0] = 10240;
                for (int i = 1; i < 4; i++) {
                    x.ne[i] = dst.ne[i] = 1;
                }
                ggml_webgpu_shader_lib_context context{};
                context.src0 = &a;
                context.src1 = &x;
                context.dst = &dst;
                context.supports_subgroups = true;
                context.min_subgroup_size = context.max_subgroup_size = sg;
                context.max_wg_size = 1024;
                auto automatic = RecordingLibrary::make_mul_mat_vec_key(context, false, true);
                check(automatic.subgroup_size == sg && automatic.workgroup_size == 128);
                check(!library.find_q8_diagnostic_pipeline(automatic));
                const auto normal = library.get_mul_mat_vec_pipeline(context, false, true);
                check(normal.wgsl == reference.get_mul_mat_vec_pipeline(context, false, true).wgsl);
                check(!library.find_q8_diagnostic_pipeline(automatic)); // A normal cache entry is not PREPARE.
                std::set<unsigned> ids;
                for (unsigned n = 0; n < 5; n++) {
                    const unsigned variant = reverse ? 5 - n : n + 1;
                    auto key = automatic;
                    key.workgroup_size = variant == 1 ? 256 : 128;
                    key.subgroup_size = variant >= 2 && variant <= 4 ? sg : 0;
                    key.q8_rows_per_tile = variant == 3 ? 1 : variant >= 4 ? 2 : 4;
                    check(!library.find_q8_diagnostic_pipeline(key));
                    const auto a = library.prepare_q8_diagnostic_pipeline(context, key);
                    const auto b = library.find_q8_diagnostic_pipeline(key);
                    check(b && a.id == b->id);
                    check(library.prepare_q8_diagnostic_pipeline(context, key).id == a.id);
                    ids.insert(a.id);
                    if (variant == 2) {
                        check(a.id == normal.id && a.wgsl == normal.wgsl);
                    }
                    if (variant == 1) {
                        auto legacy_a = *context.src0, legacy_x = *context.src1;
                        legacy_a.ne[0] = legacy_x.ne[0] = 10240;
                        auto legacy_context = context;
                        legacy_context.src0 = &legacy_a;
                        legacy_context.src1 = &legacy_x;
                        check(a.wgsl == reference.get_mul_mat_vec_pipeline(legacy_context, false, true).wgsl);
                    }
                    const auto * d = static_cast<ggml_webgpu_mul_mat_vec_shader_decisions *>(a.context.get());
                    check(d->wg_size == key.workgroup_size &&
                          d->outputs_per_wg ==
                              (key.subgroup_size ? key.q8_rows_per_tile * 128 / sg : key.q8_rows_per_tile));
                    const auto name =
                        std::string(argv[1]) + "/sg" + std::to_string(sg) + "-variant" + std::to_string(variant);
                    std::ofstream(name + ".wgsl") << a.wgsl;
                    std::ofstream digest(name + ".sha256");
                    for (uint8_t byte : a.diagnostic_sha256) {
                        static const char hex[] = "0123456789abcdef";
                        digest << hex[byte >> 4] << hex[byte & 15];
                    }
                }
                check(ids.size() == 5);
            }
        // AUTO follows the real short/fallback selector instead of a forced independent shape guard.
        for (int k : {32, 320, 640, 1024, 2560, 10240})
            for (int m : {1, 320, 512, 640, 10240}) {
                ggml_tensor a{}, x{}, dst{};
                a.type = GGML_TYPE_Q8_0;
                x.type = dst.type = GGML_TYPE_F32;
                a.ne[0] = x.ne[0] = k;
                a.ne[1] = dst.ne[0] = m;
                for (int i = 2; i < 4; i++) {
                    a.ne[i] = 1;
                }
                for (int i = 1; i < 4; i++) {
                    x.ne[i] = dst.ne[i] = 1;
                }
                ggml_webgpu_shader_lib_context c{};
                c.src0 = &a;
                c.src1 = &x;
                c.dst = &dst;
                c.max_wg_size = 1024;
                c.supports_subgroups = true;
                c.min_subgroup_size = c.max_subgroup_size = 16;
                auto key = RecordingLibrary::make_mul_mat_vec_key(c, false, true);
                check(key.subgroup_size == ggml_webgpu_q8_matvec_subgroup_size(c, key.use_mmvq, false));
                if (!key.subgroup_size) {
                    check(key.workgroup_size == ggml_webgpu_q8_short_workgroup(c, key.use_mmvq, false));
                }
            }
        std::cout
            << "25 actual forced factory shaders, both cache orders, AUTO short/fallback, prepare-only cache and SHA emitted\n";
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
