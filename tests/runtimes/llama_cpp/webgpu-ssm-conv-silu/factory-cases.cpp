}
;
int main(int argc, char ** argv) {
    try {
        if (argc != 2) {
            throw std::runtime_error("output directory required");
        }
        for (bool reverse : {false, true}) {
            Library library;
            std::set<unsigned> ids;
            for (int n = 0; n < 8; n++) {
                const int bits = reverse ? 7 - n : n;
                const bool fused = bits & 1, vec4 = bits & 2, single = bits & 4;
                ggml_tensor w{}, dst{};
                w.ne[0] = vec4 ? 4 : 3;
                dst.type = GGML_TYPE_F32;
                dst.ne[1] = single ? 1 : 9;
                ggml_webgpu_shader_lib_context c;
                c.src1 = &w;
                c.dst = &dst;
                auto a = library.get_ssm_conv_pipeline(c, fused), b = library.get_ssm_conv_pipeline(c, fused);
                if (a.id != b.id || a.code.find('#') != std::string::npos) {
                    throw std::runtime_error("cache/preprocessor regression");
                }
                const auto * d = static_cast<ggml_webgpu_ssm_conv_shader_decisions *>(a.context.get());
                if (d->block_size != 32 || d->tokens_per_wg != (single ? 1u : 8u)) {
                    throw std::runtime_error("workgroup decision regression");
                }
                if ((a.name.find("_silu") != std::string::npos) != fused ||
                    (a.name.find("_vec4") != std::string::npos) != vec4) {
                    throw std::runtime_error("variant label collision");
                }
                if ((a.code.find("exp(-sum)") != std::string::npos) != fused) {
                    throw std::runtime_error("epilogue cache collision");
                }
                ids.insert(a.id);
                std::ofstream(std::string(argv[1]) + "/" + (fused ? "fused" : "plain") + "-" +
                              (vec4 ? "vec4" : "scalar") + "-" + (single ? "single" : "multi") + ".wgsl")
                    << a.code;
            }
            if (ids.size() != 8) {
                throw std::runtime_error("pipeline key collision");
            }
        }
        std::cout << "8 actual factory variants, both cache orders, canonical and fused shaders emitted\n";
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
