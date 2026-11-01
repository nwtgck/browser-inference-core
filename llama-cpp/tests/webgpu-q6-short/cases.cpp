static size_t checks = 0;
static void require(bool value) {
    ++checks;
    if (!value) {
        throw std::runtime_error("Q6 short-workgroup contract failed");
    }
}

static ggml_tensor tensor(ggml_type type, int64_t k, int64_t m) {
    ggml_tensor t{};
    t.type = type;
    t.ne[0] = k;
    t.ne[1] = m;
    t.ne[2] = t.ne[3] = 1;
    return t;
}

int main(int argc, char ** argv) {
    require(argc == 2);
    std::filesystem::create_directories(argv[1]);
    ggml_tensor a = tensor(GGML_TYPE_Q6_K, 1024, 5), x = tensor(GGML_TYPE_F32, 1024, 1),
                dst = tensor(GGML_TYPE_F32, 5, 1), ids = tensor(GGML_TYPE_I32, 1, 1);
    for (uint32_t minimum : {4u, 8u, 16u, 32u, 64u, 128u}) {
        for (uint32_t maximum : {4u, 8u, 16u, 32u, 64u, 128u}) {
            if (minimum > maximum) {
                continue;
            }
            baseline::Library old;
            candidate::Library current;
            baseline::Library::Context before{};
            candidate::Library::Context after{};
            before.src0 = after.src0 = &a;
            before.src1 = after.src1 = &x;
            before.dst = after.dst = &dst;
            before.supports_subgroups = after.supports_subgroups = true;
            before.max_wg_size = after.max_wg_size = 256;
            before.min_subgroup_size = after.min_subgroup_size = minimum;
            before.max_subgroup_size = after.max_subgroup_size = maximum;
            // Return to both short widths after long fallback and reverse order.
            for (int64_t k : {256, 512, 768, 1024, 1280, 2048, 2304, 4096, 4352, 7680, 10752, 2048, 1024, 2048}) {
                a.ne[0] = x.ne[0] = k;
                for (int64_t m : {1, 3, 4, 5, 257}) {
                    a.ne[1] = dst.ne[0] = m;
                    uint32_t expected = std::max(32u, maximum);
                    while (expected < uint32_t(k / 16)) {
                        expected *= 2;
                    }
                    if (k > 4096) {
                        expected = 256;
                    }
                    require(candidate::Library::select(after) == expected);
                    require(candidate::Library::select(after, true) == 256);
                    auto previous = old.get_mul_mat_vec_pipeline(before, false, false);
                    auto selected = current.get_mul_mat_vec_pipeline(after, false, false);
                    auto decisions = std::static_pointer_cast<candidate::Library::Decisions>(selected.context);
                    require(decisions->wg_size == expected && decisions->outputs_per_wg == 4);
                    require(current.get_mul_mat_vec_pipeline(after, false, false).id == selected.id);
                    if (expected == 256) {
                        require(previous.wgsl == selected.wgsl && previous.name == selected.name);
                    }
                    if (minimum == 16 && maximum == 16) {
                        std::ofstream(std::filesystem::path(argv[1]) / ("q6-wg" + std::to_string(expected) + ".wgsl"))
                            << selected.wgsl;
                    }
                }
            }
            a.ne[0] = x.ne[0] = 1024;
            require(candidate::Library::select(after, false, true) == 256);
            require(candidate::Library::select(after, false, false, true) == 256);
            for (int64_t k : {0, 255, 257, 1023, 1025, 2047, 2049, 4097}) {
                a.ne[0] = x.ne[0] = k;
                require(candidate::Library::select(after) == 256);
            }
            a.ne[0] = x.ne[0] = 1024;
            for (int n : {2, 4}) {
                x.ne[1] = dst.ne[1] = n;
                require(candidate::Library::select(after) == 256);
                require(old.get_mul_mat_vec_pipeline(before, false, false).wgsl ==
                        current.get_mul_mat_vec_pipeline(after, false, false).wgsl);
            }
            x.ne[1] = dst.ne[1] = 1;
            require(old.get_mul_mat_vec_pipeline(before, true, false).wgsl ==
                    current.get_mul_mat_vec_pipeline(after, true, false).wgsl);
            before.supports_dot_product = after.supports_dot_product = true;
            before.vendor = after.vendor = "intel";
            auto mmvq = current.get_mul_mat_vec_pipeline(after, false, false);
            require(std::static_pointer_cast<candidate::Library::Decisions>(mmvq.context)->wg_size == 256);
            require(old.get_mul_mat_vec_pipeline(before, false, false).wgsl == mmvq.wgsl);
            before.supports_dot_product = after.supports_dot_product = false;
            before.vendor = after.vendor = "";
            a.type = GGML_TYPE_Q4_K;
            require(baseline::Library::select(before) == candidate::Library::select(after));
            require(old.get_mul_mat_vec_pipeline(before, false, false).wgsl ==
                    current.get_mul_mat_vec_pipeline(after, false, false).wgsl);
            a.type = GGML_TYPE_Q6_K;
            after.src2 = &ids;
            // Indexed admission is explicitly excluded even if the shared helper is called.
            require(candidate::Library::select(after, true) == 256);
            after.src2 = nullptr;
        }
    }
    candidate::Library::Context c{};
    c.src0 = &a;
    c.src1 = &x;
    c.dst = &dst;
    c.supports_subgroups = true;
    c.max_wg_size = 256;
    c.min_subgroup_size = 16;
    c.max_subgroup_size = 32;
    for (uint32_t bad : {0u, 3u, 129u}) {
        c.max_subgroup_size = bad;
        require(candidate::Library::select(c) == 256);
    }
    c.max_subgroup_size = 32;
    c.max_wg_size = 32;
    require(candidate::Library::select(c) == 256);
    c.max_wg_size = 256;
    c.supports_subgroups = false;
    require(candidate::Library::select(c) == 256);
    c.supports_subgroups = true;
    x.ne[0] = 512;
    require(candidate::Library::select(c) == 256);
    x.ne[0] = 1024;
    for (auto type : {GGML_TYPE_F16, GGML_TYPE_BF16}) {
        x.type = type;
        require(candidate::Library::select(c) == 256);
    }
    x.type = GGML_TYPE_F32;
    dst.type = GGML_TYPE_F16;
    require(candidate::Library::select(c) == 256);
    dst.type = GGML_TYPE_F32;
    a.ne[2] = 2;
    x.ne[2] = dst.ne[2] = 3;
    require(candidate::Library::select(c) == 256);
    x.ne[2] = dst.ne[2] = 4;
    require(candidate::Library::select(c) == 64);
    std::cout << "{\"checks\":" << checks << ",\"shaders\":4,\"status\":\"passed\"}\n";
}
