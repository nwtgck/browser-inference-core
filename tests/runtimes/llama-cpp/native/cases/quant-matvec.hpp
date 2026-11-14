// Current complete-header Q4/Q6 recording. Historical before/after byte oracles
// remain in the separate frozen tests; no factory or selector is reconstructed.
#include <source_location>

namespace q4_current {

static size_t checks = 0;
static void require(bool condition, const char * message) {
    ++checks;
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct Fixture {
    ggml_tensor a{}, x{}, dst{}, ids{};
    uint32_t min_sg = 16, max_sg = 16, max_wg = 256;
    bool subgroups = true, dot = false, indexed = false;
    unsigned post = 0;
    std::string vendor = "mesa";

    Fixture(ggml_type type = GGML_TYPE_Q4_0, int64_t k = 1536, int64_t m = 4096) {
        a.type = type;
        x.type = dst.type = GGML_TYPE_F32;
        a.ne[0] = x.ne[0] = k;
        a.ne[1] = dst.ne[0] = m;
        x.ne[1] = dst.ne[1] = 1;
        for (int axis = 2; axis < 4; ++axis) {
            a.ne[axis] = x.ne[axis] = dst.ne[axis] = 1;
        }
    }

    template <class Context> Context context() {
        Context result{};
        result.src0 = &a;
        result.src1 = &x;
        result.src2 = indexed ? &ids : nullptr;
        result.dst = &dst;
        result.max_wg_size = max_wg;
        result.supports_subgroups = subgroups;
        result.min_subgroup_size = min_sg;
        result.max_subgroup_size = max_sg;
        result.supports_dot_product = dot;
        result.vendor = vendor;
        result.matvec_epilogue = static_cast<typename Context::matvec_post_op>(post);
        return result;
    }

    bool eligible(bool mmvq = false, bool overlap = false) {
        return ggml_webgpu_q4_matvec_subgroup_eligible(context<ggml_webgpu_shader_lib_context>(), mmvq, overlap);
    }
    uint32_t select(bool mmvq = false, bool overlap = false) {
        return ggml_webgpu_q4_matvec_subgroup_size(context<ggml_webgpu_shader_lib_context>(), mmvq, overlap);
    }
};

static ggml_webgpu_shader_lib * new_library;
static ggml_webgpu_shader_lib * scalar_library;

static void factory(Fixture & f, bool selected, bool overlap, const std::filesystem::path & output) {
    // Subgroup capability is immutable for a device/cache. Do not reuse a
    // subgroup device cache to simulate a different, scalar-only device.
    auto & selected_library = f.subgroups ? *new_library : *scalar_library;
    const auto current =
        selected_library.get_mul_mat_vec_pipeline(f.context<ggml_webgpu_shader_lib_context>(), overlap, false);
    const auto repeated =
        selected_library.get_mul_mat_vec_pipeline(f.context<ggml_webgpu_shader_lib_context>(), overlap, false);
    require(current.pipeline.Get()->id == repeated.pipeline.Get()->id,
            "Repeated effective key uses the same pipeline object");
    const auto after = std::static_pointer_cast<ggml_webgpu_mul_mat_vec_shader_decisions>(current.context);
    require(bool(after), "Actual current factory decisions");
    const auto key =
        ggml_webgpu_shader_lib::make_mul_mat_vec_key(f.context<ggml_webgpu_shader_lib_context>(), overlap, false);
    require(after->wg_size == key.workgroup_size && after->vec_size == 1 && key.src_overlap == overlap,
            "Current factory follows its effective key and scalar layout");
    require((current.name.find("_src_overlap") != std::string::npos) == overlap,
            "Current shader variant preserves the requested binding layout");
    if (selected) {
        require(after->wg_size == 128 && after->outputs_per_wg == 4 * (128 / f.min_sg),
                "Four rows per independent subgroup in WG128");
        require(current.name.find("_independent_sg" + std::to_string(f.min_sg) + "_wg128") != std::string::npos,
                "Distinct actual profile label");
        require(current.pipeline.Get()->code.find("workgroupBarrier(") == std::string::npos,
                "Independent main does not use workgroup barriers");
        const auto dtype = f.a.type == GGML_TYPE_Q4_0 ? "q4_0" : "q4_k";
        if (!output.empty()) {
            std::ofstream(output / (std::string(dtype) + "-sg" + std::to_string(f.min_sg) + ".wgsl"))
                << current.pipeline.Get()->code;
        }
    } else {
        if (f.a.type == GGML_TYPE_Q4_0 || f.a.type == GGML_TYPE_Q4_K) {
            require(after->logical_subgroup_size == 0 && after->requested_subgroup_size == 0,
                    "Unsupported Q4 admission must not select a subgroup pipeline");
        }
    }
    for (const uint32_t dimension : {65535u, 64u}) {
        const auto grid = production_dispatch(f.context<ggml_webgpu_shader_lib_context>(), dimension, overlap);
        const uint64_t groups =
            (f.dst.ne[0] + after->outputs_per_wg - 1) / after->outputs_per_wg * f.dst.ne[2] * f.dst.ne[3];
        if (!(grid[0] <= dimension && grid[1] <= dimension && uint64_t(grid[0]) * grid[1] >= groups)) {
            std::cerr << "dispatch type=" << ggml_type_name(f.a.type) << " k=" << f.a.ne[0] << " m=" << f.a.ne[1]
                      << " subgroup=" << f.min_sg << " dot=" << f.dot << " expected groups=" << groups
                      << " actual=" << grid[0] << 'x' << grid[1] << '\n';
        }
        require(grid[0] <= dimension && grid[1] <= dimension && uint64_t(grid[0]) * grid[1] >= groups,
                "Actual host helper covers every output group with bounded rectangle");
    }
}

size_t run(ggml_webgpu_shader_lib & library, ggml_webgpu_shader_lib & scalar, const std::filesystem::path & output) {
    new_library = &library;
    scalar_library = &scalar;
    if (!output.empty()) {
        std::filesystem::create_directories(output);
    }
    for (const auto type : {GGML_TYPE_Q4_0, GGML_TYPE_Q4_K}) {
        for (const uint32_t sg : {16u, 32u}) {
            for (const int64_t k : {1024, 1536, 2048, 2560, 4096}) {
                for (const int64_t m : {4096, 4097, 12288}) {
                    Fixture f(type, k, m);
                    f.min_sg = f.max_sg = sg;
                    require(f.eligible() && f.select() == sg, "Guarded shape selects its advertised fixed width");
                    factory(f, true, false, output);
                }
            }
            for (const int64_t k : {-1, 0, 992, 1025, 4097, 17664}) {
                Fixture f(type, k);
                f.min_sg = f.max_sg = sg;
                require(!f.eligible() && f.select() == 0, "K boundary and partial-block rejection");
            }
            for (const int64_t k : {4352, 5120, 10240, 17408}) {
                for (const int64_t m : {2560, 2561, 5120, 17408}) {
                    Fixture f(type, k, m);
                    f.min_sg = f.max_sg = sg;
                    require(f.eligible() && f.select() == sg, "Long K complete blocks admitted");
                    factory(f, true, false, output);
                }
                Fixture below(type, k, 2559);
                require(!below.eligible(), "Long K row boundary");
                for (int tensor = 0; tensor < 3; ++tensor)
                    for (int axis = 2; axis < 4; ++axis) {
                        Fixture b(type, k, 2560);
                        (tensor == 0 ? b.a : tensor == 1 ? b.x : b.dst).ne[axis] = 2;
                        require(!b.eligible(), "Long K requires all six batch axes one");
                    }
                Fixture b(type, k, 2560);
                b.a.ne[2] = 2;
                b.x.ne[2] = b.dst.ne[2] = 4;
                require(!b.eligible(), "Long K broadcast remains canonical");
            }
            {
                Fixture edge(type, type == GGML_TYPE_Q4_0 ? 4128 : 4352, 2560);
                require(edge.eligible(), "First long complete block");
            }
            {
                Fixture edge(type, type == GGML_TYPE_Q4_0 ? 17440 : 17664, 2560);
                require(!edge.eligible(), "First complete block after maximum");
            }
            for (const int64_t m :
                 std::array<int64_t, 5>{-1, 0, 4095, int64_t(UINT32_MAX) - 126, int64_t(UINT32_MAX) + 1}) {
                Fixture f(type, 2048, m);
                require(!f.eligible() && f.select() == 0, "M boundary and padded u32 rejection");
            }
            Fixture largest(type, 2048, int64_t(UINT32_MAX) - 127);
            require(largest.eligible(), "Same maximum row bound as the existing Q8 path");
            for (int which = 0; which < 16; ++which) {
                Fixture f(type, 2048);
                f.min_sg = f.max_sg = sg;
                switch (which) {
                case 0:
                    f.x.type = GGML_TYPE_F16;
                    break;
                case 1:
                    f.dst.type = GGML_TYPE_F16;
                    break;
                case 2:
                    f.indexed = true;
                    break;
                case 3:
                    f.post = 1;
                    break;
                case 4:
                    f.x.ne[0]--;
                    break;
                case 5:
                    f.dst.ne[0]--;
                    break;
                case 6:
                    f.x.ne[1] = 2;
                    break;
                case 7:
                    f.dst.ne[1] = 2;
                    break;
                case 8:
                    f.a.ne[2] = 0;
                    break;
                case 9:
                    f.a.ne[3] = -1;
                    break;
                case 10:
                    f.x.ne[2] = f.dst.ne[2] = 0;
                    break;
                case 11:
                    f.x.ne[3] = f.dst.ne[3] = int64_t(UINT32_MAX) + 1;
                    break;
                case 12:
                    f.a.ne[2] = 3;
                    f.x.ne[2] = f.dst.ne[2] = 4;
                    break;
                case 13:
                    f.x.ne[3] = 2;
                    break;
                case 14:
                    f.x.ne[2] = f.dst.ne[2] = UINT32_MAX;
                    break;
                case 15:
                    f.max_wg = 127;
                    break;
                }
                require(!f.eligible() && f.select() == 0, "Dtype, shape, batch, device and graph exclusions");
            }
            Fixture f(type, 2048);
            require(!f.eligible(true) && !f.eligible(false, true), "MMVQ and overlap rejection");
            for (int axis = 2; axis < 4; ++axis) {
                Fixture b(type, 2048);
                b.min_sg = b.max_sg = sg;
                b.a.ne[axis] = 2;
                b.x.ne[axis] = b.dst.ne[axis] = 4;
                require(b.eligible(), "Broadcast batch accepted within u32 bound");
                factory(b, true, false, output);
            }
        }
        for (const auto & widths : {std::pair{4u, 4u}, std::pair{8u, 8u}, std::pair{64u, 64u}, std::pair{8u, 32u}}) {
            Fixture f(type, 2048);
            f.min_sg = widths.first;
            f.max_sg = widths.second;
            require(f.eligible() && f.select() == 0, "Shape eligibility does not fabricate a fixed subgroup width");
            factory(f, false, false, output);
        }
        for (const int64_t k : {768, 17664}) {
            Fixture f(type, k);
            require(f.select() == 0, "Out-of-range complete blocks keep old selection");
            factory(f, false, false, output);
        }
        Fixture small(type, 2048, 4095);
        factory(small, false, false, output);
        Fixture mmvq(type, 2048);
        mmvq.vendor = "intel";
        mmvq.dot = true;
        factory(mmvq, false, false, output);
        Fixture overlap(type, 2048);
        factory(overlap, false, true, output);
        Fixture scalar(type, 2048);
        scalar.subgroups = false;
        require(scalar.select() == 0, "Subgroup capability required by width policy");
        factory(scalar, false, false, output);
    }
    for (const auto type : {GGML_TYPE_Q8_0, GGML_TYPE_Q6_K, GGML_TYPE_F32, GGML_TYPE_F16}) {
        for (const auto sg : {8u, 16u, 32u, 64u}) {
            Fixture f(type, 2048);
            f.min_sg = f.max_sg = sg;
            require(!f.eligible() && f.select() == 0, "Other dtypes remain outside Q4 admission");
            factory(f, false, false, output);
        }
    }
    {
        Fixture f;
        auto c = f.context<ggml_webgpu_shader_lib_context>();
        c.src0 = nullptr;
        require(!ggml_webgpu_q4_matvec_subgroup_eligible(c, false, false), "Null weights rejected");
        c = f.context<ggml_webgpu_shader_lib_context>();
        c.src1 = nullptr;
        require(!ggml_webgpu_q4_matvec_subgroup_eligible(c, false, false), "Null input rejected");
        c = f.context<ggml_webgpu_shader_lib_context>();
        c.dst = nullptr;
        require(!ggml_webgpu_q4_matvec_subgroup_eligible(c, false, false), "Null output rejected");
    }
    return checks;
}

} // namespace q4_current
namespace q6_current {
static uint32_t select(const ggml_webgpu_shader_lib_context & context, bool indexed = false, bool mmvq = false,
                       bool overlap = false) {
    return ggml_webgpu_q4_short_workgroup(context, indexed, mmvq, overlap);
}
static size_t checks = 0;
static void require(bool value, std::source_location where = std::source_location::current()) {
    ++checks;
    if (!value) {
        throw std::runtime_error("Q6 short-workgroup contract failed at line " + std::to_string(where.line()));
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

size_t run(wgpu::Device device, wgpu::Instance instance, const std::filesystem::path & output) {
    std::filesystem::create_directories(output);
    ggml_tensor a = tensor(GGML_TYPE_Q6_K, 1024, 5), x = tensor(GGML_TYPE_F32, 1024, 1),
                dst = tensor(GGML_TYPE_F32, 5, 1), ids = tensor(GGML_TYPE_I32, 1, 1);
    for (uint32_t minimum : {4u, 8u, 16u, 32u, 64u, 128u}) {
        for (uint32_t maximum : {4u, 8u, 16u, 32u, 64u, 128u}) {
            if (minimum > maximum) {
                continue;
            }
            auto state = std::make_shared<ggml_webgpu_optional::DeviceGeneration>();
            ggml_webgpu_shader_lib current(device, instance, state);
            ggml_webgpu_shader_lib_context after{};
            after.src0 = &a;
            after.src1 = &x;
            after.dst = &dst;
            after.supports_subgroups = true;
            after.max_wg_size = 256;
            after.min_subgroup_size = minimum;
            after.max_subgroup_size = maximum;
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
                    require(select(after) == expected);
                    require(select(after, true) == 256);
                    auto selected = current.get_mul_mat_vec_pipeline(after, false, false);
                    auto decisions =
                        std::static_pointer_cast<ggml_webgpu_mul_mat_vec_shader_decisions>(selected.context);
                    require(decisions->wg_size == expected && decisions->outputs_per_wg == 4);
                    require(current.get_mul_mat_vec_pipeline(after, false, false).pipeline.Get()->id ==
                            selected.pipeline.Get()->id);
                    require(selected.pipeline.Get()->code.find("@workgroup_size(" + std::to_string(expected)) !=
                            std::string::npos);
                    if (minimum == 16 && maximum == 16) {
                        std::ofstream(output / ("q6-wg" + std::to_string(expected) + ".wgsl"))
                            << selected.pipeline.Get()->code;
                    }
                }
            }
            a.ne[0] = x.ne[0] = 1024;
            require(select(after, false, true) == 256);
            require(select(after, false, false, true) == 256);
            for (int64_t k : {0, 255, 257, 1023, 1025, 2047, 2049, 4097}) {
                a.ne[0] = x.ne[0] = k;
                require(select(after) == 256);
            }
            a.ne[0] = x.ne[0] = 1024;
            for (int n : {2, 4}) {
                x.ne[1] = dst.ne[1] = n;
                require(select(after) == 256);
                require(std::static_pointer_cast<ggml_webgpu_mul_mat_vec_shader_decisions>(
                            current.get_mul_mat_vec_pipeline(after, false, false).context)
                            ->wg_size == 256);
            }
            x.ne[1] = dst.ne[1] = 1;
            require(std::static_pointer_cast<ggml_webgpu_mul_mat_vec_shader_decisions>(
                        current.get_mul_mat_vec_pipeline(after, true, false).context)
                        ->wg_size == 256);
            after.supports_dot_product = true;
            after.vendor = "intel";
            auto mmvq = current.get_mul_mat_vec_pipeline(after, false, false);
            require(std::static_pointer_cast<ggml_webgpu_mul_mat_vec_shader_decisions>(mmvq.context)->wg_size == 256);
            require(mmvq.pipeline.Get()->code.find("#include") == std::string::npos);
            after.supports_dot_product = false;
            after.vendor = "";
            a.type = GGML_TYPE_Q4_K;
            require(ggml_webgpu_q4_short_workgroup(after, false, false, false) == std::max(64u, maximum));
            require(std::static_pointer_cast<ggml_webgpu_mul_mat_vec_shader_decisions>(
                        current.get_mul_mat_vec_pipeline(after, false, false).context)
                        ->wg_size == std::max(64u, maximum));
            a.type = GGML_TYPE_Q6_K;
            after.src2 = &ids;
            // Indexed admission is explicitly excluded even if the shared helper is called.
            require(select(after, true) == 256);
            after.src2 = nullptr;
        }
    }
    ggml_webgpu_shader_lib_context c{};
    c.src0 = &a;
    c.src1 = &x;
    c.dst = &dst;
    c.supports_subgroups = true;
    c.max_wg_size = 256;
    c.min_subgroup_size = 16;
    c.max_subgroup_size = 32;
    for (uint32_t bad : {0u, 3u, 129u}) {
        c.max_subgroup_size = bad;
        require(select(c) == 256);
    }
    c.max_subgroup_size = 32;
    c.max_wg_size = 32;
    require(select(c) == 256);
    c.max_wg_size = 256;
    c.supports_subgroups = false;
    require(select(c) == 256);
    c.supports_subgroups = true;
    x.ne[0] = 512;
    require(select(c) == 256);
    x.ne[0] = 1024;
    for (auto type : {GGML_TYPE_F16, GGML_TYPE_BF16}) {
        x.type = type;
        require(select(c) == 256);
    }
    x.type = GGML_TYPE_F32;
    dst.type = GGML_TYPE_F16;
    require(select(c) == 256);
    dst.type = GGML_TYPE_F32;
    a.ne[2] = 2;
    x.ne[2] = dst.ne[2] = 3;
    require(select(c) == 256);
    x.ne[2] = dst.ne[2] = 4;
    require(select(c) == 64);
    return checks;
}

} // namespace q6_current
static int run_quant_matvec_cases(int argc, char ** argv) {
    try {
        if (argc != 2)
            throw std::runtime_error("Expected shader directory");
        auto instance = wgpu::Instance::Acquire(new WGPUInstanceImpl);
        auto device = wgpu::Device::Acquire(new WGPUDeviceImpl);
        auto state = std::make_shared<ggml_webgpu_optional::DeviceGeneration>();
        ggml_webgpu_shader_lib library(device, instance, state);
        auto scalar_device = wgpu::Device::Acquire(new WGPUDeviceImpl);
        ggml_webgpu_shader_lib scalar(scalar_device, instance,
                                      std::make_shared<ggml_webgpu_optional::DeviceGeneration>());
        const size_t q4_checks = q4_current::run(library, scalar, argv[1]);
        const size_t q6_checks = q6_current::run(device, instance, argv[1]);
        BICORE_CHECK(q4_checks + q6_checks > 9000 && created > 100);
        std::cout << "{\"status\":\"passed\",\"checks\":" << q4_checks + q6_checks << ",\"pipelines\":" << created
                  << "}\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
