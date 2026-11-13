#include "../webgpu-native-recording/api.hpp"

// Standard/SDK/ggml/preprocessor declarations are loaded globally first. The
// two complete shader libraries then occupy separate test-only namespaces.
#include OPTIONAL_HEADER
#if RECORD_PROFILE
#include "ggml-webgpu-q8-diagnostic-sha256.hpp"
#include DIAGNOSTIC_HEADER
#endif
namespace baseline {
#include BASE_HEADER
}
#undef GGML_WEBGPU_SHADER_LIB_HPP
namespace candidate {
#include CANDIDATE_HEADER
}

struct Fixture {
    ggml_tensor a{}, x{}, dst{}, ids{};
    Fixture() {
        for (int axis = 0; axis < 4; axis++)
            a.ne[axis] = x.ne[axis] = dst.ne[axis] = 1;
        a.type = GGML_TYPE_Q8_0;
        x.type = dst.type = GGML_TYPE_F32;
        a.ne[0] = x.ne[0] = 6144;
        a.ne[1] = dst.ne[0] = 5120;
    }
    template <class Context> Context context(unsigned minimum, unsigned maximum, bool feature) {
        Context c{};
        c.src0 = &a;
        c.src1 = &x;
        c.dst = &dst;
        c.max_wg_size = 256;
        c.supports_subgroups = true;
        c.min_subgroup_size = minimum;
        c.max_subgroup_size = maximum;
        if constexpr (requires { c.subgroup_size_control_enabled; })
            c.subgroup_size_control_enabled = feature;
        return c;
    }
};
static unsigned checks = 0;
static void require(bool okay, const char * message) {
    ++checks;
    if (!okay)
        throw std::runtime_error(message);
}
static const candidate::ggml_webgpu_mul_mat_vec_shader_decisions &
decisions(const candidate::webgpu_pipeline & pipeline) {
    return *static_cast<const candidate::ggml_webgpu_mul_mat_vec_shader_decisions *>(pipeline.context.get());
}
#include HOST_FRAGMENTS

static Fixture fixture(ggml_type type) {
    Fixture f;
    f.a.type = type;
    f.a.ne[0] = f.x.ne[0] = 2048;
    f.a.ne[1] = f.dst.ne[0] = 4097;
    return f;
}

static std::string without_attribute(std::string code) {
    for (const std::string line : {"enable subgroup_size_control;\n", "@subgroup_size(16)\n"}) {
        const auto at = code.find(line);
        require(at != std::string::npos, "Requested enable/attribute absent");
        code.erase(at, line.size());
    }
    return code;
}

static void write(const std::filesystem::path & path, const std::string & code) {
    std::ofstream output(path);
    output << code;
    require(bool(output), "Emission write failed");
}

int main(int argc, char ** argv) {
    try {
        require(argc == 2, "Missing emission directory");
        const std::filesystem::path output = argv[1];
        std::filesystem::create_directories(output);
        auto instance = wgpu::Instance::Acquire(new WGPUInstanceImpl);
        auto device = wgpu::Device::Acquire(new WGPUDeviceImpl);
        auto state = std::make_shared<ggml_webgpu_optional::DeviceGeneration>();
        baseline::ggml_webgpu_shader_lib original(device);
        candidate::ggml_webgpu_shader_lib patched(device, instance, state);
        std::unordered_map<candidate::ggml_webgpu_mul_mat_vec_pipeline_key, int,
                           candidate::ggml_webgpu_mul_mat_vec_pipeline_key_hash>
            keys;
        for (const auto type : {GGML_TYPE_Q4_0, GGML_TYPE_Q4_K}) {
            auto f = fixture(type);
            const std::string name = type == GGML_TYPE_Q4_0 ? "q4_0" : "q4_K";
            for (unsigned width : {16u, 32u}) {
                const auto before = pushes;
                auto old_context = f.context<baseline::ggml_webgpu_shader_lib_context>(width, width, false);
                auto new_context = f.context<candidate::ggml_webgpu_shader_lib_context>(width, width, true);
                const auto a = original.get_mul_mat_vec_pipeline(old_context, false, false);
                const auto b = patched.get_mul_mat_vec_pipeline(new_context, false, false);
                require(a.pipeline.Get()->code == b.pipeline.Get()->code && a.name == b.name, "Fixed Q4 AUTO changed");
                require(pushes == before, "Fixed Q4 gained optional scopes");
            }
            auto context = f.context<candidate::ggml_webgpu_shader_lib_context>(8, 32, true);
            const auto requested = patched.get_mul_mat_vec_pipeline(context, false, false);
            const auto key = candidate::ggml_webgpu_shader_lib::make_mul_mat_vec_key(context, false, false);
            keys[key] = type;
            require(decisions(requested).logical_subgroup_size == 16 &&
                        decisions(requested).requested_subgroup_size == 16 && decisions(requested).wg_size == 128 &&
                        decisions(requested).outputs_per_wg == 32,
                    "Q4 requested decisions");
            const auto fixed = original.get_mul_mat_vec_pipeline(
                f.context<baseline::ggml_webgpu_shader_lib_context>(16, 16, false), false, false);
            require(without_attribute(requested.pipeline.Get()->code) == fixed.pipeline.Get()->code,
                    "Q4 requested arithmetic changed");
            const auto creates_before = created, pushes_before = pushes, waits_before = waits;
            require(patched.get_mul_mat_vec_pipeline(context, false, false).pipeline.Get() == requested.pipeline.Get(),
                    "Q4 warm identity");
            require(created == creates_before && pushes == pushes_before && waits == waits_before,
                    "Q4 warm compile/scope/wait");
            require(context.min_subgroup_size == 8 && context.max_subgroup_size == 32, "Advertised range changed");
            write(output / (name + "-requested.wgsl"), requested.pipeline.Get()->code);
            auto absent = context;
            absent.subgroup_size_control_enabled = false;
            const auto canonical = patched.get_mul_mat_vec_pipeline(absent, false, false);
            const auto old = original.get_mul_mat_vec_pipeline(
                f.context<baseline::ggml_webgpu_shader_lib_context>(8, 32, false), false, false);
            require(canonical.pipeline.Get()->code == old.pipeline.Get()->code && canonical.name == old.name,
                    "Q4 feature absence changed canonical");
            candidate::ggml_webgpu_shader_lib rejected(device, instance, state);
            inject = WGPUErrorType_Validation;
            const auto fallback = rejected.get_mul_mat_vec_pipeline(context, false, false);
            const auto & d = decisions(fallback);
            require(fallback.pipeline.Get()->code == old.pipeline.Get()->code && fallback.name == old.name,
                    "Q4 fallback dtype/source");
            require(d.requested_subgroup_size == 0 && d.logical_subgroup_size == 0 && d.outputs_per_wg == 4 &&
                        d.wg_size == (type == GGML_TYPE_Q4_0 ? 256u : 128u),
                    "Q4 fallback whole geometry");
            require(production_dispatch(requested, &f.dst, 65535)[0] == 129 &&
                        production_dispatch(fallback, &f.dst, 65535)[0] == 1025,
                    "Q4 actual caller grid handoff");
            const auto rejected_created = created, rejected_pushes = pushes, rejected_waits = waits;
            require(rejected.get_mul_mat_vec_pipeline(context, false, false).pipeline.Get() == fallback.pipeline.Get(),
                    "Q4 failed cache identity");
            require(created == rejected_created && pushes == rejected_pushes && waits == rejected_waits,
                    "Q4 failed key retried");
            write(output / (name + "-canonical.wgsl"), fallback.pipeline.Get()->code);
            if (type == GGML_TYPE_Q4_K) {
                f.a.ne[0] = f.x.ne[0] = 1024;
                require(candidate::ggml_webgpu_shader_lib::make_mul_mat_vec_key(context, false, false) == key,
                        "Q4 requested key unexpectedly depends on K");
                const auto expected = original.get_mul_mat_vec_pipeline(
                    f.context<baseline::ggml_webgpu_shader_lib_context>(8, 32, false), false, false);
                const auto before_create = created, before_push = pushes, before_wait = waits;
                const auto smaller = rejected.get_mul_mat_vec_pipeline(context, false, false);
                require(decisions(smaller).wg_size == 64 && decisions(smaller).outputs_per_wg == 4 &&
                            smaller.pipeline.Get()->code == expected.pipeline.Get()->code,
                        "Rejected Q4 key reused stale canonical geometry");
                require(created == before_create + 1 && pushes == before_push && waits == before_wait,
                        "Changed canonical context retried requested pipeline");
                f.a.ne[0] = f.x.ne[0] = 2048;
                require(rejected.get_mul_mat_vec_pipeline(context, false, false).pipeline.Get() ==
                            fallback.pipeline.Get(),
                        "Q4 canonical geometry did not restore from its own key");
                require(created == before_create + 1 && pushes == before_push && waits == before_wait,
                        "Q4 canonical revisit added compile/scope/wait");
            }
            f.a.ne[0] = f.x.ne[0] = type == GGML_TYPE_Q4_0 ? 10240 : 17408;
            f.a.ne[1] = f.dst.ne[0] = 2560;
            require(candidate::ggml_webgpu_q4_requested_subgroup_size(context, false, false) == 16,
                    "Q4 long K requested eligibility");
            require(patched.get_mul_mat_vec_pipeline(context, false, false).pipeline.Get() == requested.pipeline.Get(),
                    "Q4 long K must reuse requested shader key");
            const auto old_long = original.get_mul_mat_vec_pipeline(
                f.context<baseline::ggml_webgpu_shader_lib_context>(8, 32, false), false, false);
            const auto push_long = pushes, wait_long = waits;
            const auto fallback_long = rejected.get_mul_mat_vec_pipeline(context, false, false);
            require(decisions(fallback_long).wg_size == 256 &&
                        fallback_long.pipeline.Get()->code == old_long.pipeline.Get()->code && pushes == push_long &&
                        waits == wait_long,
                    "Q4 long K rejection must rebuild canonical without retry");
            for (int tensor = 0; tensor < 3; ++tensor)
                for (int axis = 2; axis < 4; ++axis) {
                    auto & value = (tensor == 0 ? f.a : tensor == 1 ? f.x : f.dst).ne[axis];
                    value = 2;
                    require(candidate::ggml_webgpu_q4_requested_subgroup_size(context, false, false) == 0,
                            "Q4 long requested rejects each batch axis");
                    value = 1;
                }
            f.a.ne[0] = f.x.ne[0] = 2048;
            f.a.ne[1] = f.dst.ne[0] = 4097;
            for (const auto range : {std::pair{0u, 32u}, {16u, 16u}, {32u, 64u}, {4u, 8u}, {32u, 8u}}) {
                auto c = f.context<candidate::ggml_webgpu_shader_lib_context>(range.first, range.second, true);
                require(candidate::ggml_webgpu_q4_requested_subgroup_size(c, false, false) == 0,
                        "Q4 requested range widened");
            }
            for (unsigned which = 0; which < 10; ++which) {
                auto bad = fixture(type);
                auto c = bad.context<candidate::ggml_webgpu_shader_lib_context>(8, 32, true);
                bool overlap = false, mmvq = false;
                switch (which) {
                case 0:
                    bad.a.ne[0] = bad.x.ne[0] = 1023;
                    break;
                case 1:
                    bad.a.ne[0] = bad.x.ne[0] = 4097;
                    break;
                case 2:
                    bad.a.ne[1] = bad.dst.ne[0] = 4095;
                    break;
                case 3:
                    bad.x.ne[1] = bad.dst.ne[1] = 2;
                    break;
                case 4:
                    bad.x.type = GGML_TYPE_F16;
                    break;
                case 5:
                    bad.dst.type = GGML_TYPE_F16;
                    break;
                case 6:
                    c.src2 = &bad.ids;
                    break;
                case 7:
                    c.max_wg_size = 64;
                    break;
                case 8:
                    overlap = true;
                    break;
                case 9:
                    mmvq = true;
                    break;
                }
                require(candidate::ggml_webgpu_q4_requested_subgroup_size(c, mmvq, overlap) == 0,
                        "Q4 ordinary admission widened");
            }
        }
        require(keys.size() == 2, "Q4 dtype/requested cache collision");
        auto q8 = Fixture{};
        const auto fixed_old = original.get_mul_mat_vec_pipeline(
            q8.context<baseline::ggml_webgpu_shader_lib_context>(16, 16, false), false, false);
        const auto fixed_new = patched.get_mul_mat_vec_pipeline(
            q8.context<candidate::ggml_webgpu_shader_lib_context>(16, 16, true), false, false);
        require(fixed_old.pipeline.Get()->code == fixed_new.pipeline.Get()->code && fixed_old.name == fixed_new.name,
                "Composition changed fixed Q8");
        // New dtype admission uses the same complete factories, typed cache and
        // canonical callback as existing Q4/Q8; no duplicate selector model.
        {
            auto f = fixture(GGML_TYPE_Q6_K);
            f.a.ne[1] = f.dst.ne[0] = 2048;
            for (unsigned sg : {16u, 32u}) {
                auto c = f.context<candidate::ggml_webgpu_shader_lib_context>(sg, sg, true);
                const auto p = patched.get_mul_mat_vec_pipeline(c, false, false);
                require(decisions(p).wg_size == 128 && decisions(p).logical_subgroup_size == sg &&
                            decisions(p).outputs_per_wg == 4 * (128 / sg),
                        "Q6 fixed geometry");
                require(patched.get_mul_mat_vec_pipeline(c, false, false).pipeline.Get() == p.pipeline.Get(),
                        "Q6 warm identity");
                write(output / ("q6_K-fixed-sg" + std::to_string(sg) + ".wgsl"), p.pipeline.Get()->code);
            }
            auto c = f.context<candidate::ggml_webgpu_shader_lib_context>(8, 32, true);
            const auto requested = patched.get_mul_mat_vec_pipeline(c, false, false);
            const auto fixed = patched.get_mul_mat_vec_pipeline(
                f.context<candidate::ggml_webgpu_shader_lib_context>(16, 16, true), false, false);
            require(without_attribute(requested.pipeline.Get()->code) == fixed.pipeline.Get()->code,
                    "Q6 requested changed arithmetic");
            write(output / "q6_K-requested.wgsl", requested.pipeline.Get()->code);
            const auto requested_key = candidate::ggml_webgpu_shader_lib::make_mul_mat_vec_key(c, false, false);
            for (unsigned sg : {16u, 32u}) {
                f.a.ne[0] = f.x.ne[0] = 2048;
                auto fixed_context = f.context<candidate::ggml_webgpu_shader_lib_context>(sg, sg, true);
                const auto previous = patched.get_mul_mat_vec_pipeline(fixed_context, false, false);
                for (int64_t k : {1536, 1792}) {
                    f.a.ne[0] = f.x.ne[0] = k;
                    const auto smaller = patched.get_mul_mat_vec_pipeline(fixed_context, false, false);
                    require(smaller.pipeline.Get() == previous.pipeline.Get() &&
                                decisions(smaller).logical_subgroup_size == sg &&
                                production_dispatch(smaller, &f.dst, 65535) ==
                                    std::array<uint32_t, 2>{2048u / (4u * (128u / sg)), 1},
                            "Q6 lower-K fixed source/cache/grid reuse");
                    require(
                        patched.get_mul_mat_vec_pipeline(c, false, false).pipeline.Get() == requested.pipeline.Get() &&
                            candidate::ggml_webgpu_shader_lib::make_mul_mat_vec_key(c, false, false) == requested_key,
                        "Q6 lower-K requested source/cache reuse");
                }
            }
            f.a.ne[0] = f.x.ne[0] = 2048;
            candidate::ggml_webgpu_shader_lib rejected(device, instance, state);
            inject = WGPUErrorType_Validation;
            const auto first = rejected.get_mul_mat_vec_pipeline(c, false, false);
            require(decisions(first).wg_size == 128 && decisions(first).outputs_per_wg == 4 &&
                        decisions(first).logical_subgroup_size == 0,
                    "Q6 rejected request must preserve short-K cooperative geometry");
            const auto old = original.get_mul_mat_vec_pipeline(
                f.context<baseline::ggml_webgpu_shader_lib_context>(8, 32, false), false, false);
            require(first.pipeline.Get()->code == old.pipeline.Get()->code && first.name == old.name,
                    "Q6 short fallback bytes");
            f.a.ne[0] = f.x.ne[0] = 2560;
            require(candidate::ggml_webgpu_shader_lib::make_mul_mat_vec_key(c, false, false) == requested_key,
                    "Q6 requested shape-independent key");
            const auto saved_push = pushes, saved_wait = waits;
            const auto longer = rejected.get_mul_mat_vec_pipeline(c, false, false);
            require(decisions(longer).wg_size == 256 && decisions(longer).outputs_per_wg == 4 && pushes == saved_push &&
                        waits == saved_wait,
                    "Rejected Q6 key must rebuild canonical context without retry");
            const auto old_long = original.get_mul_mat_vec_pipeline(
                f.context<baseline::ggml_webgpu_shader_lib_context>(8, 32, false), false, false);
            require(longer.pipeline.Get()->code == old_long.pipeline.Get()->code, "Q6 long fallback bytes");
            f.a.ne[0] = f.x.ne[0] = 1536;
            require(rejected.get_mul_mat_vec_pipeline(c, false, false).pipeline.Get() == first.pipeline.Get() &&
                        pushes == saved_push && waits == saved_wait,
                    "Q6 lower-K failed request must reuse canonical short pipeline without retry");
            f.a.ne[0] = f.x.ne[0] = 2048;
            require(rejected.get_mul_mat_vec_pipeline(c, false, false).pipeline.Get() == first.pipeline.Get(),
                    "Q6 fallback restore");
            require(production_dispatch(requested, &f.dst, 65535) == std::array<uint32_t, 2>{64, 1} &&
                        production_dispatch(first, &f.dst, 65535) == std::array<uint32_t, 2>{512, 1},
                    "Q6 actual host grid");
            for (int64_t k : {1280, 1535, 1536, 1537, 1792, 2047, 2048, 2049, 2560, 10751, 10752, 11008})
                for (int64_t m : {2047, 2048, 2049, 65536}) {
                    f.a.ne[0] = f.x.ne[0] = k;
                    f.a.ne[1] = f.dst.ne[0] = m;
                    const bool yes = (k >= 1536 && k <= 10752 && k % 256 == 0 && m >= 2048);
                    require(candidate::ggml_webgpu_q6_matvec_subgroup_eligible(c, false, false) == yes,
                            "Q6 K/M boundary");
                    require(candidate::ggml_webgpu_q6_requested_subgroup_size(c, false, false) == (yes ? 16u : 0u),
                            "Q6 shared requested eligibility");
                }
            for (unsigned which = 0; which < 20; ++which) {
                auto bad = fixture(GGML_TYPE_Q6_K);
                auto b = bad.context<candidate::ggml_webgpu_shader_lib_context>(16, 16, true);
                bool mmvq = false, overlap = false;
                switch (which) {
                case 0:
                    bad.a.type = GGML_TYPE_Q4_K;
                    break;
                case 1:
                    bad.x.type = GGML_TYPE_F16;
                    break;
                case 2:
                    bad.dst.type = GGML_TYPE_F16;
                    break;
                case 3:
                    b.src2 = &bad.ids;
                    break;
                case 4:
                    bad.x.ne[1] = bad.dst.ne[1] = 2;
                    break;
                case 5:
                    b.max_wg_size = 127;
                    break;
                case 6:
                    mmvq = true;
                    break;
                case 7:
                    overlap = true;
                    break;
                case 8:
                    bad.a.ne[2] = 0;
                    break;
                case 9:
                    bad.a.ne[3] = -1;
                    break;
                case 10:
                    bad.x.ne[2] = bad.dst.ne[2] = 0;
                    break;
                case 11:
                    bad.x.ne[3] = bad.dst.ne[3] = int64_t(UINT32_MAX) + 1;
                    break;
                case 12:
                    bad.a.ne[2] = 3;
                    bad.x.ne[2] = bad.dst.ne[2] = 4;
                    break;
                case 13:
                    bad.x.ne[3] = 2;
                    break;
                case 14:
                    bad.x.ne[2] = bad.dst.ne[2] = UINT32_MAX;
                    break;
                case 15:
                    bad.a.ne[1] = bad.dst.ne[0] = int64_t(UINT32_MAX) - 126;
                    break;
                case 16:
                    b.src0 = nullptr;
                    break;
                case 17:
                    b.src1 = nullptr;
                    break;
                case 18:
                    b.dst = nullptr;
                    break;
                case 19:
                    b.matvec_epilogue = decltype(b.matvec_epilogue)(1);
                    break;
                }
                require(!candidate::ggml_webgpu_q6_matvec_subgroup_eligible(b, mmvq, overlap), "Q6 exclusion");
            }
            f = fixture(GGML_TYPE_Q6_K);
            c = f.context<candidate::ggml_webgpu_shader_lib_context>(8, 32, true);
            for (int axis = 2; axis < 4; ++axis) {
                f.a.ne[axis] = 2;
                f.x.ne[axis] = f.dst.ne[axis] = 4;
                require(candidate::ggml_webgpu_q6_matvec_subgroup_eligible(c, false, false), "Q6 legal broadcast");
                f.a.ne[axis] = f.x.ne[axis] = f.dst.ne[axis] = 1;
            }
            for (const auto range : {std::pair{0u, 32u}, std::pair{16u, 16u}, std::pair{32u, 64u}, std::pair{4u, 8u},
                                     std::pair{32u, 8u}}) {
                auto b = f.context<candidate::ggml_webgpu_shader_lib_context>(range.first, range.second, true);
                require(candidate::ggml_webgpu_q6_requested_subgroup_size(b, false, false) == 0,
                        "Q6 invalid requested range");
            }
            for (unsigned sg : {8u, 64u}) {
                auto b = f.context<candidate::ggml_webgpu_shader_lib_context>(sg, sg, true);
                require(candidate::ggml_webgpu_q6_matvec_subgroup_size(b, false, false) == 0,
                        "Q6 unsupported fixed width");
            }
            c.subgroup_size_control_enabled = false;
            require(candidate::ggml_webgpu_q6_requested_subgroup_size(c, false, false) == 0, "Q6 feature absence");
            c.supports_subgroups = false;
            require(candidate::ggml_webgpu_q6_matvec_subgroup_size(c, false, false) == 0, "Q6 subgroups absence");
            auto q4 = fixture(GGML_TYPE_Q4_K);
            auto q8 = Fixture{};
            require(!(requested_key ==
                      candidate::ggml_webgpu_shader_lib::make_mul_mat_vec_key(
                          q4.context<candidate::ggml_webgpu_shader_lib_context>(8, 32, true), false, false)) &&
                        !(requested_key ==
                          candidate::ggml_webgpu_shader_lib::make_mul_mat_vec_key(
                              q8.context<candidate::ggml_webgpu_shader_lib_context>(8, 32, true), false, false)),
                    "Q6 dtype cache separation");
        }
        {
            auto f = fixture(GGML_TYPE_Q6_K);
            f.a.ne[0] = f.x.ne[0] = 256;
            f.a.ne[1] = f.dst.ne[0] = 512;
            f.x.ne[1] = f.dst.ne[1] = 32;
            auto c = f.context<candidate::ggml_webgpu_shader_lib_context>(16, 16, true);
            const auto selected = patched.get_mul_mat_fast_pipeline(c, false, true);
            using D = candidate::ggml_webgpu_mul_mat_shader_decisions;
            const auto & d = *static_cast<const D *>(selected.context.get());
            require(d.tile_m == 8 && d.tile_n == 4 && d.tile_k == 32 && d.wg_size == 64 &&
                        selected.name.find("_half_pairs_m8") != std::string::npos,
                    "Q6 C8 final current factory at grid8");
            write(output / "q6_K-c8.wgsl", selected.pipeline.Get()->code);
            f.a.ne[1] = f.dst.ne[0] = 448;
            const auto scalar = patched.get_mul_mat_fast_pipeline(c, false, true);
            require(static_cast<const D *>(scalar.context.get())->tile_m == 4 &&
                        scalar.name.find("_half_pairs") == std::string::npos,
                    "Q6 grid7 scalar fallback");
            f.a.ne[1] = f.dst.ne[0] = 512;
            require(patched.get_mul_mat_fast_pipeline(c, false, true).pipeline.Get() == selected.pipeline.Get(),
                    "Q6 C8 cache return");
        }
        require(futures.empty(), "Pending mock callbacks");
        std::cout << "{\"passed\":true,\"checks\":" << checks << ",\"created\":" << created << ",\"pushes\":" << pushes
                  << ",\"pops\":" << pops << ",\"waits\":" << waits << ",\"gpu\":false}\n";
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
