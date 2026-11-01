#include <filesystem>
#include <fstream>

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
        return candidate::ggml_webgpu_q4_matvec_subgroup_eligible(context<candidate::Library::Context>(), mmvq,
                                                                  overlap);
    }
    uint32_t select(bool mmvq = false, bool overlap = false) {
        return candidate::ggml_webgpu_q4_matvec_subgroup_size(context<candidate::Library::Context>(), mmvq, overlap);
    }
};

static baseline::Library old_library;
static candidate::Library new_library;

static void factory(Fixture & f, bool selected, bool overlap, const std::filesystem::path & output) {
    const auto old = old_library.get_mul_mat_vec_pipeline(f.context<baseline::Library::Context>(), overlap, false);
    const auto current = new_library.get_mul_mat_vec_pipeline(f.context<candidate::Library::Context>(), overlap, false);
    const auto repeated =
        new_library.get_mul_mat_vec_pipeline(f.context<candidate::Library::Context>(), overlap, false);
    require(current.id == repeated.id, "Repeated effective key uses the same pipeline object");
    const auto before = std::static_pointer_cast<baseline::Library::Decisions>(old.context);
    const auto after = std::static_pointer_cast<candidate::Library::Decisions>(current.context);
    require(before && after, "Actual factory decisions");
    require(after->src_overlap == before->src_overlap && after->vec_size == before->vec_size,
            "Binding/vector decisions unchanged");
    if (selected) {
        require(after->wg_size == 128 && after->outputs_per_wg == 4 * (128 / f.min_sg),
                "Four rows per independent subgroup in WG128");
        require(current.name.find("_independent_sg" + std::to_string(f.min_sg) + "_wg128") != std::string::npos,
                "Distinct actual profile label");
        require(current.wgsl.find("workgroupBarrier(") == std::string::npos,
                "Independent main does not use workgroup barriers");
        const auto dtype = f.a.type == GGML_TYPE_Q4_0 ? "q4_0" : "q4_k";
        if (!output.empty()) {
            std::ofstream(output / (std::string(dtype) + "-sg" + std::to_string(f.min_sg) + ".wgsl")) << current.wgsl;
        }
    } else {
        require(old.wgsl == current.wgsl && old.name == current.name, "Every unsupported path retains canonical bytes");
        require(before->wg_size == after->wg_size && before->outputs_per_wg == after->outputs_per_wg,
                "Every unsupported path retains canonical decisions");
    }
    for (const uint32_t dimension : {65535u, 64u}) {
        const auto grid = production_dispatch(current, &f.dst, dimension);
        const uint64_t groups =
            (f.dst.ne[0] + after->outputs_per_wg - 1) / after->outputs_per_wg * f.dst.ne[2] * f.dst.ne[3];
        require(grid[0] <= dimension && grid[1] <= dimension && uint64_t(grid[0]) * grid[1] >= groups,
                "Actual host helper covers every output group with bounded rectangle");
    }
}

int main(int argc, char ** argv) {
    const std::filesystem::path output = argc > 1 ? argv[1] : "";
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
            for (const int64_t k : {-1, 0, 992, 1025, 4097, 4352}) {
                Fixture f(type, k);
                f.min_sg = f.max_sg = sg;
                require(!f.eligible() && f.select() == 0, "K boundary and partial-block rejection");
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
        for (const int64_t k : {768, 4352}) {
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
        auto c = f.context<candidate::Library::Context>();
        c.src0 = nullptr;
        require(!candidate::ggml_webgpu_q4_matvec_subgroup_eligible(c, false, false), "Null weights rejected");
        c = f.context<candidate::Library::Context>();
        c.src1 = nullptr;
        require(!candidate::ggml_webgpu_q4_matvec_subgroup_eligible(c, false, false), "Null input rejected");
        c = f.context<candidate::Library::Context>();
        c.dst = nullptr;
        require(!candidate::ggml_webgpu_q4_matvec_subgroup_eligible(c, false, false), "Null output rejected");
    }
    std::cout << "{\"status\":\"passed\",\"checks\":" << checks << "}\n";
    return 0;
}
