namespace {
size_t grids = 0, builders = 0, guards = 0, rejected = 0;
void check(bool ok, const char * what) {
    if (!ok)
        throw std::runtime_error(what);
}
using ShaderGuard = bool (*)(uint32_t, uint32_t, U3, U3, U3, uint32_t &);
void verify_mapping(uint32_t count, uint32_t wg, uint32_t x, uint32_t y, ShaderGuard shader, bool rows) {
    const uint64_t logical = rows ? count : 1 + (uint64_t(count) - 1) / wg;
    const uint64_t padded = uint64_t(x) * y;
    std::vector<uint64_t> samples{0, logical - 1, padded - 1};
    for (uint64_t i = 0; i < std::min(padded, uint64_t(19)); ++i)
        samples.push_back(i);
    if (logical < padded)
        samples.push_back(logical);
    if (padded <= 200000) {
        samples.clear();
        for (uint64_t i = 0; i < padded; ++i)
            samples.push_back(i);
    }
    for (uint64_t group : samples)
        for (uint32_t lane : {0u, wg - 1}) {
            uint32_t index = 0xdeadbeef;
            const uint64_t expected = rows ? group : group * wg + lane;
            bool active =
                shader(count, wg, U3{uint32_t(group % x), uint32_t(group / x), 0}, U3{x, y, 1}, U3{lane, 0, 0}, index);
            check(active == (expected < count), "padding guard or coverage");
            if (active)
                check(uint64_t(index) == expected, "flattened coordinate or overflow");
            else
                check(index == 0xdeadbeef, "padding exposed an index");
            ++guards;
        }
}
void grid(uint32_t logical, uint32_t limit) {
    uint32_t x = 0, y = 0;
    ggml_webgpu_linear_workgroups(logical, limit, x, y);
    check(x > 0 && y > 0 && x <= limit && y <= limit, "axis exceeds device limit");
    check(uint64_t(x) * y >= logical && uint64_t(x) * y <= UINT32_MAX, "invalid u32 grid domain");
    if (logical <= std::min(limit, 65535u))
        check(x == logical && y == 1, "small dispatch changed shape");
    for (auto shader : {shader_row_norm, shader_rms_norm_mul, shader_soft_max})
        verify_mapping(logical, 1, x, y, shader, true);
    ++grids;
}
ggml_tensor tensor(ggml_type type, uint32_t cols, uint32_t rows, uint32_t heads = 1, uint32_t batch = 1) {
    ggml_tensor t;
    t.type = type;
    t.ne[0] = cols;
    t.ne[1] = rows;
    t.ne[2] = heads;
    t.ne[3] = batch;
    t.nb[0] = ggml_type_size(type);
    for (int i = 1; i < 4; ++i)
        t.nb[i] = t.nb[i - 1] * t.ne[i - 1];
    float eps = 1e-5f;
    std::memcpy(&t.op_params[0], &eps, 4);
    return t;
}
void check_packet(const webgpu_encoded_op & op, uint32_t count, uint32_t cap, size_t words, bool rows, uint32_t wg,
                  ShaderGuard guard) {
    check(op.params.size() == words && 4 * words <= 128, "WGSL/host scalar record layout mismatch");
    if (rows)
        check(op.params.back() == count, "logical row count uniform mismatch");
    uint32_t x, y;
    ggml_webgpu_linear_workgroups(rows ? count : 1 + (uint64_t(count) - 1) / wg, cap, x, y);
    check(op.x == x && op.y == y, "builder discarded grid Y");
    verify_mapping(count, wg, x, y, guard, rows);
    ++builders;
}
void packets() {
    auto ctx = std::make_shared<Context>();
    for (uint32_t rows : {1u, 9u, 257u, 65535u, 65536u, 65537u, 131360u})
        for (uint32_t wg : {64u, 256u, 1024u})
            for (bool inplace : {false, true})
                for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
                    ctx->shader_lib->inplace = inplace;
                    ctx->shader_lib->overlap = false;
                    ctx->shader_lib->src_overlap = false;
                    ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup = wg;
                    auto src = tensor(type, 4, rows), dst = src, mask = tensor(type, 4, rows),
                         sink = tensor(GGML_TYPE_F32, 1, 1);
                    auto out = inplace ? &src : &dst;
                    check_packet(ggml_webgpu_row_norm(ctx, &src, out), rows, 65535, row_norm_uniform_words, true, wg,
                                 shader_row_norm);
                    if (type == GGML_TYPE_F32)
                        for (bool has_mask : {false, true})
                            for (bool has_sink : {false, true})
                                check_packet(ggml_webgpu_soft_max(ctx, &src, has_mask ? &mask : nullptr,
                                                                  has_sink ? &sink : nullptr, out),
                                             rows, 65535, soft_max_uniform_words, true, wg, shader_soft_max);
                }
    // Fused normalization covers all binding decision paths, not just out-of-place.
    for (int mode = 0; mode < 4; ++mode)
        for (uint32_t rows : {7u, 65537u})
            for (bool swapped : {false, true}) {
                ctx->shader_lib->inplace = mode == 0;
                ctx->shader_lib->overlap = mode == 1;
                ctx->shader_lib->src_overlap = mode == 2;
                auto src = tensor(GGML_TYPE_F32, 8, rows), rn_dst = src, mul = tensor(GGML_TYPE_F32, 8, 1), dst = src;
                auto r = ggml_webgpu_rms_norm_mul(ctx, &src, &rn_dst, swapped ? &mul : &rn_dst,
                                                  swapped ? &rn_dst : &mul, &dst);
                check(r.has_value(), "fused builder absent");
                check_packet(*r, rows, 65535, rms_norm_mul_uniform_words, true, 1024, shader_rms_norm_mul);
            }
    // Large GLU domains are checked without allocating hundreds of MB. The
    // requested-backend probe separately checks arithmetic on small real tensors.
    for (uint32_t wg : {1u, 32u, 256u, 1024u})
        for (uint32_t ne : {1u, 17u, 65536u, 16777472u, 268435455u, UINT32_MAX})
            for (int mode = 0; mode < 3; ++mode) {
                uint64_t groups = 1 + (uint64_t(ne) - 1) / wg;
                if (groups > uint64_t(65535) * 65535)
                    continue;
                ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup = wg;
                ctx->shader_lib->src_overlap = mode == 1;
                auto src = tensor(GGML_TYPE_F32, ne, 1), src1 = src, dst = src;
                if (mode == 0) {
                    src.ne[0] *= 2;
                    for (int i = 1; i < 4; ++i)
                        src.nb[i] *= 2;
                }
                check_packet(ggml_webgpu_glu(ctx, &src, mode == 0 ? nullptr : &src1, &dst), ne, 65535,
                             glu_uniform_words, false, wg, shader_glu);
            }
}
} // namespace
int main() {
    try {
        for (uint32_t cap : {1u, 2u, 3u, 7u, 31u, 255u, 65535u, 131072u, UINT32_MAX}) {
            uint64_t lim = std::min(cap, 65535u), max = lim * lim;
            std::vector<uint64_t> ns{1, lim, max};
            for (uint64_t n : {lim - 1, lim + 1, 2 * lim + 1, 65536ul, 65537ul, 131360ul, 2147483647ul, 4294836225ul})
                if (n > 0 && n <= max)
                    ns.push_back(n);
            for (auto n : ns)
                grid(uint32_t(n), cap);
        }
        // Deterministic large-domain probes catch overflow that a 65,536-row smoke
        // would miss. Sample every padding edge and the first/last real groups.
        uint32_t rng = 0xa571923u;
        for (int i = 0; i < 2000; ++i) {
            rng = rng * 1664525u + 1013904223u;
            uint32_t n = 1 + (uint64_t(rng) % 4294836225ull);
            grid(n, 65535);
        }
        for (auto bad : std::vector<std::pair<uint64_t, uint32_t>>{
                 {0, 65535}, {1, 0}, {50, 7}, {4294836226ull, 65535}, {UINT64_MAX, UINT32_MAX}}) {
            bool caught = false;
            try {
                uint32_t x, y;
                ggml_webgpu_linear_workgroups(bad.first, bad.second, x, y);
            } catch (const std::runtime_error &) {
                caught = true;
            }
            check(caught, "invalid grid was accepted");
            ++rejected;
        }
        packets();
        std::cout << "{\"grid_cases\":" << grids << ",\"builder_packets\":" << builders
                  << ",\"guard_invocations\":" << guards << ",\"invalid_domains\":" << rejected
                  << ",\"gpuExecuted\":false}\n";
    } catch (const std::exception & e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
