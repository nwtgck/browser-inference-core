static ggml_tensor tensor(int64_t d, int64_t h = 1, int64_t t = 1) {
    ggml_tensor r = {};
    r.type = GGML_TYPE_F32;
    const int64_t dims[] = {d, h, t, 1};
    size_t stride = 4;
    for (int i = 0; i < 4; ++i) {
        r.ne[i] = dims[i];
        r.nb[i] = stride;
        stride *= size_t(dims[i]);
    }
    return r;
}
int main() {
    MockGlobal global;
    MockLib lib;
    auto ctx = std::make_shared<MockContext>(MockContext{&global, &lib});
    unsigned cases = 0;
    for (int64_t d : {1, 63, 65, 3072})
        for (int64_t n : {1, 3, 17})
            for (uint32_t max : {7u, 127u, 65535u})
                for (uint32_t residue : {0u, 1u, 63u})
                    for (float scale : {0.25f, -0.0f, -1.125f})
                        for (bool pre : {true, false}) {
                            auto x = pre ? tensor(d, 4, n) : tensor(d, n);
                            auto y = tensor(d, 4, n);
                            auto post = tensor(4, n);
                            auto out = pre ? tensor(d, n) : tensor(d, 4, n);
                            out.op = pre ? GGML_OP_DSV4_HC_PRE : GGML_OP_DSV4_HC_POST;
                            out.src[0] = &x;
                            out.src[1] = &y;
                            out.src[2] = pre ? nullptr : &post;
                            std::memcpy(out.op_params, &scale, 4);
                            int32_t gated = 1;
                            std::memcpy(reinterpret_cast<char *>(out.op_params) + 4, &gated, 4);
                            const ggml_webgpu_hc_limits limits = {1ull << 30, 256, max, 256, 256, 8};
                            if (!ggml_webgpu_hc_supported(&out, limits))
                                continue;
                            x.view_offs = residue * 4;
                            y.view_offs = ((residue + 7) % 64) * 4;
                            post.view_offs = ((residue + 14) % 64) * 4;
                            out.view_offs = ((residue + 21) % 64) * 4;
                            global.capabilities.limits.maxComputeWorkgroupsPerDimension = max;
                            const auto result = ggml_webgpu_hc(ctx, &out);
                            assert(result.pipeline.pre == pre);
                            assert(result.params.size() == (pre ? 7u : 6u));
                            assert(result.params[0] == x.view_offs / 4);
                            assert(result.params[1] == y.view_offs / 4);
                            if (pre) {
                                assert(result.params[2] == out.view_offs / 4);
                                assert(result.params[3] == d);
                                assert(result.params[4] == n);
                                assert(result.params[5] == ggml_webgpu_u32_from_f32(scale));
                                assert(result.params[6] == 0);
                            } else {
                                assert(result.params[2] == post.view_offs / 4);
                                assert(result.params[3] == out.view_offs / 4);
                                assert(result.params[4] == d);
                                assert(result.params[5] == n);
                            }
                            assert(result.entries.size() == (pre ? 3u : 4u));
                            for (size_t i = 0; i < result.entries.size(); ++i) {
                                assert(result.entries[i].binding == i);
                                assert(result.entries[i].tensor ==
                                       (i + 1 == result.entries.size() ? &out : out.src[i]));
                            }
                            uint32_t groups = uint32_t((ggml_nelements(&out) + 63) / 64);
                            assert(result.x > 0 && result.y > 0 && result.x <= max && result.y <= max);
                            assert(uint64_t(result.x) * result.y >= groups);
                            assert(uint64_t(result.x) * result.y - groups < result.y);
                            ++cases;
                        }
    std::cout << cases << " actual dispatcher cases passed (mock WGPU)\n";
}
