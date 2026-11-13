#include "ggml-webgpu-hc.hpp"
#include <cassert>
#include <iostream>
#include <limits>

static int checks = 0;
static void expect(bool value) {
    ++checks;
    if (!value) {
        std::cerr << "check " << checks << " failed\n";
        std::abort();
    }
}
static ggml_tensor tensor(int64_t a, int64_t b = 1, int64_t c = 1, int64_t d = 1) {
    ggml_tensor t = {};
    t.type = GGML_TYPE_F32;
    const int64_t dims[] = {a, b, c, d};
    size_t stride = 4;
    for (int i = 0; i < 4; ++i) {
        t.ne[i] = dims[i];
        t.nb[i] = stride;
        stride *= size_t(dims[i]);
    }
    return t;
}
static void reject_shared_roots(ggml_tensor & out, const ggml_webgpu_hc_limits & limits) {
    ggml_tensor root = tensor(65536);
    ggml_tensor * inputs[] = {&out, out.src[0], out.src[1], out.src[2]};
    const int n = out.op == GGML_OP_DSV4_HC_PRE ? 3 : 4;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            inputs[i]->view_src = inputs[j]->view_src = &root;
            // Also reject disjoint views conservatively: the binding prefixes
            // can overlap and placement is not available during the probe.
            for (size_t offset : {size_t(0), size_t(4), size_t(256)}) {
                inputs[i]->view_offs = offset;
                expect(!ggml_webgpu_hc_supported(&out, limits));
            }
            inputs[i]->view_src = inputs[j]->view_src = nullptr;
            inputs[i]->view_offs = 0;
            expect(ggml_webgpu_hc_supported(&out, limits));
        }
    }
}
static void mode(ggml_tensor & t, int32_t value) {
    std::memcpy(reinterpret_cast<char *>(t.op_params) + 4, &value, 4);
}
int main() {
    const ggml_webgpu_hc_limits limits = {1ull << 30, 256, 65535, 256, 256, 8};
    for (int64_t d : {1, 3, 63, 64, 65, 320, 3072})
        for (int64_t n : {1, 2, 17}) {
            auto x = tensor(d, 4, n), g = tensor(d, 4, n), out = tensor(d, n);
            out.op = GGML_OP_DSV4_HC_PRE;
            out.src[0] = &x;
            out.src[1] = &g;
            mode(out, 1);
            expect(ggml_webgpu_hc_supported(&out, limits));
            reject_shared_roots(out, limits);
            auto * saved_gate = out.src[1];
            out.src[1] = out.src[0];
            expect(!ggml_webgpu_hc_supported(&out, limits));
            out.src[1] = saved_gate;
            for (int i = 2; i < GGML_MAX_SRC; ++i) {
                out.src[i] = &x;
                expect(!ggml_webgpu_hc_supported(&out, limits));
                out.src[i] = nullptr;
            }
            mode(out, 0);
            expect(!ggml_webgpu_hc_supported(&out, limits));
            mode(out, 1);
            x.type = GGML_TYPE_F16;
            expect(!ggml_webgpu_hc_supported(&out, limits));
            x.type = GGML_TYPE_F32;
            x.nb[0] = 8;
            expect(!ggml_webgpu_hc_supported(&out, limits));
            x.nb[0] = 4;
            x.ne[1] = 3;
            expect(!ggml_webgpu_hc_supported(&out, limits));
            x.ne[1] = 4;
            for (int field = 0; field < 6; ++field) {
                auto bad = limits;
                switch (field) {
                case 0:
                    bad.max_binding_size = 16;
                    break;
                case 1:
                    bad.storage_alignment = 3;
                    break;
                case 2:
                    bad.max_workgroups = 0;
                    break;
                case 3:
                    bad.max_invocations = 63;
                    break;
                case 4:
                    bad.max_workgroup_x = 63;
                    break;
                case 5:
                    bad.max_storage_buffers = 2;
                    break;
                }
                expect(!ggml_webgpu_hc_supported(&out, bad));
            }
            auto block = tensor(d, n), res = tensor(d, 4, n), post = tensor(4, n), dst = tensor(d, 4, n);
            dst.op = GGML_OP_DSV4_HC_POST;
            dst.src[0] = &block;
            dst.src[1] = &res;
            dst.src[2] = &post;
            expect(ggml_webgpu_hc_supported(&dst, limits));
            reject_shared_roots(dst, limits);
            for (int i = 3; i < GGML_MAX_SRC; ++i) {
                dst.src[i] = &x;
                expect(!ggml_webgpu_hc_supported(&dst, limits));
                dst.src[i] = nullptr;
            }
            post.ne[0] = 3;
            expect(!ggml_webgpu_hc_supported(&dst, limits));
            post.ne[0] = 4;
            dst.src[2] = nullptr;
            expect(!ggml_webgpu_hc_supported(&dst, limits));
            dst.src[2] = &post;
            dst.op = GGML_OP_ADD;
            expect(!ggml_webgpu_hc_supported(&dst, limits));
        }
    expect(!ggml_webgpu_hc_supported(nullptr, limits));
    auto huge = tensor(1);
    huge.ne[0] = int64_t(UINT32_MAX) + 1;
    expect(!ggml_webgpu_hc_tensor_supported(&huge, limits));
    huge = tensor(1);
    huge.ne[0] = 0;
    expect(!ggml_webgpu_hc_tensor_supported(&huge, limits));
    huge = tensor(1);
    huge.ne[0] = -1;
    expect(!ggml_webgpu_hc_tensor_supported(&huge, limits));
    std::cout << checks << " host admission checks passed\n";
}
