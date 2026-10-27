#include "fixture.hpp"
#include "ggml-alloc.h"
#include "ggml-backend-impl.h"
#include <cstdlib>
#include <iostream>
#include <vector>
static unsigned checks = 0;
static void check(bool value) { ++checks; if (!value) { throw std::runtime_error("admission check " + std::to_string(checks)); } }
static void release(ggml_backend_buffer_t b) { std::free(b->context); }
static void * base(ggml_backend_buffer_t b) { return b->context; }
static ggml_backend_buffer_t allocate(ggml_backend_buffer_type_t type, size_t bytes) {
    ggml_backend_buffer_i iface{}; iface.get_base = base; iface.free_buffer = release;
    void * p = std::aligned_alloc(256, (bytes + 255) / 256 * 256); if (!p) { throw std::bad_alloc(); }
    return ggml_backend_buffer_init(type, iface, p, bytes);
}
static const char * name(ggml_backend_buffer_type_t) { return "ssm-conv-silu-test"; }
static size_t alignment(ggml_backend_buffer_type_t) { return 256; }
static ggml_backend_buffer_type memory_type = [] { ggml_backend_buffer_type t{}; t.iface.get_name=name; t.iface.get_alignment=alignment; t.iface.alloc_buffer=allocate; return t; }();
int main() { try {
    unsigned reused = 0;
    for (int nc : {1, 3, 4, 5}) for (int channels : {1, 31, 32, 33, 320})
    for (int tokens : {1, 2, 7, 8, 9}) for (int sequences : {1, 3}) for (bool padded : {false, true}) {
        fixture f(nc, channels, tokens, sequences, padded); ssm_conv_silu_plan p;
        const auto ok = [&] { return plan_ssm_conv_silu(f.graph, f.index, p); };
        check(ok()); check(p.history == f.history && p.weights == f.weights && p.conv == f.conv && p.output == f.output);
        check(p.history_bytes == ggml_nbytes(f.history) && p.weights_bytes == ggml_nbytes(f.weights) && p.output_bytes == ggml_nbytes(f.output));
        check(!plan_ssm_conv_silu(nullptr, 0, p)); check(!plan_ssm_conv_silu(f.graph, -1, p));
        check(!plan_ssm_conv_silu(f.graph, f.graph->n_nodes - 1, p)); check(ok());
        const auto flags = f.conv->flags; ggml_set_output(f.conv); check(!ok()); f.conv->flags = flags;
        f.conv->flags &= ~GGML_TENSOR_FLAG_COMPUTE; check(!ok()); f.conv->flags = flags;
        const auto of = f.output->flags; f.output->flags &= ~GGML_TENSOR_FLAG_COMPUTE; check(!ok()); f.output->flags = of;
        f.conv->src[2] = f.history; check(!ok()); f.conv->src[2] = nullptr;
        f.output->src[1] = f.weights; check(!ok()); f.output->src[1] = nullptr;
        f.output->src[0] = f.history; check(!ok()); f.output->src[0] = f.conv;
        f.output->view_src = f.conv; check(!ok()); f.output->view_src = nullptr;
        const auto type = f.weights->type; f.weights->type = GGML_TYPE_F16; check(!ok()); f.weights->type = type;
        const auto nb = f.history->nb[1]; f.history->nb[1] = 1; check(!ok()); f.history->nb[1] = nb;
        f.output->nb[1] += 4; check(!ok()); f.output->nb[1] -= 4;
        f.history->ne[0]++; check(!ok()); f.history->ne[0]--;
        const auto unary = f.output->op_params[0]; f.output->op_params[0] = GGML_UNARY_OP_SIGMOID; check(!ok()); f.output->op_params[0] = unary;
        // Scheduler evaluation of an observed intermediate ends the graph view here.
        auto slice = ggml_graph_view(f.graph, 0, f.index + 1); check(!plan_ssm_conv_silu(&slice, f.index, p));
        auto * allocator = ggml_gallocr_new(&memory_type); check(ggml_gallocr_alloc_graph(allocator, f.graph)); check(ok());
        // UNARY is in-place eligible: skipping the convolution node must still write this reused allocation.
        if (f.conv->data == f.output->data) { ++reused; }
        check(ssm_conv_silu_disjoint(uintptr_t(f.history->data), ggml_nbytes(f.history), uintptr_t(f.output->data), ggml_nbytes(f.output)));
        check(ssm_conv_silu_disjoint(uintptr_t(f.weights->data), ggml_nbytes(f.weights), uintptr_t(f.output->data), ggml_nbytes(f.output)));
        ggml_gallocr_free(allocator);
        auto * extra = ggml_scale(f.context, f.conv, 2); ggml_build_forward_expand(f.graph, extra); check(!ok());
    }
    check(reused > 0);
    // Views/observer uses of the intermediate must prevent removal, even when no explicit output flag is set.
    { fixture f(4, 33, 1, 1); auto * v=ggml_view_1d(f.context,f.conv,33,0); ggml_set_output(v); ggml_build_forward_expand(f.graph,v); ssm_conv_silu_plan p; check(!plan_ssm_conv_silu(f.graph,f.index,p)); }
    { fixture f(4, 33, 1, 1); auto * v=ggml_view_1d(f.context,f.output,33,0); ggml_set_output(v); ggml_build_forward_expand(f.graph,v); ssm_conv_silu_plan p; check(plan_ssm_conv_silu(f.graph,f.index,p)); }
    // Checked span/narrowing without allocating huge tensors.
    { ggml_tensor t{}; t.type=GGML_TYPE_F32; for (int i=0;i<4;i++) { t.ne[i]=1; t.nb[i]=4; } uint64_t bytes;
      check(ssm_conv_silu_layout(&t,false,bytes) && bytes==4); t.ne[0]=INT64_MAX; check(!ssm_conv_silu_layout(&t,false,bytes));
      t.ne[0]=1; t.nb[1]=UINT64_MAX-3; check(!ssm_conv_silu_layout(&t,false,bytes));
      t.nb[1]=4; t.ne[1]=UINT32_MAX; check(!ssm_conv_silu_layout(&t,false,bytes)); }
    std::cout << checks << " source-bound graph/layout/allocator checks; reused=" << reused << '\n';
} catch(const std::exception & e) { std::cerr << e.what() << '\n'; return 1; } }
