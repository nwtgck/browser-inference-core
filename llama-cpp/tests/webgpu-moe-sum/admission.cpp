#include "ggml-webgpu-moe-sum.hpp"
#include <iostream>
#include <stdexcept>
#include <vector>
#include <memory>
#include "ggml-alloc.h"
#include "ggml-backend-impl.h"
#include <cstdlib>
static int checks = 0;
static void check(bool value) { ++checks; if (!value) { throw std::runtime_error("admission failure at " + std::to_string(checks)); } }
struct FreeContext { void operator()(ggml_context * p) const { ggml_free(p); } };
struct Fixture {
    std::unique_ptr<ggml_context, FreeContext> context;
    ggml_cgraph * graph;
    ggml_tensor * root;
    std::vector<ggml_tensor *> views, adds;
    int start;
    Fixture(int width, int experts, int tokens, bool produced = false): context(ggml_init({ 1 << 20, nullptr, true })) {
        if (!context) { throw std::runtime_error("ggml init"); }
        graph = ggml_new_graph_custom(context.get(), 1024, false);
        root = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, width, experts, tokens);
        ggml_set_input(root);
        if (produced) { root = ggml_scale(context.get(), root, 1.25f); }
        for (int i=0; i<experts; ++i) {
            views.push_back(ggml_view_2d(context.get(), root, width, tokens, root->nb[2], i*root->nb[1]));
            ggml_build_forward_expand(graph, views.back());
        }
        start = graph->n_nodes;
        ggml_tensor * acc = views[0];
        for (int i=1; i<experts; ++i) {
            acc = ggml_add(context.get(), acc, views[i]); adds.push_back(acc);
            ggml_build_forward_expand(graph, acc);
        }
        ggml_set_output(acc);
    }
};

static void free_memory(ggml_backend_buffer_t b) { std::free(b->context); }
static void * base_memory(ggml_backend_buffer_t b) { return b->context; }
static ggml_backend_buffer_t alloc_memory(ggml_backend_buffer_type_t t, size_t size) {
    ggml_backend_buffer_i iface{}; iface.free_buffer=free_memory; iface.get_base=base_memory;
    void * p = std::aligned_alloc(256, ((size+255)/256)*256);
    if (!p) throw std::bad_alloc();
    return ggml_backend_buffer_init(t, iface, p, size);
}
static const char * memory_name(ggml_backend_buffer_type_t) { return "admission-test-only"; }
static size_t memory_alignment(ggml_backend_buffer_type_t) { return 256; }
static ggml_backend_buffer_type memory_type = []{
    ggml_backend_buffer_type t{}; t.iface.get_name=memory_name;
    t.iface.alloc_buffer=alloc_memory; t.iface.get_alignment=memory_alignment; return t;
}();
struct FreeAllocator { void operator()(ggml_gallocr_t a) const { ggml_gallocr_free(a); } };

int main() {
    for (int w: {4, 8, 128, 2560, 4096}) for (int e: {3, 4, 8, 10, 16, 32}) for (int t: {1, 2, 31}) {
        Fixture f(w,e,t); moe_sum_plan p;
        check(plan_moe_sum(f.graph,f.start,p)); check(p.operations == e-1); check(p.width_vec4 == uint32_t(w/4));
        check(p.source == f.root && p.output == f.adds.back());
        for (int j=0; j<e-2; ++j) {
            auto flags=f.adds[j]->flags;ggml_set_output(f.adds[j]);check(!plan_moe_sum(f.graph,f.start,p));f.adds[j]->flags=flags;
        }
        auto ptr=f.adds[1]->src[0];f.adds[1]->src[0]=f.root;check(!plan_moe_sum(f.graph,f.start,p));f.adds[1]->src[0]=ptr;
        f.adds[0]->src[2]=f.root;check(!plan_moe_sum(f.graph,f.start,p));f.adds[0]->src[2]=nullptr;
        auto offset=f.views[1]->view_offs;f.views[1]->view_offs=0;check(!plan_moe_sum(f.graph,f.start,p));f.views[1]->view_offs=offset;
        auto count=f.graph->n_nodes;f.graph->n_nodes--;check(!plan_moe_sum(f.graph,f.start,p));f.graph->n_nodes=count;
        check(!plan_moe_sum(f.graph,-1,p));check(!plan_moe_sum(f.graph,count,p));check(plan_moe_sum(f.graph,f.start,p));
        auto * side=ggml_scale(f.context.get(), f.adds[0], 2.0f);
        ggml_build_forward_expand(f.graph, side);check(!plan_moe_sum(f.graph,f.start,p));
    }
    for (int e: {2,33}) { Fixture f(2560,e,1);moe_sum_plan p;check(!plan_moe_sum(f.graph,f.start,p)); }
    for (int w: {1,3,257}) { Fixture f(w,10,1);moe_sum_plan p;check(!plan_moe_sum(f.graph,f.start,p)); }
    check(moe_sum_disjoint(0,256,256,256));check(!moe_sum_disjoint(0,257,256,256));
    check(!moe_sum_disjoint(256,256,0,257));check(!moe_sum_disjoint(0,1,0,1));
    check(moe_sum_disjoint(UINT64_MAX-100,100,0,128));check(!moe_sum_disjoint(0,0,1,1));
    for (bool produced : {false, true}) for (int width : {4, 12, 128, 2560, 4096}) for (int experts : {3, 4, 10, 32}) for (int tokens : {1, 3, 31}) {
        Fixture f(width,experts,tokens,produced); moe_sum_plan p;
        std::unique_ptr<ggml_gallocr, FreeAllocator> allocator(ggml_gallocr_new(&memory_type));
        check(ggml_gallocr_alloc_graph(allocator.get(), f.graph));
        check(plan_moe_sum(f.graph, f.start, p));
        check(p.source->data && p.output->data && p.source->buffer && p.output->buffer);
        const auto a = reinterpret_cast<uintptr_t>(p.source->data);
        const auto z = reinterpret_cast<uintptr_t>(p.output->data);
        check(moe_sum_disjoint(a, ggml_nbytes(p.source), z, ggml_nbytes(p.output)));
        check(p.source->nb[2] == uint64_t(width)*experts*4);
    }
    for (uint32_t limit : {1u, 2u, 3u, 7u, 65535u}) for (uint64_t count : {1ull, 128ull, 129ull, 640ull, 4096ull, 0xffffffffull}) {
        uint32_t x,y; bool accepted=moe_sum_grid(count,limit,x,y);
        if (accepted) { check(x<=limit && y<=limit); check(uint64_t(x)*y*128>=count); check(uint64_t(x)*y*128<=UINT32_MAX); }
        else { check(x==0 && y==0); }
    }
    uint32_t x,y; check(!moe_sum_grid(0,65535,x,y)); check(!moe_sum_grid(1,0,x,y));
    std::cout << checks << " actual ggml graph admission checks passed\n";
}
