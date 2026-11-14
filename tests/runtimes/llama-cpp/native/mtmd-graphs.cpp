// Pinned upstream model graph builders + synthetic metadata. Checks the real
// retained-role/graph contracts on CPU-backed ggml storage; not model inference,
// image preprocessing, WebGPU placement or performance acceptance.
#include "mtmd.h"
#include "mtmd-impl.h"
#include "mtmd-model.h"
#include "mtmd-graph.h"
#include "models/models.h"
#include <algorithm>
#include "support/check.hpp"
#include <cstring>
#include <iostream>
#include <set>
#include <unordered_map>
#include <functional>
#include <limits>
// Use the real private class definition across every TU. This fixture-only
// constructor supplies metadata without model I/O, backends or scheduling.
mtmd_backends::mtmd_backends(ggml_backend_dev_t, int, int, ggml_backend_sched_eval_callback, void *) {}
mtmd_backends::~mtmd_backends() = default;
mtmd_subcontext::mtmd_subcontext(const mtmd_model & model, mtmd_backends & shared, const mtmd_subctx_params & params)
    : model(model), backends(shared), max_nodes(model.max_nodes), flash_attn_type(params.flash_attn_type) {}
#include "mtmd-retained-graph.h"
#include "mtmd-retained-roles.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
static ggml_tensor * root(ggml_tensor * p) {
    while (p && p->view_src)
        p = p->view_src;
    return p;
}
static void probe(std::string kind, int side, int height) {
    const bool unified = kind == "gemma4uv" || kind == "gemma4uv-12b";
    const bool real_shape = kind == "gemma4uv-12b";
    if (real_shape) {
        // Reviewed fixture dimensions, not a claim about every upstream image shape.
        if (side < 48 || height < 48 || side % 48 || height % 48 || side > 1680 || height > 1680 ||
            int64_t(side) * height > 2580480)
            throw std::runtime_error("unreviewed Gemma4UV fixture dimensions");
    } else if ((side != 32 && side != 48) || height != side)
        throw std::runtime_error("unreviewed fixture image side");
    mtmd_model model{};
    mtmd_subctx_params params{};
    params.flash_attn_type = MTMD_FLASH_ATTN_TYPE_DISABLED;
    mtmd_backends shared(nullptr, 1, model.max_nodes, nullptr, nullptr);
    mtmd_subcontext ctx(model, shared, params);
    ctx.buf_compute_meta.resize(ggml_tensor_overhead() * ctx.max_nodes +
                                ggml_graph_overhead_custom(ctx.max_nodes, false));
    auto & h = model.hparams;
    const int e = real_shape ? 3840 : 32, ff = unified ? 0 : 64, p = real_shape ? 48 : 8;
    const int proj = real_shape ? 3840 : 64, l = unified ? 0 : 2, np = (side / p) * (height / p);
    h.image_size = real_shape ? 224 : kind == "gemma3" ? side : 32;
    h.patch_size = p;
    h.n_embd = e;
    h.n_ff = ff;
    h.n_head = unified ? 0 : 4;
    h.n_head_kv = unified ? 0 : 4;
    h.n_layer = l;
    h.projection_dim = proj;
    h.n_merge = unified ? 1 : 2;
    h.eps = 1e-6;
    h.ffn_op = FFN_GELU;
    h.rope_theta = 10000;
    ggml_init_params ip = {};
    ip.mem_size = 1024 * 1024;
    ip.no_alloc = true;
    ggml_context_ptr weights(ggml_init(ip));
    if (!weights)
        throw std::runtime_error("weight metadata allocation failed");
    std::vector<ggml_tensor *> wt;
    auto tensor = [&](std::string name, ggml_type type, std::initializer_list<int64_t> dims) {
        std::vector<int64_t> shape(dims);
        auto * t = ggml_new_tensor(weights.get(), type, shape.size(), shape.data());
        ggml_set_name(t, name.c_str());
        wt.push_back(t);
        return t;
    };
    auto mat = [&](std::string n, int64_t a, int64_t b) { return tensor(n, GGML_TYPE_BF16, {a, b}); };
    auto vec = [&](std::string n, int64_t a) { return tensor(n, GGML_TYPE_F32, {a}); };
    if (unified)
        model.patch_embeddings_0 = tensor("v.patch_embd.weight", GGML_TYPE_F32, {p * p * 3, e});
    else
        model.patch_embeddings_0 =
            tensor("v.patch_embd.weight", kind == "lfm3vl" ? GGML_TYPE_F16 : GGML_TYPE_F32, {p, p, 3, e});
    if (unified)
        model.position_embeddings = tensor("v.position_embd.weight", GGML_TYPE_F32, {e, real_shape ? 1120 : 16, 2});
    else if (kind == "gemma4v")
        model.position_embeddings = tensor("v.gemma4_position.weight", GGML_TYPE_F32, {e, 16, 2});
    else
        model.position_embeddings =
            tensor("v.position_embd.weight", GGML_TYPE_F32, {e, kind == "llava" || kind == "gemma3" ? np : 16});
    model.layers.resize(l);
    for (int i = 0; i < l; i++) {
        auto & b = model.layers[i];
        std::string n = "v.blk." + std::to_string(i) + ".";
        b.ln_1_w = vec(n + "ln1.weight", e);
        b.ln_1_b = vec(n + "ln1.bias", e);
        b.ln_2_w = vec(n + "ln2.weight", e);
        b.ln_2_b = vec(n + "ln2.bias", e);
        if (kind == "qwen3vl") {
            b.qkv_w = mat(n + "attn_qkv.weight", e, 3 * e);
            b.qkv_b = vec(n + "attn_qkv.bias", 3 * e);
        } else {
            b.q_w = mat(n + "attn_q.weight", e, e);
            b.k_w = mat(n + "attn_k.weight", e, e);
            b.v_w = mat(n + "attn_v.weight", e, e);
            b.q_b = vec(n + "attn_q.bias", e);
            b.k_b = vec(n + "attn_k.bias", e);
            b.v_b = vec(n + "attn_v.bias", e);
        }
        b.o_w = mat(n + "attn_out.weight", e, e);
        b.o_b = vec(n + "attn_out.bias", e);
        b.ff_up_w = mat(n + "ffn_up.weight", e, ff);
        b.ff_down_w = mat(n + "ffn_down.weight", ff, e);
        b.ff_up_b = vec(n + "ffn_up.bias", ff);
        b.ff_down_b = vec(n + "ffn_down.bias", e);
    }
    std::unique_ptr<mtmd_graph> graph;
    mtmd_image_f32 image;
    image.set_size({side, height}, true, false);
    if (kind == "llava") {
        model.proj_type = PROJECTOR_TYPE_MLP;
        h.has_llava_projector = true;
        h.feature_layers = {l};
        model.mm_0_w = mat("mm.0.weight", e, proj);
        model.mm_0_b = vec("mm.0.bias", proj);
        model.mm_2_w = mat("mm.2.weight", proj, proj);
        model.mm_2_b = vec("mm.2.bias", proj);
        graph = std::make_unique<mtmd_graph_llava>(&ctx, image);
    } else if (kind == "qwen2vl" || kind == "qwen3vl") {
        model.proj_type = kind == "qwen2vl" ? PROJECTOR_TYPE_QWEN2VL : PROJECTOR_TYPE_QWEN3VL;
        model.patch_embeddings_1 = tensor("v.patch_embd.weight.1", GGML_TYPE_F32, {p, p, 3, e});
        model.mm_0_w = mat("mm.0.weight", e * 4, e * 4);
        model.mm_0_b = vec("mm.0.bias", e * 4);
        model.mm_1_w = mat("mm.1.weight", e * 4, proj);
        model.mm_1_b = vec("mm.1.bias", proj);
        if (kind == "qwen2vl")
            graph = std::make_unique<mtmd_graph_qwen2vl>(&ctx, image);
        else {
            model.patch_bias = vec("v.patch_embd.bias", e);
            graph = std::make_unique<mtmd_graph_qwen3vl>(&ctx, image);
        }
    } else if (kind == "gemma3") {
        model.proj_type = PROJECTOR_TYPE_GEMMA3;
        model.mm_soft_emb_norm_w = vec("mm.soft_emb_norm.weight", e);
        model.mm_input_proj_w = tensor("mm.input_projection.weight", GGML_TYPE_F16, {proj, e});
        graph = std::make_unique<mtmd_graph_siglip>(&ctx, image);
    } else if (unified) {
        model.proj_type = PROJECTOR_TYPE_GEMMA4UV;
        model.patch_bias = vec("v.patch_embd.bias", e);
        model.patch_norm_1_w = vec("v.patch_norm.1.weight", p * p * 3);
        model.patch_norm_1_b = vec("v.patch_norm.1.bias", p * p * 3);
        model.patch_norm_2_w = vec("v.patch_norm.2.weight", e);
        model.patch_norm_2_b = vec("v.patch_norm.2.bias", e);
        model.patch_norm_3_w = vec("v.patch_norm.3.weight", e);
        model.patch_norm_3_b = vec("v.patch_norm.3.bias", e);
        model.mm_input_proj_w = mat("mm.input_projection.weight", e, proj);
        graph = std::make_unique<mtmd_graph_gemma4uv>(&ctx, image);
    } else if (kind == "gemma4v") {
        model.proj_type = PROJECTOR_TYPE_GEMMA4V;
        model.mm_input_proj_w = mat("mm.input_projection.weight", e, proj);
        graph = std::make_unique<mtmd_graph_gemma4v>(&ctx, image);
    } else if (kind == "lfm3vl") {
        model.proj_type = PROJECTOR_TYPE_LFM3VL;
        model.lfm3vl_embedding_norm_w = vec("v.embedding_norm.weight", e);
        model.mm_input_norm_w = vec("mm.input_norm.weight", e);
        model.mm_input_norm_b = vec("mm.input_norm.bias", e);
        model.mm_0_w = mat("mm.0.weight", e * 4, e * 4);
        model.mm_0_b = vec("mm.0.bias", e * 4);
        model.mm_1_w = mat("mm.1.weight", e * 4, proj);
        model.mm_1_b = vec("mm.1.bias", proj);
        // One hybrid conv block and one attention block. Depthwise weights are
        // F32, matching conversion/lfm3.py's explicit conv2d_spatial rule.
        auto & b = model.layers[0];
        b.conv2d_in_proj_w = mat("v.blk.0.conv2d_in_proj.weight", e, e * 3);
        b.conv2d_spatial_w = tensor("v.blk.0.conv2d_spatial.weight", GGML_TYPE_F32, {3, 3, 1, e});
        b.conv2d_out_proj_w = mat("v.blk.0.conv2d_out_proj.weight", e, e);
        graph = std::make_unique<mtmd_graph_lfm3vl>(&ctx, image);
    } else if (kind == "lfm2") {
        model.proj_type = PROJECTOR_TYPE_LFM2;
        model.mm_1_w = mat("mm.1.weight", e * 4, proj);
        model.mm_1_b = vec("mm.1.bias", proj);
        model.mm_2_w = mat("mm.2.weight", proj, proj);
        model.mm_2_b = vec("mm.2.bias", proj);
        graph = std::make_unique<mtmd_graph_siglip>(&ctx, image);
    } else
        throw std::runtime_error("bad model kind");
    // Roles are checked before graph construction so custom BF16 position
    // tables are rejected before upstream interpolation asserts on their type.
    lcb_mtmd_check_retained_roles(model, weights.get());
    auto * g = graph->build();
    if (unified) {
        int projection_uses = 0;
        for (int i = 0; i < ggml_graph_n_nodes(g); ++i) {
            auto * node = ggml_graph_node(g, i);
            for (int j = 0; j < GGML_MAX_SRC; ++j) {
                if (root(node->src[j]) != model.mm_input_proj_w)
                    continue;
                if (node->op != GGML_OP_MUL_MAT || j != 0 || node->src[j] != model.mm_input_proj_w)
                    throw std::runtime_error("Gemma4UV projection is not a direct retained left operand");
                ++projection_uses;
            }
        }
        auto * output = ggml_graph_node(g, ggml_graph_n_nodes(g) - 1);
        if (projection_uses != 1 || output->ne[0] != proj || output->ne[1] != np || wt.size() != 10)
            throw std::runtime_error("Gemma4UV projection/output contract changed");
    }
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu)
        throw std::runtime_error("CPU fixture backend missing");
    struct release_backend {
        ggml_backend_t p;
        ~release_backend() {
            ggml_backend_free(p);
        }
    } free_backend{cpu};
    auto * buffer = ggml_backend_alloc_ctx_tensors(weights.get(), cpu);
    if (!buffer)
        throw std::runtime_error("CPU fixture weight allocation failed");
    struct release_buffer {
        ggml_backend_buffer_t p;
        ~release_buffer() {
            ggml_backend_buffer_free(p);
        }
    } free_buffer{buffer};
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    auto * sched = ggml_backend_sched_new(&cpu, nullptr, 1, 8192, false, false);
    if (!sched)
        throw std::runtime_error("CPU fixture scheduler missing");
    struct release_scheduler {
        ggml_backend_sched_t p;
        ~release_scheduler() {
            ggml_backend_sched_free(p);
        }
    } free_scheduler{sched};
    lcb_mtmd_retained_graph plan;
    plan.prepare(weights.get(), g, sched, cpu, true);
    if (plan.consumer_count() == 0)
        throw std::runtime_error("fixture has no retained BF16 consumers");
    if (unified && plan.consumer_count() != 1)
        throw std::runtime_error("Gemma4UV must retain exactly the image projection");
    if (!ggml_backend_sched_reserve(sched, g))
        throw std::runtime_error("reserve failed");
    plan.verify_sources();
    plan.prepare(weights.get(), g, sched, cpu, true);
    if (!ggml_backend_sched_alloc_graph(sched, g))
        throw std::runtime_error("allocation failed");
    plan.verify_allocated(sched);
    std::set<ggml_tensor *> used;
    for (int i = 0; i < ggml_graph_n_nodes(g); i++)
        for (auto * s : ggml_graph_node(g, i)->src)
            if (s)
                used.insert(root(s));
    size_t bytes = 0;
    for (auto * t : wt)
        if (used.count(t))
            bytes += ggml_nbytes(t);
    std::cout << "{\"kind\":\"" << kind << "\",\"policy\":\"standard-converter\",\"imageSide\":" << side
              << ",\"imageHeight\":" << height << ",\"weightsBytes\":" << bytes
              << ",\"nodeCount\":" << ggml_graph_n_nodes(g) << ",\"checkedConsumers\":" << plan.consumer_count()
              << ",\"target\":\"CPU test fixture\"}\n";
}
int main(int argc, char ** argv) {
    try {
        if (argc != 3 && argc != 4)
            throw std::runtime_error("kind image-width [image-height] required");
        const int width = std::stoi(argv[2]);
        probe(argv[1], width, argc == 4 ? std::stoi(argv[3]) : width);
    } catch (const std::exception & e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
