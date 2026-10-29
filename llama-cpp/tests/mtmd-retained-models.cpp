// Pinned upstream model graph builders + synthetic metadata. Checks the real
// retained-role/graph contracts on CPU-backed ggml storage; not model inference,
// image preprocessing, WebGPU placement or performance acceptance.
#include "clip.h"
#include "clip-impl.h"
#include "clip-model.h"
#include "clip-graph.h"
#include "models/models.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <set>
#include <unordered_map>
#include <functional>
#include <limits>
// Minimal context supplies the exact fields used by upstream clip_graph constructors.
// It deliberately does not pretend to be a real clip_ctx/backend/scheduler.
struct clip_ctx {
    clip_model model{};
    std::vector<uint8_t> buf_compute_meta;
    int max_nodes=8192;
    clip_flash_attn_type flash_attn_type=CLIP_FLASH_ATTN_TYPE_DISABLED;
    projector_type proj_type() const {return model.proj_type;}
};
int clip_n_mmproj_embd(const clip_ctx *ctx){return ctx->model.hparams.projection_dim;}
#include "clip-graph-base.inc"
#include "mtmd-retained-graph.h"
#include "mtmd-retained-roles.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
static ggml_tensor * root(ggml_tensor *p){while(p && p->view_src)p=p->view_src;return p;}
static void probe(std::string kind,int side){
    if(side!=32&&side!=48)throw std::runtime_error("unreviewed fixture image side");
    clip_ctx ctx;ctx.buf_compute_meta.resize(ggml_tensor_overhead()*ctx.max_nodes+ggml_graph_overhead_custom(ctx.max_nodes,false));
    auto &model=ctx.model;auto &h=model.hparams;
    const int e=32,ff=64,p=8,proj=64,l=2,np=(side/p)*(side/p);
    h.image_size=kind=="gemma3"?side:32;h.patch_size=p;h.n_embd=e;h.n_ff=ff;h.n_head=4;h.n_head_kv=4;h.n_layer=l;h.projection_dim=proj;h.n_merge=2;
    h.eps=1e-6;h.ffn_op=FFN_GELU;h.rope_theta=10000;
    ggml_init_params ip={};ip.mem_size=1024*1024;ip.no_alloc=true;
    ggml_context_ptr weights(ggml_init(ip));if(!weights)throw std::runtime_error("weight metadata allocation failed");
    std::vector<ggml_tensor*> wt;
    auto tensor=[&](std::string name,ggml_type type,std::initializer_list<int64_t> dims){
        std::vector<int64_t> shape(dims);auto *t=ggml_new_tensor(weights.get(),type,shape.size(),shape.data());
        ggml_set_name(t,name.c_str());wt.push_back(t);return t;
    };
    auto mat=[&](std::string n,int64_t a,int64_t b){return tensor(n,GGML_TYPE_BF16,{a,b});};
    auto vec=[&](std::string n,int64_t a){return tensor(n,GGML_TYPE_F32,{a});};
    model.patch_embeddings_0=tensor("v.patch_embd.weight",kind=="lfm3vl"?GGML_TYPE_F16:GGML_TYPE_F32,{p,p,3,e});
    if(kind=="gemma4v")model.position_embeddings=tensor("v.gemma4_position.weight",GGML_TYPE_F32,{e,16,2});
    else model.position_embeddings=tensor("v.position_embd.weight",GGML_TYPE_F32,{e,kind=="llava"||kind=="gemma3"?np:16});
    model.layers.resize(l);
    for(int i=0;i<l;i++){
        auto &b=model.layers[i];std::string n="v.blk."+std::to_string(i)+".";
        b.ln_1_w=vec(n+"ln1.weight",e);b.ln_1_b=vec(n+"ln1.bias",e);b.ln_2_w=vec(n+"ln2.weight",e);b.ln_2_b=vec(n+"ln2.bias",e);
        if(kind=="qwen3vl"){b.qkv_w=mat(n+"attn_qkv.weight",e,3*e);b.qkv_b=vec(n+"attn_qkv.bias",3*e);}
        else {b.q_w=mat(n+"attn_q.weight",e,e);b.k_w=mat(n+"attn_k.weight",e,e);b.v_w=mat(n+"attn_v.weight",e,e);b.q_b=vec(n+"attn_q.bias",e);b.k_b=vec(n+"attn_k.bias",e);b.v_b=vec(n+"attn_v.bias",e);}
        b.o_w=mat(n+"attn_out.weight",e,e);b.o_b=vec(n+"attn_out.bias",e);
        b.ff_up_w=mat(n+"ffn_up.weight",e,ff);b.ff_down_w=mat(n+"ffn_down.weight",ff,e);b.ff_up_b=vec(n+"ffn_up.bias",ff);b.ff_down_b=vec(n+"ffn_down.bias",e);
    }
    std::unique_ptr<clip_graph> graph;
    clip_image_f32 image;image.set_size({side,side},true,false);
    if(kind=="llava"){
        model.proj_type=PROJECTOR_TYPE_MLP;h.has_llava_projector=true;h.feature_layers={l};
        model.mm_0_w=mat("mm.0.weight",e,proj);model.mm_0_b=vec("mm.0.bias",proj);model.mm_2_w=mat("mm.2.weight",proj,proj);model.mm_2_b=vec("mm.2.bias",proj);
        graph=std::make_unique<clip_graph_llava>(&ctx,image);
    }else if(kind=="qwen2vl"||kind=="qwen3vl"){
        model.proj_type=kind=="qwen2vl"?PROJECTOR_TYPE_QWEN2VL:PROJECTOR_TYPE_QWEN3VL;
        model.patch_embeddings_1=tensor("v.patch_embd.weight.1",GGML_TYPE_F32,{p,p,3,e});
        model.mm_0_w=mat("mm.0.weight",e*4,e*4);model.mm_0_b=vec("mm.0.bias",e*4);model.mm_1_w=mat("mm.1.weight",e*4,proj);model.mm_1_b=vec("mm.1.bias",proj);
        if(kind=="qwen2vl")graph=std::make_unique<clip_graph_qwen2vl>(&ctx,image);
        else{model.patch_bias=vec("v.patch_embd.bias",e);graph=std::make_unique<clip_graph_qwen3vl>(&ctx,image);}
    }else if(kind=="gemma3"){
        model.proj_type=PROJECTOR_TYPE_GEMMA3;model.mm_soft_emb_norm_w=vec("mm.soft_emb_norm.weight",e);
        model.mm_input_proj_w=tensor("mm.input_projection.weight",GGML_TYPE_F16,{proj,e});
        graph=std::make_unique<clip_graph_siglip>(&ctx,image);
    }else if(kind=="gemma4v"){
        model.proj_type=PROJECTOR_TYPE_GEMMA4V;
        model.mm_input_proj_w=mat("mm.input_projection.weight",e,proj);
        graph=std::make_unique<clip_graph_gemma4v>(&ctx,image);
    }else if(kind=="lfm3vl"){
        model.proj_type=PROJECTOR_TYPE_LFM3VL;
        model.lfm3vl_embedding_norm_w=vec("v.embedding_norm.weight",e);
        model.mm_input_norm_w=vec("mm.input_norm.weight",e);model.mm_input_norm_b=vec("mm.input_norm.bias",e);
        model.mm_0_w=mat("mm.0.weight",e*4,e*4);model.mm_0_b=vec("mm.0.bias",e*4);
        model.mm_1_w=mat("mm.1.weight",e*4,proj);model.mm_1_b=vec("mm.1.bias",proj);
        // One hybrid conv block and one attention block. Depthwise weights are
        // F32, matching conversion/lfm3.py's explicit conv2d_spatial rule.
        auto &b=model.layers[0];b.conv2d_in_proj_w=mat("v.blk.0.conv2d_in_proj.weight",e,e*3);
        b.conv2d_spatial_w=tensor("v.blk.0.conv2d_spatial.weight",GGML_TYPE_F32,{3,3,1,e});
        b.conv2d_out_proj_w=mat("v.blk.0.conv2d_out_proj.weight",e,e);
        graph=std::make_unique<clip_graph_lfm3vl>(&ctx,image);
    }else if(kind=="lfm2"){
        model.proj_type=PROJECTOR_TYPE_LFM2;model.mm_1_w=mat("mm.1.weight",e*4,proj);model.mm_1_b=vec("mm.1.bias",proj);model.mm_2_w=mat("mm.2.weight",proj,proj);model.mm_2_b=vec("mm.2.bias",proj);
        graph=std::make_unique<clip_graph_siglip>(&ctx,image);
    }else throw std::runtime_error("bad model kind");
    // Roles are checked before graph construction so custom BF16 position
    // tables are rejected before upstream interpolation asserts on their type.
    lcb_mtmd_check_retained_roles(model, weights.get());
    auto *g=graph->build();
    ggml_backend_t cpu=ggml_backend_cpu_init();
    if(!cpu)throw std::runtime_error("CPU fixture backend missing");
    struct release_backend {ggml_backend_t p;~release_backend(){ggml_backend_free(p);}} free_backend{cpu};
    auto *buffer=ggml_backend_alloc_ctx_tensors(weights.get(),cpu);
    if(!buffer)throw std::runtime_error("CPU fixture weight allocation failed");
    struct release_buffer {ggml_backend_buffer_t p;~release_buffer(){ggml_backend_buffer_free(p);}} free_buffer{buffer};
    ggml_backend_buffer_set_usage(buffer,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    auto *sched=ggml_backend_sched_new(&cpu,nullptr,1,8192,false,false);
    if(!sched)throw std::runtime_error("CPU fixture scheduler missing");
    struct release_scheduler {ggml_backend_sched_t p;~release_scheduler(){ggml_backend_sched_free(p);}} free_scheduler{sched};
    lcb_mtmd_retained_graph plan;
    plan.prepare(weights.get(),g,sched,cpu,true);
    if(plan.consumer_count()==0)throw std::runtime_error("fixture has no retained BF16 consumers");
    if(!ggml_backend_sched_reserve(sched,g))throw std::runtime_error("reserve failed");
    plan.verify_sources();
    plan.prepare(weights.get(),g,sched,cpu,true);
    if(!ggml_backend_sched_alloc_graph(sched,g))throw std::runtime_error("allocation failed");
    plan.verify_allocated(sched);
    std::set<ggml_tensor*> used;
    for(int i=0;i<ggml_graph_n_nodes(g);i++)for(auto *s:ggml_graph_node(g,i)->src)if(s)used.insert(root(s));
    size_t bytes=0;for(auto *t:wt)if(used.count(t))bytes+=ggml_nbytes(t);
    std::cout<<"{\"kind\":\""<<kind<<"\",\"policy\":\"standard-converter\",\"imageSide\":"<<side
             <<",\"weightsBytes\":"<<bytes<<",\"nodeCount\":"<<ggml_graph_n_nodes(g)
             <<",\"checkedConsumers\":"<<plan.consumer_count()<<",\"target\":\"CPU test fixture\"}\n";
}
int main(int argc,char **argv){try{if(argc!=3)throw std::runtime_error("kind image-side required");probe(argv[1],std::stoi(argv[2]));}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
