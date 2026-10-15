#include "image-performance-test-utils.h"
#include "core/ggml_extend.h"
#include "ggml-backend-impl.h"
#include "model/diffusion/qwen_image_2_1.hpp"
#include "rope-reference.h"
#include <iostream>
#include <map>
#include <unordered_set>
using namespace image_test;
namespace {
struct WebgpuName {
    ggml_backend backend;
    ggml_backend_device device;
    ggml_backend_t real;
    explicit WebgpuName(ggml_backend_t cpu) : backend(*cpu), device(*cpu->device), real(cpu) {
        device.context = this;
        device.iface.get_name = [](ggml_backend_dev_t) { return "WebGPU-rope-construction"; };
        device.iface.supports_op = [](ggml_backend_dev_t dev, const ggml_tensor* op) {
            return ggml_backend_supports_op(static_cast<WebgpuName*>(dev->context)->real,op);
        };
        backend.device = &device;
    }
};
int preparations(ggml_cgraph* graph, ggml_tensor* source) {
    int n = 0;
    for (int i=0;i<ggml_graph_n_nodes(graph);++i) {
        auto node=ggml_graph_node(graph,i);
        if(node->op==GGML_OP_CONT && node->src[0]->op==GGML_OP_PERMUTE && node->src[0]->src[0]==source) ++n;
    }
    return n;
}
void numerical(ggml_backend_t cpu,int width,int heads,int positions,int batch,bool interleaved,int consumers) {
    auto data=context(),reference=context(),actual=context();
    auto x=ggml_new_tensor_4d(data.get(),GGML_TYPE_F32,width,heads,positions,batch);
    auto pe=ggml_new_tensor_4d(data.get(),GGML_TYPE_F32,2,2,width/2,positions);
    Buffer buffer(ggml_backend_alloc_ctx_tensors(data.get(),cpu),ggml_backend_buffer_free);
    require(buffer!=nullptr,"rope data allocation");
    auto prepared=Rope::prepare_matrix(actual.get(),pe);
    ggml_tensor* expected=nullptr;ggml_tensor* result=nullptr;
    for(int i=0;i<consumers;++i) {
        auto a=reference_apply_rope(reference.get(),ggml_scale(reference.get(),x,1.f+float(i)/100.f),pe,interleaved);
        auto b=Rope::apply_rope_prepared(actual.get(),ggml_scale(actual.get(),x,1.f+float(i)/100.f),prepared,interleaved);
        expected=expected?ggml_concat(reference.get(),expected,a,1):a;
        result=result?ggml_concat(actual.get(),result,b,1):b;
    }
    auto rg=graph(reference.get(),expected),ag=graph(actual.get(),result);
    require(preparations(rg,pe)==consumers && preparations(ag,pe)==1,"rope preparation is not graph-local shared work");
    auto ra=allocate(cpu,rg),aa=allocate(cpu,ag);
    for(unsigned seed:{1u,19u}) {
        write(x,values(ggml_nelements(x),seed,32));
        // Change the SAME matrix storage between executions: not a persistent
        // pointer-keyed cache, and not a stale device upload after generation 1.
        write(pe,values(ggml_nelements(pe),seed+11,16));
        compute(cpu,rg);compute(cpu,ag);
        close(read(expected),read(result),0.f,0.f);
    }
}
void model_graph(ggml_backend_t cpu,int layers,bool sharing,bool reuse) {
    auto params=context(),computation=context(); WebgpuName gpu(cpu);
    Qwen::QwenImage21Config config;
    config.hidden_size=32;config.context_dim=32;config.head_dim=8;config.intermediate_size=48;
    config.in_channels=config.out_channels=4;config.num_layers=layers;config.axes_dim={2,2,4};
    Qwen::QwenImage21Model model(config);model.init(params.get());
    auto x=ggml_new_tensor_4d(params.get(),GGML_TYPE_F32,2,2,4,1);
    auto time=ggml_new_tensor_1d(params.get(),GGML_TYPE_F32,1);
    auto text=ggml_new_tensor_3d(params.get(),GGML_TYPE_F32,32,5,1);
    const auto layout=Qwen::QwenImage21Layout::build(5,{},{{2,2}});
    auto pe=ggml_new_tensor_4d(params.get(),GGML_TYPE_F32,2,2,4,reuse?4:9);
    auto mask=ggml_new_tensor_2d(params.get(),GGML_TYPE_F32,5,5);
    GGMLRunnerContext context;context.ggml_ctx=computation.get();context.backend=sharing?&gpu.backend:cpu;
    auto one=ggml_new_tensor_1d(computation.get(),GGML_TYPE_F32,1);ggml_set_name(one,"ggml_runner_build_in_tensor:one");
    Qwen::QwenImage21PrefixCache cache;
    cache.mode=reuse?Qwen::QwenImage21PrefixCache::Mode::REUSE:Qwen::QwenImage21PrefixCache::Mode::NONE;
    cache.prefix_length=5;
    context.get_cache_tensor=[&](const std::string& name) {
        return name.back()=='k' ? ggml_new_tensor_4d(params.get(),GGML_TYPE_F32,8,5,4,1) : ggml_new_tensor_4d(params.get(),GGML_TYPE_F32,8,4,5,1);
    };
    auto output=model.forward(&context,x,time,reuse?nullptr:text,{},pe,layout,{mask,nullptr},cache);
    auto g=graph(computation.get(),output,16384);
    require(preparations(g,pe)==(sharing?1:layers*2),"Qwen model did not share exactly one PE preparation");
    std::map<std::string,ggml_tensor*> weights;
    model.get_param_tensors(weights);
    std::unordered_set<const ggml_tensor*> parameters;
    for (const auto& [name,tensor] : weights) parameters.insert(tensor);
    // The position producer is shared only within the graph. A memory-driven
    // cut may recompute it in later segments; its dependency must not disappear
    // or be mistaken for a persistent cached value.
    auto plan=sd::ggml_graph_cut::build_plan(cpu,g,parameters,nullptr);
    require(plan.available && plan.has_cuts && plan.valid,"shared position producer invalidated the cut plan");
    require(plan.segments.size()>=size_t(layers),"missing Qwen layer cuts");
    for (const auto& segment : plan.segments) {
        ggml_context* segment_context=nullptr;
        auto segment_graph=sd::ggml_graph_cut::build_segment_graph(g,segment,&segment_context);
        require(segment_graph!=nullptr,"cut graph construction failed");
        ggml_free(segment_context);
    }
    std::cout<<"{\"kind\":\"qwen21-rope-graph\",\"layers\":"<<layers<<",\"prefix_reuse\":"<<(reuse?"true":"false")<<",\"shared\":"<<(sharing?"true":"false")<<",\"preparations\":"<<preparations(g,pe)<<",\"nodes\":"<<ggml_graph_n_nodes(g)<<"}\n";
}
}
int main(){try {
    auto cpu=image_test::cpu();int cases=0;
    for(int width:{8,32})for(int positions:{3,17})for(int batch:{1,2})for(bool interleaved:{false,true})for(int consumers:{2,8}) {
        numerical(cpu.get(),width,2,positions,batch,interleaved,consumers);++cases;
    }
    numerical(cpu.get(),128,2,5,1,true,64);++cases;
    for(int layers:{2,32})for(bool reuse:{false,true})for(bool sharing:{false,true}){model_graph(cpu.get(),layers,sharing,reuse);++cases;}
    std::cout<<"{\"cases\":"<<cases<<",\"execution\":\"CPU numerical and graph construction\",\"passed\":true}\n";
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
