#include "image-performance-test-utils.h"
#include "core/ggml_extend.h"
#include "webgpu-policy-fixture.h"
#include "model/diffusion/anima.hpp"
#include "model/diffusion/qwen_image_2_1.hpp"
#include "model/vae/wan_vae.hpp"
#include <iostream>
#include <map>
using namespace image_test;
struct Policy {
    WebgpuTestLimits limits;
    ggml_op reject=GGML_OP_NONE;
    ggml_backend_device device{};
    ggml_backend backend{};
    Policy(WebgpuTestLimits input={}) : limits(input) {
        device.context=this;device.iface.get_name=[](ggml_backend_dev_t){return "WebGPU-policy-replay";};
        device.iface.supports_op=[](ggml_backend_dev_t d,const ggml_tensor* t){auto& p=*static_cast<Policy*>(d->context);return t->op!=p.reject&&webgpu_test_supports(t,p.limits);};
        backend.device=&device;
    }
};
int cases=0,selected=0;
bool materialized(const ggml_tensor* node) {return node->op!=GGML_OP_NONE&&node->op!=GGML_OP_VIEW&&node->op!=GGML_OP_RESHAPE&&node->op!=GGML_OP_PERMUTE&&node->op!=GGML_OP_TRANSPOSE;}

void numerical(ggml_backend_t cpu,int width,int height,int kernel,int stride,int dilation,int padding,int batch,ggml_type type,bool circular_x,bool circular_y,float scale,bool reject) {
    auto data=context();auto x=ggml_new_tensor_4d(data.get(),GGML_TYPE_F32,width,height,4,batch);
    auto w=ggml_new_tensor_4d(data.get(),type,kernel,kernel,4,4);auto b=ggml_new_tensor_1d(data.get(),GGML_TYPE_F32,4);
    Buffer buffer(ggml_backend_alloc_ctx_tensors(data.get(),cpu),ggml_backend_buffer_free);
    write(w,values(ggml_nelements(w),7,128));write(b,values(4,3,128));
    auto ref=context(),act=context();Policy policy({size_t(batch)*32768,32768,true,false});
    if(reject)policy.reject=GGML_OP_SET;
    auto expected=ggml_ext_conv_2d(ref.get(),x,w,b,stride,stride,padding,padding,dilation,dilation,false,circular_x,circular_y,scale);
    auto actual=ggml_ext_conv_2d(act.get(),x,w,b,stride,stride,padding,padding,dilation,dilation,false,circular_x,circular_y,scale,&policy.backend);
    auto rg=graph(ref.get(),expected),ag=graph(act.get(),actual);
    const bool split=count(ag,GGML_OP_SET)>0;
    if(reject)require(!split,"rejected SET must keep original graph");
    if(split) {
        ++selected;
        for(int i=0;i<ggml_graph_n_nodes(ag);++i)require(webgpu_test_supports(ggml_graph_node(ag,i),policy.limits),std::string("introduced unsupported node: ")+ggml_op_name(ggml_graph_node(ag,i)->op));
        require(count(ag,GGML_OP_IM2COL)>1,"expected bounded expansion");
    }
    auto ra=allocate(cpu,rg),aa=allocate(cpu,ag);
    for(unsigned iteration:{1u,2u}){write(x,values(ggml_nelements(x),iteration,64));compute(cpu,rg);compute(cpu,ag);close(read(expected),read(actual),2e-6f,2e-5f);}
    ++cases;
}
void shape_budget() {
    auto c=context();auto x=ggml_new_tensor_4d(c.get(),GGML_TYPE_F32,256,256,288,1);auto w=ggml_new_tensor_4d(c.get(),GGML_TYPE_F16,3,3,288,288);
    for(size_t mib:{128u,256u,512u}) {
        auto gc=context();Policy p({mib*1024*1024,32768,true,false});auto out=ggml_ext_conv_2d(gc.get(),x,w,nullptr,1,1,1,1,1,1,false,false,false,1,&p.backend);
        auto g=graph(gc.get(),out);size_t largest=0;int unsupported=0;
        for(int i=0;i<ggml_graph_n_nodes(g);++i){auto n=ggml_graph_node(g,i);if(n->op==GGML_OP_IM2COL)largest=std::max(largest,ggml_nbytes(n));unsupported+=!webgpu_test_supports(n,p.limits);}
        require(unsupported==0,"budget-aware graph still rejects materialized nodes");
        require(largest==(mib==128?81u:mib==256?162u:324u)*1024*1024,"unexpected im2col budget");
        std::cout<<"{\"kind\":\"qwen-conv-policy\",\"binding_mib\":"<<mib<<",\"max_im2col_mib\":"<<largest/1024/1024<<",\"stripes\":"<<count(g,GGML_OP_SET)<<",\"unsupported\":"<<unsupported<<"}\n";
        ++cases;
    }
}
void audit_graph(ggml_cgraph* graph,const std::string& label,const WebgpuTestLimits& limits,bool must_support) {
    std::map<std::string,int> unsupported;int count=0;
    for(int i=0;i<ggml_graph_n_nodes(graph);++i){auto n=ggml_graph_node(graph,i);if(materialized(n)&&!webgpu_test_supports(n,limits)){++count;++unsupported[ggml_op_name(n->op)];}}
    std::cout<<"{\"kind\":\"policy-audit\",\"label\":\""<<label<<"\",\"nodes\":"<<ggml_graph_n_nodes(graph)<<",\"unsupported\":"<<count<<",\"ops\":{";
    bool first=true;for(const auto& [op,n]:unsupported){if(!first)std::cout<<",";first=false;std::cout<<"\""<<op<<"\":"<<n;}std::cout<<"}}\n";
    if(must_support)require(count==0,"unexpected WebGPU rejection: "+label);++cases;
}
void attention_numerical(ggml_backend_t cpu, int batch, int mask_mode, bool sink, bool reject, ggml_type type) {
    constexpr int head=8, heads=2, queries=65, keys=97;
    auto data=context(),ref=context(),act=context();
    auto q=ggml_new_tensor_3d(data.get(),GGML_TYPE_F32,head*heads,queries,batch);
    auto k=ggml_new_tensor_3d(data.get(),type,head*heads,keys,batch);
    auto v=ggml_new_tensor_3d(data.get(),type,head*heads,keys,batch);
    auto mask=mask_mode==0?nullptr:ggml_new_tensor_3d(data.get(),GGML_TYPE_F32,keys,mask_mode==1?1:queries,mask_mode==3?heads*batch:1);
    auto sinks=sink?ggml_new_tensor_1d(data.get(),GGML_TYPE_F32,heads*batch):nullptr;
    Buffer buffer(ggml_backend_alloc_ctx_tensors(data.get(),cpu),ggml_backend_buffer_free);
    write(k,values(ggml_nelements(k),12,32));write(v,values(ggml_nelements(v),13,32));
    if(sinks)write(sinks,std::vector<float>(heads*batch,std::log(435.f)));
    if(mask){auto m=values(ggml_nelements(mask),3,8);for(size_t i=0;i<m.size();++i)if(i%keys==3)m[i]=-INFINITY;write(mask,m);}
    Policy policy({size_t(batch)*32768,32768,true,false});if(reject)policy.reject=GGML_OP_SET;
    auto expected=ggml_ext_attention_ext(ref.get(),cpu,q,k,v,heads,mask,false,false,1,false,sinks);
    auto actual=ggml_ext_attention_ext(act.get(),&policy.backend,q,k,v,heads,mask,false,false,1,false,sinks);
    auto rg=graph(ref.get(),expected),ag=graph(act.get(),actual);const bool split=count(ag,GGML_OP_SET)>0;
    if(reject || (batch==2 && mask_mode==3))require(!split,"unsupported SET/multi-head mask span must retain unchunked attention");
    else {
        require(split,"expected bounded manual attention type="+std::string(ggml_type_name(type))+" batch="+std::to_string(batch)+" mask="+std::to_string(mask_mode)+" sink="+std::to_string(sink));
        for(int i=0;i<ggml_graph_n_nodes(ag);++i){auto node=ggml_graph_node(ag,i);if(materialized(node)&&!webgpu_test_supports(node,policy.limits)) { std::cerr<<"type="<<ggml_type_name(type)<<" batch="<<batch<<" mask="<<mask_mode<<" sink="<<sink<<" op="<<ggml_op_name(node->op)<<" out="<<ggml_nbytes(node)<<" src0="<<(node->src[0]?ggml_nbytes(node->src[0]):0)<<"\n";throw std::runtime_error("bounded attention introduced rejected node");}}
        require(count(ag,GGML_OP_FLASH_ATTN_EXT)==0,"manual setting silently enabled Flash");
    }
    auto ra=allocate(cpu,rg),aa=allocate(cpu,ag);
    for(unsigned input:{1u,2u}){write(q,values(ggml_nelements(q),input,32));compute(cpu,rg);compute(cpu,ag);close(read(expected),read(actual),3e-6f,3e-5f);}
    ++cases;
}

void attention_policies() {
    for(bool flash:{false,true}) for(bool subgroups:{false,true}) for(int length:{77,512}) {
        auto params=context(),gc=context();Policy p;p.limits.subgroups=subgroups;GGMLRunnerContext c;c.ggml_ctx=gc.get();c.backend=&p.backend;c.flash_attn_enabled=flash;
        auto one=ggml_new_tensor_1d(gc.get(),GGML_TYPE_F32,1);ggml_set_name(one,"ggml_runner_build_in_tensor:one");
        Anima::AnimaAttention layer(2048,1024,16,128);layer.init(params.get());auto x=ggml_new_tensor_3d(params.get(),GGML_TYPE_F32,2048,1024,1);auto text=ggml_new_tensor_3d(params.get(),GGML_TYPE_F32,1024,length,1);
        auto compact=Anima::finish_text_context(&c,text,16,true);auto output=layer.forward(&c,x,compact.values,nullptr,nullptr,compact.sinks);
        auto g=graph(gc.get(),output);audit_graph(g,"anima-"+std::to_string(length)+(flash?"-flash-requested":"-manual")+(subgroups?"-subgroups":"-no-subgroups"),p.limits,true);
        require(count(g,GGML_OP_FLASH_ATTN_EXT)==int(flash&&subgroups),"Flash rejection did not rebuild a supported manual graph");

    }
    for(bool flash:{false,true}) for(bool legacy:{false,true}) for(int queries:{1024,4096}) {
        auto params=context(),gc=context();Policy p;GGMLRunnerContext c;c.ggml_ctx=gc.get();c.backend=&p.backend;c.flash_attn_enabled=flash;
        if(legacy) p.device.iface.get_name=[](ggml_backend_dev_t){return "legacy-policy";};
        Qwen::QwenImage21Config config;Qwen::QwenImage21Attention layer(config);layer.init(params.get());
        auto x=ggml_new_tensor_3d(params.get(),GGML_TYPE_F32,4096,queries+77,1);auto pe=ggml_new_tensor_4d(params.get(),GGML_TYPE_F32,2,2,64,queries+77);
        std::vector<Qwen::QwenImage21Segment> segments{{0,77,0,-1},{77,queries+77,77,0}};
        std::vector<ggml_tensor*> masks{ggml_new_tensor_2d(params.get(),GGML_TYPE_F32,77,77),nullptr};
        auto output=layer.forward(&c,x,pe,segments,masks,{});auto g=graph(gc.get(),output);
        audit_graph(g,(flash?"qwen21-flash":"qwen21-manual")+std::string(legacy?"-legacy":"-bounded")+"-q"+std::to_string(queries),p.limits,flash||!legacy);
    }
    for(bool packed:{false,true}) {
        auto params=context(),gc=context();Policy p;GGMLRunnerContext c;c.ggml_ctx=gc.get();c.backend=&p.backend;
        WAN::CausalConv3d layer(96,96,{3,3,3},{1,1,1},{1,1,1});layer.init(params.get());
        auto x=ggml_new_tensor_4d(params.get(),GGML_TYPE_F32,64,64,1,96);
        ggml_tensor* output;
        if(packed) output=layer.forward(&c,x);
        else {
            // Frozen pre-003 causal path still uses the existing WebGPU-aware
            // full-depth 3D-to-2D lowering. Do not compare a CPU-only graph and
            // mislabel its IM2COL_3D rejection as an application fallback.
            std::map<std::string,ggml_tensor*> weights;layer.get_param_tensors(weights);
            auto padded=ggml_ext_pad_ext(gc.get(),&p.backend,x,1,1,1,1,2,0,0,0,false,false);
            output=ggml_ext_conv_3d(gc.get(),&p.backend,padded,weights.at("weight"),weights.at("bias"),96,
                                     1,1,1,0,0,0,1,1,1,false,false);
        }
        auto g=graph(gc.get(),output);
        audit_graph(g,packed?"003-causal-packed":"causal-pre003-webgpu",p.limits,true);
    }
}
int main(){try{auto cpu=image_test::cpu();
    for(auto type:{GGML_TYPE_F16,GGML_TYPE_F32})for(int batch:{1,2})for(int mode=0;mode<6;++mode)for(bool reject:{false,true}){
        numerical(cpu.get(),mode==0?31:32,mode==1?33:32,3,mode==2?2:1,mode==3?2:1,mode==4?0:1,batch,type,mode==5,mode==5,mode==1?1.f/128.f:1.f,reject);
    }
    for(auto type:{GGML_TYPE_F16,GGML_TYPE_F32})for(int batch:{1,2})for(int mask=0;mask<4;++mask)for(bool sink:{false,true})for(bool reject:{false,true})
        attention_numerical(cpu.get(),batch,mask,sink,reject,type);
    shape_budget();attention_policies();
    require(selected>0,"no numerical stripe cases executed");std::cout<<"{\"cases\":"<<cases<<",\"numerical_split_cases\":"<<selected<<",\"gpu_executed\":false}\n";
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
