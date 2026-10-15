#include "image-performance-test-utils.h"
#include "core/ggml_extend.h"
#include "ggml-backend-impl.h"
#include <cstring>
#include <iostream>
using namespace image_test;
namespace {
struct Policy {
    ggml_backend_t real;
    ggml_backend backend;
    ggml_backend_device device;
    size_t score_limit;
    bool reject_scaled_softmax;
    Policy(ggml_backend_t cpu,size_t limit,bool reject) : real(cpu),backend(*cpu),device(*cpu->device),score_limit(limit),reject_scaled_softmax(reject) {
        device.context=this;
        device.iface.get_name=[](ggml_backend_dev_t){return "WebGPU-softmax-construction";};
        device.iface.supports_op=[](ggml_backend_dev_t dev,const ggml_tensor* node) {
            auto& p=*static_cast<Policy*>(dev->context);
            if(node->op==GGML_OP_MUL_MAT && ggml_nbytes(node)>p.score_limit)return false;
            float scale=1.f; if(node->op==GGML_OP_SOFT_MAX)std::memcpy(&scale,node->op_params,sizeof(scale));
            if(p.reject_scaled_softmax && scale!=1.f)return false;
            return ggml_backend_supports_op(p.real,node);
        };
        backend.device=&device;
    }
};
void numerical(ggml_backend_t cpu,ggml_type type,int head,int batch,int mask_mode,bool sink,bool bounded,bool reject) {
    constexpr int heads=2,queries=33,keys=41;
    auto data=context(),reference=context(),actual=context();
    auto q=ggml_new_tensor_3d(data.get(),GGML_TYPE_F32,head*heads,queries,batch);
    auto k=ggml_new_tensor_3d(data.get(),type,head*heads,keys,batch);
    auto v=ggml_new_tensor_3d(data.get(),type,head*heads,keys,batch);
    auto mask=mask_mode?ggml_new_tensor_3d(data.get(),GGML_TYPE_F32,keys,mask_mode==1?1:queries,1):nullptr;
    auto sinks=sink?ggml_new_tensor_1d(data.get(),GGML_TYPE_F32,heads*batch):nullptr;
    Buffer buffer(ggml_backend_alloc_ctx_tensors(data.get(),cpu),ggml_backend_buffer_free);
    require(buffer!=nullptr,"attention input allocation");
    if(mask) { auto m=values(ggml_nelements(mask),9,8); for(size_t i=0;i<m.size();++i)if(i%keys==7)m[i]=-INFINITY;write(mask,m); }
    if(sinks)write(sinks,std::vector<float>(heads*batch,std::log(435.f)));
    const size_t full_scores=size_t(queries)*keys*heads*batch*sizeof(float);
    Policy policy(cpu,bounded?full_scores/2:SIZE_MAX,reject);
    // kv_scale is the Flash K/V overflow control, NOT the softmax temperature.
    // Both manual paths must still scale scores by 1/sqrt(head), after matmul.
    auto expected=ggml_ext_attention_ext(reference.get(),cpu,q,k,v,heads,mask,false,false,.125f,false,sinks);
    auto result=ggml_ext_attention_ext(actual.get(),&policy.backend,q,k,v,heads,mask,false,false,.125f,false,sinks);
    auto rg=graph(reference.get(),expected),ag=graph(actual.get(),result);
    const bool fused=mask==nullptr&&!reject;
    require(count(rg,GGML_OP_SCALE)>0,"CPU reference lost its explicit score scale");
    require((count(ag,GGML_OP_SCALE)==0)==fused,"incorrect fused admission or fallback");
    require(count(ag,GGML_OP_FLASH_ATTN_EXT)==0,"manual path enabled Flash");
    if(bounded && head<=16)require(count(ag,GGML_OP_SET)>0,"small bounded score case did not split");
    for(int i=0;i<ggml_graph_n_nodes(ag);++i) {
        auto node=ggml_graph_node(ag,i);
        if(node->op==GGML_OP_SOFT_MAX && fused) {
            require(node->src[0]->op==GGML_OP_MUL_MAT,"fusion moved scale before score accumulation");
            float scale;std::memcpy(&scale,node->op_params,sizeof(scale));
            require(scale==1.f/std::sqrt(float(head)),"wrong attention scale");
            require((node->src[2]!=nullptr)==sink,"sink was dropped");
        }
    }
    auto ra=allocate(cpu,rg),aa=allocate(cpu,ag);
    for(unsigned seed:{3u,23u}) {
        write(q,values(ggml_nelements(q),seed,8));write(k,values(ggml_nelements(k),seed+4,16));write(v,values(ggml_nelements(v),seed+9,8));
        compute(cpu,rg);compute(cpu,ag);close(read(expected),read(result),4e-6f,4e-5f);
    }
}
}
int main(){try {
    auto backend=cpu();int cases=0;
    for(auto type:{GGML_TYPE_F16,GGML_TYPE_F32})for(int head:{8,128})for(int batch:{1,2})for(int mask:{0,1,2})
        for(bool sink:{false,true})for(bool bounded:{false,true})for(bool reject:{false,true}){numerical(backend.get(),type,head,batch,mask,sink,bounded,reject);++cases;}
    std::cout<<"{\"cases\":"<<cases<<",\"inputs_per_case\":2,\"execution\":\"CPU arithmetic with WebGPU construction policy\",\"passed\":true}\n";
}catch(const std::exception& error){std::cerr<<error.what()<<"\n";return 1;}}
