// CPU arithmetic and backend-capability regression; not real H3 inference.
#include "core/ggml_extend.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

struct Filter {
    ggml_backend_t real;
    bool reject;
    ggml_backend_device device = {};
    ggml_backend backend;
    Filter(ggml_backend_t source, bool rejected) : real(source), reject(rejected), backend(*source) {
        device.context = this;
        device.iface.supports_op = [](ggml_backend_dev_t dev, const ggml_tensor* node) {
            auto self = static_cast<Filter*>(dev->context);
            if (node->op == GGML_OP_IM2COL_3D || (self->reject && node->op == GGML_OP_MUL_MAT)) return false;
            return ggml_backend_supports_op(self->real, node);
        };
        backend.device = &device;
    }
};
int check(ggml_type type, int batches, bool bias, bool reject, bool direct, bool force, int depth) {
    using Ctx = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
    using Backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
    using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
    Backend cpu(ggml_backend_cpu_init(), ggml_backend_free);
    ggml_backend_cpu_set_n_threads(cpu.get(), 1);
    Ctx ctx(ggml_init({4*1024*1024,nullptr,true}), ggml_free);
    const int W=3,H=2,IC=4,OC=5,S=W*H*depth;
    auto x=ggml_new_tensor_4d(ctx.get(),GGML_TYPE_F32,W,H,depth,IC*batches);
    auto w=ggml_new_tensor_4d(ctx.get(),type,1,1,1,IC*OC);
    auto b=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,OC);
    ggml_set_input(x);
    Filter filter(cpu.get(), reject);
    auto y=ggml_ext_conv_3d(ctx.get(),&filter.backend,x,w,bias?b:nullptr,IC,1,1,1,0,0,0,1,1,1,force,direct);
    if(y->ne[0]!=W||y->ne[1]!=H||y->ne[2]!=depth||y->ne[3]!=OC*batches) return 1;
    auto graph=ggml_new_graph_custom(ctx.get(),128,false);
    ggml_build_forward_expand(graph,y);
    int conv=0,im2col=0;
    for(int i=0;i<ggml_graph_n_nodes(graph);i++) {
        auto node=ggml_graph_node(graph,i);
        if(node->op==GGML_OP_CONV_3D)conv++;
        if(node->op==GGML_OP_IM2COL_3D)im2col++;
    }
    const bool lowered=!reject&&!direct&&!force&&depth>1;
    if(lowered && (conv||im2col))return 2;
    if((reject||direct)&&!force&&depth>1&&conv!=1)return 3;
    if(force&&im2col!=1)return 4;
    Buffer buf(ggml_backend_alloc_ctx_tensors(ctx.get(),cpu.get()),ggml_backend_buffer_free);
    if(!buf)return 5;
    std::vector<float> xv(S*IC*batches),wv(IC*OC),bv(OC),actual(S*OC*batches);
    std::vector<ggml_fp16_t> wh(wv.size());
    for(size_t i=0;i<xv.size();i++)xv[i]=(int(i%19)-9)/16.f;
    for(size_t i=0;i<wv.size();i++){wv[i]=(int(i%7)-3)/8.f;wh[i]=ggml_fp32_to_fp16(wv[i]);}
    for(int i=0;i<OC;i++)bv[i]=(i-2)/16.f;
    ggml_backend_tensor_set(x,xv.data(),0,xv.size()*sizeof(float));
    ggml_backend_tensor_set(w,type==GGML_TYPE_F16?static_cast<void*>(wh.data()):static_cast<void*>(wv.data()),0,ggml_nbytes(w));
    ggml_backend_tensor_set(b,bv.data(),0,bv.size()*sizeof(float));
    if(ggml_backend_graph_compute(cpu.get(),graph)!=GGML_STATUS_SUCCESS)return 6;
    ggml_backend_tensor_get(y,actual.data(),0,actual.size()*sizeof(float));
    for(int n=0;n<batches;n++)for(int oc=0;oc<OC;oc++)for(int pos=0;pos<S;pos++){
        float expected=bias?bv[oc]:0;
        for(int ic=0;ic<IC;ic++)expected+=wv[oc*IC+ic]*xv[(n*IC+ic)*S+pos];
        float value=actual[(n*OC+oc)*S+pos];
        if(!std::isfinite(value)||std::abs(value-expected)>0.002f)return 7;
    }
    return 0;
}
int main(){
    int cases=0;
    for(auto type:{GGML_TYPE_F16,GGML_TYPE_F32})for(int batch:{1,2})for(bool bias:{false,true})for(int mode=0;mode<5;mode++){
        const int result=check(type,batch,bias,mode==1,mode==2,mode==3,mode==4?1:7);
        if(result){std::fprintf(stderr,"case %d error %d\n",cases,result);return result;}
        cases++;
    }
    std::printf("%d temporal pointwise/fallback cases passed (CPU, synthetic)\n",cases);return 0;
}
