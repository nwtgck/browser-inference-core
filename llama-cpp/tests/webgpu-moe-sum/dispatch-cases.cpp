static int checks=0;
static void check(bool ok) {++checks;if(!ok)throw std::runtime_error("dispatch failure at "+std::to_string(checks));}
struct Fixture {
    ggml_context * context;
    ggml_cgraph * graph;
    ggml_tensor * root,* out;
    int start;
    webgpu_context ctx=std::make_shared<Context>();
    ggml_backend_buffer_type type{};
    ggml_backend_webgpu_buffer_context sc{{1,1ull<<30},ctx->global_ctx},dc{{2,1ull<<30},ctx->global_ctx};
    ggml_backend_buffer sb{},db{};
    Fixture(int width,int experts,int tokens,uint32_t prefix): context(ggml_init({1<<20,nullptr,true})) {
        if(!context)throw std::bad_alloc();
        graph=ggml_new_graph_custom(context,1024,false);
        root=ggml_new_tensor_3d(context,GGML_TYPE_F32,width,experts,tokens); ggml_set_input(root);
        std::vector<ggml_tensor*> views;
        for(int i=0;i<experts;++i){views.push_back(ggml_view_2d(context,root,width,tokens,root->nb[2],i*root->nb[1]));ggml_build_forward_expand(graph,views.back());}
        start=graph->n_nodes;out=views[0];
        for(int i=1;i<experts;++i){out=ggml_add(context,out,views[i]);ggml_build_forward_expand(graph,out);}
        ggml_set_output(out);
        type.iface.get_name=ggml_backend_webgpu_buffer_type_get_name;
        sb.context=&sc;sb.buft=&type;db.context=&dc;db.buft=&type;
        root->buffer=&sb;out->buffer=&db;root->data=reinterpret_cast<void*>(0x1000+prefix);
        out->data=reinterpret_cast<void*>(0x1000+ROUNDUP_POW2(ggml_nbytes(root)+prefix,256)+48);
    }
    ~Fixture(){ggml_free(context);}
};
int main(){
    for(int width:{4,12,128,2560,4096}) for(int experts:{3,4,10,32}) for(int tokens:{1,3,31}) for(uint32_t prefix:{0u,16u,48u,240u,256u,272u}) {
        Fixture f(width,experts,tokens,prefix);int n=1;
        auto run=[&]{return ggml_webgpu_moe_sum(f.ctx,f.graph,f.start,n);};
        auto good=run();check(bool(good));check(n==experts-1);
        check(good->params==std::vector<uint32_t>({(prefix%256)/16,3u,uint32_t(width/4),uint32_t(tokens)}));
        check(good->pipeline==uint32_t(experts));check(good->entries.size()==2);
        check(good->entries[0].tensor==f.root&&good->entries[1].tensor==f.out);
        auto fail=[&]{n=1;unsigned before=encoded,calls=f.ctx->shader_lib->calls;check(!run());check(n==1&&before==encoded&&calls==f.ctx->shader_lib->calls);};
        auto & l=f.ctx->global_ctx->capabilities.limits;
        l.maxComputeInvocationsPerWorkgroup=127;fail();l.maxComputeInvocationsPerWorkgroup=256;
        l.maxComputeWorkgroupSizeX=127;fail();l.maxComputeWorkgroupSizeX=256;
        l.maxStorageBuffersPerShaderStage=1;fail();l.maxStorageBuffersPerShaderStage=8;
        l.maxComputeWorkgroupsPerDimension=0;fail();l.maxComputeWorkgroupsPerDimension=65535;
        l.maxStorageBufferBindingSize=8;fail();l.maxStorageBufferBindingSize=1ull<<30;
        auto data=f.root->data;f.root->data=reinterpret_cast<void*>(uintptr_t(data)+4);fail();f.root->data=data;
        f.sc.buffer.size=prefix+ggml_nbytes(f.root)-1;fail();f.sc.buffer.size=1ull<<30;
        f.dc.global_ctx=std::make_shared<Global>();fail();f.dc.global_ctx=f.ctx->global_ctx;
        f.dc.buffer.id=1;check(bool(run()));
        data=f.out->data;f.out->data=f.root->data;fail();f.out->data=data;f.dc.buffer.id=2;
        f.sb.buft=nullptr;fail();f.sb.buft=&f.type;
        f.type.iface.get_name=nullptr;f.sb.context=reinterpret_cast<void*>(uintptr_t(1));fail();
        f.type.iface.get_name=ggml_backend_webgpu_buffer_type_get_name;f.sb.context=&f.sc;
        f.root->buffer=nullptr;fail();f.root->buffer=&f.sb;
        f.graph->n_nodes--;fail();f.graph->n_nodes++;
        auto * mid=f.graph->nodes[f.start];auto flags=mid->flags;ggml_set_output(mid);fail();mid->flags=flags;
        check(bool(run()));
    }
    std::cout<<checks<<" actual production dispatch checks passed\n";
}
