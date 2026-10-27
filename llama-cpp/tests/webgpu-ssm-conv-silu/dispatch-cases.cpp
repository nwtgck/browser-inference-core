static unsigned checks=0;
static void check(bool value) { ++checks; if(!value) { throw std::runtime_error("dispatch check "+std::to_string(checks)); } }
struct device_fixture : fixture {
    webgpu_context ctx=std::make_shared<Context>();
    ggml_backend_webgpu_buffer_context h_ctx{{1},ctx->global_ctx},w_ctx{{2},ctx->global_ctx},o_ctx{{3},ctx->global_ctx};
    ggml_backend_buffer h_buf{},w_buf{},o_buf{};
    device_fixture(int nc,int channels,int tokens,int seq,bool padded,uint32_t prefix):fixture(nc,channels,tokens,seq,padded) {
        for(auto * b:{&h_buf,&w_buf,&o_buf}) { b->iface.get_base=ggml_backend_webgpu_buffer_get_base; b->size=1ull<<30; }
        h_buf.context=&h_ctx; w_buf.context=&w_ctx; o_buf.context=&o_ctx;
        history_root->buffer=history->buffer=&h_buf; weights_root->buffer=weights->buffer=&w_buf;
        history_root->data=weights_root->data=reinterpret_cast<void *>(uintptr_t(0x1000+prefix));
        history->data=reinterpret_cast<void *>(uintptr_t(history_root->data)+history->view_offs);
        weights->data=reinterpret_cast<void *>(uintptr_t(weights_root->data)+weights->view_offs);
        conv->buffer=output->buffer=&o_buf; conv->data=output->data=reinterpret_cast<void *>(uintptr_t(0x1000+prefix));
    }
    auto run(int & consumed) { return ggml_webgpu_ssm_conv_silu(ctx,graph,index,consumed); }
};
int main() { try {
    for(int nc:{1,3,4,5}) for(int channels:{1,31,32,33,320}) for(int tokens:{1,2,7,8,9})
    for(bool padded:{false,true}) for(uint32_t prefix:{0u,4u,252u,256u,260u}) {
        device_fixture f(nc,channels,tokens,3,padded,prefix); int consumed=1; auto r=f.run(consumed);
        check(r && consumed==2 && r->pipeline.id==1);
        auto plain=ggml_webgpu_ssm_conv(f.ctx,f.history,f.weights,f.output);
        check(plain.pipeline.id==0 && plain.params==r->params && plain.x==r->x && plain.y==r->y);
        const uint32_t tpw=tokens==1?1:8;
        check(r->x==uint32_t((channels+31)/32) && r->y==uint32_t(((tokens+tpw-1)/tpw)*3));
        check(r->params==std::vector<uint32_t>({uint32_t((prefix+f.history->view_offs)%256/4),uint32_t((prefix+f.weights->view_offs)%256/4),prefix%256/4,
              uint32_t(f.history->nb[1]/4),uint32_t(f.history->nb[2]/4),uint32_t(f.weights->nb[1]/4),1,uint32_t(channels),uint32_t(channels*tokens),uint32_t(nc),uint32_t(channels),uint32_t(tokens),3,uint32_t((tokens+tpw-1)/tpw)}));
        check(r->entries.size()==3);
        for(int i=0;i<3;i++) { auto * t=i==0?f.history:i==1?f.weights:f.output; const uint64_t at=prefix+t->view_offs;
            check(r->entries[i].offset==at-at%256 && r->entries[i].size==at%256+ggml_nbytes(t)); }
        auto reject=[&] { const auto calls=f.ctx->shader_lib->calls,n=encoded; consumed=1; check(!f.run(consumed)); check(consumed==1 && encoded==n && f.ctx->shader_lib->calls==calls); };
        auto & l=f.ctx->global_ctx->capabilities.limits;
        l.maxComputeInvocationsPerWorkgroup=32*tpw-1; reject(); l.maxComputeInvocationsPerWorkgroup=256;
        l.maxComputeWorkgroupSizeX=31; reject(); l.maxComputeWorkgroupSizeX=256;
        l.maxComputeWorkgroupSizeY=tpw-1; reject(); l.maxComputeWorkgroupSizeY=256;
        l.maxStorageBuffersPerShaderStage=2; reject(); l.maxStorageBuffersPerShaderStage=8;
        l.maxUniformBuffersPerShaderStage=0; reject(); l.maxUniformBuffersPerShaderStage=12;
        l.maxBindingsPerBindGroup=3; reject(); l.maxBindingsPerBindGroup=1000;
        l.maxUniformBufferBindingSize=127; reject(); l.maxUniformBufferBindingSize=65536;
        l.maxStorageBufferBindingSize=3; reject(); l.maxStorageBufferBindingSize=1ull<<30;
        l.maxComputeWorkgroupsPerDimension=0; reject(); l.maxComputeWorkgroupsPerDimension=65535;
        l.minStorageBufferOffsetAlignment=3; reject(); l.minStorageBufferOffsetAlignment=256;
        f.h_buf.iface.get_base=nullptr; f.h_buf.context=reinterpret_cast<void *>(1); reject(); f.h_buf.iface.get_base=ggml_backend_webgpu_buffer_get_base; f.h_buf.context=&f.h_ctx;
        f.h_buf.context=nullptr; reject(); f.h_buf.context=&f.h_ctx;
        const auto data=f.history->data; f.history->data=reinterpret_cast<void *>(uintptr_t(data)+1); reject(); f.history->data=data;
        f.o_ctx.global_ctx=std::make_shared<Global>(); reject(); f.o_ctx.global_ctx=f.ctx->global_ctx;
        f.w_ctx.buffer.usage=wgpu::BufferUsage::None; reject(); f.w_ctx.buffer.usage=wgpu::BufferUsage::Storage;
        f.o_buf.size=prefix+ggml_nbytes(f.output)-1; reject(); f.o_buf.size=1ull<<30;
        f.o_ctx.buffer.bytes=prefix+ggml_nbytes(f.output)-1; reject(); f.o_ctx.buffer.bytes=1ull<<30;
        const auto root=f.history_root->data; f.history_root->data=reinterpret_cast<void *>(uintptr_t(0x800)); reject(); f.history_root->data=root;
        if(padded) { auto * b=f.history_root->buffer; f.history_root->buffer=nullptr; reject(); f.history_root->buffer=b; }
        // Reject payload and binding-only overlap against each read binding, even through distinct ggml wrappers.
        const auto out=f.output->data;
        for(bool weight:{false,true}) { f.o_ctx.buffer.id=weight?2:1; f.output->data=weight?f.weights->data:f.history->data; reject(); }
        f.o_ctx.buffer.id=3; f.output->data=out;
        f.w_ctx.buffer.id=1; f.weights_root->data=f.history_root->data; f.weights->data=f.history->data;
        if(!padded) { reject(); }
    }
    { device_fixture f(1,1,1,1,false,0); int used=1; f.o_ctx.buffer.id=1; f.output->data=reinterpret_cast<void *>(uintptr_t(0x1004)); check(!f.run(used) && used==1);
      f.output->data=reinterpret_cast<void *>(uintptr_t(0x1100)); check(bool(f.run(used)) && used==2); }
    { device_fixture f(4,33,9,3,false,0); int used=1; f.ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension=5; check(!f.run(used) && used==1); }
    { device_fixture f(1,1,1,1,false,0); int used=1; f.w_ctx.buffer.id=1;
      f.weights->data=reinterpret_cast<void *>(uintptr_t(0x1004)); check(!f.run(used) && used==1);
      f.weights->data=reinterpret_cast<void *>(uintptr_t(0x1100)); check(bool(f.run(used)) && used==2); }
    // Rounding token tiles can overflow the global y ID although logical element count fits u32.
    { device_fixture f(1,1,1,1,false,0); constexpr int64_t tokens=2147483644;
      for(auto * t:{f.history,f.conv,f.output}) {
          t->ne[0]=t==f.history?tokens:1; t->ne[1]=t==f.history?1:tokens; t->ne[2]=2;
          t->nb[1]=uint64_t(t->ne[0])*4; t->nb[2]=uint64_t(tokens)*4; t->nb[3]=uint64_t(tokens)*8;
      }
      ssm_conv_silu_plan p; check(plan_ssm_conv_silu(f.graph,f.index,p));
      f.ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension=UINT32_MAX;
      int used=1; const auto calls=f.ctx->shader_lib->calls; check(!f.run(used) && used==1 && f.ctx->shader_lib->calls==calls); }
    // An aligned binding prefix is part of the WGSL u32 address bound.
    { device_fixture f(1,1,1,1,false,252); auto * t=f.history; t->ne[0]=UINT32_MAX-31u;
      for(int i=1;i<4;i++) { t->nb[i]=uint64_t(t->ne[0])*4; }
      f.h_buf.size=f.h_ctx.buffer.bytes=1ull<<36;
      f.ctx->global_ctx->capabilities.limits.maxStorageBufferBindingSize=1ull<<36;
      uint64_t bytes; check(ssm_conv_silu_layout(t,false,bytes)); webgpu_ssm_conv_silu_binding b;
      check(!ggml_webgpu_ssm_conv_silu_binding(f.ctx,t,bytes,b)); }
    std::cout << checks << " actual host binding/dispatch/cache-admission checks\n";
} catch(const std::exception & e) { std::cerr << e.what() << '\n'; return 1; } }
