static unsigned check_dispatch() {
    unsigned checks=0;
    auto check=[&](bool ok){++checks;if(!ok)throw std::runtime_error("actual expert dispatcher mismatch");};
    for(ggml_type type:{GGML_TYPE_Q5_1,GGML_TYPE_IQ4_NL})for(uint32_t wg_limit:{64,128,256})
    for(uint32_t subgroup:{0,3,4,8,16,32,64,128,256}) {
        webgpu_context ctx=std::make_shared<RecordedContext>();
        ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup=wg_limit;
        ctx->global_ctx->capabilities.min_subgroup_size=subgroup;
        ctx->global_ctx->capabilities.max_subgroup_size=subgroup;
        for(int k:{32,320,640,1024,1088})for(int m:{256,257,2560})for(int slots:{1,10})
        for(bool shared:{false,true})for(uint32_t max_groups:{128,65535}) {
            ggml_tensor a{},x{},ids{},dst{};a.type=type;x.type=dst.type=GGML_TYPE_F32;ids.type=GGML_TYPE_I32;
            for(auto * t:{&a,&x,&ids,&dst})for(auto & n:t->ne)n=1;
            a.ne[0]=x.ne[0]=k;a.ne[1]=dst.ne[0]=m;a.ne[2]=512;
            ids.ne[0]=dst.ne[1]=slots;x.ne[1]=shared?1:slots;
            a.view_offs=256+ggml_type_size(type);x.view_offs=512+12;ids.view_offs=8;dst.view_offs=768+28;
            a.nb[1]=(k/32+1)*ggml_type_size(type);a.nb[2]=(m*(k/32+1)+3)*ggml_type_size(type);
            x.nb[1]=(k+5)*4;x.nb[2]=x.ne[1]*x.nb[1];
            ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension=max_groups;
            const auto op=ggml_webgpu_mul_mat_id_vec(ctx,&a,&x,&ids,&dst);
            const auto * decision=static_cast<ggml_webgpu_mul_mat_vec_shader_decisions *>(op.pipeline.context.get());
            check(decision->wg_size==(wg_limit>=128 && k<=1024 && subgroup>=4 && subgroup<=128?128u:256u));
            check(decision->outputs_per_wg==4);
            check(op.params==std::vector<uint32_t>({1,3,2,7,uint32_t(k),uint32_t(m),512,uint32_t(slots),uint32_t(x.ne[1]),uint32_t(k/32+1),uint32_t(k+5),uint32_t(m*(k/32+1)+3),uint32_t(x.ne[1]*(k+5))}));
            check(op.entries.size()==4 && op.entries[0].offset==256 && op.entries[1].offset==512 && op.entries[2].offset==0 && op.entries[3].offset==768);
            const uint32_t groups=((m+3)/4)*slots;
            check(op.y==CEIL_DIV(groups,max_groups));check(op.x==CEIL_DIV(groups,op.y));
            check(op.x*op.y>=groups && op.x<=max_groups && op.y<=max_groups);
        }
    }
    std::cout<<checks<<" actual expert dispatch checks passed\n";
    return checks;
}
