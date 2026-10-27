static unsigned check_dispatch() {
    unsigned checks=0;
    auto check=[&](bool ok){++checks;if(!ok)throw std::runtime_error("actual Q4 dispatcher mismatch");};
    for(bool indexed:{false,true})for(uint32_t sg:{4,8,16,32,64,128})for(uint32_t limit:{32,64,128,256})
    for(int k:{256,512,768,1024,1280,2048,2304,4096,4352})for(int m:{1,3,4,5,257,512,2048})
    for(bool shared:{false,true})for(uint32_t max_groups:{17,65535}) {
        webgpu_context ctx=std::make_shared<RecordedContext>();
        auto & caps=ctx->global_ctx->capabilities;
        caps.limits.maxComputeInvocationsPerWorkgroup=limit;caps.limits.maxComputeWorkgroupsPerDimension=max_groups;
        caps.min_subgroup_size=caps.max_subgroup_size=sg;
        ggml_tensor a{},x{},ids{},dst{};a.type=GGML_TYPE_Q4_K;x.type=dst.type=GGML_TYPE_F32;ids.type=GGML_TYPE_I32;
        for(auto * t:{&a,&x,&ids,&dst})for(auto & n:t->ne)n=1;
        a.ne[0]=x.ne[0]=k;a.ne[1]=dst.ne[0]=m;
        if(indexed){a.ne[2]=11;ids.ne[0]=dst.ne[1]=8;x.ne[1]=shared?1:8;}
        else {a.ne[2]=2;a.ne[3]=2;x.ne[2]=dst.ne[2]=shared?4:2;x.ne[3]=dst.ne[3]=shared?6:2;}
        a.view_offs=256+144;x.view_offs=512+12;ids.view_offs=8;dst.view_offs=768+28;
        a.nb[1]=(k/256+1)*144;a.nb[2]=(m*(k/256+1)+3)*144;a.nb[3]=a.ne[2]*a.nb[2]+144;
        x.nb[1]=(k+5)*4;x.nb[2]=x.ne[1]*x.nb[1]+12;x.nb[3]=x.ne[2]*x.nb[2]+20;
        const auto op=indexed?ggml_webgpu_mul_mat_id_vec(ctx,&a,&x,&ids,&dst):ggml_webgpu_mul_mat(ctx,&a,&x,&dst);
        const auto * decision=static_cast<ggml_webgpu_mul_mat_vec_shader_decisions *>(op.pipeline.context.get());
        uint32_t expected=std::max(32u,sg);while(expected<uint32_t(k/16))expected*=2;
        if(k>4096 || expected>limit)expected=256;
        check(decision->wg_size==expected);check(decision->outputs_per_wg==4);
        if(indexed)check(op.params==std::vector<uint32_t>({1,3,2,7,uint32_t(k),uint32_t(m),11,8,uint32_t(x.ne[1]),uint32_t(k/256+1),uint32_t(k+5),uint32_t(m*(k/256+1)+3),uint32_t(x.ne[1]*(k+5)+3)}));
        else check(op.params==std::vector<uint32_t>({1,3,7,uint32_t(m),1,uint32_t(k),uint32_t(k/256+1),uint32_t(k+5),uint32_t(a.nb[2]/144),uint32_t(x.nb[2]/4),uint32_t(a.nb[3]/144),uint32_t(x.nb[3]/4),2,2,uint32_t(x.ne[2]/2),uint32_t(x.ne[3]/2)}));
        check(op.entries.size()==(indexed?4u:3u));
        check(op.entries[0].offset==256 && op.entries[1].offset==512 && op.entries.back().offset==768);
        if(indexed)check(op.entries[2].offset==0);
        const uint32_t groups=((m+3)/4)*(indexed?8:dst.ne[2]*dst.ne[3]);
        check(op.y==CEIL_DIV(groups,max_groups));check(op.x==CEIL_DIV(groups,op.y));
        check(op.x*op.y>=groups && op.x<=max_groups);
    }
    std::cout<<checks<<" actual Q4 dispatch checks passed\n";
    return checks;
}
