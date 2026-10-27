static size_t checks=0;
static void require(bool condition,const char * message) {
    ++checks;
    if(!condition)throw std::runtime_error(message);
}
struct Fixture {
    ggml_tensor a{},x{},ids{},dst{};
    ggml_webgpu_shader_lib_context c{};
    Fixture(){reset();}
    void reset(){
        a={};x={};ids={};dst={};c={};
        a.type=GGML_TYPE_IQ3_S;x.type=dst.type=GGML_TYPE_F32;ids.type=GGML_TYPE_I32;
        c.src0=&a;c.src1=&x;c.src2=&ids;c.dst=&dst;
        c.supports_subgroups=true;c.min_subgroup_size=c.max_subgroup_size=16;c.max_wg_size=256;
        c.max_workgroups_per_dim=65535;c.max_storage_binding_bytes=1ull<<30;
        layout(2560,640,11,10,false,false);
    }
    void layout(int64_t k,int64_t m,int64_t experts,int64_t slots,bool shared,bool padded){
        for(auto * t:{&a,&x,&ids,&dst}){
            for(unsigned axis=0;axis<4;++axis){t->ne[axis]=1;t->nb[axis]=0;}
            t->view_offs=0;
        }
        a.ne[0]=x.ne[0]=k;a.ne[1]=dst.ne[0]=m;a.ne[2]=experts;
        x.ne[1]=shared?1:slots;ids.ne[0]=dst.ne[1]=slots;
        const size_t bytes=ggml_type_size(a.type);
        const size_t block=a.type==GGML_TYPE_IQ3_S?256:(a.type==GGML_TYPE_Q5_1||a.type==GGML_TYPE_IQ4_NL)?32:1;
        a.nb[0]=bytes;a.nb[1]=(static_cast<size_t>(k)/block+(padded?1:0))*bytes;
        a.nb[2]=(static_cast<size_t>(m)*(a.nb[1]/bytes)+(padded?3:0))*bytes;a.nb[3]=a.nb[2]*experts;
        x.nb[0]=ggml_type_size(x.type);x.nb[1]=(static_cast<size_t>(k)+(padded?5:0))*x.nb[0];
        x.nb[2]=x.ne[1]*x.nb[1]+(padded?12:0);x.nb[3]=x.nb[2];
        ids.nb[0]=4;ids.nb[1]=ids.nb[2]=ids.nb[3]=slots*4;
        dst.nb[0]=4;dst.nb[1]=m*4;dst.nb[2]=dst.nb[3]=m*slots*4;
        if(padded){a.view_offs=256+bytes;x.view_offs=512+12;ids.view_offs=256+8;dst.view_offs=768+28;}
        const ggml_tensor * tensors[4]={&a,&x,&ids,&dst};
        for(unsigned i=0;i<4;++i){
            c.indexed_prefix_bytes[i]=tensors[i]->view_offs%256;
            c.indexed_binding_bytes[i]=(tensor_span(tensors[i])+c.indexed_prefix_bytes[i]+3)/4*4;
            c.indexed_buffer_remaining[i]=c.indexed_binding_bytes[i];
        }
    }
    uint32_t select()const{return ggml_webgpu_iq3s_id_subgroup_size(c);}
};
static const ggml_webgpu_mul_mat_vec_shader_decisions & decision(const webgpu_pipeline & p){
    return *static_cast<ggml_webgpu_mul_mat_vec_shader_decisions *>(p.context.get());
}
int main(int argc,char ** argv){
    try{
        for(uint32_t sg:{16u,32u,64u})for(int k:{2048,2304,2560,2816,4096})
        for(int m:{256,257,639,640,641})for(bool shared:{false,true})for(bool padded:{false,true}){
            Fixture f;f.c.min_subgroup_size=f.c.max_subgroup_size=sg;f.layout(k,m,11,10,shared,padded);
            require(f.select()==sg,"eligible IQ3_S shape rejected");
        }
        for(int64_t k:{int64_t(-256),int64_t(0),int64_t(256),int64_t(512),int64_t(1792),int64_t(2047),int64_t(2049),int64_t(4097),int64_t(4352),INT64_MAX}){
            Fixture f;f.a.ne[0]=f.x.ne[0]=k;require(f.select()==0,"K fallback");
        }
        for(int64_t m:{int64_t(-1),int64_t(0),int64_t(7),int64_t(255),int64_t(UINT32_MAX),INT64_MAX}){
            Fixture f;f.a.ne[1]=f.dst.ne[0]=m;require(f.select()==0,"M fallback");
        }
        for(const auto & limits:std::vector<std::pair<uint32_t,uint32_t>>{{0,0},{4,4},{8,8},{15,15},{16,32},{32,16},{32,64},{64,128},{128,128}}){
            Fixture f;f.c.min_subgroup_size=limits.first;f.c.max_subgroup_size=limits.second;require(f.select()==0,"subgroup fallback");
        }
        {Fixture f;f.c.supports_subgroups=false;require(f.select()==0,"no subgroup support fallback");}
        {Fixture f;f.c.max_wg_size=127;require(f.select()==0,"small workgroup limit fallback");}
        {Fixture f;f.c.max_wg_size=128;require(f.select()==16,"exact workgroup limit admission");}
        {Fixture f;f.c.max_workgroups_per_dim=0;require(f.select()==0,"unknown grid limit fallback");}
        {Fixture f;f.c.max_workgroups_per_dim=1;require(f.select()==0,"grid capacity fallback");}
        {Fixture f;f.layout(2048,256,11,1,true,false);f.c.max_workgroups_per_dim=3;require(f.select()==16,"grid square boundary admission");
            f.layout(2048,320,11,1,true,false);require(f.select()==0,"grid square overflow fallback");}
        {Fixture f;f.c.max_storage_binding_bytes=0;require(f.select()==0,"unknown binding limit fallback");}
        {Fixture f;f.layout(2560,640,512,10,false,true);require(f.select()==16,"large expert tensor shape admission");}
        for(auto type:{GGML_TYPE_Q5_1,GGML_TYPE_IQ4_NL,GGML_TYPE_Q8_0,GGML_TYPE_F32,GGML_TYPE_F16}){
            Fixture f;f.a.type=type;require(f.select()==0,"weight type fallback");
        }
        {Fixture f;f.x.type=GGML_TYPE_F16;require(f.select()==0,"F16 input fallback");}
        {Fixture f;f.dst.type=GGML_TYPE_F16;require(f.select()==0,"F16 output fallback");}
        {Fixture f;f.ids.type=GGML_TYPE_F32;require(f.select()==0,"IDs type fallback");}
        {Fixture f;f.c.src0=nullptr;require(f.select()==0,"null matrix fallback");}
        {Fixture f;f.c.src1=nullptr;require(f.select()==0,"null input fallback");}
        {Fixture f;f.c.src2=nullptr;require(f.select()==0,"ordinary context fallback");}
        {Fixture f;f.c.dst=nullptr;require(f.select()==0,"null output fallback");}
        {Fixture f;f.x.ne[0]++;require(f.select()==0,"input K mismatch fallback");}
        {Fixture f;f.dst.ne[0]++;require(f.select()==0,"output M mismatch fallback");}
        {Fixture f;f.ids.ne[0]--;require(f.select()==0,"ID count mismatch fallback");}
        {Fixture f;f.x.ne[1]=2;require(f.select()==0,"input slot shape fallback");}
        for(int64_t count:{int64_t(-1),int64_t(0),int64_t(UINT32_MAX)+1,INT64_MAX}){
            Fixture f;f.a.ne[2]=count;require(f.select()==0,"invalid expert count fallback");
            Fixture g;g.dst.ne[1]=g.ids.ne[0]=count;require(g.select()==0,"invalid active count fallback");
        }
        {Fixture f;f.dst.ne[1]=f.ids.ne[0]=12;require(f.select()==0,"active exceeds experts fallback");}
        for(auto * label:{"x2","x3","a3","ids1","ids2","ids3","dst2","dst3"}){
            Fixture f;
            if(std::string(label)=="x2")f.x.ne[2]=2;else if(std::string(label)=="x3")f.x.ne[3]=2;
            else if(std::string(label)=="a3")f.a.ne[3]=2;else if(std::string(label)=="ids1")f.ids.ne[1]=2;
            else if(std::string(label)=="ids2")f.ids.ne[2]=2;else if(std::string(label)=="ids3")f.ids.ne[3]=2;
            else if(std::string(label)=="dst2")f.dst.ne[2]=2;else f.dst.ne[3]=2;
            require(f.select()==0,"multiple token/dimension fallback");
        }
        {Fixture f;f.a.nb[0]=4;require(f.select()==0,"quant block stride fallback");}
        {Fixture f;f.a.nb[1]++;require(f.select()==0,"unaligned weight row stride fallback");}
        {Fixture f;f.a.nb[2]++;require(f.select()==0,"unaligned expert stride fallback");}
        {Fixture f;f.a.nb[1]=110;require(f.select()==0,"overlapping weight row fallback");}
        {Fixture f;f.a.nb[2]=110;require(f.select()==0,"overlapping expert fallback");}
        {Fixture f;f.a.nb[1]=std::numeric_limits<size_t>::max();require(f.select()==0,"weight row stride overflow fallback");}
        {Fixture f;f.a.nb[2]=std::numeric_limits<size_t>::max();require(f.select()==0,"expert stride overflow fallback");}
        {Fixture f;f.x.nb[0]=8;require(f.select()==0,"input scalar stride fallback");}
        {Fixture f;f.x.nb[1]++;require(f.select()==0,"input row alignment fallback");}
        {Fixture f;f.x.nb[2]=(uint64_t(UINT32_MAX)+1)*4;require(f.select()==0,"input token stride overflow fallback");}
        {Fixture f;f.x.nb[1]=4;require(f.select()==0,"overlapping input row fallback");}
        {Fixture f;f.ids.nb[0]=8;require(f.select()==0,"ID element stride fallback");}
        {Fixture f;f.dst.nb[1]+=4;require(f.select()==0,"nonpacked output fallback");}
        {Fixture f;f.a.ne[2]=UINT32_MAX;require(f.select()==0,"expert extent overflow fallback");}
        {Fixture f;f.dst.ne[1]=f.ids.ne[0]=f.a.ne[2]=UINT32_MAX;require(f.select()==0,"output extent overflow fallback");}
        for(unsigned i=0;i<4;++i){
            {Fixture f;f.c.indexed_prefix_bytes[i]++;require(f.select()==0,"prefix alignment fallback");}
            {Fixture f;f.c.indexed_prefix_bytes[i]=UINT64_MAX;require(f.select()==0,"prefix overflow fallback");}
            {Fixture f;f.c.indexed_binding_bytes[i]--;require(f.select()==0,"binding extent fallback");}
            {Fixture f;f.c.indexed_buffer_remaining[i]--;require(f.select()==0,"physical buffer extent fallback");}
            {Fixture f;f.c.max_storage_binding_bytes=f.c.indexed_binding_bytes[i]-1;require(f.select()==0,"device binding cap fallback");}
        }
        for(bool tuned_first:{false,true}){
            Fixture f;RecordingLibrary lib;
            auto get=[&](bool tuned){f.c.min_subgroup_size=f.c.max_subgroup_size=tuned?16:8;return lib.get_mul_mat_id_vec_pipeline(f.c);};
            const auto first=get(tuned_first),second=get(!tuned_first);
            require(first.id!=second.id,"tuned/fallback cache identity");
            require(get(tuned_first).id==first.id&&get(!tuned_first).id==second.id,"mixed order cache reuse");
        }
        Fixture f;RecordingLibrary lib;std::vector<unsigned> pipelines;
        if(argc>=2)std::filesystem::create_directories(argv[1]);
        for(uint32_t sg:{16u,32u,64u}){
            f.c.min_subgroup_size=f.c.max_subgroup_size=sg;
            const auto p=lib.get_mul_mat_id_vec_pipeline(f.c);
            require(std::find(pipelines.begin(),pipelines.end(),p.id)==pipelines.end(),"fixed subgroup cache identity");pipelines.push_back(p.id);
            require(decision(p).wg_size==128&&decision(p).outputs_per_wg==4*(128/sg),"subgroup decisions");
            require(p.name=="mul_mat_id_vec_iq3_s_f32_independent_sg"+std::to_string(sg)+"_wg128","profile identity");
            require(p.wgsl.find("@workgroup_size(128)")!=std::string::npos,"hardware workgroup specialization");
            require(p.wgsl.find("let num_block_groups: u32 = "+std::to_string(sg)+" / 16")!=std::string::npos,"accumulator block stride specialization");
            require(p.wgsl.find("var<workgroup>")==std::string::npos&&p.wgsl.find("workgroupBarrier")==std::string::npos,"shared reduction or LUT introduced");
            require(p.wgsl.find("accumulate_vec_dot(subgroup_invocation_id,")!=std::string::npos,"subgroup lane assignment");
            require(p.wgsl.find("if (num_subgroups == expected_subgroups)")!=std::string::npos,"full-packed fast path");
            require(p.wgsl.find("let active_count = subgroupAdd(1u)")!=std::string::npos,"partial active width");
            require(p.wgsl.find("virtual_tid += active_count")!=std::string::npos,"partial logical lane completion");
            require(p.wgsl.find("if (owns_rows && subgroup_invocation_id == 0u")!=std::string::npos,"extra subgroup output guard");
            require(lib.get_mul_mat_id_vec_pipeline(f.c).id==p.id,"repeat subgroup cache hit");
            if(argc>=2)std::ofstream(std::filesystem::path(argv[1])/("iq3s-independent-sg"+std::to_string(sg)+".wgsl"))<<p.wgsl;
        }
        f.c.min_subgroup_size=f.c.max_subgroup_size=8;
        const auto baseline=lib.get_mul_mat_id_vec_pipeline(f.c);
        require(decision(baseline).wg_size==256&&decision(baseline).outputs_per_wg==4,"baseline decisions retained");
        require(baseline.name=="mul_mat_id_vec_iq3_s_f32_sg_reduce","baseline profile retained");
        ReferenceLibrary reference;
        require(reference.get_mul_mat_id_vec_pipeline(f.c).wgsl==baseline.wgsl,"fallback differs from actual pristine factory");
        if(argc>=2)std::ofstream(std::filesystem::path(argv[1])/"iq3s-baseline.wgsl")<<baseline.wgsl;
        ggml_webgpu_mul_mat_id_pipeline_key key0{},key16{};key0.workgroup_size=key16.workgroup_size=128;key16.subgroup_size=16;
        require(!(key0==key16),"same-WG subgroup key equality");
        std::unordered_map<ggml_webgpu_mul_mat_id_pipeline_key,int,ggml_webgpu_mul_mat_id_pipeline_key_hash> keys;
        keys[key0]=1;keys[key16]=2;require(keys.size()==2&&keys[key0]==1&&keys[key16]==2,"same-WG subgroup key hashing");
        for(uint32_t sg:{16u,32u,64u})for(uint32_t m:{639u,640u,641u})for(bool shared:{false,true}){
            const uint32_t rows=4*(128/sg),groups=(m+rows-1)/rows,total=groups*10;
            uint32_t nx=0,ny=0;compute_2d_workgroups(total,64,nx,ny);
            std::vector<unsigned> visits(m*10);
            const uint32_t selected[12]={999,998,10,3,7,0,9,1,8,2,6,10};
            for(uint32_t linear=0;linear<nx*ny;++linear){
                if(linear>=total)continue;
                const uint32_t slot=linear/groups,expert=selected[2+slot];
                require(slot<10&&expert<11,"ID slot/tail address");
                for(uint32_t subgroup=0;subgroup<128/sg;++subgroup)for(uint32_t row=0;row<4;++row){
                    const uint32_t output=(linear%groups)*rows+subgroup*4+row;
                    if(output>=m)continue;
                    const uint64_t a=1+expert*uint64_t(m*11+3)+output*11;
                    const uint64_t x=3+(slot%(shared?1:10))*2565;
                    const uint64_t dst=7+slot*m+output;
                    require(a==1+selected[2+slot]*uint64_t(m*11+3)+output*11&&
                            x==3+(shared?0:slot)*2565&&dst<7+m*10,"indexed broadcast/duplicate address");
                    ++visits[dst-7];
                }
            }
            for(unsigned count:visits)require(count==1,"indexed output written more or less than once");
        }
        for(uint32_t sg:{4u,8u,16u,32u,64u})for(int k:{1024,2048,2560,4096,4352})
        for(int m:{255,256,640,641})for(int slots:{1,10})for(bool shared:{false,true})for(bool padded:{false,true}){
            Fixture shape;shape.layout(k,m,11,slots,shared,padded);
            webgpu_context ctx=std::make_shared<RecordedContext>();auto & cap=ctx->global_ctx->capabilities;
            cap.min_subgroup_size=cap.max_subgroup_size=sg;cap.limits.maxComputeWorkgroupsPerDimension=64;
            buffer_size_queries=0;
            const auto op=ggml_webgpu_mul_mat_id_vec(ctx,&shape.a,&shape.x,&shape.ids,&shape.dst);
            const bool admitted=sg>=16&&k>=2048&&k<=4096&&m>=256;
            const uint32_t rows=admitted?4*(128/sg):4;
            require(decision(op.pipeline).wg_size==(admitted?128u:256u),"dispatcher selection");
            require(decision(op.pipeline).outputs_per_wg==rows,"dispatcher row decisions");
            require(buffer_size_queries==(admitted?4u:0u),"ineligible bounds queries added");
            require(op.params==std::vector<uint32_t>({padded?1u:0u,padded?3u:0u,padded?2u:0u,padded?7u:0u,
                uint32_t(k),uint32_t(m),11,uint32_t(slots),shared?1u:uint32_t(slots),uint32_t(k/256+(padded?1:0)),
                uint32_t(k+(padded?5:0)),uint32_t(m*(k/256+(padded?1:0))+(padded?3:0)),uint32_t((shared?1:slots)*(k+(padded?5:0))+(padded?3:0))}),"actual uniform layout");
            const uint32_t groups=((m+rows-1)/rows)*slots;
            require(op.y==CEIL_DIV(groups,64u)&&op.x==CEIL_DIV(groups,op.y)&&op.x*op.y>=groups,"actual dispatch grid");
            require(op.entries.size()==4&&op.entries[0].binding==0&&op.entries[3].binding==3,"actual bindings");
        }
        for(auto type:{GGML_TYPE_Q5_1,GGML_TYPE_IQ4_NL}){
            Fixture shape;shape.a.type=type;shape.layout(320,640,11,10,false,true);
            webgpu_context ctx=std::make_shared<RecordedContext>();buffer_size_queries=0;
            const auto op=ggml_webgpu_mul_mat_id_vec(ctx,&shape.a,&shape.x,&shape.ids,&shape.dst);
            require(decision(op.pipeline).wg_size==128&&decision(op.pipeline).outputs_per_wg==4,"existing expert-short fallback changed");
            require(buffer_size_queries==0,"non-IQ3 binding queries added");
        }
        {Fixture shape;webgpu_context ctx=std::make_shared<RecordedContext>();physical_override=1;
            const auto op=ggml_webgpu_mul_mat_id_vec(ctx,&shape.a,&shape.x,&shape.ids,&shape.dst);physical_override=0;
            require(decision(op.pipeline).wg_size==256,"actual physical bounds not enforced");}
        {Fixture shape;webgpu_context ctx=std::make_shared<RecordedContext>();ctx->global_ctx->capabilities.limits.maxStorageBufferBindingSize=1;
            const auto op=ggml_webgpu_mul_mat_id_vec(ctx,&shape.a,&shape.x,&shape.ids,&shape.dst);
            require(decision(op.pipeline).wg_size==256,"actual device binding limit not enforced");}
        if(argc==4){
            const uint32_t sg=static_cast<uint32_t>(std::stoul(argv[3]));
            require(sg==16||sg==32||sg==64,"invalid profile subgroup");
            std::ifstream profiles(argv[2]);require(bool(profiles),"cannot read fixture profiles");
            std::ofstream selected(std::filesystem::path(argv[1])/"case-decisions.json");selected<<"[\n";
            std::string name;bool first=true;
            while(profiles>>name){
                uint32_t p[13];for(auto & value:p)require(bool(profiles>>value),"invalid fixture profile params");
                uint64_t bytes[3];for(auto & value:bytes)require(bool(profiles>>value),"invalid fixture profile bytes");
                Fixture shape;shape.layout(p[4],p[5],p[6],p[7],p[8]==1,p[0]!=0);
                webgpu_context ctx=std::make_shared<RecordedContext>();
                ctx->global_ctx->capabilities.min_subgroup_size=ctx->global_ctx->capabilities.max_subgroup_size=sg;
                const auto op=ggml_webgpu_mul_mat_id_vec(ctx,&shape.a,&shape.x,&shape.ids,&shape.dst);
                require(op.params==std::vector<uint32_t>(p,p+13),"profile params differ from actual dispatcher");
                for(unsigned i=0;i<3;++i)require(op.entries[i].size<=bytes[i],"profile buffer shorter than actual binding");
                const auto & d=decision(op.pipeline);
                const bool tuned=d.wg_size==128;
                if(!first)selected<<",\n";
                first=false;
                selected<<"  {\"name\":\""<<name<<"\",\"selectedFile\":\""
                    <<(tuned?"iq3s-independent-sg"+std::to_string(sg)+".wgsl":"iq3s-baseline.wgsl")
                    <<"\",\"selectedName\":\""<<op.pipeline.name<<"\",\"wgSize\":"<<d.wg_size
                    <<",\"outputsPerWg\":"<<d.outputs_per_wg<<"}";
            }
            selected<<"\n]\n";
        }
        std::cout<<checks<<" actual IQ3_S selector/cache/factory/dispatch checks passed; "<<created<<" recorded pipelines\n";
    }catch(const std::exception & e){std::cerr<<e.what()<<'\n';return 1;}
}
