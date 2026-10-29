static size_t checks=0;
static void check(bool value,const char * message) {++checks;if(!value)throw std::runtime_error(message);}
struct Fixture {
    ggml_backend_webgpu_buffer_context ac{{11}},xc{{12}},dc{{13}},ic{{14}};
    ggml_backend_buffer ab{&ac},xb{&xc},db{&dc},ib{&ic};
    ggml_tensor ar{},xr{},dr{},ir{},a{},x{},dst{},ids{};
    bool indexed;
    Fixture(int mode,ggml_type at,ggml_type xt,int64_t k):indexed(mode>=2) {
        for(auto * t:{&a,&x,&dst,&ids}) for(auto & n:t->ne)n=1;
        a.type=at;x.type=xt;dst.type=GGML_TYPE_F32;ids.type=GGML_TYPE_I32;
        a.ne[0]=x.ne[0]=k;a.ne[1]=dst.ne[0]=(mode==0||mode==2)?33:4;
        if(indexed) {a.ne[2]=3;x.ne[1]=dst.ne[1]=ids.ne[0]=2;x.ne[2]=dst.ne[2]=ids.ne[1]=mode==2?1:2;}
        else {x.ne[1]=dst.ne[1]=mode==0?1:5;a.ne[2]=2;x.ne[2]=dst.ne[2]=4;a.ne[3]=2;x.ne[3]=dst.ne[3]=2;}
        dense();
        ar.data=(void*)uintptr_t(0x1000+4096);xr.data=(void*)uintptr_t(0x1000+8192);
        dr.data=(void*)uintptr_t(0x1000+12288);ir.data=(void*)uintptr_t(0x1000+16384);
        a.view_src=&ar;x.view_src=&xr;dst.view_src=&dr;ids.view_src=&ir;
        a.buffer=&ab;x.buffer=&xb;dst.buffer=&db;ids.buffer=&ib;
        dst.src[0]=&a;dst.src[1]=&x;dst.src[2]=&ids;
    }
    void dense() {
        for(auto * t:{&a,&x,&dst,&ids}) {t->nb[0]=ggml_type_size(t->type);t->nb[1]=t->ne[0]/ggml_blck_size(t->type)*t->nb[0];for(int i=2;i<4;++i)t->nb[i]=t->nb[i-1]*t->ne[i-1];}
    }
    void scenario(int id) {
        if(id>=1&&id<=3)a.view_offs=id*ggml_type_size(a.type);
        else if(id>=4&&id<=6)x.view_offs=(id-3)*ggml_type_size(x.type);
        else if(id>=7&&id<=9)dst.view_offs=(id-6)*4;
        else if(id>=10&&id<=15) {auto * t=id<13?&a:&x;const int axis=(id-10)%3+1;t->nb[axis]+=ggml_type_size(t->type);}
        else if(id==16) {ar.data=(void*)(uintptr_t(ar.data)+ggml_type_size(a.type));a.view_offs=0;}
        else if(id==17) {xr.data=(void*)(uintptr_t(xr.data)+ggml_type_size(x.type));x.view_offs=0;}
        else if(id==18) { // All higher strides are unused on this vector graph.
            a.ne[1]=dst.ne[0]=1;a.ne[2]=a.ne[3]=x.ne[1]=x.ne[2]=x.ne[3]=dst.ne[1]=dst.ne[2]=dst.ne[3]=1;
            ids.ne[0]=ids.ne[1]=1;dense();for(int i=1;i<4;++i) {a.nb[i]+=ggml_type_size(a.type);x.nb[i]+=ggml_type_size(x.type);}
        } else if(id==19) { // Shared input buffer, with nonzero merged-relative bases.
            x.buffer=&ab;xr.data=ar.data;x.view_offs=16;a.view_offs=4;
        } else if(id==20) {x.buffer=&ab;xr.data=ar.data;x.view_offs=32;a.view_offs=16;}
        else if(id==21&&indexed) {a.ne[2]=11;x.ne[1]=dst.ne[1]=ids.ne[0]=4;dense();a.view_offs=ggml_type_size(a.type);x.view_offs=3*ggml_type_size(x.type);dst.view_offs=ids.view_offs=12;a.nb[1]=33*ggml_type_size(a.type);a.nb[2]=1092*ggml_type_size(a.type);x.nb[1]=37*ggml_type_size(x.type);x.nb[2]=151*ggml_type_size(x.type);}
        else if(id==22) {x.ne[1]=dst.ne[1]=4;dense();}
        else if(id==23) {x.ne[1]=1;dense();x.nb[1]+=ggml_type_size(x.type);}
        else if(id==24)a.view_offs=ggml_type_size(a.type);
    }
};
static uint64_t payload(const ggml_tensor& t) {return uintptr_t(t.view_src->data)-0x1000+t.view_offs;}
// Independent address oracle: enumerate actual logical starts, instead of
// reproducing the production stride predicate or assuming an aligned root.
static bool starts_aligned(const ggml_tensor& t,uint64_t binding_offset) {
    const auto size=ggml_type_size(t.type);
    for(int64_t i3=0;i3<t.ne[3];++i3)for(int64_t i2=0;i2<t.ne[2];++i2)for(int64_t i1=0;i1<t.ne[1];++i1)
        if(((payload(t)-binding_offset+i1*t.nb[1]+i2*t.nb[2]+i3*t.nb[3])/size)%4)return false;
    return true;
}
static void vector_json(const std::vector<uint32_t>& values) {std::cout<<'[';for(size_t i=0;i<values.size();++i){if(i)std::cout<<',';std::cout<<values[i];}std::cout<<']';}
int main(int argc,char **argv) {
    try {
        check(argc==2,"output directory required");std::filesystem::create_directories(argv[1]);
        auto old=std::make_shared<baseline::Context>();auto actual=std::make_shared<candidate::Context>();
        size_t count=0,changed=0,aligned=0,unused=0,overlaps=0,mmvq=0,sg_matrix=0,nonfloat=0;
        std::unordered_map<std::string,std::string> saved;
        std::cout<<"{\"rows\":[";
        for(int order:{0,1})for(int mode=0;mode<4;++mode)for(auto at:{GGML_TYPE_F32,GGML_TYPE_F16,GGML_TYPE_Q8_0,GGML_TYPE_BF16})for(auto xt:{GGML_TYPE_F32,GGML_TYPE_F16}) {
            if(at!=GGML_TYPE_F16&&xt==GGML_TYPE_F16)continue;
            for(int64_t k:{31,32,33,128})for(size_t alignment:{4u,8u,16u,256u})for(int item=0;item<27;++item) {
                const int scenario=order?26-item:item;
                if(ggml_is_quantized(at)&&k%ggml_blck_size(at))continue;
                if(scenario==18&&(mode==1||mode==3))continue;
                if((scenario==19||scenario==20)&&(mode>=2||at!=xt))continue;
                if(scenario==21&&mode!=2)continue;
                if(scenario==22&&mode!=0)continue;
                if(scenario==23&&mode!=2)continue;
                if((scenario==24||scenario==25)&&mode!=1)continue;
                if(scenario==26&&(mode!=0||at!=GGML_TYPE_Q8_0))continue;
                Fixture f(mode,at,xt,k);f.scenario(scenario);
                actual->global_ctx->capabilities.limits.minStorageBufferOffsetAlignment=alignment;
                old->global_ctx->capabilities.limits.minStorageBufferOffsetAlignment=alignment;
                const auto set_caps=[&](auto& caps) {caps.supports_subgroup_matrix=scenario==24||scenario==25;caps.supports_dot_product=scenario==26;};
                set_caps(old->global_ctx->capabilities);set_caps(actual->global_ctx->capabilities);
                old->global_ctx->vendor=actual->global_ctx->vendor=scenario==26?"nvidia":"Recording";
                const auto before=baseline::run(old,&f.a,&f.x,&f.ids,&f.dst,f.indexed);
                const auto after=candidate::run(actual,&f.a,&f.x,&f.ids,&f.dst,f.indexed);
                const auto again=candidate::run(actual,&f.a,&f.x,&f.ids,&f.dst,f.indexed);
                check(before.size()==after.size(),"dispatch count changed");
                for(size_t i=0;i<before.size();++i) {check(before[i].params==after[i].params,"uniform bytes changed");check(before[i].bind_group_entries==after[i].bind_group_entries,"binding changed");check(before[i].workgroups==after[i].workgroups,"grid changed");check(after[i].pipeline.id==again[i].pipeline.id,"cache repeat changed");}
                const auto & d=after.back();const auto & entries=d.bind_group_entries;
                const bool matrix=mode==1||mode==3;
                const bool merged=!f.indexed&&entries.size()==2;
                const size_t bx=merged?0:1,bd=f.indexed?(matrix?2:3):(merged?1:2);
                const auto oa=(payload(f.a)-entries[0].offset)/ggml_type_size(at);
                const auto ox=(payload(f.x)-entries[bx].offset)/ggml_type_size(xt);
                const auto od=(payload(f.dst)-entries[bd].offset)/4;
                check(d.params[0]==oa&&(scenario==26||d.params[1]==ox),"wire offsets disagree with real binding window");
                check(d.params[f.indexed&&!matrix?3:2]==od,"destination wire offset disagrees with binding window");
                const bool expected=(at==GGML_TYPE_F32||at==GGML_TYPE_F16)&&f.a.ne[0]%4==0&&(!matrix||f.a.ne[1]%4==0)&&starts_aligned(f.a,entries[0].offset)&&starts_aligned(f.x,entries[bx].offset)&&(!matrix||od%4==0);
                const bool selected=d.pipeline.name.find("_vectorized")!=std::string::npos;
                check(selected==expected,"production factory/caller selected incorrect layout");
                const bool old_vector=before.back().pipeline.name.find("_vectorized")!=std::string::npos;
                if(expected||!old_vector)check(d.pipeline.wgsl==before.back().pipeline.wgsl,"unchanged path shader drift");
                if(expected)++aligned;
                if(scenario==18&&expected)++unused;
                if(merged)++overlaps;
                if(scenario==26) {check(after.size()==2,"MMVQ route not exercised");++mmvq;}
                if(scenario==24||scenario==25)++sg_matrix;
                if(at==GGML_TYPE_Q8_0||at==GGML_TYPE_BF16)++nonfloat;
                if(old_vector&&!selected)++changed;
                auto found=saved.find(d.pipeline.wgsl);
                if(found==saved.end()) {const std::string file=std::to_string(saved.size())+".wgsl";std::ofstream(std::filesystem::path(argv[1])/file)<<d.pipeline.wgsl;found=saved.emplace(d.pipeline.wgsl,file).first;}
                const std::string& filename=found->second;
                if(count)std::cout<<',';
                std::cout<<"{\"mode\":"<<mode<<",\"scenario\":"<<scenario<<",\"alignment\":"<<alignment<<",\"vectorized\":"<<(selected?"true":"false")<<",\"label\":\""<<d.pipeline.name<<"\",\"shader\":\""<<filename<<"\",\"params\":";
                vector_json(d.params);std::cout<<",\"grid\":";vector_json(d.workgroups);std::cout<<'}';++count;
            }
        }
        check(changed>100&&aligned>100&&unused>0&&overlaps>0&&mmvq>0&&sg_matrix>0&&nonfloat>0,"layout coverage incomplete");
        std::cout<<"],\"cases\":"<<count<<",\"checks\":"<<checks<<",\"newScalarFallbacks\":"<<changed<<",\"aligned\":"<<aligned<<",\"unusedStride\":"<<unused<<",\"mergedBindings\":"<<overlaps<<",\"mmvq\":"<<mmvq<<",\"subgroupMatrix\":"<<sg_matrix<<",\"nonfloat\":"<<nonfloat<<"}\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
