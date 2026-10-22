static size_t checks = 0;
static void require(bool condition, const char * message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
int main(int argc, char ** argv) {
    try {
        ggml_tensor a{}, b{}, dst{};
        a.type = GGML_TYPE_Q8_0; b.type = GGML_TYPE_F32; dst.type = GGML_TYPE_F32;
        for (int i=0; i<4; ++i) a.ne[i]=b.ne[i]=dst.ne[i]=1;
        ggml_webgpu_shader_lib_context c{};
        c.src0=&a; c.src1=&b; c.dst=&dst;
        c.max_wg_size=256; c.supports_subgroups=true;
        c.min_subgroup_size=8; c.max_subgroup_size=8;
        a.ne[0]=b.ne[0]=320; a.ne[1]=dst.ne[0]=10240;
        for (const uint32_t sg : {4u,8u,16u,32u,64u,128u}) {
            c.min_subgroup_size=c.max_subgroup_size=sg;
            for (int64_t k=32;k<=1024;k+=32) {
                a.ne[0]=b.ne[0]=k;
                const uint32_t bucket=k<=256?32:k<=512?64:128;
                const uint32_t expected=std::max(bucket,sg);
                require(ggml_webgpu_q8_short_workgroup(c,false,false)==expected,"workgroup bucket");
                // Every useful lane owns exactly its original block and eight elements.
                for (uint32_t lane=0;lane<expected;++lane) {
                    for (uint32_t block=lane/4;block<static_cast<uint32_t>(k/32);block+=expected/4) {
                        require(block==lane/4,"lane visited a different block");
                        for (uint32_t j=0;j<8;++j) {
                            const uint32_t element=block*32+(lane%4)*8+j;
                            require(element==lane*8+j && element<static_cast<uint32_t>(k),"element mapping");
                        }
                    }
                }
            }
        }
        c.min_subgroup_size=c.max_subgroup_size=32;
        a.ne[0]=b.ne[0]=320;
        for (const int64_t m : {1,1023,1024,1025,10240}) {
            dst.ne[0]=a.ne[1]=m;
            require(ggml_webgpu_q8_short_workgroup(c,false,false)==(m<1024?256:64),"output rows admission");
        }
        dst.ne[0]=a.ne[1]=10240;
        for (const int64_t k : {int64_t(-32),int64_t(0),int64_t(1),int64_t(31),int64_t(33),int64_t(1025),INT64_MAX}) {
            a.ne[0]=k;require(ggml_webgpu_q8_short_workgroup(c,false,false)==256,"invalid/long K fallback");
        }
        a.ne[0]=320;
        for (const auto t : {GGML_TYPE_F32,GGML_TYPE_F16,GGML_TYPE_Q4_0,GGML_TYPE_IQ4_XS}) {
            a.type=t;require(ggml_webgpu_q8_short_workgroup(c,false,false)==256,"other weights fallback");
        }
        a.type=GGML_TYPE_Q8_0;b.type=GGML_TYPE_F16;
        require(ggml_webgpu_q8_short_workgroup(c,false,false)==256,"F16 input fallback");b.type=GGML_TYPE_F32;
        for (int n=2;n<=5;++n) {dst.ne[1]=n;require(ggml_webgpu_q8_short_workgroup(c,false,false)==256,"multiple columns fallback");}
        dst.ne[1]=1;
        require(ggml_webgpu_q8_short_workgroup(c,true,false)==256,"integer-dot fallback");
        require(ggml_webgpu_q8_short_workgroup(c,false,true)==256,"overlap fallback");
        c.supports_subgroups=false;require(ggml_webgpu_q8_short_workgroup(c,false,false)==256,"workgroup reduction fallback");c.supports_subgroups=true;
        for (auto limits : std::vector<std::pair<uint32_t,uint32_t>>{{0,0},{0,32},{2,32},{32,0},{32,16},{8,24},{24,32},{32,256}}) {
            c.min_subgroup_size=limits.first;c.max_subgroup_size=limits.second;
            require(ggml_webgpu_q8_short_workgroup(c,false,false)==256,"invalid subgroup range");
        }
        c.min_subgroup_size=4;c.max_subgroup_size=128;
        require(ggml_webgpu_q8_short_workgroup(c,false,false)==128,"full-subgroup floor");
        c.max_wg_size=64;require(ggml_webgpu_q8_short_workgroup(c,false,false)==256,"unsupported workgroup fallback");
        c.max_wg_size=256;c.min_subgroup_size=c.max_subgroup_size=8;
        RecordingLibrary lib;
        auto get = [&](int64_t k, int64_t m=10240, int64_t n=1, bool overlap=false) {
            a.ne[0]=b.ne[0]=k; a.ne[1]=dst.ne[0]=m; b.ne[1]=dst.ne[1]=n;
            return lib.get_mul_mat_vec_pipeline(c,overlap);
        };
        const auto p64=get(320),p128=get(640),p256=get(2560),p32=get(256);
        for (const auto & p : {p32,p64,p128,p256}) {
            const auto decision=static_cast<ggml_webgpu_mul_mat_vec_shader_decisions *>(p.context.get());
            require(p.wgsl.find("@workgroup_size("+std::to_string(decision->wg_size)+")")!=std::string::npos,"compiled size differs from decision");
            require(decision->outputs_per_wg==4,"output tiling changed");
        }
        require(p64.id!=p128.id && p128.id!=p256.id && p32.id!=p64.id,"distinct shape pipeline identity");
        require(get(320).id==p64.id && get(384).id==p64.id && get(640).id==p128.id,"repeat/bucket cache hit");
        require(get(320,1023).id==p256.id,"fallback must not reuse short pipeline");
        require(get(320,10240,2).id!=p64.id,"column cache distinction");
        require(get(320,10240,1,true).id!=p64.id,"overlap cache distinction");
        require(p64.name=="mul_mat_vec_q8_0_f32_sg_reduce_wg64","profile suffix");
        require(p256.name=="mul_mat_vec_q8_0_f32_sg_reduce","default profile label preserved");
        b.type=GGML_TYPE_F16;
        require(static_cast<ggml_webgpu_mul_mat_vec_shader_decisions *>(get(320).context.get())->wg_size==256,"factory F16 exclusion");
        b.type=GGML_TYPE_F32;a.type=GGML_TYPE_Q4_0;
        require(static_cast<ggml_webgpu_mul_mat_vec_shader_decisions *>(get(320).context.get())->wg_size==256,"factory other quant exclusion");
        a.type=GGML_TYPE_Q8_0;
        require(get(320).id==p64.id,"type fallback must not pollute Q8 cache");
        c.vendor="nvidia";c.supports_dot_product=true;
        const auto integer_dot=get(320);
        require(static_cast<ggml_webgpu_mul_mat_vec_shader_decisions *>(integer_dot.context.get())->wg_size==256,"factory integer-dot exclusion");
        require(integer_dot.id!=p64.id,"integer-dot cache distinction");
        if (argc==2) {
            std::filesystem::create_directories(argv[1]);
            for (const auto & p : {p32,p64,p128,p256}) {
                const auto decision=static_cast<ggml_webgpu_mul_mat_vec_shader_decisions *>(p.context.get());
                std::ofstream(std::filesystem::path(argv[1])/("q8-"+std::to_string(decision->wg_size)+".wgsl"))<<p.wgsl;
            }
        }
        std::cout << checks << " actual selector/cache/factory checks passed; " << created << " recorded pipelines\n";
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
