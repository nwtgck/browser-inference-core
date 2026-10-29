static void numbers(const std::vector<uint32_t>& v){std::cout<<'[';for(size_t i=0;i<v.size();++i){if(i)std::cout<<',';std::cout<<v[i];}std::cout<<']';}
int main(int argc,char**argv){if(argc<19 || argc>21)throw std::runtime_error("19 args");
std::string out=argv[1],name=argv[2];uint32_t p[16];for(int i=0;i<16;++i)p[i]=std::stoul(argv[i+3]);
ggml_tensor a{},x{},ids{},d{};for(auto*t:{&a,&x,&ids,&d})for(auto&n:t->ne)n=1;
a.type=p[0]==0?GGML_TYPE_F32:p[0]==1?GGML_TYPE_Q8_0:GGML_TYPE_IQ3_S;x.type=(argc>=20 && std::string(argv[19])=="f16")?GGML_TYPE_F16:GGML_TYPE_F32;d.type=GGML_TYPE_F32;ids.type=GGML_TYPE_I32;
a.ne[0]=p[1];a.ne[1]=p[2];a.ne[2]=p[3];x.ne[0]=p[1];x.ne[1]=p[6];x.ne[2]=p[5];d.ne[0]=p[2];d.ne[1]=p[4];d.ne[2]=p[5];d.src[0]=&a;
a.view_offs=p[7]*ggml_type_size(a.type);x.view_offs=p[8]*ggml_type_size(x.type);d.view_offs=p[9]*4;ids.view_offs=p[10]*4;
a.nb[1]=p[11]*ggml_type_size(a.type);a.nb[2]=p[12]*ggml_type_size(a.type);x.nb[1]=p[13]*ggml_type_size(x.type);x.nb[2]=p[14]*ggml_type_size(x.type);ids.nb[1]=p[15]*4;
ggml_webgpu_shader_lib_context c{};c.src0=&a;c.src1=&x;c.src2=&ids;c.dst=&d;c.max_wg_size=argc==21?std::stoul(argv[20]):256;
Canonical gather_lib;auto gather=gather_lib.get_mul_mat_id_gather_pipeline(c);
Candidate changed_gather_lib;auto same_gather=changed_gather_lib.get_mul_mat_id_gather_pipeline(c);
auto gather_hit=changed_gather_lib.get_mul_mat_id_gather_pipeline(c);
if(gather.name!=same_gather.name||gather_hit.id!=same_gather.id)throw std::runtime_error("gather cache/name changed");
const auto expert_count=a.ne[2];a.ne[2]=expert_count==1?2:1;
auto other_experts=changed_gather_lib.get_mul_mat_id_gather_pipeline(c);a.ne[2]=expert_count;
if(other_experts.id==same_gather.id||changed_gather_lib.get_mul_mat_id_gather_pipeline(c).id!=same_gather.id)
    throw std::runtime_error("gather expert key/return cache changed");
const auto old_m=d.ne[0],old_used=d.ne[1],old_tokens=d.ne[2];
d.ne[0]=old_m+1;d.ne[1]=old_used+1;d.ne[2]=old_tokens+1;
if(changed_gather_lib.get_mul_mat_id_gather_pipeline(c).id!=same_gather.id)
    throw std::runtime_error("gather shape polluted expert cache");
d.ne[0]=old_m;d.ne[1]=old_used;d.ne[2]=old_tokens;
#ifdef TEST_LOGICAL32_GATHER
const auto old_zero=gather.code.find("    // the matmul writes only the rows that an expert owns");
const auto new_zero=same_gather.code.find("    let zero_lanes = min(32u,");
if(old_zero==std::string::npos||new_zero==std::string::npos||gather.code.substr(0,old_zero)!=same_gather.code.substr(0,new_zero)||gather.code==same_gather.code)
    throw std::runtime_error("gather changed outside zeroing or candidate was not selected");
#else
if(gather.code!=same_gather.code)throw std::runtime_error("historical gather changed");
#endif
// The primary output is the actual prepared factory. Keep the canonical
// baseline separately so a numerical runner cannot silently execute only it.
std::ofstream(out+"/"+name+"-gather.wgsl")<<same_gather.code;
std::ofstream(out+"/"+name+"-gather-baseline.wgsl")<<gather.code;
Canonical canonical;Candidate candidate;
auto *src0=&a,*src1=&x,*src2=&ids,*dst=&d;int ctx=0;
// @ACTUAL_LAYOUT@
auto plain=canonical.get_mul_mat_id_pipeline(c,vec4_aligned);auto changed=candidate.get_mul_mat_id_pipeline(c,vec4_aligned);
auto plain_hit=canonical.get_mul_mat_id_pipeline(c,vec4_aligned);auto changed_hit=candidate.get_mul_mat_id_pipeline(c,vec4_aligned);
if(plain.id!=plain_hit.id||changed.id!=changed_hit.id||plain.name!=changed.name)throw std::runtime_error("factory/cache mismatch");
std::string expected=plain.code;const std::string anchor="        if (is_valid) {\n            let k_end";
auto at=expected.find(anchor);if(at==std::string::npos||expected.find(anchor,at+1)!=std::string::npos)throw std::runtime_error("canonical anchor drift");
expected.replace(at,anchor.size(),"        if (is_valid && output_col_base < rest_token_n) {\n            let k_end");
if(changed.code!=expected)throw std::runtime_error("factory changed more than inactive-group calculation");
auto a_decisions=static_cast<ggml_webgpu_mul_mat_shader_decisions*>(plain.context.get());
auto b_decisions=static_cast<ggml_webgpu_mul_mat_shader_decisions*>(changed.context.get());
if(a_decisions->tile_k!=b_decisions->tile_k||a_decisions->tile_m!=b_decisions->tile_m||a_decisions->tile_n!=b_decisions->tile_n||a_decisions->wg_size!=b_decisions->wg_size)throw std::runtime_error("tiling changed");
std::ofstream(out+"/"+name+"-baseline.wgsl")<<plain.code;std::ofstream(out+"/"+name+"-candidate.wgsl")<<changed.code;
std::shared_ptr<void> main_context=changed.context;
uint32_t param_n_expert=src0->ne[2],param_n_expert_used=dst->ne[1],param_n_tokens=dst->ne[2];
// @ACTUAL_PARAMETER_VECTORS@
struct Caps {struct Limits {uint32_t maxComputeWorkgroupsPerDimension=17;}limits;};struct Global {Caps capabilities;};struct Context{Global*global_ctx;};Global global;Context context{&global};auto* context_ptr=&context;
auto*main_decisions=static_cast<ggml_webgpu_mul_mat_shader_decisions*>(main_context.get());uint32_t wg_x=1,wg_y=1;
// @ACTUAL_GRID@

std::cout<<"{\"gatherWorkgroupSize\":"<<c.max_wg_size<<",\"gatherParams\":";numbers(gather_params);std::cout<<",\"mainParams\":";numbers(main_params);std::cout<<",\"grid\":["<<wg_x<<','<<wg_y<<"],\"maxTiles\":"<<ggml_webgpu_mul_mat_id_max_tiles(dst)<<",\"tileK\":"<<main_decisions->tile_k<<",\"vectorized\":"<<(changed.name.find("_vectorized")!=std::string::npos?"true":"false")<<"}\n";
}
