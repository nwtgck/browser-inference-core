static unsigned checks=0;
static void check(bool value){++checks;if(!value)throw std::runtime_error("provider check "+std::to_string(checks));}
struct Fixture {
 webgpu_context ctx=std::make_shared<Context>();ggml_backend_webgpu_context bc{ctx};ggml_backend backend{};
 ggml_backend_webgpu_buffer_context buffers[3];ggml_backend_buffer owners[3]{};ggml_tensor a{},x{},dst{};
 Fixture(int k,int m,uint32_t sg=16,uint32_t prefix=0){
  ctx->global_ctx->capabilities.min_subgroup_size=ctx->global_ctx->capabilities.max_subgroup_size=sg;
  backend.guid=ggml_backend_webgpu_guid();backend.context=&bc;
  a.type=GGML_TYPE_Q8_0;x.type=dst.type=GGML_TYPE_F32;a.ne[0]=x.ne[0]=k;a.ne[1]=dst.ne[0]=m;
  for(int i=2;i<4;i++){a.ne[i]=1;}
  for(int i=1;i<4;i++){x.ne[i]=dst.ne[i]=1;}
  const uint64_t row=uint64_t(k/32)*34;
  a.nb[0]=34;a.nb[1]=row;a.nb[2]=a.nb[3]=row*m;
  x.nb[0]=dst.nb[0]=4;x.nb[1]=x.nb[2]=x.nb[3]=uint64_t(k)*4;dst.nb[1]=dst.nb[2]=dst.nb[3]=uint64_t(m)*4;
  ggml_tensor * tensors[3]={&a,&x,&dst};
  for(unsigned i=0;i<3;i++){buffers[i]={{i+1},ctx->global_ctx};owners[i].context=&buffers[i];owners[i].iface.get_base=ggml_backend_webgpu_buffer_get_base;owners[i].size=1ull<<30;tensors[i]->buffer=&owners[i];tensors[i]->data=reinterpret_cast<void *>(uintptr_t(0x1000+prefix));}
  dst.op=GGML_OP_MUL_MAT;dst.src[0]=&a;dst.src[1]=&x;
 }
 int run(uint32_t variant,uint32_t phase,uint32_t reps,ggml_webgpu_q8_diagnostic_report_v1 & out){
  ggml_webgpu_q8_diagnostic_request_v1 request{1,sizeof(request),variant,phase,reps,0};
  return ggml_backend_webgpu_q8_diagnostic_v1(&backend,&dst,&request,&out,sizeof(out));
 }
};
int main(){try{
 for(uint32_t sg:{4u,8u,16u,32u,64u})for(auto shape:{std::pair<int,int>{32,1},{320,10240},{640,2560},{2560,512},{2560,640},{2560,10240},{10240,320}})for(uint32_t variant:{0u,1u,2u,3u,4u,5u}){
  Fixture f(shape.first,shape.second,sg,256);ggml_webgpu_q8_diagnostic_report_v1 out;
  const unsigned created_before=created,dispatched=wgpu::dispatches;
  check(f.run(variant,2,1,out)==4 && created==created_before && wgpu::dispatches==dispatched);
  check(f.run(variant,1,0,out)==0 && out.dispatches==0 && wgpu::dispatches==dispatched);
  const auto key=RecordingLibrary::make_mul_mat_vec_key(ggml_webgpu_mat_context(f.ctx,&f.a,&f.x,&f.dst),false);
  const bool independent=(variant>=2&&variant<=4)||(variant==0&&key.subgroup_size);
  check(out.actual_algorithm==uint32_t(independent?2:1));
  const unsigned prepared=created;const auto digest=std::vector<uint8_t>(out.shader_sha256,out.shader_sha256+32);
  check(std::any_of(digest.begin(),digest.end(),[](uint8_t x){return x!=0;}));
  const uint32_t gx=out.grid_x,gy=out.grid_y;
  const auto begins=f.ctx->bind_group_cache.begins,ends=f.ctx->bind_group_cache.ends;
  const auto param_count=actual_params.size();
  check(f.run(variant,2,9,out)==0 && created==prepared && out.dispatches==9);
  check(wgpu::dispatches==dispatched+9 && !f.ctx->operation_in_progress && !f.ctx->active_command_encoder);
  check(f.ctx->bind_group_cache.begins==begins+1 && f.ctx->bind_group_cache.ends==ends+1);
  check(f.ctx->param_arena.cursor==0 && f.ctx->param_arena.resets>0 && f.ctx->param_arena.flushes>0);
  check(actual_params.size()==param_count+9 && out.grid_x==gx && out.grid_y==gy);
  for(size_t i=param_count;i<actual_params.size();i++)check(actual_params[i]==actual_params[param_count]);
  const auto & p=actual_params.back();
  check(p.size()==16 && p[0]==0 && p[1]==0 && p[2]==0 && p[3]==uint32_t(shape.second) && p[4]==1 && p[5]==uint32_t(shape.first));
  check(p[6]==uint32_t(shape.first/32) && p[12]==1 && p[13]==1 && p[14]==1 && p[15]==1);
  check(wgpu::grids.back()==std::make_pair(gx,gy));
 }
 {
  Fixture f(2560,640);ggml_webgpu_q8_diagnostic_report_v1 out;unsigned reentries=0;
  auto reenter=[&]{++reentries;check(f.ctx->operation_in_progress);ggml_webgpu_q8_diagnostic_report_v1 nested;check(f.run(1,1,0,nested)==5);check(f.run(1,2,1,nested)==5);ggml_cgraph graph{};check(ggml_backend_webgpu_graph_compute(&f.backend,&graph)==GGML_STATUS_FAILED);};
  prepare_hook=reenter;check(f.run(1,1,0,out)==0 && reentries==1);
  suspend_hook=[&]{check(!f.ctx->active_command_encoder);reenter();};check(f.run(1,2,1,out)==0 && reentries==2 && !f.ctx->operation_in_progress);
 }
 {
  Fixture f(2560,640);ggml_webgpu_q8_diagnostic_report_v1 out;const auto start=created,n=wgpu::dispatches;
  auto reject=[&]{check(f.run(2,1,0,out)==3 && created==start && wgpu::dispatches==n && !f.ctx->operation_in_progress);};
  f.ctx->global_ctx->capabilities.max_subgroup_size=32;reject();f.ctx->global_ctx->capabilities.max_subgroup_size=16;
  f.ctx->global_ctx->capabilities.supports_subgroups=false;reject();f.ctx->global_ctx->capabilities.supports_subgroups=true;
  f.ctx->global_ctx->device.f16=false;reject();f.ctx->global_ctx->device.f16=true;
  auto & l=f.ctx->global_ctx->capabilities.limits;l.maxComputeWorkgroupsPerDimension=0;reject();l.maxComputeWorkgroupsPerDimension=65535;
  l.maxComputeInvocationsPerWorkgroup=127;reject();l.maxComputeInvocationsPerWorkgroup=1024;
  l.maxStorageBufferBindingSize=4;reject();l.maxStorageBufferBindingSize=1ull<<30;
  l.maxUniformBufferBindingSize=127;reject();l.maxUniformBufferBindingSize=128;reject();l.maxUniformBufferBindingSize=65536;
  l.maxStorageBuffersPerShaderStage=2;reject();l.maxStorageBuffersPerShaderStage=8;
  l.maxBindingsPerBindGroup=3;reject();l.maxBindingsPerBindGroup=1000;
  f.owners[0].iface.get_base=nullptr;reject();f.owners[0].iface.get_base=ggml_backend_webgpu_buffer_get_base;
  f.buffers[0].global_ctx=std::make_shared<Global>();reject();f.buffers[0].global_ctx=f.ctx->global_ctx;
  f.buffers[0].buffer.usage=wgpu::BufferUsage::None;reject();f.buffers[0].buffer.usage=wgpu::BufferUsage::Storage;
  f.buffers[0].buffer.bytes=1;reject();f.buffers[0].buffer.bytes=1ull<<30;
  f.owners[0].size=1;reject();f.owners[0].size=1ull<<30;
  f.buffers[2].buffer.id=1;reject();f.buffers[2].buffer.id=3;
  f.a.view_src=&f.x;reject();f.a.view_src=nullptr;
  f.a.ne[0]=16385;reject();f.a.ne[0]=2560;
  f.dst.src[2]=&f.x;reject();f.dst.src[2]=nullptr;
  f.ctx->global_ctx->capabilities.supports_dot_product=true;f.ctx->global_ctx->vendor="nvidia";
  check(f.run(0,1,0,out)==3 && created==start && wgpu::dispatches==n);
  check(f.run(2,1,0,out)==0); // Explicit non-MMVQ selection does not reinterpret its F32 input.
 }
 {Fixture f(32,1);ggml_webgpu_q8_diagnostic_report_v1 out;
  f.a.data=reinterpret_cast<void *>(uintptr_t(0x1000+34));f.x.data=reinterpret_cast<void *>(uintptr_t(0x1000+260));f.dst.data=reinterpret_cast<void *>(uintptr_t(0x1000+516));
  check(f.run(1,1,0,out)==0 && f.run(1,2,1,out)==0);
  check(actual_params.back()[0]==1 && actual_params.back()[1]==1 && actual_params.back()[2]==1);}
 {Fixture f(2560,640);ggml_webgpu_q8_diagnostic_report_v1 out;f.ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension=32;
  check(f.run(3,1,0,out)==0 && out.grid_x==27 && out.grid_y==3);check(f.run(3,2,1,out)==0 && wgpu::grids.back()==std::make_pair(27u,3u));}
 {Fixture f(32,1);ggml_webgpu_q8_diagnostic_report_v1 out;f.buffers[2].buffer.id=2;
  f.x.data=reinterpret_cast<void *>(uintptr_t(0x1100));f.dst.data=reinterpret_cast<void *>(uintptr_t(0x1190));
  const auto n=created;check(f.run(1,1,0,out)==3 && n==created);}
 {Fixture f(320,10240);ggml_webgpu_q8_diagnostic_report_v1 out;f.ctx->global_ctx->capabilities.supports_subgroups=false;
  f.ctx->global_ctx->capabilities.min_subgroup_size=f.ctx->global_ctx->capabilities.max_subgroup_size=0;
  check(f.run(0,1,0,out)==0 && out.actual_algorithm==1 && !out.cooperative_subgroup_reduction);check(f.run(0,2,1,out)==0);}
 {Fixture f(16384,3855);ggml_webgpu_q8_diagnostic_report_v1 out;check(f.run(1,1,0,out)==0);check(f.run(1,2,1,out)==0);}
 for(auto shape:{std::pair<int,int>{16384,65536},{16384,4096},{31,1},{16385,1},{32,65537}}){Fixture f(shape.first,shape.second);ggml_webgpu_q8_diagnostic_report_v1 out;const auto n=created;check(f.run(1,1,0,out)==3 && n==created);}
 {Fixture f(32,1);ggml_webgpu_q8_diagnostic_request_v1 req{1,sizeof(req),1,1,0,0};ggml_webgpu_q8_diagnostic_report_v1 out;std::memset(&out,0xa5,sizeof(out));auto before=out;check(ggml_backend_webgpu_q8_diagnostic_v1(&f.backend,&f.dst,&req,&out,sizeof(out)-1)==2 && std::memcmp(&before,&out,sizeof(out))==0);
  for(auto * field:{&req.abi_version,&req.struct_size,&req.variant,&req.phase,&req.reserved}){const auto old=*field;*field=999;const auto n=created;check(ggml_backend_webgpu_q8_diagnostic_v1(&f.backend,&f.dst,&req,&out,sizeof(out))==2 && n==created);*field=old;}
  check(f.run(1,1,1,out)==2);check(f.run(1,2,0,out)==2);check(f.run(1,2,129,out)==2);}
 std::cout<<checks<<" actual provider/bindings/params/build/submit/map/guard checks; dispatches="<<wgpu::dispatches<<'\n';
}catch(const std::exception & e){std::cerr<<e.what()<<'\n';return 1;}}
