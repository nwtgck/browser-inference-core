
int main(){
 for(bool passes:{false,true})for(size_t alignment:{128,256,512}) {
  auto ctx=std::make_shared<context>();ctx->global_ctx=std::make_shared<global_context>();ctx->batch_compute_passes=passes;
  ctx->param_arena.init({},128,74,alignment);ggml_backend_webgpu_context bc{ctx};backend b{&bc};
  size_t total=0,expected_submits=0;
  for(int count:{0,1,63,64,65,128,129,300}) {
   std::vector<ggml_tensor> nodes;std::vector<ggml_tensor*> ptrs;size_t batch=0;
   for(int i=0;i<count;i++) {
    int kernels=(i%7==0)?0:(i%4)+1;nodes.push_back({i==3?GGML_OP_SET_ROWS:1,kernels,uint32_t(count*23+i*17)});
    total+=kernels;batch+=kernels;if(batch>=64){expected_submits++;batch=0;}
   }
   if(batch)expected_submits++;
   for(auto &n:nodes)ptrs.push_back(&n);ggml_cgraph g{count,ptrs.data()};
   assert(ggml_backend_webgpu_graph_compute(&b,&g)==GGML_STATUS_SUCCESS);
   assert(ctx->param_arena.next_slot==0);
  }
  auto &q=ctx->global_ctx->queue;assert(q.submits==expected_submits);
#ifdef GGML_WEBGPU_BATCH_PARAM_UPLOADS
  assert(q.writes==q.submits);
#else
  assert(q.writes==total);
#endif
  q.drain();printf("passes=%d alignment=%zu kernels=%zu submits=%zu writes=%zu graph passed\n",passes,alignment,total,q.submits,q.writes);
 }
 assert(checks==36);
}
