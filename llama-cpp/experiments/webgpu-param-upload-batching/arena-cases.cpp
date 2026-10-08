
int main(int argc,char **argv){
 if(argc>1){
  webgpu_param_arena a; a.init({},128,74,256); uint32_t x=1;
  int mode=atoi(argv[1]);
  if(mode==1)a.alloc_slot(132);
  if(mode==2)for(int i=0;i<75;i++)a.alloc_slot(4);
#ifdef GGML_WEBGPU_BATCH_PARAM_UPLOADS
  if(mode==3)a.stage(0,&x,4);
  a.alloc_slot(4);
  if(mode==4)a.stage(1,&x,4);
  if(mode==5)a.stage(0,&x,132);
  if(mode==6){a.upload_data.resize(8);a.stage(0,&x,4);}
#endif
  return 0;
 }

 for(size_t alignment:{128,256,512}) {
  webgpu_param_arena a; a.init({},128,74,alignment); wgpu::Queue q;
  size_t total=0;
#ifdef GGML_WEBGPU_BATCH_PARAM_UPLOADS
  a.flush(q); assert(q.writes==0);
  assert(a.upload_data.size()==74*alignment);
#endif
  for(int graph=0;graph<40;graph++) {
   std::vector<std::pair<size_t,std::vector<uint8_t>>> expected;
   auto submit=[&](){
#ifdef GGML_WEBGPU_BATCH_PARAM_UPLOADS
    a.flush(q);
#endif
    q.Submit(a.buffer,expected,a.slot_stride); a.reset(); expected.clear();
   };
   int limit=graph==0 ? 0 : graph==1 ? 64 : graph==2 ? 65 : graph*7;
   int kernels=0;
   for(int op=0;kernels<limit;op++) {
    int n=std::min(1+(op%4),limit-kernels);
    for(int j=0;j<n;j++) {
     size_t size=((total*13)%33)*4,off=a.alloc_slot(size);
     std::vector<uint8_t> p(size,uint8_t(total%251+1)); total++; kernels++;
#ifdef GGML_WEBGPU_BATCH_PARAM_UPLOADS
     a.stage(off,p.data(),p.size());
#else
     q.WriteBuffer(a.buffer,off,p.data(),p.size());
#endif
     expected.push_back({off,p}); std::fill(p.begin(),p.end(),0xff);
    }
    if(expected.size()>=64)submit();
   }
   if(!expected.empty())submit();
  }
#ifdef GGML_WEBGPU_BATCH_PARAM_UPLOADS
  assert(q.writes==q.submits);
#else
  assert(q.writes==total);
#endif
  q.drain(); printf("alignment=%zu kernels=%zu submissions=%zu writes=%zu passed\n",alignment,total,q.submits,q.writes);
 }
}
