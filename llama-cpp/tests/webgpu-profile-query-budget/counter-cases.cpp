
struct test_case { const char * name; std::vector<size_t> counts; int expected_profile_failure; };
static std::vector<size_t> repeat(size_t n,size_t value) { return std::vector<size_t>(n,value); }
static void run_case(const test_case & test,bool & first) {
 fixture f;
#ifdef GGML_WEBGPU_GPU_PROFILE
 f.ctx->batch_compute_passes=false;
#endif
 assert(f.ctx->param_arena.slot_count==74 && f.ctx->profile_timestamp_query_count==0);
 collected_queries.clear();
 std::vector<ggml_tensor> tensors(test.counts.size());
 std::vector<ggml_tensor *> nodes;
 std::vector<webgpu_dispatch_desc> dispatches;
 std::vector<uint32_t> expected;
 for(size_t i=0;i<test.counts.size();++i){
  nodes.push_back(&tensors[i]);
  dispatches.push_back({{f.pipeline,"argsort"},{uint32_t(17+i)},f.entries(),{1,1}});
  for(size_t j=0;j<test.counts[i];++j)expected.push_back(uint32_t(17+i));
 }
 ggml_cgraph graph{int(nodes.size()),nodes.data(),dispatches,test.counts};
 bool failed=false;std::string failure;
 try{assert(ggml_backend_webgpu_graph_compute(&f.backend,&graph)==GGML_STATUS_SUCCESS);}
 catch(const std::logic_error & e){failed=true;failure=e.what();}
 catch(const std::runtime_error & e){failed=true;failure=e.what();}
 const bool query_failure=failure.find("profile_timestamp_query_count + 2")!=std::string::npos;
 const bool arena_failure=failure.find("parameter arena exhausted")!=std::string::npos;
#ifdef GGML_WEBGPU_GPU_PROFILE
 #ifdef EXPECT_SAFE_QUERY_BUDGET
 const bool expected_failure=test.expected_profile_failure==2;
 #else
 const bool expected_failure=test.expected_profile_failure!=0;
 #endif
#else
 const bool expected_failure=test.expected_profile_failure==2;
#endif
 assert(failed==expected_failure);
 if(failed){
  if(test.expected_profile_failure==2)assert(arena_failure);
  else{assert(query_failure && f.ctx->profile_timestamp_query_count==4096 && collected_queries.empty());}
 }else{
  f.global->stats->drain();
  assert(f.global->stats->observed_params==expected);
  assert(f.ctx->param_arena.next_slot==0);
  for(auto q:collected_queries)assert(q>0 && q<=4096 && q%2==0);
 }
 if(failed){
  // Drain the CPU model's already-submitted closures before dropping the model.
  // This is test cleanup, not a proposed runtime recovery from a native trap.
  f.global->stats->drain();
  const auto & observed=f.global->stats->observed_params;
  assert(observed.size()<=expected.size() && std::equal(observed.begin(),observed.end(),expected.begin()));
 }
 if(!first){std::cout<<',';}
 first=false;
 std::cout<<"{\"name\":\""<<test.name<<"\",\"nodes\":"<<test.counts.size()<<",\"plannedDispatches\":"<<expected.size()
  <<",\"failed\":"<<(failed?"true":"false")<<",\"failureKind\":\""<<(query_failure?"query":arena_failure?"arena":"none")
  <<"\",\"queryCounterAtReturn\":"<<f.ctx->profile_timestamp_query_count<<",\"arenaUsedAtReturn\":"<<f.ctx->param_arena.next_slot<<",\"collectedQueries\":[";
 for(size_t i=0;i<collected_queries.size();++i){if(i)std::cout<<',';std::cout<<collected_queries[i];}
 std::cout<<"]}";
}
int main(){
#ifdef GGML_WEBGPU_GPU_PROFILE
 {
  fixture f;
  for(uint32_t capacity : {74u,2048u,2049u}){
   f.ctx->param_arena.slot_count=capacity;bool rejected=false;
   try{validate_extracted_profile_capacity(f.ctx);}
   catch(const std::logic_error &){rejected=true;}
#ifdef EXPECT_SAFE_QUERY_BUDGET
   assert(rejected==(capacity>2048));
#else
   assert(!rejected);
#endif
  }
 }
#endif
 std::vector<test_case> cases={{"empty",{},0},{"one",{1},0},{"boundary64",repeat(64,1),0},{"boundary65",repeat(65,1),0},
 {"single1983",repeat(1983,1),0},{"single1984",repeat(1984,1),0},{"single2048",repeat(2048,1),0}};
 auto prefix=repeat(185,11);prefix.push_back(8);
 cases.push_back({"prefix186-q4086",prefix,0});
 auto exact=prefix;exact.push_back(5);cases.push_back({"exact2048-dispatches",exact,0});
 auto first_over=prefix;first_over.push_back(6);cases.push_back({"first2049-dispatches",first_over,1});
 auto maximum=prefix;maximum.push_back(11);cases.push_back({"minimal187nodes-max74slots",maximum,1});
 auto slots=repeat(63,1);slots.push_back(11);cases.push_back({"exact74-arena-slots",slots,0});
 auto invalid=repeat(63,1);invalid.push_back(12);cases.push_back({"existing75-arena-rejection",invalid,2});
 std::vector<size_t> small;
 for(int batch=0;batch<21;++batch){for(int i=0;i<15;++i)small.push_back(4);small.push_back(3);small.push_back(4);}
 for(int batch=0;batch<9;++batch)for(int i=0;i<16;++i)small.push_back(4);
 for(int i=0;i<15;++i){small.push_back(4);}
 small.push_back(3);small.push_back(4);
 cases.push_back({"small-width518nodes",small,1});
 bool first=true;std::cout<<"{\"status\":\"passed\",\"cases\":[";
 for(const auto & test:cases)run_case(test,first);
 assert(live_resources==0);std::cout<<"]}\n";
}
