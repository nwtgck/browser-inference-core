
#include <vector>
#include <memory>
#include <functional>
#include <cassert>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#define GGML_ASSERT assert
#define GGML_ABORT(...) abort()
#define ROUNDUP_POW2(a,b) (((a)+(b)-1)&~((b)-1))
namespace wgpu {
struct Device {};
enum BufferUsage { CopyDst=1, Uniform=2 };
struct Buffer {
 std::shared_ptr<std::vector<uint8_t>> v;
 explicit operator bool() const {return bool(v);}
 void Destroy() {v.reset();}
 Buffer & operator=(std::nullptr_t) {v.reset(); return *this;}
};
struct Queue {
 std::vector<std::function<void()>> pending;
 size_t writes=0, submits=0;
 void WriteBuffer(Buffer b,size_t offset,const void *p,size_t n) {
  std::vector<uint8_t> copy(n); if(n) memcpy(copy.data(),p,n);
  pending.push_back([b,offset,copy](){ assert(offset+copy.size()<=b.v->size()); if(!copy.empty()) memcpy(b.v->data()+offset,copy.data(),copy.size()); }); writes++;
 }
 void Submit(Buffer b,std::vector<std::pair<size_t,std::vector<uint8_t>>> expected,size_t stride) {
  pending.push_back([b,expected,stride](){for(auto & e:expected) {
   if(!e.second.empty()) assert(memcmp(b.v->data()+e.first,e.second.data(),e.second.size())==0);
#ifdef GGML_WEBGPU_BATCH_PARAM_UPLOADS
   for(size_t i=e.second.size();i<stride;i++) assert((*b.v)[e.first+i]==0);
#endif
  }}); submits++;
 }
 void drain() {for(auto & f:pending)f();pending.clear();}
};
}
static void ggml_webgpu_create_buffer(wgpu::Device &,wgpu::Buffer & b,size_t n,int,const char*) {b.v=std::make_shared<std::vector<uint8_t>>(n,0);}
