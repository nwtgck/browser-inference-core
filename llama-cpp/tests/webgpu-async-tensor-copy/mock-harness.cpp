// Execute production helpers against deferred CPU queues. This models ordering,
// not Dawn validation, JSPI, device throughput, or a real model's copy counts.
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
static int checks=0;
#define CHECK(x) do { ++checks; if (!(x)) throw std::runtime_error(std::string(#x)+" at "+std::to_string(__LINE__)); } while (0)
#define GGML_ASSERT(x) assert(x)
#define GGML_UNUSED(x) (void)(x)
#define GGML_LOG_DEBUG(...) ((void)0)
#define WEBGPU_LOG_DEBUG(...) ((void)0)
#define WEBGPU_CPU_PROFILE_TOTAL_START(...) ((void)0)
#define WEBGPU_CPU_PROFILE_TOTAL_END(...) ((void)0)
namespace wgpu {
enum class BufferUsage { None=0, CopySrc=1, CopyDst=2 };
BufferUsage operator&(BufferUsage a,BufferUsage b) { return BufferUsage(int(a)&int(b)); }
struct Storage { std::vector<uint8_t> bytes; BufferUsage usage; bool destroyed=false; };
struct Buffer {
    std::shared_ptr<Storage> p;
    explicit Buffer(size_t size=64,BufferUsage usage=BufferUsage(3)):p(std::make_shared<Storage>(Storage{std::vector<uint8_t>(size,0xcc),usage,false})) {}
    explicit operator bool() const { return bool(p); }
    Storage* Get() const { return p.get(); }
    size_t GetSize() const { return p->bytes.size(); }
    BufferUsage GetUsage() const { return p->usage; }
    void Destroy() { p->destroyed=true; }
};
struct CommandBuffer { std::function<void()> run; };
struct CommandEncoder {
    CommandBuffer command;
    void CopyBufferToBuffer(Buffer a,size_t ao,Buffer b,size_t bo,size_t n) {
        CHECK(a.Get()!=b.Get());CHECK(!a.p->destroyed&&!b.p->destroyed);
        CHECK(ao%4==0&&bo%4==0&&n%4==0);
        CHECK(ao<=a.GetSize()&&n<=a.GetSize()-ao&&bo<=b.GetSize()&&n<=b.GetSize()-bo);
        command.run=[=]{std::copy_n(a.p->bytes.begin()+ao,n,b.p->bytes.begin()+bo);};
    }
    CommandBuffer Finish() { return command; }
};
struct Device {
    bool fail_encoder=false;
    CommandEncoder CreateCommandEncoder() { if(fail_encoder)throw std::runtime_error("encoder");return {}; }
};
struct Queue {
    int writes=0,submissions=0,waits=0;
    bool fail_write=false,fail_wait=false;
    std::vector<std::function<void()>> pending;
    void Submit(size_t n,const CommandBuffer* commands) { for(size_t i=0;i<n;i++){++submissions;pending.push_back(commands[i].run);} }
    void WriteBuffer(Buffer b,size_t offset,const void* data,size_t n) {
        if(fail_write)throw std::runtime_error("write");
        CHECK(offset%4==0&&n%4==0);CHECK(offset<=b.GetSize()&&n<=b.GetSize()-offset);
        ++writes;
        // Queue.writeBuffer consumes caller bytes on the content timeline.
        std::vector<uint8_t> snapshot((const uint8_t*)data,(const uint8_t*)data+n);
        pending.push_back([=]{std::copy(snapshot.begin(),snapshot.end(),b.p->bytes.begin()+offset);});
    }
    void drain() { if(fail_wait)throw std::runtime_error("queue");auto work=std::move(pending);pending.clear();for(auto& f:work)f(); }
};
}
struct global_context { wgpu::Device device;wgpu::Queue queue; };
struct webgpu_context { std::shared_ptr<global_context> global_ctx; };
struct ggml_backend_webgpu_context { std::shared_ptr<webgpu_context> webgpu_ctx; };
struct ggml_backend_webgpu_buffer_context { wgpu::Buffer buffer;std::shared_ptr<global_context> global_ctx; };
struct ggml_backend;using ggml_backend_t=ggml_backend*;
struct ggml_backend_buffer;using ggml_backend_buffer_t=ggml_backend_buffer*;
struct ggml_backend_device;using ggml_backend_dev_t=ggml_backend_device*;
struct ggml_tensor {
    ggml_tensor* view_src=nullptr;ggml_backend_buffer_t buffer=nullptr;void* data=nullptr;
    size_t view_offs=0,nbytes=16;int layout=0,flags=0;
};
enum ggml_backend_dev_type { GGML_BACKEND_DEVICE_TYPE_CPU,GGML_BACKEND_DEVICE_TYPE_GPU };
struct device_iface { enum ggml_backend_dev_type(*get_type)(ggml_backend_dev_t); };
struct ggml_backend_device { device_iface iface;enum ggml_backend_dev_type type; };
struct backend_iface {
    const char*(*get_name)(ggml_backend_t)=nullptr;
    bool(*cpy_tensor_async)(ggml_backend_t,ggml_backend_t,const ggml_tensor*,ggml_tensor*)=nullptr;
    void(*synchronize)(ggml_backend_t)=nullptr;
};
struct ggml_backend { backend_iface iface;void* context=nullptr;ggml_backend_dev_t device=nullptr; };
struct ggml_backend_buffer_type { bool host; };using ggml_backend_buffer_type_t=ggml_backend_buffer_type*;
enum ggml_backend_buffer_usage { GGML_BACKEND_BUFFER_USAGE_ANY,GGML_BACKEND_BUFFER_USAGE_WEIGHTS };
struct buffer_iface {
    void*(*get_base)(ggml_backend_buffer_t)=nullptr;
    void(*set_tensor)(ggml_backend_buffer_t,ggml_tensor*,const void*,size_t,size_t)=nullptr;
    void(*get_tensor)(ggml_backend_buffer_t,const ggml_tensor*,void*,size_t,size_t)=nullptr;
    bool(*cpy_tensor)(ggml_backend_buffer_t,const ggml_tensor*,ggml_tensor*)=nullptr;
};
struct ggml_backend_buffer { buffer_iface iface;ggml_backend_buffer_type_t buft;void* context;size_t size;enum ggml_backend_buffer_usage usage=GGML_BACKEND_BUFFER_USAGE_ANY; };
static void* const webgpu_ptr_base=(void*)(uintptr_t)0x1000;
static bool ggml_are_same_layout(const ggml_tensor* a,const ggml_tensor* b){return a->nbytes==b->nbytes&&a->layout==b->layout;}
static size_t ggml_nbytes(const ggml_tensor* t){return t->nbytes;}
static bool ggml_backend_buffer_is_host(ggml_backend_buffer_t b){return b->buft->host;}
static bool ggml_backend_buffer_is_meta(ggml_backend_buffer_t){return false;}
static enum ggml_backend_dev_type device_type(ggml_backend_dev_t d){return d->type;}
static const char* ggml_backend_webgpu_name(ggml_backend_t){return "WebGPU";}
static const char* foreign_name(ggml_backend_t){return "WebGPU";} // Same string is not private-interface identity.
// @ACTUAL_OFFSET@
// @ACTUAL_BUFFER_BASE@
// @ACTUAL_GENERIC_BASE@
// @ACTUAL_DEVICE_ACCESS@
static void ggml_backend_webgpu_wait_queue(std::shared_ptr<global_context> g){++g->queue.waits;g->queue.drain();}
// @ACTUAL_WEBGPU_SYNCHRONIZE@
// @ACTUAL_SYNCHRONIZE@
static void ggml_backend_webgpu_buffer_memset(std::shared_ptr<global_context> g,wgpu::Buffer b,uint32_t value,size_t offset,size_t n){
    const size_t lane=offset%4;
    std::vector<uint8_t> bytes(n);for(size_t i=0;i<n;i++)bytes[i]=((uint8_t*)&value)[lane+i];
    g->queue.pending.push_back([=]{std::copy(bytes.begin(),bytes.end(),b.p->bytes.begin()+offset);});
}
// @ACTUAL_WEBGPU_SET@
static int gets=0;
static void gpu_get(ggml_backend_buffer_t b,const ggml_tensor* t,void* data,size_t offset,size_t n){
    ++gets;auto* ctx=(ggml_backend_webgpu_buffer_context*)b->context;ctx->global_ctx->queue.drain();
    std::memcpy(data,ctx->buffer.p->bytes.data()+ggml_webgpu_tensor_offset(t)+offset,n);
}
static void host_set(ggml_backend_buffer_t,ggml_tensor* t,const void* data,size_t off,size_t n){std::memcpy((uint8_t*)t->data+off,data,n);}
static void host_get(ggml_backend_buffer_t,const ggml_tensor* t,void* data,size_t off,size_t n){std::memcpy(data,(uint8_t*)t->data+off,n);}
// @ACTUAL_GENERIC_SET_GET@
// @ACTUAL_BUFFER_COPY@
// @ACTUAL_GENERIC_BUFFER_COPY@
// @ACTUAL_HOOK@
// @ACTUAL_GENERIC_COPY@
// @ACTUAL_GENERIC_ASYNC_COPY@

struct CpuState { int waits=0;bool fail=false;std::function<void()> complete; };
static void cpu_sync(ggml_backend_t b){auto* s=(CpuState*)b->context;++s->waits;if(s->fail)throw std::runtime_error("cpu");if(s->complete){auto f=std::move(s->complete);s->complete={};f();}}
static void* host_base(ggml_backend_buffer_t b){return b->context;}
static void* foreign_base(ggml_backend_buffer_t){return (void*)0x4000;}
struct Fixture {
    std::shared_ptr<global_context> g=std::make_shared<global_context>();
    ggml_backend_webgpu_context gpu_ctx{std::make_shared<webgpu_context>(webgpu_context{g})};
    ggml_backend_webgpu_buffer_context a{wgpu::Buffer(),g},b{wgpu::Buffer(),g};
    ggml_backend_buffer_type host_type{true},gpu_type{false};
    ggml_backend_device cpu_device{{device_type},GGML_BACKEND_DEVICE_TYPE_CPU},gpu_device{{device_type},GGML_BACKEND_DEVICE_TYPE_GPU};
    CpuState cpu_state;
    ggml_backend cpu{{foreign_name,nullptr,cpu_sync},&cpu_state,&cpu_device};
    ggml_backend gpu{{ggml_backend_webgpu_name,// @REGISTERED_HOOK@
                      ggml_backend_webgpu_synchronize},&gpu_ctx,&gpu_device};
    std::vector<uint8_t> host=std::vector<uint8_t>(64);
    ggml_backend_buffer hb{{host_base,host_set,host_get,nullptr},&host_type,host.data(),64};
    ggml_backend_buffer ab{{ggml_backend_webgpu_buffer_get_base,ggml_backend_webgpu_buffer_set_tensor,gpu_get,ggml_backend_webgpu_buffer_cpy_tensor},&gpu_type,&a,64};
    ggml_backend_buffer bb{{ggml_backend_webgpu_buffer_get_base,ggml_backend_webgpu_buffer_set_tensor,gpu_get,ggml_backend_webgpu_buffer_cpy_tensor},&gpu_type,&b,64};
    ggml_tensor src{nullptr,&hb,host.data()+4,0,16,0,0},gpu_src{nullptr,&ab,(void*)0x1004,0,16,0,0},dst{nullptr,&bb,(void*)0x1008,0,16,0,0};
    Fixture(){for(int i=0;i<64;i++){host[i]=uint8_t(i+1);a.buffer.p->bytes[i]=uint8_t(i+33);}}
    bool host_copy(){return ggml_backend_webgpu_cpy_tensor_async(&cpu,&gpu,&src,&dst);}
    bool gpu_copy(){return ggml_backend_webgpu_cpy_tensor_async(&gpu,&gpu,&gpu_src,&dst);}
    void guard(uint8_t left=0xcc,uint8_t right=0xcc){CHECK(b.buffer.p->bytes[7]==left);CHECK(b.buffer.p->bytes[24]==right);}
    void copied(int start){for(int i=0;i<16;i++)CHECK(b.buffer.p->bytes[8+i]==uint8_t(start+i));guard();}
};

// Only the scheduler bookkeeping is modeled. The branching, callback priority,
// waits and copying below are the complete pinned production copy_input helper.
constexpr int GGML_TENSOR_FLAG_INPUT=1;
struct Event { int syncs=0,waits=0;ggml_backend_t backend=nullptr; };
using ggml_backend_event_t=Event*;
struct ggml_cgraph {};
struct ggml_backend_sched_split { int backend_id=0;ggml_cgraph graph; };
struct ggml_backend_sched {
    ggml_backend_t backends[1]{};ggml_backend_event_t events[1][1]{};int cur_copy=0;
    ggml_backend_t source=nullptr;ggml_tensor* copy=nullptr;
    bool(*callback_copy)(ggml_backend_t,ggml_tensor*,ggml_tensor*,ggml_cgraph*,void*)=nullptr;
    void* callback_copy_user_data=nullptr;
};
using ggml_backend_sched_t=ggml_backend_sched*;
static ggml_backend_t ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t s,ggml_tensor*){return s->source;}
static ggml_tensor* scheduler_tensor_copy(ggml_backend_sched_t s,ggml_tensor*,int,int){return s->copy;}
#define tensor_copy(t,b,c) scheduler_tensor_copy(sched,t,b,c)
static void ggml_backend_event_synchronize(ggml_backend_event_t e){++e->syncs;ggml_backend_synchronize(e->backend);}
static void ggml_backend_event_wait(ggml_backend_t b,ggml_backend_event_t e){++e->waits;ggml_backend_synchronize(b);}
// @ACTUAL_GENERIC_USAGE@
// @ACTUAL_HOST_WEIGHT@
// @ACTUAL_SCHEDULER_COPY@
#undef tensor_copy
static void throws(const std::function<void()>& f,const char* expected){bool caught=false;try{f();}catch(const std::runtime_error& e){caught=e.what()==std::string(expected);}CHECK(caught);}
static void host_success_and_lifetime(){
    Fixture f;std::vector<int> observed;
    f.g->queue.pending.push_back([&]{observed.push_back(f.b.buffer.p->bytes[8]);});
    f.cpu_state.complete=[&]{std::fill_n(f.host.begin()+4,16,77);};
    CHECK(f.host_copy());CHECK(f.cpu_state.waits==1);CHECK(f.g->queue.waits==0);CHECK(f.g->queue.writes==1);
    CHECK(f.b.buffer.p->bytes[8]==0xcc);std::fill(f.host.begin(),f.host.end(),99);
    f.g->queue.pending.push_back([&]{observed.push_back(f.b.buffer.p->bytes[8]);});
    ggml_backend_synchronize(&f.gpu);CHECK(observed==std::vector<int>({0xcc,77}));
    for(int i=0;i<16;i++){CHECK(f.b.buffer.p->bytes[8+i]==77);}f.guard();
    {Fixture z;z.cpu.iface.synchronize=nullptr;CHECK(z.host_copy());CHECK(z.cpu_state.waits==0);z.g->queue.drain();z.copied(5);}
    {Fixture z;z.src.data=z.host.data()+5;CHECK(z.host_copy());z.g->queue.drain();z.copied(6);}
    {Fixture z;z.src.layout=z.dst.layout=17;CHECK(z.host_copy());z.g->queue.drain();z.copied(5);} // identical strided byte layouts
    {Fixture z;ggml_backend_tensor_copy_async(&z.cpu,&z.gpu,&z.src,&z.dst);CHECK(z.g->queue.waits==0&&z.cpu_state.waits==1);z.host.clear();z.host.shrink_to_fit();z.g->queue.drain();z.copied(5);}
    {Fixture z;std::weak_ptr<wgpu::Storage> alive=z.b.buffer.p;CHECK(z.host_copy());z.b.buffer.Destroy();z.b.buffer.p.reset();CHECK(!alive.expired());z.g->queue.drain();CHECK(alive.expired());}
}
static void host_guards(){
    for(int test=0;test<23;test++){
        Fixture f;
        switch(test){
            case 0:f.cpu.device=nullptr;break;
            case 1:f.cpu_device.type=GGML_BACKEND_DEVICE_TYPE_GPU;break;
            case 2:f.host_type.host=false;break;
            case 3:f.hb.buft=nullptr;break;
            case 4:f.hb.iface.get_base=nullptr;break;
            case 5:f.src.data=(void*)((uintptr_t)f.host.data()-1);break;
            case 6:f.src.data=(void*)((uintptr_t)f.host.data()+65);break;
            case 7:f.src.data=f.host.data()+52;break;
            case 8:f.dst.data=(void*)0x1009;break;
            case 9:f.src.nbytes=f.dst.nbytes=15;break;
            case 10:f.src.nbytes=f.dst.nbytes=0;break;
            case 11:f.dst.data=(void*)0x1040;break;
            case 12:f.dst.data=(void*)0x0fff;break;
            case 13:f.dst.view_offs=std::numeric_limits<size_t>::max();break;
            case 14:f.dst.layout=4;break;
            case 15:f.b.buffer.p->usage=wgpu::BufferUsage::CopySrc;break;
            case 16:f.bb.size=23;break;
            case 17:f.b.buffer.p->bytes.resize(23);break;
            case 18:f.b.global_ctx=std::make_shared<global_context>();break;
            case 19:f.bb.iface.get_base=foreign_base;f.bb.context=(void*)1;break;
            case 20:f.gpu.iface.get_name=foreign_name;f.gpu.context=(void*)1;break;
            case 21:f.src.data=nullptr;break;
            case 22:f.dst.data=nullptr;break;
        }
        CHECK(!f.host_copy());CHECK(f.g->queue.waits==0&&f.g->queue.writes==0&&f.cpu_state.waits==0);
    }
    {Fixture f;CHECK(!ggml_backend_webgpu_cpy_tensor_async(nullptr,&f.gpu,&f.src,&f.dst));CHECK(!ggml_backend_webgpu_cpy_tensor_async(&f.cpu,nullptr,&f.src,&f.dst));CHECK(!ggml_backend_webgpu_cpy_tensor_async(&f.cpu,&f.gpu,nullptr,&f.dst));CHECK(!ggml_backend_webgpu_cpy_tensor_async(&f.cpu,&f.gpu,&f.src,nullptr));}
    {Fixture f;f.gpu_ctx.webgpu_ctx.reset();CHECK(!f.host_copy());}
    {Fixture f;f.b.buffer.p.reset();CHECK(!f.host_copy());}
    {Fixture f;f.bb.context=nullptr;CHECK(!f.host_copy());}
    {Fixture f;f.hb.context=nullptr;f.hb.iface.get_base=nullptr;CHECK(!f.host_copy());}
    {Fixture f;f.src.nbytes=f.dst.nbytes=std::numeric_limits<size_t>::max();CHECK(!f.host_copy());}
    {Fixture f;f.src.data=f.host.data()+48;f.dst.data=(void*)0x1030;CHECK(f.host_copy());f.g->queue.drain();for(int i=0;i<16;i++)CHECK(f.b.buffer.p->bytes[48+i]==49+i);CHECK(f.b.buffer.p->bytes[47]==0xcc);}

    {Fixture f;ggml_tensor sb=f.src,db=f.dst;f.src.view_src=&sb;f.src.data=f.host.data()+12;f.src.view_offs=8;f.src.buffer=nullptr;f.dst.view_src=&db;f.dst.data=(void*)0xdead;f.dst.view_offs=12;f.dst.buffer=nullptr;CHECK(f.host_copy());f.g->queue.drain();for(int i=0;i<16;i++)CHECK(f.b.buffer.p->bytes[20+i]==13+i);CHECK(f.b.buffer.p->bytes[19]==0xcc&&f.b.buffer.p->bytes[36]==0xcc);}
    // GPU or mapped-host producers are synchronized by generic fallback, never
    // mistaken for completed CPU data merely because their buffer is host-visible.
    {Fixture f;f.cpu_device.type=GGML_BACKEND_DEVICE_TYPE_GPU;f.cpu_state.complete=[&]{std::fill_n(f.host.begin()+4,16,83);};ggml_backend_tensor_copy_async(&f.cpu,&f.gpu,&f.src,&f.dst);CHECK(f.cpu_state.waits==1&&f.g->queue.waits==1);f.g->queue.drain();CHECK(f.b.buffer.p->bytes[8]==83);}
    // The legacy set helper continues to own byte heads/tails on fallback.
    for(int k=0;k<2;k++){Fixture f;if(k==0)f.dst.data=(void*)0x1009;else f.src.nbytes=f.dst.nbytes=15;ggml_backend_tensor_copy_async(&f.cpu,&f.gpu,&f.src,&f.dst);CHECK(f.g->queue.waits==1&&f.cpu_state.waits==1);f.g->queue.drain();size_t begin=k?8:9,end=begin+f.src.nbytes;for(size_t i=begin;i<end;i++)CHECK(f.b.buffer.p->bytes[i]==5+i-begin);CHECK(f.b.buffer.p->bytes[begin-1]==0xcc&&f.b.buffer.p->bytes[end]==0xcc);}
    {Fixture f;f.src.nbytes=f.dst.nbytes=0;ggml_backend_tensor_copy_async(&f.cpu,&f.gpu,&f.src,&f.dst);CHECK(f.g->queue.waits==1&&f.cpu_state.waits==1&&f.g->queue.writes==0);}
}
static void gpu_order_guards_and_fallback(){
    {Fixture f;ggml_backend_webgpu_context same{std::make_shared<webgpu_context>(webgpu_context{f.g})};ggml_backend source=f.gpu;source.context=&same;CHECK(ggml_backend_webgpu_cpy_tensor_async(&source,&f.gpu,&f.gpu_src,&f.dst));CHECK(f.g->queue.waits==0);f.g->queue.drain();f.copied(37);}
    {Fixture f;f.gpu_src.nbytes=f.dst.nbytes=0;CHECK(!f.gpu_copy());ggml_backend_tensor_copy_async(&f.gpu,&f.gpu,&f.gpu_src,&f.dst);CHECK(f.g->queue.waits==2&&f.g->queue.submissions==0);}

    {Fixture f;uint8_t producer[16];std::fill_n(producer,16,71);f.g->queue.WriteBuffer(f.a.buffer,4,producer,16);CHECK(f.gpu_copy());CHECK(f.g->queue.waits==0&&f.g->queue.submissions==1);std::fill_n(producer,16,99);f.g->queue.WriteBuffer(f.a.buffer,4,producer,16);f.g->queue.drain();for(int i=0;i<16;i++)CHECK(f.b.buffer.p->bytes[8+i]==71);CHECK(f.a.buffer.p->bytes[4]==99);f.guard();}
    {Fixture f;std::weak_ptr<wgpu::Storage> a=f.a.buffer.p,b=f.b.buffer.p;CHECK(f.gpu_copy());f.a.buffer.Destroy();f.b.buffer.Destroy();f.a.buffer.p.reset();f.b.buffer.p.reset();CHECK(!a.expired()&&!b.expired());f.g->queue.drain();CHECK(a.expired()&&b.expired());}
    {Fixture f;ggml_tensor a=f.gpu_src,b=f.dst;f.gpu_src.view_src=&a;f.gpu_src.buffer=nullptr;f.gpu_src.view_offs=8;f.gpu_src.data=(void*)0xdead;f.dst.view_src=&b;f.dst.buffer=nullptr;f.dst.view_offs=12;f.dst.data=(void*)0xbeef;CHECK(f.gpu_copy());f.g->queue.drain();for(int i=0;i<16;i++)CHECK(f.b.buffer.p->bytes[20+i]==45+i);}
    for(int k=0;k<18;k++){Fixture f;ggml_backend source=f.gpu;ggml_backend_webgpu_context alternate=f.gpu_ctx;
        switch(k){
            case 0:source.iface.get_name=foreign_name;source.context=(void*)1;break;
            case 1:source.context=nullptr;break;
            case 2:f.ab.context=nullptr;break;
            case 3:f.a.global_ctx=std::make_shared<global_context>();break;
            case 4:alternate.webgpu_ctx=std::make_shared<webgpu_context>(webgpu_context{std::make_shared<global_context>()});source.context=&alternate;break;
            case 5:f.gpu_src.data=(void*)0x1005;break;
            case 6:f.a.buffer.p->usage=wgpu::BufferUsage::CopyDst;break;
            case 7:f.b.buffer.p->usage=wgpu::BufferUsage::CopySrc;break;
            case 8:f.gpu_src.data=(void*)0x1040;break;
            case 9:f.ab.size=19;break;
            case 10:f.a.buffer.p->bytes.resize(19);break;
            case 11:f.gpu_src.view_offs=std::numeric_limits<size_t>::max();break;
            case 12:f.a.buffer.p.reset();break;
            case 13:f.gpu_src.buffer=nullptr;break;
            case 14:f.dst.buffer=nullptr;break;
            case 15:f.a.global_ctx.reset();break;
            case 16:f.ab.iface.get_base=foreign_base;f.ab.context=(void*)1;break;
            case 17:f.gpu_src.data=(void*)0x0fff;break;
        }
        CHECK(!ggml_backend_webgpu_cpy_tensor_async(&source,&f.gpu,&f.gpu_src,&f.dst));CHECK(f.g->queue.submissions==0&&f.g->queue.waits==0);
    }
    // Alias no-op remains valid; partial overlap and same-buffer disjoint copy
    // keep the older portable host-staging fallback, even on the same queue.
    for(int k=0;k<3;k++){Fixture f;f.b.buffer=f.a.buffer;f.dst.data=k==0?f.gpu_src.data:(void*)(uintptr_t)(k==1?0x1008:0x1024);if(k==0){CHECK(f.gpu_copy());CHECK(f.g->queue.submissions==0);}else{CHECK(!f.gpu_copy());gets=0;ggml_backend_tensor_copy_async(&f.gpu,&f.gpu,&f.gpu_src,&f.dst);CHECK(f.g->queue.waits==2&&gets==1);f.g->queue.drain();size_t off=k==1?8:36;for(int i=0;i<16;i++)CHECK(f.b.buffer.p->bytes[off+i]==37+i);}}
    {Fixture f;f.b.buffer=f.a.buffer;f.dst.data=f.gpu_src.data=(void*)0x1005;f.a.buffer.p->usage=wgpu::BufferUsage::None;CHECK(f.gpu_copy());CHECK(f.g->queue.submissions==0);} // exact no-op needs neither aligned transfer nor usage
    {Fixture f;auto other=std::make_shared<global_context>();f.a.global_ctx=other;ggml_backend_webgpu_context sc{std::make_shared<webgpu_context>(webgpu_context{other})};ggml_backend source=f.gpu;source.context=&sc;uint8_t v[16];std::fill_n(v,16,86);other->queue.WriteBuffer(f.a.buffer,4,v,16);gets=0;ggml_backend_tensor_copy_async(&source,&f.gpu,&f.gpu_src,&f.dst);CHECK(other->queue.waits==1&&f.g->queue.waits==1&&gets==1);f.g->queue.drain();CHECK(f.b.buffer.p->bytes[8]==86);}
}
static void scheduler_and_cpu_mode(){
    for(bool enabled:{false,true}){int total=0;for(int i=0;i<10;i++){Fixture f;if(!enabled)f.gpu.iface.cpy_tensor_async=nullptr;f.src.flags=(i==1||i==3)?0:GGML_TENSOR_FLAG_INPUT;ggml_backend_sched s;s.backends[0]=&f.gpu;s.source=&f.cpu;s.copy=&f.dst;ggml_backend_sched_split split;ggml_backend_sched_copy_input(&s,&split,&f.src);CHECK(f.g->queue.waits==(f.src.flags?1:(enabled?1:2)));CHECK(f.cpu_state.waits==(f.src.flags?0:1));total+=f.g->queue.waits;std::fill(f.host.begin(),f.host.end(),99);f.g->queue.drain();f.copied(5);}CHECK(total==(enabled?10:12));}
    // Callback remains first after the original destination wait. Host-weight
    // callback success never enters the new hook or synchronizes the producer.
    for(bool handled:{false,true}){Fixture f;int called=0;f.hb.usage=GGML_BACKEND_BUFFER_USAGE_WEIGHTS;struct User{int* count;bool handled;};User u{&called,handled};ggml_backend_sched s;s.backends[0]=&f.gpu;s.source=&f.cpu;s.copy=&f.dst;s.callback_copy_user_data=&u;s.callback_copy=[](ggml_backend_t,ggml_tensor*,ggml_tensor*,ggml_cgraph*,void* p){auto* u=(User*)p;++*u->count;return u->handled;};ggml_backend_sched_split split;ggml_backend_sched_copy_input(&s,&split,&f.src);CHECK(called==1&&f.g->queue.waits==1);CHECK(f.cpu_state.waits==(handled?0:1));CHECK(f.g->queue.writes==(handled?0:1));}
    for(bool input:{false,true})for(bool accepted:{false,true}){Fixture f;f.src.flags=input?GGML_TENSOR_FLAG_INPUT:0;if(!accepted)f.gpu.iface.cpy_tensor_async=nullptr;Event e{0,0,&f.gpu};ggml_backend_sched s;s.backends[0]=&f.gpu;s.events[0][0]=&e;s.source=&f.cpu;s.copy=&f.dst;ggml_backend_sched_split split;ggml_backend_sched_copy_input(&s,&split,&f.src);CHECK(e.waits==(input?0:1));CHECK(e.syncs==(input?1:(accepted?0:1)));f.g->queue.drain();f.copied(5);}
    {Fixture f;std::vector<uint8_t> dst(64,0);ggml_backend_buffer hb{{host_base,host_set,host_get,nullptr},&f.host_type,dst.data(),64};ggml_tensor t{nullptr,&hb,dst.data()+8,0,16,0,0};CpuState ds;ggml_backend cpu=f.cpu;cpu.context=&ds;ggml_backend_tensor_copy_async(&f.cpu,&cpu,&f.src,&t);CHECK(f.cpu_state.waits==1&&ds.waits==1&&f.g->queue.waits==0&&f.g->queue.writes==0);for(int i=0;i<16;i++)CHECK(dst[8+i]==5+i);}
    {Fixture f;ggml_backend_tensor_copy_async(&f.cpu,&f.gpu,&f.src,&f.src);CHECK(f.cpu_state.waits==0&&f.g->queue.waits==0&&f.g->queue.writes==0);}
}
static void errors(){
    {Fixture f;f.cpu_state.fail=true;throws([&]{f.host_copy();},"cpu");CHECK(f.cpu_state.waits==1&&f.g->queue.writes==0&&f.g->queue.waits==0);}
    {Fixture f;f.g->queue.fail_write=true;throws([&]{f.host_copy();},"write");CHECK(f.cpu_state.waits==1&&f.g->queue.writes==0);}
    {Fixture f;f.g->device.fail_encoder=true;throws([&]{f.gpu_copy();},"encoder");CHECK(f.g->queue.submissions==0);}
    {Fixture f;f.g->queue.fail_wait=true;CHECK(f.host_copy());throws([&]{ggml_backend_synchronize(&f.gpu);},"queue");CHECK(f.g->queue.waits==1);}
}
int main(){host_success_and_lifetime();host_guards();gpu_order_guards_and_fallback();scheduler_and_cpu_mode();errors();std::cout<<"PASS: "<<checks<<" actual-helper ordering/lifetime/guard/fallback/scheduler checks (deferred CPU queue)\n";}
