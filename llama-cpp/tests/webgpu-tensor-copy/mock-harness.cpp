// CPU-only instrumented WebGPU model. This does not validate Dawn or browser execution.
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>
#include <limits>
namespace wgpu {
enum class BufferUsage { None=0, CopySrc=1, CopyDst=2 };
BufferUsage operator&(BufferUsage a, BufferUsage b) { return BufferUsage(int(a)&int(b)); }
struct Storage { std::vector<uint8_t> bytes; bool destroyed=false; BufferUsage usage; };
struct Buffer {
    std::shared_ptr<Storage> p;
    Buffer(size_t n=64, BufferUsage usage=BufferUsage(3)):p(std::make_shared<Storage>(Storage{std::vector<uint8_t>(n),false,usage})) {}
    Storage * Get() const { return p.get(); }
    BufferUsage GetUsage() const { return p->usage; }
    void Destroy() { p->destroyed=true; }
};
struct CommandBuffer { std::function<void()> run; };
struct CommandEncoder {
    CommandBuffer cmd;
    void CopyBufferToBuffer(Buffer a,size_t x,Buffer b,size_t y,size_t n) {
        assert(!a.p->destroyed && !b.p->destroyed && a.Get()!=b.Get());
        assert(x%4==0 && y%4==0 && n%4==0);
        assert(x<=a.p->bytes.size() && n<=a.p->bytes.size()-x);
        assert(y<=b.p->bytes.size() && n<=b.p->bytes.size()-y);
        cmd.run=[=]{ std::copy_n(a.p->bytes.begin()+x,n,b.p->bytes.begin()+y); };
    }
    CommandBuffer Finish() { return cmd; }
};
struct Device { CommandEncoder CreateCommandEncoder() { return {}; } };
struct Queue {
    int submissions=0;
    std::vector<std::function<void()>> pending;
    void Submit(size_t n, const CommandBuffer * c) { for(size_t i=0;i<n;i++) { pending.push_back(c[i].run); submissions++; } }
    void WriteBuffer(Buffer b,size_t off,const void *data,size_t n) {
        std::vector<uint8_t> bytes((const uint8_t*)data,(const uint8_t*)data+n);
        pending.push_back([=]{std::copy(bytes.begin(),bytes.end(),b.p->bytes.begin()+off);});
    }
    void drain() { auto commands=std::move(pending); pending.clear(); for(auto &f:commands) f(); }
};
}
struct global_context { wgpu::Device device; wgpu::Queue queue; };
struct ggml_backend_webgpu_buffer_context { wgpu::Buffer buffer; std::shared_ptr<global_context> global_ctx; };
struct ggml_backend_buffer;
using ggml_backend_buffer_t=ggml_backend_buffer*;
struct ggml_tensor { ggml_tensor* view_src=nullptr; ggml_backend_buffer_t buffer=nullptr; void* data=nullptr; size_t view_offs=0; size_t nbytes=16; int layout=0; };
struct iface_t { void*(*get_base)(ggml_backend_buffer_t); bool(*cpy_tensor)(ggml_backend_buffer_t,const ggml_tensor*,ggml_tensor*); };
struct ggml_backend_buffer { iface_t iface; void* context; size_t size; };
static void* const webgpu_ptr_base=(void*)(uintptr_t)0x1000;
static size_t ggml_nbytes(const ggml_tensor* t) { return t->nbytes; }
static bool ggml_are_same_layout(const ggml_tensor* a,const ggml_tensor* b) { return a->nbytes==b->nbytes && a->layout==b->layout; }
#define GGML_UNUSED(x) (void)(x)
#define GGML_ASSERT(x) assert(x)
#define GGML_LOG_DEBUG(...) ((void)0)
// @EXTRACTED_HOOK@
int gets=0,sets=0;
static bool ggml_backend_buffer_is_host(ggml_backend_buffer_t) { return false; }
static void ggml_backend_tensor_get(const ggml_tensor* t,void* data,size_t off,size_t n) {
    gets++;
    auto* b=t->view_src?t->view_src->buffer:t->buffer;
    auto* c=(ggml_backend_webgpu_buffer_context*)b->context;
    c->global_ctx->queue.drain();
    std::memcpy(data,c->buffer.p->bytes.data()+ggml_webgpu_tensor_offset(t)+off,n);
}
static void ggml_backend_tensor_set(ggml_tensor* t,const void* data,size_t off,size_t n) {
    sets++;
    auto* b=t->view_src?t->view_src->buffer:t->buffer;
    auto* c=(ggml_backend_webgpu_buffer_context*)b->context;
    c->global_ctx->queue.WriteBuffer(c->buffer,ggml_webgpu_tensor_offset(t)+off,data,n);
}
// @EXTRACTED_GENERIC_COPY@
struct Fixture {
    std::shared_ptr<global_context> g=std::make_shared<global_context>();
    ggml_backend_webgpu_buffer_context a{wgpu::Buffer(),g},b{wgpu::Buffer(),g};
    ggml_backend_buffer ab{{ggml_backend_webgpu_buffer_get_base,ggml_backend_webgpu_buffer_cpy_tensor},&a,64};
    ggml_backend_buffer bb{{ggml_backend_webgpu_buffer_get_base,ggml_backend_webgpu_buffer_cpy_tensor},&b,64};
    ggml_tensor src{nullptr,&ab,(void*)0x1004,0,16,0},dst{nullptr,&bb,(void*)0x1008,0,16,0};
    Fixture() { for(size_t i=0;i<64;i++) a.buffer.p->bytes[i]=uint8_t(i+1); std::fill(b.buffer.p->bytes.begin(),b.buffer.p->bytes.end(),0xcc); }
    bool copy() { return ggml_backend_webgpu_buffer_cpy_tensor(&bb,&src,&dst); }
};
static void* foreign_base(ggml_backend_buffer_t) { return nullptr; }
int main() {
    { Fixture f; assert(f.copy()); assert(f.g->queue.submissions==1); assert(f.b.buffer.p->bytes[8]==0xcc); f.g->queue.drain(); for(int i=0;i<16;i++) assert(f.b.buffer.p->bytes[8+i]==5+i); assert(f.b.buffer.p->bytes[7]==0xcc&&f.b.buffer.p->bytes[24]==0xcc); }
    { Fixture f; uint8_t x[16]; std::fill_n(x,16,77); f.g->queue.WriteBuffer(f.a.buffer,4,x,16); assert(f.copy()); std::fill_n(x,16,99); f.g->queue.WriteBuffer(f.a.buffer,4,x,16); f.a.buffer.Destroy(); f.g->queue.drain(); assert(f.b.buffer.p->bytes[8]==77); assert(f.a.buffer.p->bytes[4]==99); }
    { Fixture f; ggml_tensor base=f.src; f.src.view_src=&base; f.src.buffer=nullptr; f.src.data=(void*)0xdead; f.src.view_offs=8; ggml_tensor db=f.dst; f.dst.view_src=&db; f.dst.buffer=nullptr; f.dst.data=(void*)0xbeef; f.dst.view_offs=12; assert(f.copy()); f.g->queue.drain(); assert(f.b.buffer.p->bytes[20]==13); }
    { Fixture f; f.a.global_ctx=std::make_shared<global_context>(); assert(!f.copy()); assert(f.g->queue.submissions==0); }
    { Fixture f; f.ab.iface.get_base=foreign_base; f.ab.context=nullptr; assert(!f.copy()); }
    { Fixture f; f.src.buffer=nullptr; assert(!f.copy()); }
    { Fixture f; f.dst.layout=1; assert(!f.copy()); }
    { Fixture f; f.src.nbytes=f.dst.nbytes=0; assert(f.copy()); assert(f.g->queue.submissions==0); }
    { Fixture f; f.b.buffer=f.a.buffer; f.dst.data=f.src.data; assert(f.copy()); assert(f.g->queue.submissions==0); f.dst.data=(void*)0x1024; assert(!f.copy()); f.dst.data=(void*)0x1008; assert(!f.copy()); }
    for(int c=0;c<3;c++) { Fixture f; if(c==0) f.src.data=(void*)0x1005; if(c==1) f.dst.data=(void*)0x1009; if(c==2) f.src.nbytes=f.dst.nbytes=15; assert(!f.copy()); assert(f.g->queue.submissions==0); }
    for(int c=0;c<4;c++) { Fixture f; if(c==0) f.src.data=(void*)0x1044; if(c==1) f.dst.data=(void*)0x1044; if(c==2) f.src.nbytes=f.dst.nbytes=std::numeric_limits<size_t>::max(); if(c==3) f.src.view_offs=std::numeric_limits<size_t>::max(); assert(!f.copy()); assert(f.g->queue.submissions==0); }
    for(int c=0;c<2;c++) { Fixture f; (c==0?f.a:f.b).buffer.p->usage=wgpu::BufferUsage::None; assert(!f.copy()); }
    { Fixture f; f.src.data=f.dst.data=(void*)0x1030; assert(f.copy()); f.g->queue.drain(); assert(f.b.buffer.p->bytes[63]==64); }
    // Exercise the actual generic fallback, including overlapping source/destination.
    { Fixture f; f.b.buffer=f.a.buffer; gets=sets=0; ggml_backend_tensor_copy(&f.src,&f.dst); assert(gets==1&&sets==1); f.g->queue.drain(); for(int i=0;i<16;i++) assert(f.b.buffer.p->bytes[8+i]==5+i); }
    { Fixture f; f.src.data=(void*)0x1005; gets=sets=0; ggml_backend_tensor_copy(&f.src,&f.dst); assert(gets==1&&sets==1); f.g->queue.drain(); assert(f.b.buffer.p->bytes[8]==6); }
    { Fixture f; gets=sets=0; ggml_backend_tensor_copy(&f.src,&f.dst); assert(gets==0&&sets==0); uint8_t out[16]; ggml_backend_tensor_get(&f.dst,out,0,16); assert(out[0]==5&&out[15]==20); }
    { Fixture f; f.a.global_ctx=std::make_shared<global_context>(); gets=sets=0; uint8_t x[16]; std::fill_n(x,16,42); f.a.global_ctx->queue.WriteBuffer(f.a.buffer,4,x,16); ggml_backend_tensor_copy(&f.src,&f.dst); assert(gets==1&&sets==1); f.g->queue.drain(); assert(f.b.buffer.p->bytes[8]==42); }
    { Fixture f; f.src.nbytes=f.dst.nbytes=15; gets=sets=0; ggml_backend_tensor_copy(&f.src,&f.dst); assert(gets==1&&sets==1); f.g->queue.drain(); assert(f.b.buffer.p->bytes[22]==19&&f.b.buffer.p->bytes[23]==0xcc); }
    // Copy snapshot, mutate live state, restore snapshot, then read the live tensor.
    { Fixture f; gets=sets=0; ggml_backend_tensor_copy(&f.src,&f.dst); uint8_t x[16]={}; f.g->queue.WriteBuffer(f.a.buffer,4,x,16); ggml_backend_tensor_copy(&f.dst,&f.src); uint8_t out[16]; ggml_backend_tensor_get(&f.src,out,0,16); assert(out[0]==5&&out[15]==20); assert(gets==1&&sets==0&&f.g->queue.submissions==2); }
    std::cout << "PASS: hook/fallback/queue scenarios (CPU mock; no WebGPU runtime)\n";
}
