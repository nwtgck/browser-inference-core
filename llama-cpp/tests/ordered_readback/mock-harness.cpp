// Deterministic CPU queue model; this does not validate a WebGPU implementation.
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace wgpu {
enum class BufferUsage { None = 0, CopySrc = 1, CopyDst = 2, MapRead = 4 };
static BufferUsage operator|(BufferUsage a, BufferUsage b) {
    return BufferUsage(int(a) | int(b));
}
enum class MapMode { Read };
enum class CallbackMode { AllowSpontaneous };
enum class WaitStatus { Success, TimedOut, Error };
enum class MapAsyncStatus { Success, Error, Aborted, CallbackCancelled };
enum class QueueWorkDoneStatus { Success, Error, CallbackCancelled };
using StringView = std::string;
struct Future {
    std::function<void()> complete;
    bool mapping;
};
} // namespace wgpu
struct runtime {
    std::vector<std::function<void()>> pending;
    std::vector<std::string> trace;
    int submissions = 0, queue_waits = 0, map_waits = 0, allocations = 0, destroys = 0;
    int mapped_reads = 0, unmaps = 0;
    wgpu::WaitStatus wait_status = wgpu::WaitStatus::Success;
    wgpu::MapAsyncStatus map_status = wgpu::MapAsyncStatus::Success;
    wgpu::QueueWorkDoneStatus queue_status = wgpu::QueueWorkDoneStatus::Success;
    std::function<void()> before_map_completion;
    void drain() {
        auto work = std::move(pending);
        pending.clear();
        for (auto & f : work) {
            f();
        }
    }
};
namespace wgpu {
struct storage {
    runtime * r;
    std::vector<uint8_t> bytes;
    BufferUsage usage;
    bool mapped = false, map_pending = false, destroyed = false;
};
struct Buffer {
    std::shared_ptr<storage> p;
    Buffer() = default;
    Buffer(runtime & r, size_t size, BufferUsage usage)
        : p(std::make_shared<storage>(storage{&r, std::vector<uint8_t>(size), usage})) {}
    bool operator==(std::nullptr_t) const {
        return !p;
    }
    explicit operator bool() const {
        return bool(p);
    }
    size_t GetSize() const {
        return p->bytes.size();
    }
    void Destroy() {
        assert(!p->mapped && !p->map_pending);
        p->destroyed = true;
        ++p->r->destroys;
    }
    template <class Callback>
    Future MapAsync(MapMode, size_t offset, size_t size, CallbackMode, Callback callback) const {
        assert(!p->destroyed && !p->mapped && !p->map_pending);
        assert(offset % 8 == 0 && size % 4 == 0 && offset + size <= p->bytes.size());
        assert((int(p->usage) & int(BufferUsage::MapRead)) != 0);
        p->r->trace.push_back("map-request");
        p->map_pending = true;
        return {[s = p, callback] {
                    s->r->drain();
                    if (s->r->before_map_completion) {
                        s->r->before_map_completion();
                    }
                    s->r->trace.push_back("map-complete");
                    s->map_pending = false;
                    s->mapped = s->r->map_status == MapAsyncStatus::Success;
                    callback(s->r->map_status, "map diagnostic");
                },
                true};
    }
    const void * GetConstMappedRange(size_t offset, size_t size) const {
        assert(p->mapped && !p->map_pending && !p->destroyed && offset + size <= p->bytes.size());
        ++p->r->mapped_reads;
        p->r->trace.push_back("mapped-range");
        return p->bytes.data() + offset;
    }
    void Unmap() {
        assert(p->mapped);
        p->mapped = false;
        ++p->r->unmaps;
        p->r->trace.push_back("unmap");
    }
};
struct BufferDescriptor {
    size_t size;
    BufferUsage usage;
    const char * label;
    bool mappedAtCreation;
};
struct CommandBuffer {
    std::function<void()> run;
};
struct CommandEncoder {
    runtime * r;
    CommandBuffer command;
    void CopyBufferToBuffer(Buffer src, size_t src_offset, Buffer dst, size_t dst_offset, size_t size) {
        assert(src.p->r == r && dst.p->r == r);
        assert(!src.p->destroyed && !dst.p->destroyed && !dst.p->mapped && !dst.p->map_pending);
        assert(src_offset % 4 == 0 && dst_offset % 4 == 0 && size % 4 == 0);
        assert(src_offset + size <= src.GetSize() && dst_offset + size <= dst.GetSize());
        assert((int(src.p->usage) & int(BufferUsage::CopySrc)) != 0);
        assert((int(dst.p->usage) & int(BufferUsage::CopyDst)) != 0);
        r->trace.push_back("encode-copy");
        command.run = [=] {
            src.p->r->trace.push_back("execute-copy");
            std::copy_n(src.p->bytes.begin() + src_offset, size, dst.p->bytes.begin() + dst_offset);
        };
    }
    CommandBuffer Finish() {
        return command;
    }
};
struct Device {
    runtime * r;
    Buffer CreateBuffer(const BufferDescriptor * desc) {
        assert(!desc->mappedAtCreation);
        ++r->allocations;
        return Buffer(*r, desc->size, desc->usage);
    }
    CommandEncoder CreateCommandEncoder() {
        return {r, {}};
    }
};
struct Queue {
    runtime * r;
    void Submit(size_t count, const CommandBuffer * commands) {
        for (size_t i = 0; i < count; ++i) {
            ++r->submissions;
            r->trace.push_back("submit-copy");
            r->pending.push_back(commands[i].run);
        }
    }
    template <class Callback> Future OnSubmittedWorkDone(CallbackMode, Callback callback) {
        r->trace.push_back("queue-wait-request");
        return {[r = r, callback] {
                    r->drain();
                    callback(r->queue_status, "queue diagnostic");
                },
                false};
    }
};
struct Instance {
    runtime * r;
    WaitStatus WaitAny(Future future, uint64_t) {
        (future.mapping ? r->map_waits : r->queue_waits)++;
        r->trace.push_back(future.mapping ? "map-wait" : "queue-wait");
        if (r->wait_status == WaitStatus::Success) {
            future.complete();
        }
        return r->wait_status;
    }
};
} // namespace wgpu
struct webgpu_global_context_struct {
    runtime r;
    wgpu::Instance instance{&r};
    wgpu::Device device{&r};
    wgpu::Queue queue{&r};
    wgpu::Buffer get_tensor_staging_buf;
    std::recursive_mutex mutex;
};
using webgpu_global_context = std::shared_ptr<webgpu_global_context_struct>;
struct webgpu_context_struct {
    webgpu_global_context global_ctx;
};
struct ggml_backend_webgpu_context {
    std::shared_ptr<webgpu_context_struct> webgpu_ctx;
};
struct ggml_backend_webgpu_buffer_context {
    wgpu::Buffer buffer;
    std::string label;
    webgpu_global_context global_ctx;
};
struct ggml_backend_buffer;
struct ggml_backend;
using ggml_backend_buffer_t = ggml_backend_buffer *;
using ggml_backend_t = ggml_backend *;
struct ggml_tensor {
    ggml_tensor * view_src = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    void * data = nullptr;
    size_t view_offs = 0, nbytes = 32;
};
using get_hook = void (*)(ggml_backend_t, const ggml_tensor *, void *, size_t, size_t);
struct ggml_backend_buffer {
    struct {
        void (*get_tensor)(ggml_backend_buffer_t, const ggml_tensor *, void *, size_t, size_t);
    } iface;
    void * context;
};
struct ggml_backend {
    struct {
        get_hook get_tensor_async;
        void (*synchronize)(ggml_backend_t);
    } iface;
    void * context;
};
static void * const webgpu_ptr_base = (void *)(uintptr_t)0x1000;
static size_t ggml_nbytes(const ggml_tensor * tensor) {
    return tensor->nbytes;
}
template <class... Args> [[noreturn]] static void mock_abort(const char * format, Args... args) {
    char message[512];
    std::snprintf(message, sizeof(message), format, args...);
    throw std::runtime_error(message);
}
#define GGML_ASSERT(condition)                                                                                         \
    do {                                                                                                               \
        if (!(condition))                                                                                              \
            throw std::logic_error(#condition);                                                                        \
    } while (0)
#define GGML_ABORT(...) mock_abort(__VA_ARGS__)
#define WEBGPU_LOG_DEBUG(message) ((void)0)
#define WEBGPU_CPU_PROFILE_TOTAL_START(id)
#define WEBGPU_CPU_PROFILE_TOTAL_END(id, ctx)
#define WEBGPU_RUNTIME_WAIT_TIMEOUT_MS 30000u
#define WEBGPU_RUNTIME_WAIT_TIMEOUT_NS 30000000000ull
// @ACTUAL_HELPERS@
// @ACTUAL_GENERIC_GET@
// @ACTUAL_SYNCHRONIZE@
// @ACTUAL_HOOK@
// @ACTUAL_ASYNC_ENTRY@

struct fixture {
    webgpu_global_context g = std::make_shared<webgpu_global_context_struct>();
    ggml_backend_webgpu_context backend_ctx{std::make_shared<webgpu_context_struct>(webgpu_context_struct{g})};
    ggml_backend backend{{// @REGISTERED_HOOK@
                          , ggml_backend_webgpu_synchronize},
                         &backend_ctx};
    ggml_backend_webgpu_buffer_context buffer_ctx{wgpu::Buffer(g->r, 128, wgpu::BufferUsage::CopySrc), "tensor", g};
    ggml_backend_buffer buffer{{ggml_backend_webgpu_buffer_get_tensor}, &buffer_ctx};
    ggml_tensor tensor{nullptr, &buffer, (void *)0x1008, 0, 32};
    fixture() {
        for (size_t i = 0; i < 128; ++i) {
            buffer_ctx.buffer.p->bytes[i] = uint8_t(i);
        }
    }
    void producer(uint8_t value) {
        g->r.pending.push_back([this, value] {
            g->r.trace.push_back("execute-producer");
            std::fill(buffer_ctx.buffer.p->bytes.begin(), buffer_ctx.buffer.p->bytes.end(), value);
        });
    }
    void get(void * data, size_t offset = 0, size_t size = 16) {
        ggml_backend_tensor_get_async(&backend, &tensor, data, offset, size);
    }
};
static size_t position(const runtime & r, const char * event) {
    const auto it = std::find(r.trace.begin(), r.trace.end(), event);
    assert(it != r.trace.end());
    return size_t(it - r.trace.begin());
}
static void assert_success_order(const fixture & f) {
    const runtime & r = f.g->r;
    assert(position(r, "submit-copy") < position(r, "map-request"));
    assert(position(r, "map-request") < position(r, "map-wait"));
    assert(position(r, "execute-producer") < position(r, "execute-copy"));
    assert(position(r, "execute-copy") < position(r, "map-complete"));
    assert(position(r, "map-complete") < position(r, "mapped-range"));
    assert(position(r, "mapped-range") < position(r, "unmap"));
    assert(r.submissions == 1 && r.map_waits == 1 && r.unmaps == 1 && r.mapped_reads == 1);
}
static void foreign_get(ggml_backend_buffer_t buffer, const ggml_tensor *, void * data, size_t, size_t size) {
    auto * r = static_cast<runtime *>(buffer->context);
    assert(r->queue_waits == 1 && r->pending.empty());
    r->trace.push_back("foreign-get");
    std::memset(data, 93, size);
}
template <class Action> static void expect_runtime_failure(Action action, const char * message) {
    bool failed = false;
    try {
        action();
    } catch (const std::runtime_error & error) {
        failed = true;
        assert(std::string(error.what()).find(message) != std::string::npos);
    }
    assert(failed);
}
int main() {
    // The actual optional callback eliminates the redundant queue-wait callback,
    // but both paths retain the actual staging-copy/map/unmap and output bytes.
    for (bool hook : {false, true}) {
        fixture f;
        if (!hook) {
            f.backend.iface.get_tensor_async = nullptr;
        }
        uint8_t output[20];
        std::fill_n(output, 20, 0xcc);
        bool inside_call = true;
        f.g->r.before_map_completion = [&] {
            assert(inside_call);
            for (uint8_t byte : output) {
                assert(byte == 0xcc);
            }
        };
        f.producer(77);
        f.get(output + 2);
        inside_call = false;
        assert(f.g->r.queue_waits == (hook ? 0 : 1));
        assert_success_order(f);
        for (int i = 0; i < 20; ++i) {
            assert(output[i] == (i >= 2 && i < 18 ? 77 : 0xcc));
        }
        assert(f.g->r.pending.empty());
        // No pending operation owns the caller's output. Immediate reuse after return is safe.
        std::fill_n(output, 20, 42);
        f.g->r.drain();
        for (uint8_t byte : output) {
            assert(byte == 42);
        }
    }
    // Every byte alignment and tail is handled by the unchanged real buffer getter.
    for (size_t tensor_offset = 0; tensor_offset < 4; ++tensor_offset) {
        for (size_t offset = 0; offset < 4; ++offset) {
            for (size_t size = 1; size <= 17; ++size) {
                fixture f;
                f.tensor.data = (void *)(uintptr_t)(0x1008 + tensor_offset);
                uint8_t output[21];
                std::fill_n(output, 21, 0xcc);
                f.get(output + 2, offset, size);
                for (size_t i = 0; i < 21; ++i) {
                    assert(output[i] == (i >= 2 && i < size + 2 ? uint8_t(8 + tensor_offset + offset + i - 2) : 0xcc));
                }
                assert(f.g->r.queue_waits == 0 && f.g->r.map_waits == 1);
            }
        }
    }
    // A view uses view_src's buffer and data even when its own buffer is foreign.
    {
        fixture f;
        ggml_tensor base = f.tensor;
        ggml_backend_buffer foreign{{foreign_get}, nullptr};
        f.tensor.view_src = &base;
        f.tensor.buffer = &foreign;
        f.tensor.data = (void *)0xdead;
        f.tensor.view_offs = 3;
        uint8_t output[5];
        f.get(output, 2, 5);
        assert(output[0] == 13 && output[4] == 17 && f.g->r.queue_waits == 0);
    }
    // Repeated reads unmap before reuse; growth destroys only the old unmapped staging buffer.
    {
        fixture f;
        uint8_t output[32];
        f.get(output, 0, 4);
        f.producer(19);
        f.get(output, 0, 3);
        assert(output[0] == 19 && f.g->r.allocations == 1 && f.g->r.destroys == 0);
        f.get(output, 0, 32);
        assert(f.g->r.allocations == 2 && f.g->r.destroys == 1 && f.g->r.unmaps == 3);
        assert(!f.g->get_tensor_staging_buf.p->mapped && !f.g->get_tensor_staging_buf.p->map_pending);
    }
    // Zero-byte reads retain the previous synchronization without allocation or mapping.
    {
        fixture f;
        f.producer(21);
        f.get(nullptr, 0, 0);
        assert(f.g->r.queue_waits == 1 && f.g->r.pending.empty());
        assert(f.g->r.submissions == 0 && f.g->r.map_waits == 0 && f.g->r.allocations == 0);
    }
    // Foreign buffers are never cast to a WebGPU context, and retain the wait-before-get.
    {
        fixture f;
        ggml_backend_buffer foreign{{foreign_get}, &f.g->r};
        f.tensor.buffer = &foreign;
        f.producer(22);
        uint8_t output[4];
        f.get(output, 0, 4);
        assert(output[0] == 93 && f.g->r.map_waits == 0);
        assert(position(f.g->r, "queue-wait") < position(f.g->r, "foreign-get"));
    }
    // A WebGPU tensor owned by another context preserves both contexts' waits.
    {
        fixture f, other;
        f.tensor.buffer = &other.buffer;
        f.producer(23);
        other.producer(24);
        uint8_t output[4];
        f.get(output, 0, 4);
        assert(output[0] == 24 && f.g->r.queue_waits == 1 && other.g->r.map_waits == 1);
        assert(f.g->r.pending.empty() && other.g->r.pending.empty());
    }
    // Actual public entry assertions reject an invalid destination range before any queue work.
    {
        fixture f;
        uint8_t output[4] = {};
        bool failed = false;
        try {
            f.get(output, 31, 4);
        } catch (const std::logic_error &) {
            failed = true;
        }
        assert(failed && f.g->r.submissions == 0 && f.g->r.queue_waits == 0 && f.g->r.map_waits == 0);
    }
    // Existing map/wait status checks must abort before mapped access or host memcpy.
    for (auto status :
         {wgpu::MapAsyncStatus::Error, wgpu::MapAsyncStatus::Aborted, wgpu::MapAsyncStatus::CallbackCancelled}) {
        fixture f;
        f.g->r.map_status = status;
        uint8_t output[4] = {1, 2, 3, 4};
        expect_runtime_failure([&] { f.get(output, 0, 4); }, "Buffer map failed with status");
        assert(output[0] == 1 && output[3] == 4 && f.g->r.mapped_reads == 0 && f.g->r.unmaps == 0);
    }
    for (auto status : {wgpu::WaitStatus::TimedOut, wgpu::WaitStatus::Error}) {
        fixture f;
        f.g->r.wait_status = status;
        uint8_t output[4] = {1, 2, 3, 4};
        expect_runtime_failure([&] { f.get(output, 0, 4); }, status == wgpu::WaitStatus::TimedOut
                                                                 ? "Buffer map wait timed out"
                                                                 : "Buffer map wait failed");
        assert(output[0] == 1 && output[3] == 4 && f.g->r.mapped_reads == 0);
    }
    // Fallback errors retain their existing queue-work diagnostic and do not read the foreign buffer.
    {
        fixture f;
        ggml_backend_buffer foreign{{foreign_get}, &f.g->r};
        f.tensor.buffer = &foreign;
        f.g->r.queue_status = wgpu::QueueWorkDoneStatus::Error;
        uint8_t output[4] = {1, 2, 3, 4};
        expect_runtime_failure([&] { f.get(output, 0, 4); }, "Queue work failed with status");
        assert(output[0] == 1 && output[3] == 4 && f.g->r.map_waits == 0);
    }
    std::cout
        << "PASS: actual ordered_readback hook, generic entry, staging getter and wait/map helpers (CPU model only)\n";
}
