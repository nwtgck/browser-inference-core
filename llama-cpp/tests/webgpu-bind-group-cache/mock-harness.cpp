// CPU-only deferred queue and reference-counted native-object model.
// This executes actual extracted cache/build/graph/free functions, not a GPU.
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <functional>

static uintptr_t next_handle = 16;
static std::vector<uintptr_t> reusable_handles;
static int live_resources = 0;
struct resource {
    uintptr_t handle;
    int device;
    bool destroyed = false;
    std::vector<uint8_t> bytes;
    explicit resource(int device, size_t size = 0) : device(device), bytes(size) {
        if (reusable_handles.empty()) { handle = next_handle++; }
        else { handle = reusable_handles.back(); reusable_handles.pop_back(); }
        ++live_resources;
    }
    ~resource() { reusable_handles.push_back(handle); --live_resources; }
};
struct statistics {
    int device = 1;
    size_t layouts = 0, groups = 0, writes = 0, submissions = 0, dispatches = 0;
    bool fail_group = false, throw_group = false;
    std::vector<std::function<void()>> pending;
    std::vector<uint32_t> observed_params;
    void drain() { auto commands = std::move(pending); pending.clear(); for (auto & command : commands) command(); }
};
namespace wgpu {
enum class BufferUsage { CopyDst = 1, Uniform = 2 };
static BufferUsage operator|(BufferUsage a, BufferUsage b) { return BufferUsage(int(a) | int(b)); }
struct Buffer {
    std::shared_ptr<resource> p;
    Buffer(std::nullptr_t = nullptr) {}
    explicit Buffer(std::shared_ptr<resource> p) : p(std::move(p)) {}
    explicit operator bool() const { return bool(p); }
    bool operator!=(std::nullptr_t) const { return bool(p); }
    void * Get() const { return p ? reinterpret_cast<void *>(p->handle) : nullptr; }
    size_t GetSize() const { return p->bytes.size(); }
    void Destroy() { p->destroyed = true; }
};
struct Opaque {
    bool present = false;
    explicit operator bool() const { return present; }
};
struct BindGroupLayout;
struct ComputePipeline {
    std::shared_ptr<resource> p;
    statistics * stats = nullptr;
    explicit operator bool() const { return bool(p); }
    void * Get() const { return p ? reinterpret_cast<void *>(p->handle) : nullptr; }
    BindGroupLayout GetBindGroupLayout(uint32_t index) const;
};
struct BindGroupLayout { ComputePipeline pipeline; uint32_t index; };
BindGroupLayout ComputePipeline::GetBindGroupLayout(uint32_t index) const {
    assert(p);
    ++stats->layouts;
    return {*this, index};
}
struct BindGroupEntry {
    const void * nextInChain = nullptr;
    uint32_t binding = 0;
    Buffer buffer;
    uint64_t offset = 0, size = std::numeric_limits<uint64_t>::max();
    Opaque sampler, textureView;
};
struct group {
    ComputePipeline pipeline;
    uint32_t group_index;
    std::vector<BindGroupEntry> entries;
};
struct BindGroup {
    std::shared_ptr<group> p;
    explicit operator bool() const { return bool(p); }
    void * Get() const { return p.get(); }
};
struct BindGroupDescriptor {
    BindGroupLayout layout;
    size_t entryCount;
    const BindGroupEntry * entries;
    const char * label;
};
struct BufferDescriptor { size_t size; BufferUsage usage; const char * label; bool mappedAtCreation; };
struct PassTimestampWrites { int querySet; uint32_t beginningOfPassWriteIndex, endOfPassWriteIndex; };
struct ComputePassDescriptor { const PassTimestampWrites * timestampWrites; };
struct invocation { BindGroup group; ComputePipeline pipeline; };
using commands = std::vector<invocation>;
struct ComputePassEncoder {
    std::shared_ptr<commands> encoded;
    BindGroup group;
    ComputePipeline pipeline;
    ComputePassEncoder(std::nullptr_t = nullptr) {}
    explicit ComputePassEncoder(std::shared_ptr<commands> encoded) : encoded(std::move(encoded)) {}
    explicit operator bool() const { return bool(encoded); }
    void SetPipeline(ComputePipeline value) { pipeline = std::move(value); }
    void SetBindGroup(uint32_t, BindGroup value) { group = std::move(value); }
    void DispatchWorkgroups(uint32_t, uint32_t, uint32_t) { assert(encoded && group); encoded->push_back({group, pipeline}); }
    void End() {}
};
struct CommandBuffer { std::shared_ptr<commands> encoded; };
struct CommandEncoder {
    std::shared_ptr<commands> encoded;
    CommandEncoder(std::nullptr_t = nullptr) {}
    explicit CommandEncoder(std::shared_ptr<commands> encoded) : encoded(std::move(encoded)) {}
    ComputePassEncoder BeginComputePass(const ComputePassDescriptor * = nullptr) { return ComputePassEncoder(encoded); }
    CommandBuffer Finish() { return {encoded}; }
};
struct Device {
    std::shared_ptr<statistics> stats;
    Buffer CreateBuffer(const BufferDescriptor * descriptor) {
        return Buffer(std::make_shared<resource>(stats->device, descriptor->size));
    }
    BindGroup CreateBindGroup(const BindGroupDescriptor * descriptor) {
        ++stats->groups;
        if (stats->throw_group) { stats->throw_group = false; throw std::runtime_error("create failure"); }
        if (stats->fail_group) { stats->fail_group = false; return {}; }
        assert(descriptor->layout.pipeline.p->device == stats->device);
        for (size_t i = 0; i < descriptor->entryCount; ++i) {
            const auto & entry = descriptor->entries[i];
            if (entry.buffer) { assert(entry.buffer.p->device == stats->device && !entry.buffer.p->destroyed); }
        }
        return {std::make_shared<group>(group{descriptor->layout.pipeline, descriptor->layout.index,
                                            {descriptor->entries, descriptor->entries + descriptor->entryCount}})};
    }
    CommandEncoder CreateCommandEncoder() { return CommandEncoder(std::make_shared<commands>()); }
};
struct Queue {
    std::shared_ptr<statistics> stats;
    void WriteBuffer(Buffer buffer, size_t offset, const void * data, size_t size) {
        assert(!buffer.p->destroyed && offset + size <= buffer.GetSize());
        ++stats->writes;
        std::vector<uint8_t> owned(static_cast<const uint8_t *>(data), static_cast<const uint8_t *>(data) + size);
        stats->pending.push_back([buffer, offset, owned = std::move(owned)] {
            std::copy(owned.begin(), owned.end(), buffer.p->bytes.begin() + offset);
        });
    }
    void Submit(size_t count, const CommandBuffer * submitted) {
        ++stats->submissions;
        for (size_t i = 0; i < count; ++i) {
            stats->pending.push_back([stats = stats, encoded = submitted[i].encoded] {
                for (const auto & command : *encoded) {
                    assert(command.group.p->pipeline.Get() == command.pipeline.Get());
                    const auto & entries = command.group.p->entries;
                    const auto & parameter = entries.back();
                    uint32_t value;
                    assert(!parameter.buffer.p->destroyed && parameter.size >= sizeof(value));
                    std::memcpy(&value, parameter.buffer.p->bytes.data() + parameter.offset, sizeof(value));
                    const auto & destination = entries.front();
                    assert(!destination.buffer.p->destroyed && destination.offset + sizeof(value) <= destination.buffer.GetSize());
                    std::memcpy(destination.buffer.p->bytes.data() + destination.offset, &value, sizeof(value));
                    stats->observed_params.push_back(value);
                    ++stats->dispatches;
                }
            });
        }
    }
};
}
#define GGML_ASSERT(value) do { if (!(value)) throw std::logic_error(#value); } while (0)
#define GGML_ABORT(message) throw std::runtime_error(message)
#define WEBGPU_LOG_DEBUG(message) ((void) 0)
#define WEBGPU_CPU_PROFILE_TOTAL_START(id)
#define WEBGPU_CPU_PROFILE_TOTAL_END(id, ctx)
#define ROUNDUP_POW2(x, pow2) (((x) + (pow2) - 1) & ~((pow2) - 1))
#define WEBGPU_MAX_PROFILE_QUERY_COUNT 4096u
// @ACTUAL_CREATE_BUFFER@
// @ACTUAL_PARAM_ARENA@
// @ACTUAL_CACHE@
struct webgpu_pipeline { wgpu::ComputePipeline pipeline; std::string name; };
// @ACTUAL_ENCODED_OP@
// @ACTUAL_DISPATCH_DESC@
struct webgpu_global_context_struct {
    std::shared_ptr<statistics> stats = std::make_shared<statistics>();
    wgpu::Device device{stats};
    wgpu::Queue queue{stats};
    uint32_t command_submit_batch_size = 64, max_inflight_batches = UINT32_MAX;
    std::atomic<uint64_t> buffer_generation{0};
};
using webgpu_global_context = std::shared_ptr<webgpu_global_context_struct>;
struct webgpu_context_struct {
    webgpu_global_context global_ctx;
    webgpu_param_arena param_arena;
    webgpu_bind_group_cache bind_group_cache;
    wgpu::CommandEncoder active_command_encoder;
    wgpu::ComputePassEncoder active_compute_pass;
    bool batch_compute_passes = true;
    uint32_t profile_timestamp_query_count = 0;
    int profile_timestamp_query_set = 0;
};
using webgpu_context = std::shared_ptr<webgpu_context_struct>;
struct ggml_backend_webgpu_context { webgpu_context webgpu_ctx; };
struct ggml_backend { void * context; };
using ggml_backend_t = ggml_backend *;
struct ggml_backend_webgpu_buffer_context { wgpu::Buffer buffer; webgpu_global_context global_ctx; };
struct ggml_backend_buffer { void * context; };
using ggml_backend_buffer_t = ggml_backend_buffer *;
enum ggml_status { GGML_STATUS_SUCCESS };
enum ggml_op { GGML_OP_NONE, GGML_OP_SET_ROWS };
struct ggml_tensor { ggml_op op = GGML_OP_NONE; };
struct ggml_cgraph {
    int n_nodes;
    ggml_tensor ** nodes;
    std::vector<webgpu_dispatch_desc> dispatches;
};
// @ACTUAL_MAKE_ENTRY@
// @ACTUAL_BUILD_MULTI@
static std::optional<webgpu_encoded_op> ggml_webgpu_encode(webgpu_context & ctx, ggml_cgraph * graph, int index, int & encoded) {
    encoded = 1;
    return ggml_backend_webgpu_build_multi(ctx, {graph->dispatches[index]});
}
static void ggml_backend_webgpu_wait_queue(webgpu_global_context & ctx) { ctx->stats->drain(); }
// @ACTUAL_SUBMIT@
#ifdef GGML_WEBGPU_GPU_PROFILE
static void ggml_backend_webgpu_collect_profile_results(webgpu_context &, const std::vector<std::string> &, uint32_t &) {}
#endif
static void ggml_backend_webgpu_check_set_rows(webgpu_context &, uint32_t &) {}
// @ACTUAL_GRAPH_COMPUTE@
// @ACTUAL_BUFFER_FREE@

struct fixture {
    webgpu_global_context global = std::make_shared<webgpu_global_context_struct>();
    webgpu_context ctx = std::make_shared<webgpu_context_struct>();
    ggml_backend_webgpu_context backend_context{ctx};
    ggml_backend backend{&backend_context};
    wgpu::Buffer data{std::make_shared<resource>(global->stats->device, 128)};
    wgpu::ComputePipeline pipeline{std::make_shared<resource>(global->stats->device), global->stats.get()};
    fixture() {
        ctx->global_ctx = global;
        ctx->param_arena.init(global->device, 128, 74, 256);
    }
    std::vector<wgpu::BindGroupEntry> entries() {
        return {ggml_webgpu_make_bind_group_entry(0, data, 0, 4)};
    }
    wgpu::BindGroup get(webgpu_bind_group_cache & cache, uint32_t group_index = 0) {
        return cache.get(global->device, pipeline, group_index, entries(), "test", global->buffer_generation.load());
    }
    void graph(size_t count, uint32_t seed, uint64_t offset = 0) {
        std::vector<ggml_tensor> tensors(count);
        std::vector<ggml_tensor *> nodes;
        std::vector<webgpu_dispatch_desc> dispatches;
        for (size_t i = 0; i < count; ++i) {
            nodes.push_back(&tensors[i]);
            auto binding = entries(); binding[0].offset = offset;
            dispatches.push_back({{pipeline, "test"}, {seed + uint32_t(i)}, binding, {1, 1}});
        }
        ggml_cgraph graph{int(count), nodes.data(), dispatches};
        assert(ggml_backend_webgpu_graph_compute(&backend, &graph) == GGML_STATUS_SUCCESS);
    }
};
static void assert_values(const std::vector<uint32_t> & values, size_t start, size_t count, uint32_t seed) {
    assert(values.size() >= start + count);
    for (size_t i = 0; i < count; ++i) assert(values[start + i] == seed + i);
}
int main() {
    // Every dimension in the exact key causes a miss; a repeated exact key hits.
    for (int change = 0; change < 8; ++change) {
        fixture f;
        webgpu_bind_group_cache cache;
        cache.begin_graph(0);
        auto original = f.get(cache);
        const auto groups = f.global->stats->groups, layouts = f.global->stats->layouts;
        cache.begin_graph(0);
        assert(f.get(cache).Get() == original.Get());
        assert(f.global->stats->groups == groups && f.global->stats->layouts == layouts);
        auto entries = f.entries();
        auto pipeline = f.pipeline;
        uint32_t index = 0;
        if (change == 0) pipeline = {std::make_shared<resource>(1), f.global->stats.get()};
        if (change == 1) index = 1;
        if (change == 2) entries[0].binding = 1;
        if (change == 3) entries[0].buffer = wgpu::Buffer(std::make_shared<resource>(1, 128));
        if (change == 4) entries[0].offset = 4;
        if (change == 5) entries[0].size = 8;
        if (change == 6) entries.push_back(ggml_webgpu_make_bind_group_entry(1, f.data, 4, 4));
        if (change == 7) { entries.push_back(ggml_webgpu_make_bind_group_entry(1, f.data, 4, 4)); std::swap(entries[0], entries[1]); }
        cache.begin_graph(0);
        auto changed = cache.get(f.global->device, pipeline, index, entries, "label", 0);
        assert(changed.Get() != original.Get() && f.global->stats->groups == groups + 1);
    }
    // Real build_multi and real graph loop: hit parameters still upload, 64-kernel
    // batches reset arena slices, and a changed state offset invalidates only its slots.
    {
        fixture f;
        f.graph(130, 100);
        assert(f.ctx->bind_group_cache.slots.size() == 130);
        assert(f.global->stats->groups == 130 && f.global->stats->layouts == 130);
        assert(f.ctx->bind_group_cache.slots[0].bindings.back().offset == 0);
        assert(f.ctx->bind_group_cache.slots[63].bindings.back().offset == 63 * 256);
        assert(f.ctx->bind_group_cache.slots[64].bindings.back().offset == 0);
        f.global->stats->drain(); assert_values(f.global->stats->observed_params, 0, 130, 100);
        const auto writes = f.global->stats->writes, submissions = f.global->stats->submissions;
        f.graph(130, 900);
        assert(f.global->stats->groups == 130 && f.global->stats->layouts == 130);
#ifdef GGML_WEBGPU_BATCH_PARAM_UPLOADS
        assert(f.global->stats->writes == writes + 3);
#else
        assert(f.global->stats->writes == writes + 130);
#endif
        assert(f.global->stats->submissions == submissions + 3);
        f.global->stats->drain(); assert_values(f.global->stats->observed_params, 130, 130, 900);
        f.graph(130, 1200, 4);
        assert(f.global->stats->groups == 260);
        f.global->stats->drain(); assert_values(f.global->stats->observed_params, 260, 130, 1200);
        // Long prefill-like graph to short decode-like graph drops every unused tail slot.
        std::weak_ptr<wgpu::group> tail = f.ctx->bind_group_cache.slots.back().group.p;
        f.graph(2, 1500, 4);
        assert(f.ctx->bind_group_cache.slots.size() == 2 && tail.expired());
        f.global->stats->drain();
        f.graph(0, 0);
        assert(f.ctx->bind_group_cache.slots.empty());
    }
    // One dispatch's parameter slice offset is part of the key, even when tensor bindings match.
    {
        fixture f;
        auto & cache = f.ctx->bind_group_cache;
        cache.begin_graph(0);
        auto values = f.entries();
        values.push_back(ggml_webgpu_make_bind_group_entry(1, f.ctx->param_arena.buffer, 0, 128));
        auto first = cache.get(f.global->device, f.pipeline, 0, values, "params", 0);
        cache.begin_graph(0); values.back().offset = 256;
        auto shifted = cache.get(f.global->device, f.pipeline, 0, values, "params", 0);
        assert(first.Get() != shifted.Get());
        cache.begin_graph(0); values.back().size = 64;
        assert(cache.get(f.global->device, f.pipeline, 0, values, "params", 0).Get() != shifted.Get());
    }
    // Unsupported chains/non-buffer resources and too many entries use original creation.
    for (int kind = 0; kind < 5; ++kind) {
        fixture f;
        webgpu_bind_group_cache cache;
        cache.begin_graph(0); f.get(cache);
        auto values = f.entries();
        if (kind == 0) values[0].nextInChain = reinterpret_cast<void *>(uintptr_t(1));
        if (kind == 1) values[0].sampler.present = true;
        if (kind == 2) values[0].textureView.present = true;
        if (kind == 3) values[0].buffer = nullptr;
        if (kind == 4) while (values.size() <= cache.max_buffer_entries) values.push_back(values[0]);
        for (int repeat = 0; repeat < 2; ++repeat) {
            cache.begin_graph(0);
            cache.get(f.global->device, f.pipeline, 0, values, "bypass", 0);
            assert(cache.slots.size() == 1 && !cache.slots[0].group);
        }
        assert(f.global->stats->groups == 3);
    }
    // Hard bounds are honored for arbitrary repeated graphs past the capacity.
    {
        fixture f;
        webgpu_bind_group_cache cache;
        for (int repeat = 0; repeat < 2; ++repeat) {
            cache.begin_graph(0);
            for (size_t i = 0; i < cache.max_slots + 3; ++i) f.get(cache);
            cache.end_graph();
            assert(cache.slots.size() == cache.max_slots && cache.cursor == cache.max_slots);
        }
        assert(f.global->stats->groups == cache.max_slots + 6);
        cache.begin_graph(0); f.get(cache); cache.end_graph();
        assert(cache.slots.size() == 1);
    }
    // Strong key references prevent native-address reuse until invalidation/tail removal.
    {
        fixture f;
        webgpu_bind_group_cache cache;
        cache.begin_graph(0); f.get(cache);
        const auto old_buffer = f.data.Get(), old_pipeline = f.pipeline.Get();
        std::weak_ptr<resource> buffer_ref = f.data.p, pipeline_ref = f.pipeline.p;
        cache.slots[0].group = {}; // Isolate ownership held by the actual key.
        f.data = nullptr; f.pipeline = {};
        assert(!buffer_ref.expired() && !pipeline_ref.expired());
        auto fresh_buffer = wgpu::Buffer(std::make_shared<resource>(1, 128));
        auto fresh_pipeline = wgpu::ComputePipeline{std::make_shared<resource>(1), f.global->stats.get()};
        assert(fresh_buffer.Get() != old_buffer && fresh_pipeline.Get() != old_pipeline);
        cache.begin_graph(1);
        assert(buffer_ref.expired() && pipeline_ref.expired() && cache.slots.empty());
        // Explicit reuse of released handles is possible, but all prior keys are gone.
        bool reused = false;
        for (int i = 0; i < 4; ++i) {
            auto next = std::make_shared<resource>(1, 128);
            reused |= reinterpret_cast<void *>(next->handle) == old_buffer || reinterpret_cast<void *>(next->handle) == old_pipeline;
        }
        assert(reused);
    }
    // The actual buffer-free hook bumps the shared epoch before destroying the resource.
    {
        fixture f;
        f.graph(1, 20); f.global->stats->drain();
        auto ctx = new ggml_backend_webgpu_buffer_context{f.data, f.global};
        ggml_backend_buffer buffer{ctx};
        auto retained = f.data.p;
        ggml_backend_webgpu_buffer_free_buffer(&buffer);
        assert(f.global->buffer_generation.load() == 1 && retained->destroyed);
        // Lookup during an already-started graph also invalidates; no begin_graph shortcut.
        f.data = wgpu::Buffer(std::make_shared<resource>(1, 128));
        f.get(f.ctx->bind_group_cache);
        assert(f.ctx->bind_group_cache.generation == 1);
        assert(f.ctx->bind_group_cache.slots[0].group.Get() == nullptr);
        // The ordinal is preserved when a free happens between dispatches.
        assert(f.ctx->bind_group_cache.cursor == 2 && f.ctx->bind_group_cache.slots.size() == 2);
    }
    // New contexts/devices have independent cache state and exact native identities.
    {
        fixture a, b;
        a.graph(2, 1); b.graph(2, 3);
        a.global->stats->drain(); b.global->stats->drain();
        assert(a.global->stats->groups == 2 && b.global->stats->groups == 2);
        auto & ca = a.ctx->bind_group_cache; auto & cb = b.ctx->bind_group_cache;
        assert(ca.slots[0].pipeline.Get() != cb.slots[0].pipeline.Get());
        ca.begin_graph(5); assert(ca.slots.empty() && cb.slots.size() == 2);
        b.global->stats->device = 2;
        b.data = wgpu::Buffer(std::make_shared<resource>(2, 128));
        b.pipeline = {std::make_shared<resource>(2), b.global->stats.get()};
        cb.begin_graph(0); b.get(cb);
        assert(b.global->stats->groups == 3);
    }
    // Replacing cached groups cannot release resources still owned by queued commands.
    {
        fixture f;
        f.graph(2, 100);
        std::weak_ptr<wgpu::group> queued = f.ctx->bind_group_cache.slots[0].group.p;
        f.graph(2, 200, 4);
        assert(!queued.expired());
        f.global->stats->drain();
        assert_values(f.global->stats->observed_params, 0, 2, 100);
        assert_values(f.global->stats->observed_params, 2, 2, 200);
        assert(queued.expired());
    }
    // Null creation and exceptions do not turn a failed replacement into a cache hit.
    {
        fixture f;
        webgpu_bind_group_cache cache;
        cache.begin_graph(0); auto first = f.get(cache);
        auto changed = f.entries(); changed[0].offset = 4;
        f.global->stats->fail_group = true; cache.begin_graph(0);
        assert(!cache.get(f.global->device, f.pipeline, 0, changed, "fail", 0));
        assert(!cache.slots[0].group);
        cache.begin_graph(0); auto valid = f.get(cache);
        assert(valid.Get() != first.Get());
        f.global->stats->throw_group = true; cache.begin_graph(0);
        bool failed = false;
        try { cache.get(f.global->device, f.pipeline, 0, changed, "throw", 0); }
        catch (const std::runtime_error &) { failed = true; }
        assert(failed);
        cache.begin_graph(0); assert(f.get(cache).Get() == valid.Get());
    }
    assert(live_resources == 0);
    std::cout << "PASS: actual cache/key/graph/batching/lifetime/free/failure scenarios (CPU model only)\n";
}
