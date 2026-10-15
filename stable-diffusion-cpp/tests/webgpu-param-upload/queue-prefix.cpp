// Ordered queue API double: writes snapshot host bytes immediately; commands
// execute later in issue order. No browser, GPU timing or model claims.
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#define GGML_ASSERT assert
#define GGML_ABORT(...) std::abort()
#define ROUNDUP_POW2(a, b) (((a) + (b) - 1) & ~((b) - 1))
#define WEBGPU_LOG_DEBUG(x)
#define WEBGPU_CPU_PROFILE_TOTAL_START(x)
#define WEBGPU_CPU_PROFILE_TOTAL_END(x, y)
namespace wgpu {
struct CommandEncoder;
struct Device {
    CommandEncoder CreateCommandEncoder();
    struct BindGroup CreateBindGroup(const struct BindGroupDescriptor*);
};
enum BufferUsage { CopyDst = 1, Uniform = 2 };
struct Buffer {
    std::shared_ptr<std::vector<uint8_t>> bytes;
    explicit operator bool() const { return bool(bytes); }
    void Destroy() { bytes.reset(); }
    Buffer& operator=(std::nullptr_t) { bytes.reset(); return *this; }
};
struct BindGroupEntry { Buffer buffer; size_t offset, size; };
struct BindGroup { BindGroupEntry entry; };
struct BindGroupDescriptor { int layout; size_t entryCount; const BindGroupEntry* entries; const char* label; };
struct Pipeline {
    std::vector<uint8_t> expected;
    int GetBindGroupLayout(int) const { return 0; }
};
struct CommandBuffer { std::vector<std::function<void()>> commands; };
struct Queue {
    std::vector<std::function<void()>> pending;
    size_t writes = 0, submits = 0, uploaded_bytes = 0, waits = 0, kernels = 0;
    void WriteBuffer(Buffer buffer, size_t offset, const void* data, size_t size) {
        assert(size % 4 == 0 && offset + size <= buffer.bytes->size());
        std::vector<uint8_t> snapshot(size);
        if (size) std::memcpy(snapshot.data(), data, size);
        pending.push_back([buffer, offset, snapshot] {
            if (!snapshot.empty()) std::memcpy(buffer.bytes->data() + offset, snapshot.data(), snapshot.size());
        });
        ++writes; uploaded_bytes += size;
    }
    void Submit(size_t count, const CommandBuffer* buffers) {
        assert(count == 1);
        for (const auto& execute : buffers[0].commands) {
            pending.push_back(execute); ++kernels;
        }
        ++submits;
    }
    void drain() { for (auto& execute : pending) execute(); pending.clear(); }
};
struct ComputePassEncoder {
    std::shared_ptr<CommandBuffer> state;
    Pipeline pipeline;
    BindGroup binding;
    bool ended = false;
    explicit operator bool() const { return bool(state); }
    void SetPipeline(Pipeline value) { assert(!ended); pipeline = std::move(value); }
    void SetBindGroup(int, BindGroup value) { assert(!ended); binding = value; }
    void DispatchWorkgroups(uint32_t, uint32_t, uint32_t) {
        assert(state && !ended);
        auto entry = binding.entry;
        auto expected = pipeline.expected;
        state->commands.push_back([entry, expected] {
            assert(expected.size() <= entry.size);
            if (!expected.empty()) assert(std::memcmp(entry.buffer.bytes->data() + entry.offset,
                                                      expected.data(), expected.size()) == 0);
#ifdef GGML_WEBGPU_BATCH_PARAM_UPLOADS
            for (size_t i = expected.size(); i < entry.size; ++i)
                assert((*entry.buffer.bytes)[entry.offset + i] == 0);
#endif
        });
    }
    void End() { assert(state && !ended); ended = true; }
    ComputePassEncoder& operator=(std::nullptr_t) { state.reset(); return *this; }
};
struct CommandEncoder {
    std::shared_ptr<CommandBuffer> state = std::make_shared<CommandBuffer>();
    ComputePassEncoder BeginComputePass() { ComputePassEncoder pass; pass.state = state; return pass; }
    CommandBuffer Finish() { return *state; }
    CommandEncoder& operator=(std::nullptr_t) { state.reset(); return *this; }
};
CommandEncoder Device::CreateCommandEncoder() { return {}; }
BindGroup Device::CreateBindGroup(const BindGroupDescriptor* descriptor) {
    assert(descriptor->entryCount == 1);
    const auto& entry = descriptor->entries[0];
    assert(entry.offset + entry.size <= entry.buffer.bytes->size());
    return {entry};
}
} // namespace wgpu
static void ggml_webgpu_create_buffer(wgpu::Device&, wgpu::Buffer& buffer,
                                      size_t size, int, const char*) {
    buffer.bytes = std::make_shared<std::vector<uint8_t>>(size, 0);
}
