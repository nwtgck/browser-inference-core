#pragma once
#include "ggml-backend-tensor-get-batch.h"

namespace readback_cases {
struct Fixture {
    webgpu_context ctx = std::make_shared<webgpu_context_struct>();
    ggml_backend_webgpu_context bridge;
    std::unique_ptr<ggml_backend_webgpu_buffer_context> storage;
    ggml_backend_buffer_type type{};
    ggml_backend_buffer buffer{};
    ggml_backend backend{};
    ggml_tensor tensor{};

    explicit Fixture(size_t physical_size = 128, size_t logical_size = 32) {
        ctx->global_ctx = std::make_shared<webgpu_global_context_struct>();
        auto & global = *ctx->global_ctx;
        global.instance = wgpu::Instance::Acquire(new WGPUInstanceImpl);
        global.device = wgpu::Device::Acquire(new WGPUDeviceImpl);
        global.queue = global.device.GetQueue();
        global.optional_device = std::make_shared<ggml_webgpu_optional::DeviceGeneration>();
        global.capabilities.limits.maxBufferSize =
            std::max(physical_size, size_t(GGML_BACKEND_TENSOR_GET_BATCH_V1_MAX_BYTES));
        wgpu::BufferDescriptor descriptor{};
        descriptor.size = physical_size;
        descriptor.usage = wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Storage;
        storage = std::make_unique<ggml_backend_webgpu_buffer_context>(global.device.CreateBuffer(&descriptor),
                                                                       "readback", ctx->global_ctx);
        buffer.context = storage.get();
        buffer.buft = &type;
        buffer.size = physical_size;
        buffer.iface.get_tensor = ggml_backend_webgpu_buffer_get_tensor;
        buffer.iface.get_base = ggml_backend_webgpu_buffer_get_base;
        bridge.webgpu_ctx = ctx;
        backend.context = &bridge;
        backend.iface.get_tensor_async = ggml_backend_webgpu_get_tensor_async;
        backend.iface.synchronize = ggml_backend_webgpu_synchronize;
        tensor.type = GGML_TYPE_I8;
        tensor.ne[0] = logical_size;
        tensor.ne[1] = tensor.ne[2] = tensor.ne[3] = 1;
        tensor.nb[0] = 1;
        tensor.nb[1] = tensor.nb[2] = tensor.nb[3] = logical_size;
        tensor.data = reinterpret_cast<void *>(uintptr_t(webgpu_ptr_base) + 8);
        tensor.buffer = &buffer;
        for (size_t index = 0; index < physical_size; ++index)
            storage->buffer.Get()->bytes[index] = uint8_t(index);
        commands = {};
        readback = {};
    }
    ~Fixture() {
        drain_queue(ctx->global_ctx->queue.Get());
    }
    void produce(uint8_t value) {
        std::vector<uint8_t> values(storage->buffer.GetSize(), value);
        ctx->global_ctx->queue.WriteBuffer(storage->buffer, 0, values.data(), values.size());
        std::fill(values.begin(), values.end(), 0xde);
    }
    void get(void * output, size_t offset = 0, size_t size = 16) {
        ggml_backend_tensor_get_async(&backend, &tensor, output, offset, size);
    }
};

template <class Call> static void failure(Call call, const char * message) {
    bool caught = false;
    try {
        call();
    } catch (const std::runtime_error & error) {
        caught = true;
        BICORE_CHECK(std::string(error.what()).find(message) != std::string::npos);
    }
    BICORE_CHECK(caught);
}

static void scalar() {
    for (const bool hook : {false, true}) {
        Fixture fixture;
        if (!hook)
            fixture.backend.iface.get_tensor_async = nullptr;
        std::array<uint8_t, 20> output;
        output.fill(0xcc);
        bool inside = true;
        readback.before_map = [&] {
            BICORE_CHECK(inside);
            for (auto byte : output)
                BICORE_CHECK(byte == 0xcc);
        };
        fixture.produce(77);
        const auto previous_waits = waits;
        fixture.get(output.data() + 2);
        inside = false;
        readback.before_map = {};
        BICORE_CHECK(commands.queue_waits == (hook ? 0 : 1));
        BICORE_CHECK(commands.maps == 1 && commands.unmaps == 1 && commands.submissions == 1);
        BICORE_CHECK(waits - previous_waits == (hook ? 1 : 2));
        for (size_t index = 0; index < output.size(); ++index)
            BICORE_CHECK(output[index] == (index >= 2 && index < 18 ? 77 : 0xcc));
        output.fill(42);
        drain_queue(fixture.ctx->global_ctx->queue.Get());
        for (auto byte : output)
            BICORE_CHECK(byte == 42);
    }
    for (size_t base = 0; base < 4; ++base) {
        for (size_t offset = 0; offset < 4; ++offset) {
            for (size_t size = 1; size <= 17; ++size) {
                Fixture fixture;
                fixture.tensor.data = reinterpret_cast<void *>(uintptr_t(webgpu_ptr_base) + 8 + base);
                std::array<uint8_t, 21> output;
                output.fill(0xcc);
                fixture.get(output.data() + 2, offset, size);
                for (size_t index = 0; index < output.size(); ++index) {
                    BICORE_CHECK(output[index] ==
                                 (index >= 2 && index < size + 2 ? uint8_t(8 + base + offset + index - 2) : 0xcc));
                }
#ifdef __EMSCRIPTEN__
                const bool direct = (base + offset) % 4 == 0 && size % 4 == 0;
#else
                const bool direct = false;
#endif
                BICORE_CHECK(commands.queue_waits == 0 && commands.maps == 1 && commands.unmaps == 1);
                BICORE_CHECK(commands.direct_reads == unsigned(direct) && commands.pointer_reads == unsigned(!direct));
            }
        }
    }
    {
        Fixture fixture;
        auto base = fixture.tensor;
        ggml_backend_buffer foreign{};
        fixture.tensor.view_src = &base;
        fixture.tensor.buffer = &foreign;
        fixture.tensor.view_offs = 3;
        fixture.tensor.data = reinterpret_cast<void *>(0xdead);
        std::array<uint8_t, 5> output{};
        fixture.get(output.data(), 2, output.size());
        BICORE_CHECK(output.front() == 13 && output.back() == 17 && commands.queue_waits == 0);
    }
    {
        Fixture fixture;
        std::array<uint8_t, 32> output{};
        fixture.get(output.data(), 0, 4);
        fixture.produce(19);
        fixture.get(output.data(), 0, 3);
        BICORE_CHECK(output[0] == 19 && commands.buffers_created == 1 && commands.buffers_destroyed == 0);
        fixture.get(output.data(), 0, 32);
        BICORE_CHECK(commands.buffers_created == 2 && commands.buffers_destroyed == 1 && commands.unmaps == 3);
    }
    {
        Fixture fixture;
        fixture.produce(21);
        fixture.get(nullptr, 0, 0);
        BICORE_CHECK(commands.queue_waits == 1 && commands.maps == 0 && commands.submissions == 0 &&
                     commands.buffers_created == 0);
    }
    {
        Fixture fixture;
        ggml_backend_buffer foreign{};
        foreign.iface.get_tensor = [](ggml_backend_buffer_t, const ggml_tensor *, void * output, size_t, size_t size) {
            BICORE_CHECK(commands.queue_waits == 1);
            std::memset(output, 93, size);
        };
        fixture.tensor.buffer = &foreign;
        fixture.produce(22);
        std::array<uint8_t, 4> output{};
        fixture.get(output.data(), 0, output.size());
        BICORE_CHECK(output[0] == 93 && commands.queue_waits == 1 && commands.maps == 0);
    }
    {
        Fixture fixture, other;
        fixture.tensor.buffer = &other.buffer;
        fixture.produce(23);
        other.produce(24);
        std::array<uint8_t, 4> output{};
        fixture.get(output.data(), 0, output.size());
        BICORE_CHECK(output[0] == 24 && commands.queue_waits == 1 && commands.maps == 1);
    }
    {
        Fixture fixture;
        std::array<uint8_t, 4> output{};
        failure([&] { fixture.get(output.data(), 31, 4); }, "GGML_ASSERT");
        BICORE_CHECK(commands.maps == 0 && commands.queue_waits == 0 && commands.submissions == 0);
    }
    for (const auto status :
         {WGPUMapAsyncStatus_Error, WGPUMapAsyncStatus_Aborted, WGPUMapAsyncStatus_CallbackCancelled}) {
        Fixture fixture;
        readback.map_status = status;
        std::array<uint8_t, 4> output{1, 2, 3, 4};
        failure([&] { fixture.get(output.data(), 0, 4); }, "Buffer map failed");
        BICORE_CHECK(output[0] == 1 && output[3] == 4 && commands.direct_reads == 0 && commands.pointer_reads == 0 &&
                     commands.unmaps == 0);
    }
    for (const bool timeout : {false, true}) {
        Fixture fixture;
        wait_timeout = timeout;
        wait_error = !timeout;
        std::array<uint8_t, 4> output{1, 2, 3, 4};
        failure([&] { fixture.get(output.data(), 0, 4); },
                timeout ? "Buffer map wait timed out" : "Buffer map wait failed");
        BICORE_CHECK(output[0] == 1 && output[3] == 4 && commands.pointer_reads == 0 && commands.direct_reads == 0);
        // Abort interception cleanup, not a runtime recovery policy.
        futures.clear();
    }
    {
        Fixture fixture;
        readback.queue_status = WGPUQueueWorkDoneStatus_Error;
        ggml_backend_buffer foreign{};
        foreign.iface.get_tensor = [](ggml_backend_buffer_t, const ggml_tensor *, void *, size_t, size_t) {
            BICORE_CHECK(false);
        };
        fixture.tensor.buffer = &foreign;
        std::array<uint8_t, 4> output{1, 2, 3, 4};
        failure([&] { fixture.get(output.data(), 0, 4); }, "Queue work failed");
        BICORE_CHECK(output[0] == 1 && output[3] == 4 && commands.maps == 0);
    }
#ifdef __EMSCRIPTEN__
    {
        Fixture fixture;
        readback.read_status = WGPUStatus_Error;
        std::array<uint8_t, 6> output{1, 2, 3, 4, 5, 6};
        failure([&] { fixture.get(output.data() + 1, 0, 4); }, "reading mapped tensor data failed");
        BICORE_CHECK(output[0] == 1 && output[1] == 2 && output[5] == 6);
        BICORE_CHECK(commands.direct_reads == 1 && commands.pointer_reads == 0 && commands.unmaps == 1);
        readback.read_status = WGPUStatus_Success;
        fixture.get(output.data() + 1, 0, 4);
        BICORE_CHECK(output[1] == 8 && output[4] == 11 && commands.unmaps == 2);
    }
#endif
}

#ifdef __EMSCRIPTEN__
static void batch() {
    {
        Fixture fixture;
        const auto previous_waits = waits;
        BICORE_CHECK(ggml_backend_webgpu_try_tensor_get_batch_v1(nullptr, 0));
        BICORE_CHECK(waits == previous_waits && commands.maps == 0 && commands.submissions == 0);
        std::array<ggml_backend_tensor_get_request_v1, 256> requests{};
        std::array<std::array<uint8_t, 8>, 256> outputs;
        for (size_t index = 0; index < requests.size(); ++index) {
            outputs[index].fill(0xcc);
            requests[index] = {&fixture.tensor, outputs[index].data() + 1, index % 5, index % 7};
        }
        BICORE_CHECK(ggml_backend_webgpu_try_tensor_get_batch_v1(requests.data(), requests.size()));
        BICORE_CHECK(commands.maps == 1 && commands.submissions == 1 && commands.queue_waits == 0 &&
                     commands.unmaps == 1);
        BICORE_CHECK(waits - previous_waits == 1);
        for (size_t index = 0; index < requests.size(); ++index) {
            for (size_t byte = 0; byte < outputs[index].size(); ++byte) {
                BICORE_CHECK(outputs[index][byte] ==
                             (byte >= 1 && byte <= index % 7 ? uint8_t(8 + index % 5 + byte - 1) : 0xcc));
            }
        }
    }
    for (unsigned mutation = 0; mutation < 5; ++mutation) {
        Fixture fixture;
        std::array<uint8_t, 32> output;
        output.fill(0xcc);
        std::array<ggml_backend_tensor_get_request_v1, 2> requests{
            {{&fixture.tensor, output.data(), 0, 4}, {&fixture.tensor, output.data() + 8, 0, 4}}};
        if (mutation == 0)
            requests[1].tensor = nullptr;
        if (mutation == 1)
            requests[1].data = nullptr;
        if (mutation == 2)
            requests[1].offset = 31;
        if (mutation == 3)
            requests[1].size = GGML_BACKEND_TENSOR_GET_BATCH_V1_MAX_BYTES + 1;
        if (mutation == 4)
            fixture.storage->buffer.Get()->usage = WGPUBufferUsage_CopyDst;
        BICORE_CHECK(!ggml_backend_webgpu_try_tensor_get_batch_v1(requests.data(), requests.size()));
        BICORE_CHECK(commands.maps == 0 && commands.submissions == 0 && commands.buffers_created == 0 &&
                     commands.copies == 0);
        for (auto byte : output)
            BICORE_CHECK(byte == 0xcc);
    }
    {
        Fixture fixture;
        std::array<uint8_t, 16> output;
        output.fill(0xcc);
        const std::array<ggml_backend_tensor_get_request_v1, 2> requests{
            {{&fixture.tensor, output.data() + 2, 1, 7}, {&fixture.tensor, output.data() + 4, 9, 4}}};
        BICORE_CHECK(ggml_backend_webgpu_try_tensor_get_batch_v1(requests.data(), requests.size()));
        BICORE_CHECK(output[2] == 9 && output[3] == 10 && output[4] == 17 && output[7] == 20 && output[8] == 15);
        output.fill(42);
        drain_queue(fixture.ctx->global_ctx->queue.Get());
        for (auto byte : output)
            BICORE_CHECK(byte == 42);
    }
    {
        constexpr size_t mib = 1024 * 1024;
        Fixture fixture(5 * mib + 16, 5 * mib);
        std::vector<uint8_t> first(3 * mib), second(2 * mib);
        const std::array<ggml_backend_tensor_get_request_v1, 2> requests{
            {{&fixture.tensor, first.data(), 0, first.size()},
             {&fixture.tensor, second.data(), first.size(), second.size()}}};
        const auto previous_waits = waits;
        BICORE_CHECK(ggml_backend_webgpu_try_tensor_get_batch_v1(requests.data(), requests.size()));
        BICORE_CHECK(commands.maps == 2 && commands.submissions == 2 && commands.unmaps == 2 &&
                     commands.queue_waits == 0);
        BICORE_CHECK(waits - previous_waits == 2);
        for (size_t index = 0; index < first.size(); ++index)
            BICORE_CHECK(first[index] == uint8_t(8 + index));
        for (size_t index = 0; index < second.size(); ++index)
            BICORE_CHECK(second[index] == uint8_t(8 + first.size() + index));
    }
    {
        Fixture fixture;
        readback.read_status = WGPUStatus_Error;
        std::array<uint8_t, 4> output{1, 2, 3, 4};
        const ggml_backend_tensor_get_request_v1 request{&fixture.tensor, output.data(), 0, output.size()};
        failure([&] { (void)ggml_backend_webgpu_try_tensor_get_batch_v1(&request, 1); },
                "reading mapped batch data failed");
        BICORE_CHECK(output[0] == 1 && output[3] == 4 && commands.unmaps == 1);
    }
}
#endif
} // namespace readback_cases
