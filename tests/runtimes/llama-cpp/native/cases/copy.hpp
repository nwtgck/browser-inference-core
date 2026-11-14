#pragma once

namespace copy_cases {
struct Host {
    std::vector<uint8_t> bytes = std::vector<uint8_t>(64, 3);
    ggml_backend_buffer_type type{};
    ggml_backend_buffer buffer{};
    ggml_backend_device device{};
    ggml_backend backend{};
    ggml_tensor tensor{};
    enum ggml_backend_dev_type device_type = GGML_BACKEND_DEVICE_TYPE_CPU;
    bool host = true;
    unsigned synchronizations = 0;
    std::function<void()> complete;

    Host() {
        type.context = this;
        type.iface.is_host = [](ggml_backend_buffer_type_t value) { return static_cast<Host *>(value->context)->host; };
        buffer.buft = &type;
        buffer.context = this;
        buffer.size = bytes.size();
        buffer.iface.get_base = [](ggml_backend_buffer_t value) -> void * {
            return static_cast<Host *>(value->context)->bytes.data();
        };
        device.context = this;
        device.iface.get_type = [](ggml_backend_dev_t value) {
            return static_cast<Host *>(value->context)->device_type;
        };
        backend.device = &device;
        backend.context = this;
        backend.iface.synchronize = [](ggml_backend_t value) {
            auto & state = *static_cast<Host *>(value->context);
            ++state.synchronizations;
            if (state.complete)
                state.complete();
        };
        tensor.type = GGML_TYPE_I8;
        tensor.ne[0] = 16;
        tensor.ne[1] = tensor.ne[2] = tensor.ne[3] = 1;
        tensor.nb[0] = 1;
        tensor.nb[1] = tensor.nb[2] = tensor.nb[3] = 16;
        tensor.buffer = &buffer;
        tensor.data = bytes.data() + 4;
    }
};

static void connect(readback_cases::Fixture & gpu) {
    gpu.tensor.ne[0] = gpu.tensor.nb[1] = gpu.tensor.nb[2] = gpu.tensor.nb[3] = 16;
    gpu.backend.iface.get_name = ggml_backend_webgpu_name;
    gpu.backend.iface.cpy_tensor_async = ggml_backend_webgpu_cpy_tensor_async;
    gpu.buffer.iface.set_tensor = ggml_backend_webgpu_buffer_set_tensor;
    gpu.buffer.iface.cpy_tensor = ggml_backend_webgpu_buffer_cpy_tensor;
}

static void host_snapshot() {
    for (const bool explicit_snapshot : {false, true}) {
        Host host;
        readback_cases::Fixture gpu;
        connect(gpu);
        std::vector<uint8_t> observed;
        gpu.ctx->global_ctx->queue.Get()->pending.push_back(
            [&] { observed.push_back(gpu.storage->buffer.Get()->bytes[8]); });
        for (const uint8_t value : {17, 42, 79}) {
            host.complete = [&, value] { std::fill(host.bytes.begin() + 4, host.bytes.begin() + 20, value); };
            if (explicit_snapshot) {
                BICORE_CHECK(ggml_backend_webgpu_copy_host_input_snapshot_v1(&host.backend, &gpu.backend, &host.tensor,
                                                                             &gpu.tensor));
            } else {
                ggml_backend_tensor_copy_async(&host.backend, &gpu.backend, &host.tensor, &gpu.tensor);
            }
            std::fill(host.bytes.begin(), host.bytes.end(), 99);
            gpu.ctx->global_ctx->queue.Get()->pending.push_back(
                [&] { observed.push_back(gpu.storage->buffer.Get()->bytes[8]); });
        }
        BICORE_CHECK(observed.empty() && host.synchronizations == 3 && commands.writes.size() == 3 &&
                     commands.queue_waits == 0 && commands.submissions == 0);
        host.bytes.clear();
        host.bytes.shrink_to_fit();
        ggml_backend_synchronize(&gpu.backend);
        BICORE_CHECK(observed == std::vector<uint8_t>({8, 17, 42, 79}));
        BICORE_CHECK(commands.queue_waits == 1);
        const auto & bytes = gpu.storage->buffer.Get()->bytes;
        for (size_t index = 0; index < bytes.size(); ++index)
            BICORE_CHECK(bytes[index] == (index >= 8 && index < 24 ? 79 : uint8_t(index)));
    }
    for (unsigned mutation = 0; mutation < 15; ++mutation) {
        Host host;
        readback_cases::Fixture gpu;
        connect(gpu);
        if (mutation == 0)
            host.device_type = GGML_BACKEND_DEVICE_TYPE_GPU;
        if (mutation == 1)
            host.host = false;
        if (mutation == 2)
            host.buffer.usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
        if (mutation == 3)
            host.tensor.data = nullptr;
        if (mutation == 4)
            gpu.tensor.data = nullptr;
        if (mutation == 5)
            gpu.tensor.ne[0] = 12;
        if (mutation == 6)
            gpu.tensor.data = reinterpret_cast<void *>(uintptr_t(webgpu_ptr_base) + 9);
        if (mutation == 7)
            gpu.tensor.data = reinterpret_cast<void *>(uintptr_t(webgpu_ptr_base) + 124);
        if (mutation == 8)
            gpu.buffer.size = 16;
        if (mutation == 9)
            gpu.storage->buffer.Get()->usage = WGPUBufferUsage_CopySrc;
        if (mutation == 10)
            host.tensor.data = host.bytes.data() + 60;
        if (mutation == 11)
            host.buffer.buft = nullptr;
        if (mutation == 12)
            host.backend.device = nullptr;
        if (mutation == 13)
            gpu.backend.iface.get_name = [](ggml_backend_t) { return "WebGPU"; };
        if (mutation == 14)
            gpu.storage->global_ctx = std::make_shared<webgpu_global_context_struct>();
        BICORE_CHECK(
            !ggml_backend_webgpu_copy_host_input_snapshot_v1(&host.backend, &gpu.backend, &host.tensor, &gpu.tensor));
        BICORE_CHECK(host.synchronizations == 0 && commands.writes.empty() && commands.submissions == 0 &&
                     commands.queue_waits == 0);
    }
    {
        Host host;
        readback_cases::Fixture gpu;
        connect(gpu);
        host.buffer.usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
        BICORE_CHECK(ggml_backend_webgpu_cpy_tensor_async(&host.backend, &gpu.backend, &host.tensor, &gpu.tensor));
        BICORE_CHECK(host.synchronizations == 1 && commands.writes.size() == 1);
    }
    {
        Host host;
        readback_cases::Fixture gpu;
        connect(gpu);
        gpu.backend.iface.cpy_tensor_async = nullptr;
        host.complete = [&] { std::fill(host.bytes.begin() + 4, host.bytes.begin() + 20, 53); };
        ggml_backend_tensor_copy_async(&host.backend, &gpu.backend, &host.tensor, &gpu.tensor);
        BICORE_CHECK(host.synchronizations == 1 && commands.queue_waits == 1 && commands.writes.size() == 1);
        std::fill(host.bytes.begin(), host.bytes.end(), 99);
        drain_queue(gpu.ctx->global_ctx->queue.Get());
        for (size_t index = 8; index < 24; ++index)
            BICORE_CHECK(gpu.storage->buffer.Get()->bytes[index] == 53);
    }
    {
        Host host;
        readback_cases::Fixture gpu;
        connect(gpu);
        auto parent = gpu.tensor;
        gpu.tensor.view_src = &parent;
        gpu.tensor.view_offs = 4;
        gpu.tensor.data = reinterpret_cast<void *>(0xdead);
        ggml_backend_buffer foreign{};
        gpu.tensor.buffer = &foreign;
        BICORE_CHECK(
            ggml_backend_webgpu_copy_host_input_snapshot_v1(&host.backend, &gpu.backend, &host.tensor, &gpu.tensor));
        drain_queue(gpu.ctx->global_ctx->queue.Get());
        BICORE_CHECK(commands.writes[0].offset == 12 && commands.writes[0].bytes.size() == 16);
    }
}

static void device_copy() {
    for (unsigned mutation = 0; mutation < 11; ++mutation) {
        readback_cases::Fixture source, destination;
        connect(source);
        connect(destination);
        const auto global = source.ctx->global_ctx;
        // Two real buffer contexts on the same device/queue, without inventing
        // an alternative private class or executing a shader.
        destination.ctx->global_ctx = global;
        destination.storage->global_ctx = global;
        wgpu::BufferDescriptor descriptor{};
        descriptor.size = 128;
        descriptor.usage = wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst;
        destination.storage->buffer = global->device.CreateBuffer(&descriptor);
        auto src = source.tensor, dst = destination.tensor;
        if (mutation == 1)
            dst = src; // Exact alias is a no-op.
        if (mutation == 2) {
            dst = src;
            dst.data = reinterpret_cast<void *>(uintptr_t(webgpu_ptr_base) + 12);
        }
        if (mutation == 3)
            src.data = reinterpret_cast<void *>(uintptr_t(webgpu_ptr_base) + 9);
        if (mutation == 4)
            src.ne[0] = dst.ne[0] = 15;
        if (mutation == 5)
            source.storage->buffer.Get()->usage = WGPUBufferUsage_CopyDst;
        if (mutation == 6)
            destination.storage->buffer.Get()->usage = WGPUBufferUsage_CopySrc;
        if (mutation == 7)
            source.buffer.size = 16;
        if (mutation == 8)
            destination.storage->global_ctx = std::make_shared<webgpu_global_context_struct>();
        if (mutation == 9)
            src.ne[0] = 12;
        if (mutation == 10)
            dst.data = reinterpret_cast<void *>(uintptr_t(webgpu_ptr_base) + 124);
        commands = {};
        const bool accepted = ggml_backend_webgpu_cpy_tensor_async(&source.backend, &destination.backend, &src, &dst);
        BICORE_CHECK(accepted == (mutation <= 1));
        BICORE_CHECK(commands.queue_waits == 0 && commands.writes.empty());
        BICORE_CHECK(commands.submissions == unsigned(mutation == 0) && commands.copies == unsigned(mutation == 0));
        if (mutation == 0) {
            source.produce(61);
            // The earlier submitted copy precedes this later source write.
            drain_queue(global->queue.Get());
            const auto & output = destination.storage->buffer.Get()->bytes;
            for (size_t index = 8; index < 24; ++index)
                BICORE_CHECK(output[index] == uint8_t(index));
            BICORE_CHECK(ggml_backend_webgpu_cpy_tensor_async(&source.backend, &destination.backend, &src, &dst));
            source.produce(92);
            drain_queue(global->queue.Get());
            for (size_t index = 8; index < 24; ++index)
                BICORE_CHECK(output[index] == 61);
        }
    }
    {
        readback_cases::Fixture source, destination;
        connect(source);
        connect(destination);
        source.produce(47);
        ggml_backend_tensor_copy_async(&source.backend, &destination.backend, &source.tensor, &destination.tensor);
        BICORE_CHECK(commands.copies == 1 && commands.maps == 1 && commands.queue_waits == 2);
        drain_queue(destination.ctx->global_ctx->queue.Get());
        for (size_t index = 8; index < 24; ++index)
            BICORE_CHECK(destination.storage->buffer.Get()->bytes[index] == 47);
    }
}
static void all() {
    host_snapshot();
    device_copy();
}
} // namespace copy_cases
