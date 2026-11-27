#pragma once

namespace expert_storage_cases {
enum class Result { fits, exhausted, validation_and_exhaustion, device_lost };

static void allocate(Result result) {
    transfer_cases::Backend gpu(transfer_cases::BackendKind::webgpu);
    gpu.type.iface.get_name = ggml_backend_webgpu_buffer_type_get_name;
    gpu.type.iface.get_max_size = ggml_backend_webgpu_buffer_type_get_max_size;
    gpu.type.iface.alloc_buffer = ggml_backend_webgpu_buffer_type_alloc_buffer;
    auto & global = *gpu.context->global_ctx;
    global.capabilities.limits.maxBufferSize = 32 * 1024;
    global.capabilities.limits.maxStorageBufferBindingSize = 16 * 1024;
    const transfer_cases::Memory context(ggml_init({128 * 1024, nullptr, true}), ggml_free);
    BICORE_CHECK(context);
    constexpr uint64_t bank_bytes = 18 * 1024;
    for (int i = 0; i < 3; ++i) {
        auto * bank = ggml_new_tensor_3d(context.get(), GGML_TYPE_Q4_0, 128, 128, 2);
        BICORE_CHECK(ggml_nbytes(bank) == bank_bytes);
    }

    const auto get = reinterpret_cast<ggml_webgpu_get_expert_storage_v1_t>(
        ggml_backend_reg_get_proc_address(&gpu.registry, GGML_WEBGPU_EXPERT_STORAGE_V1));
    BICORE_CHECK(get);
    unsigned allocations = 0;
    resources.buffer_error = [&](uint64_t bytes) {
        BICORE_CHECK(bytes == bank_bytes);
        ++allocations;
        if (result == Result::validation_and_exhaustion && allocations == 2)
            return WGPUErrorType_Validation;
        if (result != Result::fits && allocations > 2)
            return WGPUErrorType_OutOfMemory;
        return WGPUErrorType_NoError;
    };
    if (result == Result::device_lost) {
        on_wait_once = [&] { global.optional_device->control.lost.store(true); };
    }
    commands = {};
    const auto pushes_before = pushes;
    const auto pops_before = pops;
    const auto allocation = get()->allocate(&gpu.type, context.get());
    resources.buffer_error = {};
    BICORE_CHECK(allocations == 3 && pushes - pushes_before == 3 && pops - pops_before == 3);
    BICORE_CHECK(allocation.out_of_memory == (result == Result::exhausted));
    BICORE_CHECK(commands.writes.empty() && commands.submissions == 0);
    if (result == Result::fits) {
        BICORE_CHECK(allocation.buffer && commands.buffers_destroyed == 0);
        BICORE_CHECK(ggml_backend_buffer_get_size(allocation.buffer) == 3 * bank_bytes);
        ggml_backend_buffer_free(allocation.buffer);
    } else {
        BICORE_CHECK(!allocation.buffer);
    }
    BICORE_CHECK(commands.buffers_created == 3 && commands.buffers_destroyed == 3 &&
                 global.optional_device->control.healthy() == (result != Result::device_lost));
}

static void context_allocations(Result result, bool tracked) {
    transfer_cases::Backend gpu(transfer_cases::BackendKind::webgpu);
    gpu.type.iface.alloc_buffer = ggml_backend_webgpu_buffer_type_alloc_buffer;
    const auto get = reinterpret_cast<ggml_webgpu_get_context_storage_v1_t>(
        ggml_backend_reg_get_proc_address(&gpu.registry, GGML_WEBGPU_CONTEXT_STORAGE_V1));
    BICORE_CHECK(get);
    const auto * storage = get();
    ggml_webgpu_context_storage_status status{tracked ? &gpu.device : nullptr};
    BICORE_CHECK(storage->exchange(&status) == nullptr);

    commands = {};
    auto & global = *gpu.context->global_ctx;
    unsigned count = 0;
    resources.buffer_error = [&](uint64_t) {
        if (++count == 2) {
            switch (result) {
            case Result::fits:
                return WGPUErrorType_NoError;
            case Result::exhausted:
                return WGPUErrorType_OutOfMemory;
            case Result::validation_and_exhaustion:
                return WGPUErrorType_Validation;
            case Result::device_lost:
                return WGPUErrorType_OutOfMemory;
            }
        }
        return WGPUErrorType_NoError;
    };
    const auto before = pushes;
    ggml_backend_buffer_ptr first(ggml_backend_buft_alloc_buffer(&gpu.type, 256));
    BICORE_CHECK(first);
    bool clear_completed = false;
    if (result == Result::exhausted) {
        // Simulate previously submitted work retaining the first buffer
        // while the next allocation fails.
        global.queue.Get()->pending.push_back([&] { clear_completed = true; });
    }
    if (result == Result::device_lost) {
        on_wait_once = [&] { global.optional_device->control.lost.store(true); };
    }
    ggml_backend_buffer_ptr second(ggml_backend_buft_alloc_buffer(&gpu.type, 256));
    resources.buffer_error = {};
    BICORE_CHECK(storage->exchange(nullptr) == &status);
    BICORE_CHECK(count == 2 && pushes - before == (tracked ? 6 : 0));
    BICORE_CHECK(static_cast<bool>(second) == (result == Result::fits));
    BICORE_CHECK(status.out_of_memory == (result == Result::exhausted));
    BICORE_CHECK(status.other_failure ==
                 (result == Result::validation_and_exhaustion || result == Result::device_lost));
    BICORE_CHECK(commands.writes.empty() && commands.submissions == 0);
    if (status.out_of_memory) {
        first.reset();
        BICORE_CHECK(!clear_completed && commands.queue_waits == 0);
        BICORE_CHECK(storage->drain(&gpu.device));
        BICORE_CHECK(clear_completed && commands.queue_waits == 1 && futures.empty());
        readback.queue_status = WGPUQueueWorkDoneStatus_Error;
        BICORE_CHECK(!storage->drain(&gpu.device));
        BICORE_CHECK(!global.optional_device->control.healthy());
        readback.queue_status = WGPUQueueWorkDoneStatus_Success;
    }
}

static void all() {
    for (auto result : {Result::fits, Result::exhausted, Result::validation_and_exhaustion, Result::device_lost}) {
        allocate(result);
        context_allocations(result, true);
    }
    context_allocations(Result::fits, false);
}
} // namespace expert_storage_cases
