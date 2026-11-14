#ifdef SDCB_TEST_BROWSER_API
#define __EMSCRIPTEN__ 1
#endif
#include "browser-api.hpp"
#include "ggml-webgpu.cpp"

#include <iostream>
#include <thread>

namespace {
template <class Function> void rejects(Function function) {
    bool rejected = false;
    try {
        function();
    } catch (const std::exception &) {
        rejected = true;
    }
    SDC_CHECK(rejected);
}
webgpu_global_context global_context() {
    auto result = std::make_shared<webgpu_global_context_struct>();
    result->instance = wgpu::Instance::Acquire(new WGPUInstanceImpl);
    result->device = wgpu::Device::Acquire(new WGPUDeviceImpl);
    result->queue = wgpu::Queue::Acquire(new WGPUQueueImpl);
    result->capabilities.limits.minStorageBufferOffsetAlignment = 256;
    result->capabilities.limits.minUniformBufferOffsetAlignment = 256;
    result->capabilities.limits.maxStorageBufferBindingSize = 128 * 1024 * 1024;
    result->capabilities.limits.maxComputeWorkgroupStorageSize = 32768;
    result->capabilities.limits.maxComputeWorkgroupsPerDimension = 65535;
    result->capabilities.limits.maxComputeInvocationsPerWorkgroup = 256;
    return result;
}

void completion() {
    auto global = global_context();
    wgpu::Buffer buffer = wgpu::Buffer::Acquire(new WGPUBufferImpl{1, std::vector<unsigned char>(4), false});
    ggml_backend_webgpu_event_context event_context;
    event_context.global_ctx = global;
    ggml_backend_event event{};
    event.context = &event_context;
    auto context = std::make_shared<webgpu_context_struct>();
    context->global_ctx = global;
    ggml_backend_webgpu_context wrapper{context, "contract"};
    ggml_backend backend{};
    backend.context = &wrapper;
    size_t checks = 0;
    for (int kind = 0; kind < 3; ++kind) {
        const auto reset = [&] {
            browser_api::reset_wait();
            global->uploads = ggml_webgpu_upload_budget{};
            global->uploads.reserve(128, [] { SDC_CHECK(false); });
            if (kind == 2)
                ggml_backend_webgpu_event_record(&backend, &event);
        };
        const auto call = [&] {
            if (kind == 0)
                ggml_backend_webgpu_wait_queue(global);
            else if (kind == 1)
                ggml_backend_webgpu_map_buffer(global, buffer, wgpu::MapMode::Read, 0, 4);
            else
                ggml_backend_webgpu_device_event_synchronize(nullptr, &event);
        };
        for (uint64_t logical_milliseconds : {0ULL, 35000ULL, 181000ULL, 3600000ULL, 86400000ULL}) {
            (void)logical_milliseconds;
            reset();
            call();
            SDC_CHECK(browser_api::state.observed_timeout == UINT64_MAX);
            SDC_CHECK(browser_api::state.waits == 1 && browser_api::state.requests == 1);
            SDC_CHECK(global->uploads.pending_bytes() == (kind == 0 ? 0 : 128));
            if (kind == 2)
                SDC_CHECK(!event_context.recorded && !event_context.result);
            ++checks;
        }
        for (auto status : {WGPUWaitStatus_Error, WGPUWaitStatus_TimedOut, static_cast<WGPUWaitStatus>(123)}) {
            reset();
            browser_api::state.wait_status = status;
            rejects(call);
            SDC_CHECK(global->uploads.pending_bytes() == 128);
            global->uploads.reserve(4, [] { SDC_CHECK(false); });
            SDC_CHECK(global->uploads.pending_bytes() == 132);
            if (kind == 2)
                SDC_CHECK(event_context.recorded);
            ++checks;
        }
        for (int status : {int(WGPUQueueWorkDoneStatus_Error), int(WGPUQueueWorkDoneStatus_CallbackCancelled), 123}) {
            reset();
            if (kind == 1)
                browser_api::state.map_status = static_cast<WGPUMapAsyncStatus>(status);
            else
                browser_api::state.queue_status = static_cast<WGPUQueueWorkDoneStatus>(status);
            rejects(call);
            SDC_CHECK(global->uploads.pending_bytes() == 128);
            global->uploads.reserve(4, [] { SDC_CHECK(false); });
            SDC_CHECK(global->uploads.pending_bytes() == 132);
            ++checks;
        }
        reset();
        browser_api::state.fire_callback = false;
        rejects(call);
        SDC_CHECK(global->uploads.pending_bytes() == 128);
        global->uploads.reserve(4, [] { SDC_CHECK(false); });
        SDC_CHECK(global->uploads.pending_bytes() == 132);
        ++checks;
    }
    browser_api::reset_wait();
    global->uploads = ggml_webgpu_upload_budget{};
    global->uploads.reserve(128, [] { SDC_CHECK(false); });
    browser_api::state.during_wait = [&] {
        rejects([&] { global->uploads.reserve(4, [] { SDC_CHECK(false); }); });
        rejects([&] { ggml_backend_webgpu_wait_queue(global); });
        SDC_CHECK(global->uploads.pending_bytes() == 128);
    };
    ggml_backend_webgpu_wait_queue(global);
    SDC_CHECK(browser_api::state.waits == 1 && browser_api::state.requests == 1 &&
              global->uploads.pending_bytes() == 0);
    browser_api::reset_wait();
    ggml_backend_webgpu_event_context idle;
    idle.global_ctx = global;
    ggml_backend_event idle_event{};
    idle_event.context = &idle;
    ggml_backend_webgpu_device_event_synchronize(nullptr, &idle_event);
    SDC_CHECK(browser_api::state.waits == 0 && browser_api::state.requests == 0);
    ggml_backend_webgpu_event_record(&backend, &idle_event);
    ggml_backend_webgpu_device_event_synchronize(nullptr, &idle_event);
    ggml_backend_webgpu_device_event_synchronize(nullptr, &idle_event);
    SDC_CHECK(browser_api::state.waits == 1 && browser_api::state.requests == 1);
    ggml_backend_webgpu_event_record(&backend, &idle_event);
    const auto old_id = idle.future.id;
    auto old_callback = browser_api::state.futures.at(old_id);
    browser_api::state.futures.erase(old_id);
    std::weak_ptr<ggml_backend_webgpu_event_result> old_result = idle.result;
    ggml_backend_webgpu_event_record(&backend, &idle_event);
    const auto new_result = idle.result;
    auto observed_old = old_result.lock();
    SDC_CHECK(observed_old != nullptr);
    browser_api::state.queue_status = WGPUQueueWorkDoneStatus_CallbackCancelled;
    old_callback();
    SDC_CHECK(new_result->status == wgpu::QueueWorkDoneStatus::Error);
    SDC_CHECK(observed_old->status == wgpu::QueueWorkDoneStatus::CallbackCancelled);
    // The real SDK consumes a one-shot C++ callback when it fires. Keep an
    // explicit observer while reading its result, then verify final release.
    SDC_CHECK(observed_old.use_count() == 1);
    browser_api::state.queue_status = WGPUQueueWorkDoneStatus_Success;
    ggml_backend_webgpu_device_event_synchronize(nullptr, &idle_event);
    SDC_CHECK(new_result->status == wgpu::QueueWorkDoneStatus::Success);
    old_callback = {};
    observed_old.reset();
    SDC_CHECK(old_result.expired());
    std::function<void()> late;
    std::weak_ptr<ggml_backend_webgpu_event_result> late_result;
    {
        auto temporary = std::make_unique<ggml_backend_webgpu_event_context>();
        temporary->global_ctx = global;
        ggml_backend_event temporary_event{};
        temporary_event.context = temporary.get();
        ggml_backend_webgpu_event_record(&backend, &temporary_event);
        const auto id = temporary->future.id;
        late = browser_api::state.futures.at(id);
        browser_api::state.futures.erase(id);
        late_result = temporary->result;
    }
    auto observed_late = late_result.lock();
    SDC_CHECK(observed_late != nullptr);
    browser_api::state.queue_status = WGPUQueueWorkDoneStatus_CallbackCancelled;
    late();
    SDC_CHECK(observed_late->status == wgpu::QueueWorkDoneStatus::CallbackCancelled);
    SDC_CHECK(observed_late.use_count() == 1);
    late = {};
    observed_late.reset();
    SDC_CHECK(late_result.expired());
    browser_api::reset_wait();
    global->uploads = ggml_webgpu_upload_budget{};
    browser_api::state.futures.clear();
    std::cout << "completion failure/reentry/lifetime cases=" << checks + 5 << '\n';
}

void arena() {
    auto global = global_context();
    size_t checks = 0;
    for (size_t alignment : {128U, 256U, 512U}) {
        webgpu_param_arena arena;
        arena.init(global->device, 128, 74, alignment);
        for (uint32_t count : {0U, 1U, 63U, 64U, 65U, 74U}) {
            const auto before = browser_api::state.writes;
            for (uint32_t index = 0; index < count; ++index) {
                const auto offset = arena.alloc_slot(4);
                SDC_CHECK(offset == index * alignment);
                std::vector<uint32_t> value{index + 123};
#ifdef GGML_WEBGPU_BATCH_PARAM_UPLOADS
                arena.stage(offset, value.data(), 4);
#else
                global->queue.WriteBuffer(arena.buffer, offset, value.data(), 4);
#endif
                value[0] = 0; // staging owns bytes independently of its caller.
            }
#ifdef GGML_WEBGPU_BATCH_PARAM_UPLOADS
            arena.flush(global->queue);
            SDC_CHECK(browser_api::state.writes - before == unsigned(count != 0));
#else
            SDC_CHECK(browser_api::state.writes - before == count);
#endif
            browser_api::drain();
            for (uint32_t index = 0; index < count; ++index) {
                uint32_t value;
                std::memcpy(&value, arena.buffer.Get()->bytes.data() + index * alignment, 4);
                SDC_CHECK(value == index + 123);
                for (size_t padding = 4; padding < 128; ++padding)
                    SDC_CHECK(arena.buffer.Get()->bytes[index * alignment + padding] == 0);
            }
            arena.reset();
            SDC_CHECK(arena.next_slot == 0);
            ++checks;
        }
        rejects([&] { arena.alloc_slot(132); });
        for (uint32_t i = 0; i < 74; ++i)
            arena.alloc_slot(4);
        rejects([&] { arena.alloc_slot(4); });
#ifdef GGML_WEBGPU_BATCH_PARAM_UPLOADS
        arena.reset();
        uint32_t value = 1;
        rejects([&] { arena.stage(0, &value, 4); });
        arena.alloc_slot(4);
        rejects([&] { arena.stage(1, &value, 4); });
        rejects([&] { arena.stage(0, &value, 132); });
        arena.upload_data.resize(8);
        rejects([&] { arena.stage(0, &value, 4); });
#endif
    }
    std::cout << "arena ownership/alignment/bounds cases=" << checks << '\n';
}

void grid() {
    size_t checks = 0;
    const auto verify = [&](uint64_t count, uint32_t cap) {
        uint32_t x = 0, y = 0;
        ggml_webgpu_linear_workgroups(count, cap, x, y);
        const uint64_t limit = std::min(cap, 65535U);
        SDC_CHECK(x > 0 && y > 0 && x <= limit && y <= limit);
        SDC_CHECK(uint64_t(x) * y >= count && uint64_t(x) * y <= UINT32_MAX);
        if (count <= limit)
            SDC_CHECK(x == count && y == 1);
        ++checks;
    };
    for (uint32_t cap : {1U, 2U, 3U, 7U, 31U, 255U, 65535U, 131072U, UINT32_MAX}) {
        const uint64_t limit = std::min(cap, 65535U), maximum = limit * limit;
        for (uint64_t count :
             std::array<uint64_t, 8>{1, maximum, limit, limit + 1, 65536, 131360, 2147483647, 4294836225})
            if (count <= maximum)
                verify(count, cap);
    }
    uint32_t random = 0xa571923;
    for (int index = 0; index < 2000; ++index) {
        random = random * 1664525U + 1013904223U;
        verify(1 + uint64_t(random) % 4294836225ULL, 65535);
    }
    for (auto invalid : std::vector<std::pair<uint64_t, uint32_t>>{
             {0, 65535}, {1, 0}, {50, 7}, {4294836226ULL, 65535}, {UINT64_MAX, UINT32_MAX}})
        rejects([&] {
            uint32_t x, y;
            ggml_webgpu_linear_workgroups(invalid.first, invalid.second, x, y);
        });
    std::cout << "complete-backend dispatch grid cases=" << checks << '\n';
}
void graph_queue() {
    size_t cases = 0;
    for (bool passes : {false, true})
        for (size_t alignment : {128U, 256U, 512U})
            for (uint32_t batch : {1U, 8U, 64U})
                for (uint32_t inflight : {1U, 3U, 1000U}) {
                    auto global = global_context();
                    global->command_submit_batch_size = batch;
                    global->max_inflight_batches = inflight;
                    auto runtime = std::make_shared<webgpu_context_struct>();
                    runtime->global_ctx = global;
                    runtime->shader_lib = std::make_unique<ggml_webgpu_shader_lib>(global->device);
                    runtime->batch_compute_passes = passes;
                    runtime->param_arena.init(global->device, 128, batch + 10, alignment);
                    ggml_backend_webgpu_context wrapper{runtime, "contract"};
                    ggml_backend backend{};
                    backend.context = &wrapper;
                    const auto metadata = ggml_init({2 * 1024 * 1024, nullptr, true});
                    SDC_CHECK(metadata != nullptr);
                    const auto buffer =
                        wgpu::Buffer::Acquire(new WGPUBufferImpl{1, std::vector<unsigned char>(4096), false});
                    ggml_backend_webgpu_buffer_context buffer_context(buffer, "contract", global);
                    ggml_backend_buffer backing{};
                    backing.context = &buffer_context;
                    backing.size = 4096;
                    auto input = ggml_new_tensor_1d(metadata, GGML_TYPE_F32, 16);
                    input->data = webgpu_ptr_base;
                    input->buffer = &backing;
                    for (unsigned count : {0U, 1U, 63U, 64U, 65U, 128U, 129U, 300U}) {
                        auto graph = ggml_new_graph_custom(metadata, 512, false);
                        auto output = input;
                        for (unsigned i = 0; i < count; ++i) {
                            output = ggml_scale(metadata, output, float(i + 2));
                            output->data = reinterpret_cast<void *>(uintptr_t(webgpu_ptr_base) + 256);
                            output->buffer = &backing;
                        }
                        if (count)
                            ggml_build_forward_expand(graph, output);
                        browser_api::state.uniforms.clear();
                        browser_api::state.dispatches.clear();
                        const auto before_submits = browser_api::state.submissions;
                        const auto before_writes = browser_api::state.writes;
                        browser_api::reset_wait();
                        SDC_CHECK(ggml_backend_webgpu_graph_compute(&backend, graph) == GGML_STATUS_SUCCESS);
                        const unsigned submits = count == 0 ? 0 : (count + batch - 1) / batch;
                        SDC_CHECK(browser_api::state.submissions - before_submits == submits);
                        SDC_CHECK(browser_api::state.waits == (submits == 0 ? 0 : (submits - 1) / inflight));
#ifdef GGML_WEBGPU_BATCH_PARAM_UPLOADS
                        SDC_CHECK(browser_api::state.writes - before_writes == submits);
#else
                        SDC_CHECK(browser_api::state.writes - before_writes == count);
#endif
                        SDC_CHECK(runtime->param_arena.next_slot == 0);
                        browser_api::drain();
                        SDC_CHECK(browser_api::state.dispatches.size() == count);
                        SDC_CHECK(browser_api::state.uniforms.size() == count);
                        // Read uniforms when queued work executes, after all
                        // intervening uploads and parameter arena resets.
                        for (unsigned i = 0; i < count; ++i) {
                            const auto & words = browser_api::state.uniforms[i];
                            const uint32_t wanted = ggml_webgpu_u32_from_f32(float(i + 2));
                            SDC_CHECK(std::find(words.begin(), words.end(), wanted) != words.end());
                            SDC_CHECK(browser_api::state.dispatches[i].first > 0);
                        }
                        ++cases;
                    }
                    ggml_free(metadata);
                }
    std::cout << "actual graph queue/submit/uniform ownership cases=" << cases << '\n';
}
void bf16_admission() {
    const auto ctx = ggml_init({1024 * 1024, nullptr, true});
    SDC_CHECK(ctx != nullptr);
    auto weight = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, 32, 5);
    auto input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 2);
    auto indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2);
    auto gathered = ggml_get_rows(ctx, weight, indices);
    auto global = global_context();
    ggml_backend_webgpu_device_context device_context{global, "contract", "external API"};
    ggml_backend_device device{};
    device.context = &device_context;
    const auto admitted = [&](ggml_tensor * tensor) { return ggml_backend_webgpu_device_supports_op(&device, tensor); };
    for (size_t offset : {0U, 2U, 254U, 256U, 258U}) {
        weight->data = reinterpret_cast<void *>(uintptr_t(webgpu_ptr_base) + offset);
        SDC_CHECK(admitted(gathered));
    }
    weight->data = reinterpret_cast<void *>(uintptr_t(webgpu_ptr_base) + 1);
    // A byte-offset BF16 tensor cannot be formed by the typed allocator. The
    // real binding helper rejects this invalid representation before admission.
    bool invalid_rejected = false;
    try {
        invalid_rejected = !admitted(gathered);
    } catch (const std::runtime_error &) {
        invalid_rejected = true;
    }
    SDC_CHECK(invalid_rejected);
    weight->data = webgpu_ptr_base;
    gathered->type = GGML_TYPE_I32;
    SDC_CHECK(!admitted(gathered));
    gathered->type = GGML_TYPE_F32;
    auto matrix = ggml_mul_mat(ctx, weight, input);
    auto experts = ggml_new_tensor_3d(ctx, GGML_TYPE_BF16, 32, 5, 3);
    auto expert_input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 1, 2);
    auto ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, 2);
    auto indexed = ggml_mul_mat_id(ctx, experts, expert_input, ids);
    for (auto operation : {matrix, indexed}) {
        auto right = operation->src[1];
        SDC_CHECK(admitted(operation));
        for (auto type : {GGML_TYPE_F16, GGML_TYPE_BF16}) {
            right->type = type;
            SDC_CHECK(!admitted(operation));
        }
        right->type = GGML_TYPE_F32;
    }
    ggml_free(ctx);
    std::cout << "real-type BF16 admission/alignment cases passed\n";
}

void registry(const std::string & mode) {
    auto & api = browser_api::state;
    api.fail_instance = mode == "null-instance";
    api.fail_adapter = mode == "null-adapter";
    api.fail_device = mode == "device-failure";
    api.f16 = mode != "no-f16";
    api.subgroups = mode != "no-subgroups";
    api.implicit_sync = mode != "no-implicit-sync";
    api.adapter_wait_error = mode == "adapter-wait-error";
    api.device_wait_error = mode == "device-wait-error";
    api.missing_adapter_callback = mode == "missing-adapter-callback";
    api.missing_device_callback = mode == "missing-device-callback";
    const bool unavailable = api.fail_instance || api.fail_adapter || !api.f16 || api.adapter_wait_error ||
                             api.missing_adapter_callback
#ifndef __EMSCRIPTEN__
                             || !api.implicit_sync
#endif
        ;
    if (mode == "concurrent") {
        std::atomic<bool> start{false};
        std::vector<ggml_backend_reg_t> registries(16);
        std::vector<ggml_backend_dev_t> devices(16);
        std::vector<std::thread> threads;
        for (size_t index = 0; index < registries.size(); ++index)
            threads.emplace_back([&, index] {
                while (!start.load(std::memory_order_acquire))
                    std::this_thread::yield();
                registries[index] = ggml_backend_webgpu_reg();
                SDC_CHECK(ggml_backend_webgpu_reg_get_device_count(registries[index]) == 1);
                devices[index] = ggml_backend_webgpu_reg_get_device(registries[index], 0);
                const auto published =
                    static_cast<ggml_backend_webgpu_device_context *>(devices[index]->context)->webgpu_global_ctx;
                SDC_CHECK(published->queue != nullptr);
            });
        start.store(true, std::memory_order_release);
        for (auto & thread : threads)
            thread.join();
        for (size_t index = 1; index < registries.size(); ++index)
            SDC_CHECK(registries[index] == registries.front() && devices[index] == devices.front());
    }
    const auto reg = ggml_backend_webgpu_reg();
    const auto context = static_cast<ggml_backend_webgpu_reg_context *>(reg->context);
    const auto complete_late = [&] {
        auto futures = std::move(api.futures);
        api.futures.clear();
        for (auto & [id, callback] : futures) {
            (void)id;
            callback();
        }
    };
    if (unavailable) {
        SDC_CHECK(ggml_backend_webgpu_reg_get_device_count(reg) == 0);
        api.fail_instance = api.fail_adapter = false;
        api.f16 = true;
        for (int i = 0; i < 50; ++i)
            SDC_CHECK(ggml_backend_webgpu_reg() == reg && ggml_backend_webgpu_reg_get_device_count(reg) == 0);
        SDC_CHECK(api.instances == 1 && api.adapters == (mode == "null-instance" ? 0 : 1) && api.devices == 0);
        complete_late();
        SDC_CHECK(context->webgpu_global_ctx->adapter == nullptr);
        SDC_CHECK(ggml_backend_webgpu_reg_get_device_count(reg) == 0);
    } else if (api.fail_device || api.device_wait_error || api.missing_device_callback) {
        SDC_CHECK(ggml_backend_webgpu_reg_get_device_count(reg) == 1);
        rejects([&] { ggml_backend_webgpu_reg_get_device(reg, 0); });
        SDC_CHECK(api.devices == 1 && api.queues == 0 && api.buffers == 0 && api.pipelines == 0 &&
                  context->webgpu_global_ctx->device == nullptr);
        complete_late();
        SDC_CHECK(context->webgpu_global_ctx->device == nullptr);
    } else {
        SDC_CHECK(ggml_backend_webgpu_reg_get_device_count(reg) == 1);
        std::vector<ggml_backend_dev_t> devices(16);
        if (mode == "concurrent") {
            std::vector<std::thread> threads;
            for (size_t index = 0; index < devices.size(); ++index)
                threads.emplace_back([&, index] {
                    for (int repeat = 0; repeat < 16; ++repeat) {
                        SDC_CHECK(ggml_backend_webgpu_reg() == reg);
                        devices[index] = ggml_backend_webgpu_reg_get_device(reg, 0);
                    }
                });
            for (auto & thread : threads)
                thread.join();
            for (const auto device : devices)
                SDC_CHECK(device == devices.front());
        } else {
            if (mode == "reordered")
                for (int i = 0; i < 50; ++i)
                    SDC_CHECK(ggml_backend_webgpu_reg_get_device_count(ggml_backend_webgpu_reg()) == 1);
            devices.front() = ggml_backend_webgpu_reg_get_device(reg, 0);
        }
        const auto device = devices.front();
        const auto global = static_cast<ggml_backend_webgpu_device_context *>(device->context)->webgpu_global_ctx;
        const auto instance = global->instance.Get();
        const auto adapter = global->adapter.Get();
        const auto gpu = global->device.Get();
        const auto queue = global->queue.Get();
        SDC_CHECK(global->capabilities.supports_subgroups == api.subgroups);
        wgpu::BufferDescriptor descriptor;
        descriptor.size = 4;
        descriptor.usage = wgpu::BufferUsage::CopyDst;
        const auto held = global->device.CreateBuffer(&descriptor);
        const auto buffers = api.buffers;
        SDC_CHECK(held.Get()->owner == gpu && queue->owner == gpu && gpu->owner == adapter &&
                  adapter->owner == instance);
        api.fail_adapter = true;
        if (mode == "device-lost")
            api.lost_callback.callback(&gpu, WGPUDeviceLostReason_Unknown, {"observer loss", WGPU_STRLEN},
                                       api.lost_callback.userdata1, api.lost_callback.userdata2);
        for (int repeat = 0; repeat < 50; ++repeat) {
            SDC_CHECK(ggml_backend_webgpu_reg() == reg);
            SDC_CHECK(ggml_backend_webgpu_reg_get_device(reg, 0) == device);
            SDC_CHECK(global->instance.Get() == instance && global->adapter.Get() == adapter &&
                      global->device.Get() == gpu && global->queue.Get() == queue);
            SDC_CHECK(held.Get()->owner == gpu && global->memset_params_buf.Get()->owner == gpu);
        }
        rejects([&] { ggml_backend_webgpu_reg_get_device(reg, 99); });
        SDC_CHECK(api.instances == 1 && api.adapters == 1 && api.devices == 1);
        SDC_CHECK(api.queues == 1 && api.pipelines == 1 && api.buffers == buffers);
    }
#ifdef __EMSCRIPTEN__
    constexpr const char * branch = "browser";
#else
    constexpr const char * branch = "native";
#endif
    std::cout << "complete backend registry " << branch << "-branch case=" << mode << '\n';
}
} // namespace

#include "dispatch-packets.hpp"
#include "mixed-queue.hpp"

int main(int argc, char ** argv) {
    try {
        if (argc == 2) {
            if (std::string(argv[1]) == "mixed-queue")
                graph_mixed_queue();
            else
                registry(argv[1]);
            return 0;
        }
        completion();
        arena();
        grid();
        dispatch_packets();
        graph_queue();
        bf16_admission();
        std::cout << "complete production backend; external API observer; no GPU executed\n";
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
