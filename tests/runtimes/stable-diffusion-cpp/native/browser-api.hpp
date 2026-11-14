#pragma once

// The pinned SDK owns every API declaration and C++ wrapper. This observer
// implements only the external C GPU boundary; production backend types and
// algorithms are compiled unchanged in the including translation unit.
#include "check.hpp"
#include <webgpu/webgpu_cpp.h>
#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace browser_api {
inline size_t live_buffers = 0;
}

struct WGPUInstanceImpl {
    unsigned refs = 1;
};
struct WGPUAdapterImpl {
    unsigned refs = 1;
    WGPUInstance owner = nullptr;
    ~WGPUAdapterImpl() {
        if (owner)
            wgpuInstanceRelease(owner);
    }
};
struct WGPUDeviceImpl {
    unsigned refs = 1;
    WGPUAdapter owner = nullptr;
    ~WGPUDeviceImpl() {
        if (owner)
            wgpuAdapterRelease(owner);
    }
};
struct WGPUQueueImpl {
    unsigned refs = 1;
    WGPUDevice owner = nullptr;
    ~WGPUQueueImpl() {
        if (owner)
            wgpuDeviceRelease(owner);
    }
};
struct WGPUBufferImpl {
    unsigned refs = 1;
    std::vector<unsigned char> bytes;
    bool destroyed = false;
    WGPUDevice owner = nullptr;
    bool observed = [] {
        ++browser_api::live_buffers;
        return true;
    }();
    uint64_t logical_size = 0; // Metadata-only bindings never expose payload bytes.
    ~WGPUBufferImpl() {
        if (owner)
            wgpuDeviceRelease(owner);
        if (observed)
            --browser_api::live_buffers;
    }
};
struct WGPUShaderModuleImpl {
    unsigned refs = 1;
    std::string code;
};
struct WGPUComputePipelineImpl {
    unsigned refs = 1;
    std::string label;
};
struct WGPUBindGroupLayoutImpl {
    unsigned refs = 1;
};
struct WGPUBindGroupImpl {
    unsigned refs = 1;
    std::vector<WGPUBindGroupEntry> entries;
};
struct WGPUCommandEncoderImpl {
    unsigned refs = 1;
    std::vector<std::function<void()>> commands;
    size_t dispatches = 0;
};
struct WGPUCommandBufferImpl {
    unsigned refs = 1;
    std::vector<std::function<void()>> commands;
    size_t dispatches = 0;
};
struct WGPUComputePassEncoderImpl {
    unsigned refs = 1;
    WGPUCommandEncoder encoder;
    WGPUBindGroup group = nullptr;
};

namespace browser_api {
template <class Handle> void retain(Handle handle) {
    SDC_CHECK(handle != nullptr);
    ++handle->refs;
}
template <class Handle> void release(Handle handle) {
    SDC_CHECK(handle != nullptr);
    if (!--handle->refs)
        delete handle;
}
inline std::string text(WGPUStringView value) {
    return value.data == nullptr         ? ""
           : value.length == WGPU_STRLEN ? std::string(value.data)
                                         : std::string(value.data, value.length);
}
struct State {
    uint64_t next_future = 1;
    std::map<uint64_t, std::function<void()>> futures;
    std::vector<std::function<void()>> pending;
    std::vector<std::vector<uint32_t>> uniforms;
    std::vector<std::pair<uint32_t, uint32_t>> dispatches;
    std::vector<size_t> binding_counts, write_sizes;
    std::vector<size_t> submission_dispatches;
    unsigned reads = 0, writes = 0, submissions = 0, waits = 0, requests = 0, copies = 0;
    int submission_budget = -1;
    unsigned instances = 0, adapters = 0, devices = 0, queues = 0, buffers = 0, pipelines = 0;
    uint64_t observed_timeout = 0;
    WGPUWaitStatus wait_status = WGPUWaitStatus_Success;
    WGPUQueueWorkDoneStatus queue_status = WGPUQueueWorkDoneStatus_Success;
    WGPUMapAsyncStatus map_status = WGPUMapAsyncStatus_Success;
    bool fire_callback = true, fail_instance = false, fail_adapter = false, fail_device = false;
    bool f16 = true, subgroups = true, implicit_sync = true;
    bool adapter_wait_error = false, device_wait_error = false;
    bool missing_adapter_callback = false, missing_device_callback = false;
    WGPUDeviceLostCallbackInfo lost_callback{};
    std::function<void()> during_wait;
    std::mutex mutex;
};
inline State state;
inline WGPUFuture future(std::function<void()> callback) {
    const auto id = state.next_future++;
    state.futures.emplace(id, std::move(callback));
    return {id};
}
inline void drain() {
    auto pending = std::move(state.pending);
    state.pending.clear();
    for (auto & command : pending)
        command();
}
inline void reset_wait() {
    state.waits = state.requests = 0;
    state.wait_status = WGPUWaitStatus_Success;
    state.queue_status = WGPUQueueWorkDoneStatus_Success;
    state.map_status = WGPUMapAsyncStatus_Success;
    state.fire_callback = true;
    state.during_wait = {};
}
} // namespace browser_api

extern "C" {
void __wrap_ggml_abort(const char *, int, const char * format, ...) {
    char message[2048];
    va_list args;
    va_start(args, format);
    std::vsnprintf(message, sizeof(message), format, args);
    va_end(args);
    throw std::runtime_error(message);
}
#define SDC_GPU_REFS(Name)                                                                                             \
    void wgpu##Name##AddRef(WGPU##Name handle) {                                                                       \
        browser_api::retain(handle);                                                                                   \
    }                                                                                                                  \
    void wgpu##Name##Release(WGPU##Name handle) {                                                                      \
        browser_api::release(handle);                                                                                  \
    }
SDC_GPU_REFS(Instance)
SDC_GPU_REFS(Adapter)
SDC_GPU_REFS(Device)
SDC_GPU_REFS(Queue)
SDC_GPU_REFS(Buffer)
SDC_GPU_REFS(ShaderModule)
SDC_GPU_REFS(ComputePipeline)
SDC_GPU_REFS(BindGroupLayout)
SDC_GPU_REFS(CommandEncoder)
SDC_GPU_REFS(CommandBuffer)
#undef SDC_GPU_REFS
void wgpuComputePassEncoderAddRef(WGPUComputePassEncoder handle) {
    browser_api::retain(handle);
}
void wgpuComputePassEncoderRelease(WGPUComputePassEncoder handle) {
    SDC_CHECK(handle != nullptr);
    if (!--handle->refs) {
        browser_api::release(handle->encoder);
        delete handle;
    }
}
void wgpuBindGroupAddRef(WGPUBindGroup handle) {
    browser_api::retain(handle);
}
void wgpuBindGroupRelease(WGPUBindGroup handle) {
    SDC_CHECK(handle != nullptr);
    if (!--handle->refs) {
        for (const auto & entry : handle->entries)
            if (entry.buffer)
                browser_api::release(entry.buffer);
        delete handle;
    }
}
WGPUInstance wgpuCreateInstance(const WGPUInstanceDescriptor *) {
    ++browser_api::state.instances;
    return browser_api::state.fail_instance ? nullptr : new WGPUInstanceImpl;
}
WGPUFuture wgpuInstanceRequestAdapter(WGPUInstance instance, const WGPURequestAdapterOptions *,
                                      WGPURequestAdapterCallbackInfo info) {
    ++browser_api::state.adapters;
    browser_api::retain(instance);
    auto owner = std::shared_ptr<WGPUInstanceImpl>(instance, wgpuInstanceRelease);
    return browser_api::future([info, owner] {
        auto adapter = browser_api::state.fail_adapter ? nullptr : new WGPUAdapterImpl{1, owner.get()};
        if (adapter)
            browser_api::retain(owner.get());
        info.callback(adapter ? WGPURequestAdapterStatus_Success : WGPURequestAdapterStatus_Unavailable, adapter,
                      {"observer adapter", WGPU_STRLEN}, info.userdata1, info.userdata2);
    });
}
WGPUBool wgpuAdapterHasFeature(WGPUAdapter, WGPUFeatureName feature) {
    if (feature == WGPUFeatureName_ShaderF16)
        return browser_api::state.f16;
    if (feature == WGPUFeatureName_Subgroups)
        return browser_api::state.subgroups;
#ifndef __EMSCRIPTEN__
    if (feature == WGPUFeatureName_ImplicitDeviceSynchronization)
        return browser_api::state.implicit_sync;
#endif
    return false;
}
WGPUBool wgpuDeviceHasFeature(WGPUDevice, WGPUFeatureName) {
    return false;
}
WGPUBool wgpuInstanceHasWGSLLanguageFeature(WGPUInstance, WGPUWGSLLanguageFeatureName) {
    return false;
}
WGPUStatus wgpuAdapterGetLimits(WGPUAdapter, WGPULimits * limits) {
    *limits = WGPULimits{};
    limits->maxStorageBufferBindingSize = 128 * 1024 * 1024;
    limits->minStorageBufferOffsetAlignment = 256;
    limits->minUniformBufferOffsetAlignment = 256;
    limits->maxComputeWorkgroupStorageSize = 32768;
    limits->maxComputeInvocationsPerWorkgroup = 256;
    limits->maxComputeWorkgroupsPerDimension = 65535;
    return WGPUStatus_Success;
}
WGPUStatus wgpuAdapterGetInfo(WGPUAdapter, WGPUAdapterInfo * info) {
    info->vendor = {"observer", WGPU_STRLEN};
    info->architecture = {"native", WGPU_STRLEN};
    info->device = {"external API", WGPU_STRLEN};
    info->description = {"no GPU", WGPU_STRLEN};
    info->subgroupMinSize = 4;
    info->subgroupMaxSize = 32;
    return WGPUStatus_Success;
}
void wgpuAdapterInfoFreeMembers(WGPUAdapterInfo) {}
#ifndef __EMSCRIPTEN__
void wgpuAdapterPropertiesSubgroupMatrixConfigsFreeMembers(WGPUAdapterPropertiesSubgroupMatrixConfigs configs) {
    SDC_CHECK(configs.configCount == 0 && configs.configs == nullptr);
}
#endif
WGPUFuture wgpuAdapterRequestDevice(WGPUAdapter adapter, const WGPUDeviceDescriptor * descriptor,
                                    WGPURequestDeviceCallbackInfo info) {
    ++browser_api::state.devices;
    browser_api::state.lost_callback = descriptor->deviceLostCallbackInfo;
    browser_api::retain(adapter);
    auto owner = std::shared_ptr<WGPUAdapterImpl>(adapter, wgpuAdapterRelease);
    return browser_api::future([info, owner] {
        auto device = browser_api::state.fail_device ? nullptr : new WGPUDeviceImpl{1, owner.get()};
        if (device)
            browser_api::retain(owner.get());
        info.callback(device ? WGPURequestDeviceStatus_Success : WGPURequestDeviceStatus_Error, device,
                      {"observer device", WGPU_STRLEN}, info.userdata1, info.userdata2);
    });
}
WGPUQueue wgpuDeviceGetQueue(WGPUDevice device) {
    ++browser_api::state.queues;
    browser_api::retain(device);
    return new WGPUQueueImpl{1, device};
}
WGPUFuture wgpuQueueOnSubmittedWorkDone(WGPUQueue, WGPUQueueWorkDoneCallbackInfo info) {
    ++browser_api::state.requests;
    return browser_api::future([info] {
        browser_api::drain();
        info.callback(browser_api::state.queue_status, {"observer queue", WGPU_STRLEN}, info.userdata1, info.userdata2);
    });
}
WGPUFuture wgpuBufferMapAsync(WGPUBuffer, WGPUMapMode, size_t, size_t, WGPUBufferMapCallbackInfo info) {
    ++browser_api::state.requests;
    ++browser_api::state.reads;
    return browser_api::future([info] {
        browser_api::drain();
        info.callback(browser_api::state.map_status, {"observer map", WGPU_STRLEN}, info.userdata1, info.userdata2);
    });
}
WGPUWaitStatus wgpuInstanceWaitAny(WGPUInstance, size_t count, WGPUFutureWaitInfo * infos, uint64_t timeout) {
    SDC_CHECK(count == 1);
    ++browser_api::state.waits;
    browser_api::state.observed_timeout = timeout;
    if (browser_api::state.devices == 0 && browser_api::state.adapter_wait_error)
        return WGPUWaitStatus_Error;
    if (browser_api::state.devices > 0 && browser_api::state.device_wait_error)
        return WGPUWaitStatus_Error;
    if (browser_api::state.devices == 0 && browser_api::state.missing_adapter_callback)
        return WGPUWaitStatus_Success;
    if (browser_api::state.devices > 0 && browser_api::state.missing_device_callback)
        return WGPUWaitStatus_Success;
    if (browser_api::state.during_wait)
        browser_api::state.during_wait();
    if (browser_api::state.wait_status != WGPUWaitStatus_Success)
        return browser_api::state.wait_status;
    if (!browser_api::state.fire_callback)
        return WGPUWaitStatus_Success;
    const auto found = browser_api::state.futures.find(infos[0].future.id);
    SDC_CHECK(found != browser_api::state.futures.end());
    const auto callback = found->second;
    browser_api::state.futures.erase(found);
    callback();
    infos[0].completed = true;
    return WGPUWaitStatus_Success;
}
WGPUBuffer wgpuDeviceCreateBuffer(WGPUDevice device, const WGPUBufferDescriptor * descriptor) {
    ++browser_api::state.buffers;
    browser_api::retain(device);
    return new WGPUBufferImpl{1, std::vector<unsigned char>(descriptor->size), false, device};
}
void wgpuBufferDestroy(WGPUBuffer buffer) {
    SDC_CHECK(!buffer->destroyed);
    buffer->destroyed = true;
}
uint64_t wgpuBufferGetSize(WGPUBuffer buffer) {
    return buffer->logical_size != 0 ? buffer->logical_size : buffer->bytes.size();
}
void * wgpuBufferGetMappedRange(WGPUBuffer buffer, size_t offset, size_t size) {
    if (size == WGPU_WHOLE_MAP_SIZE) {
        SDC_CHECK(offset <= buffer->bytes.size());
        size = buffer->bytes.size() - offset;
    }
    SDC_CHECK(offset <= buffer->bytes.size() && size <= buffer->bytes.size() - offset);
    return buffer->bytes.data() + offset;
}
const void * wgpuBufferGetConstMappedRange(WGPUBuffer buffer, size_t offset, size_t size) {
    return wgpuBufferGetMappedRange(buffer, offset, size);
}
void wgpuBufferUnmap(WGPUBuffer) {}
void wgpuQueueWriteBuffer(WGPUQueue, WGPUBuffer buffer, uint64_t offset, const void * data, size_t size) {
    SDC_CHECK(offset % 4 == 0 && size % 4 == 0 && offset <= buffer->bytes.size() &&
              size <= buffer->bytes.size() - offset);
    ++browser_api::state.writes;
    browser_api::state.write_sizes.push_back(size);
    std::vector<unsigned char> owned(static_cast<const unsigned char *>(data),
                                     static_cast<const unsigned char *>(data) + size);
    browser_api::retain(buffer);
    auto retained = std::shared_ptr<WGPUBufferImpl>(buffer, wgpuBufferRelease);
    browser_api::state.pending.push_back([retained, offset, owned] {
        SDC_CHECK(!retained->destroyed);
        std::copy(owned.begin(), owned.end(), retained->bytes.begin() + offset);
    });
}
WGPUCommandEncoder wgpuDeviceCreateCommandEncoder(WGPUDevice, const WGPUCommandEncoderDescriptor *) {
    return new WGPUCommandEncoderImpl;
}
void wgpuCommandEncoderCopyBufferToBuffer(WGPUCommandEncoder encoder, WGPUBuffer source, uint64_t from,
                                          WGPUBuffer target, uint64_t to, uint64_t size) {
    SDC_CHECK(source != target && from % 4 == 0 && to % 4 == 0 && size % 4 == 0);
    SDC_CHECK(from <= source->bytes.size() && size <= source->bytes.size() - from);
    SDC_CHECK(to <= target->bytes.size() && size <= target->bytes.size() - to);
    browser_api::retain(source);
    browser_api::retain(target);
    auto retained_source = std::shared_ptr<WGPUBufferImpl>(source, wgpuBufferRelease);
    auto retained_target = std::shared_ptr<WGPUBufferImpl>(target, wgpuBufferRelease);
    encoder->commands.push_back([retained_source, retained_target, from, to, size] {
        SDC_CHECK(!retained_source->destroyed && !retained_target->destroyed);
        std::copy_n(retained_source->bytes.begin() + from, size, retained_target->bytes.begin() + to);
        ++browser_api::state.copies;
    });
}
WGPUCommandBuffer wgpuCommandEncoderFinish(WGPUCommandEncoder encoder, const WGPUCommandBufferDescriptor *) {
    return new WGPUCommandBufferImpl{1, std::move(encoder->commands), encoder->dispatches};
}
void wgpuQueueSubmit(WGPUQueue, size_t count, const WGPUCommandBuffer * commands) {
    if (browser_api::state.submission_budget == 0)
        throw std::runtime_error("observer queue submission failure");
    if (browser_api::state.submission_budget > 0)
        --browser_api::state.submission_budget;
    ++browser_api::state.submissions;
    size_t dispatches = 0;
    for (size_t index = 0; index < count; ++index)
        for (const auto & command : commands[index]->commands) {
            browser_api::state.pending.push_back(command);
        }
    for (size_t index = 0; index < count; ++index)
        dispatches += commands[index]->dispatches;
    browser_api::state.submission_dispatches.push_back(dispatches);
}
WGPUShaderModule wgpuDeviceCreateShaderModule(WGPUDevice, const WGPUShaderModuleDescriptor * descriptor) {
    const auto * shader = reinterpret_cast<const WGPUShaderSourceWGSL *>(descriptor->nextInChain);
    return new WGPUShaderModuleImpl{1, browser_api::text(shader->code)};
}
WGPUComputePipeline wgpuDeviceCreateComputePipeline(WGPUDevice, const WGPUComputePipelineDescriptor * descriptor) {
    ++browser_api::state.pipelines;
    return new WGPUComputePipelineImpl{1, browser_api::text(descriptor->label)};
}
WGPUBindGroupLayout wgpuComputePipelineGetBindGroupLayout(WGPUComputePipeline, uint32_t) {
    return new WGPUBindGroupLayoutImpl;
}
WGPUBindGroup wgpuDeviceCreateBindGroup(WGPUDevice, const WGPUBindGroupDescriptor * descriptor) {
    auto result = new WGPUBindGroupImpl;
    result->entries.assign(descriptor->entries, descriptor->entries + descriptor->entryCount);
    for (const auto & entry : result->entries)
        if (entry.buffer) {
            const auto size = wgpuBufferGetSize(entry.buffer);
            SDC_CHECK(entry.offset <= size && entry.size <= size - entry.offset);
            browser_api::retain(entry.buffer);
        }
    return result;
}
WGPUComputePassEncoder wgpuCommandEncoderBeginComputePass(WGPUCommandEncoder encoder,
                                                          const WGPUComputePassDescriptor *) {
    browser_api::retain(encoder);
    return new WGPUComputePassEncoderImpl{1, encoder, nullptr};
}
void wgpuComputePassEncoderSetPipeline(WGPUComputePassEncoder, WGPUComputePipeline) {}
void wgpuComputePassEncoderSetBindGroup(WGPUComputePassEncoder pass, uint32_t, WGPUBindGroup group, size_t,
                                        const uint32_t *) {
    pass->group = group;
}
void wgpuComputePassEncoderDispatchWorkgroups(WGPUComputePassEncoder pass, uint32_t x, uint32_t y, uint32_t z) {
    SDC_CHECK(z == 1 && pass->group != nullptr);
    ++pass->encoder->dispatches;
    browser_api::state.dispatches.emplace_back(x, y);
    browser_api::state.binding_counts.push_back(pass->group->entries.size());
    auto group = pass->group;
    wgpuBindGroupAddRef(group);
    auto retained_group = std::shared_ptr<WGPUBindGroupImpl>(group, wgpuBindGroupRelease);
    pass->encoder->commands.push_back([retained_group] {
        const auto & params = retained_group->entries.back();
        SDC_CHECK(params.buffer != nullptr && params.offset + params.size <= params.buffer->bytes.size());
        std::vector<uint32_t> words(params.size / 4);
        std::memcpy(words.data(), params.buffer->bytes.data() + params.offset, params.size);
        browser_api::state.uniforms.push_back(std::move(words));
    });
}
void wgpuComputePassEncoderEnd(WGPUComputePassEncoder) {}
void wgpuPipelineLayoutRelease(WGPUPipelineLayout) {
    throw std::runtime_error("unexpected GPU pipeline layout");
}
void wgpuSurfaceRelease(WGPUSurface) {
    throw std::runtime_error("unexpected GPU surface");
}
void wgpuSamplerAddRef(WGPUSampler) {
    throw std::runtime_error("unexpected GPU sampler");
}
void wgpuSamplerRelease(WGPUSampler) {
    throw std::runtime_error("unexpected GPU sampler");
}
void wgpuTextureViewAddRef(WGPUTextureView) {
    throw std::runtime_error("unexpected GPU texture view");
}
void wgpuTextureViewRelease(WGPUTextureView) {
    throw std::runtime_error("unexpected GPU texture view");
}
}
