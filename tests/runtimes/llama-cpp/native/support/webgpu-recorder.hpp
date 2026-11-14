#pragma once

// Actual complete shader libraries and pinned SDK C++ bindings. Only the C GPU
// API boundary is replaced; no adapter, native runtime, browser or GPU is used.
#include <webgpu/webgpu_cpp.h>
#include "ggml-impl.h"
#include "pre_wgsl.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include "check.hpp"
#include <chrono>
#include <cctype>
#include <cstdarg>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>
#ifdef BICORE_REAL_GGML
#include "allocation-fault.hpp"
using ExternalAllocation = allocation_fault::ExternalCall;
#else
struct ExternalAllocation {};
#endif
#ifndef BICORE_REAL_GGML
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#endif

struct WGPUInstanceImpl {
    unsigned refs = 1;
};
struct WGPUAdapterImpl {
    unsigned refs = 1;
};
struct WGPUDeviceImpl {
    unsigned refs = 1;
    std::vector<std::pair<WGPUErrorFilter, WGPUErrorType>> scopes;
    WGPUQueue queue = nullptr;
};
struct WGPUShaderModuleImpl {
    unsigned refs = 1;
    std::string code;
};
struct WGPUPipelineLayoutImpl {
    unsigned refs = 1;
};
struct WGPUComputePipelineImpl {
    unsigned refs = 1;
    unsigned id;
    std::string code;
    std::string name;
};
struct WGPUBufferImpl {
    unsigned refs = 1;
    std::vector<uint8_t> bytes;
    bool destroyed = false;
    WGPUBufferUsage usage = WGPUBufferUsage_None;
    WGPUQueue queue = nullptr;
    bool mapped = false;
};
struct WGPUBindGroupLayoutImpl {
    unsigned refs = 1;
};
struct WGPUBindGroupImpl {
    unsigned refs = 1;
    unsigned id;
    std::vector<WGPUBuffer> buffers;
    std::vector<WGPUBindGroupEntry> bindings;
};
struct WGPUSamplerImpl {
    unsigned refs = 1;
};
struct WGPUTextureViewImpl {
    unsigned refs = 1;
};
struct WGPUQueueImpl {
    unsigned refs = 1;
    uint64_t completed = 0;
    std::vector<std::function<void()>> pending;
};
struct WGPUCommandEncoderImpl {
    unsigned refs = 1;
    std::shared_ptr<std::vector<std::function<void()>>> operations =
        std::make_shared<std::vector<std::function<void()>>>();
};
struct WGPUCommandBufferImpl {
    unsigned refs = 1;
    std::shared_ptr<std::vector<std::function<void()>>> operations;
};
struct WGPUComputePassEncoderImpl {
    unsigned refs = 1;
    WGPUComputePipeline pipeline = nullptr;
    WGPUBindGroup group = nullptr;
    bool ended = false;
    std::shared_ptr<std::vector<std::function<void()>>> operations;
};
struct WGPUQuerySetImpl {
    unsigned refs = 1;
    bool destroyed = false;
    uint32_t count = 0;
};
struct RecordedDispatch {
    unsigned pipeline;
    unsigned group;
    uint32_t x, y, z;
};
struct RecordedWrite {
    WGPUBuffer buffer;
    uint64_t offset;
    std::vector<uint8_t> bytes;
};
struct CommandObserver {
    unsigned passes = 0, ends = 0;
    std::vector<RecordedDispatch> dispatches;
    std::vector<RecordedWrite> writes;
    std::vector<std::pair<uint32_t, uint32_t>> timestamp_pairs;
    std::vector<uint32_t> resolved_queries;
    std::vector<char> order;
    std::vector<uint32_t> observed_params;
    unsigned submissions = 0;
    unsigned executed = 0;
    unsigned queue_waits = 0;
    unsigned maps = 0;
    unsigned copies = 0, unmaps = 0, direct_reads = 0, pointer_reads = 0;
    unsigned buffers_created = 0, buffers_destroyed = 0;
};
static CommandObserver commands;
struct ReadbackControl {
    WGPUMapAsyncStatus map_status = WGPUMapAsyncStatus_Success;
    WGPUQueueWorkDoneStatus queue_status = WGPUQueueWorkDoneStatus_Success;
    WGPUStatus read_status = WGPUStatus_Success;
    std::function<void()> before_map;
};
static ReadbackControl readback;
enum class GroupFailure { none, null_result, exception };
struct ResourceObserver {
    unsigned groups_created = 0;
    unsigned groups_alive = 0;
    unsigned layouts_created = 0;
    unsigned buffers_released = 0, pipelines_released = 0;
    std::function<void(WGPUBuffer)> before_buffer_destroy;
    GroupFailure group_failure = GroupFailure::none;
};
static ResourceObserver resources;
struct Future {
    std::function<void()> complete;
};
static std::map<std::uint64_t, Future> futures;
static unsigned created = 0, pushes = 0, pops = 0, waits = 0;
static std::uint64_t next_future = 1;
static WGPUErrorType inject = WGPUErrorType_NoError;
static std::function<void()> on_wait_once;
static bool request_error = false, request_throw = false, wait_throw = false, wait_timeout = false, wait_error = false,
            wait_without_callback = false;
static unsigned device_requests = 0;
static void * last_request_owner = nullptr;
static WGPUFuture last_request_future{};
static void drain_until(WGPUQueue queue, uint64_t target) {
    while (queue->completed < target) {
        BICORE_CHECK(!queue->pending.empty());
        auto operation = std::move(queue->pending.front());
        queue->pending.erase(queue->pending.begin());
        ++queue->completed;
        operation();
    }
}
static void drain_queue(WGPUQueue queue) {
    drain_until(queue, queue->completed + queue->pending.size());
}
static std::string text(WGPUStringView view) {
    return view.data == nullptr         ? ""
           : view.length == WGPU_STRLEN ? std::string(view.data)
                                        : std::string(view.data, view.length);
}
extern "C" {
WGPUBuffer wgpuDeviceCreateBuffer(WGPUDevice device, const WGPUBufferDescriptor * descriptor) {
    ExternalAllocation external;
    BICORE_CHECK(descriptor != nullptr && descriptor->nextInChain == nullptr && !descriptor->mappedAtCreation);
    auto * result = new WGPUBufferImpl;
    ++commands.buffers_created;
    result->bytes.resize(descriptor->size);
    result->usage = descriptor->usage;
    result->queue = device->queue;
    return result;
}
void wgpuBufferDestroy(WGPUBuffer value) {
    BICORE_CHECK(!value->destroyed);
    if (resources.before_buffer_destroy)
        resources.before_buffer_destroy(value);
    value->destroyed = true;
    ++commands.buffers_destroyed;
}
uint64_t wgpuBufferGetSize(WGPUBuffer value) {
    return value->bytes.size();
}
WGPUBufferUsage wgpuBufferGetUsage(WGPUBuffer value) {
    return value->usage;
}
void wgpuQueueAddRef(WGPUQueue value) {
    ++value->refs;
}
void wgpuQueueRelease(WGPUQueue value) {
    if (!--value->refs) {
        BICORE_CHECK(value->pending.empty());
        delete value;
    }
}
WGPUQueue wgpuDeviceGetQueue(WGPUDevice device) {
    ExternalAllocation external;
    if (!device->queue)
        device->queue = new WGPUQueueImpl;
    wgpuQueueAddRef(device->queue);
    return device->queue;
}
void wgpuQueueWriteBuffer(WGPUQueue queue, WGPUBuffer buffer, uint64_t offset, const void * data, size_t size) {
    ExternalAllocation external;
    BICORE_CHECK(!buffer->destroyed && offset <= buffer->bytes.size() && size <= buffer->bytes.size() - offset);
    BICORE_CHECK((buffer->usage & WGPUBufferUsage_CopyDst) != 0);
    const auto * bytes = static_cast<const uint8_t *>(data);
    commands.writes.push_back({buffer, offset, {bytes, bytes + size}});
    wgpuBufferAddRef(buffer);
    auto retained = wgpu::Buffer::Acquire(buffer);
    queue->pending.push_back([retained, offset, copied = std::vector<uint8_t>(bytes, bytes + size)] {
        if (!copied.empty())
            std::memcpy(retained.Get()->bytes.data() + offset, copied.data(), copied.size());
    });
}
void wgpuCommandEncoderAddRef(WGPUCommandEncoder value) {
    ++value->refs;
}
void wgpuCommandEncoderRelease(WGPUCommandEncoder value) {
    if (!--value->refs)
        delete value;
}
WGPUCommandEncoder wgpuDeviceCreateCommandEncoder(WGPUDevice, const WGPUCommandEncoderDescriptor * descriptor) {
    ExternalAllocation external;
    BICORE_CHECK(descriptor == nullptr || descriptor->nextInChain == nullptr);
    return new WGPUCommandEncoderImpl;
}
WGPUComputePassEncoder wgpuCommandEncoderBeginComputePass(WGPUCommandEncoder encoder,
                                                          const WGPUComputePassDescriptor * descriptor) {
    ExternalAllocation external;
    BICORE_CHECK(descriptor == nullptr || descriptor->nextInChain == nullptr);
    ++commands.passes;
    commands.order.push_back('C');
    if (descriptor && descriptor->timestampWrites) {
        const auto & timestamps = *descriptor->timestampWrites;
        BICORE_CHECK(timestamps.querySet != nullptr && !timestamps.querySet->destroyed);
        BICORE_CHECK(timestamps.beginningOfPassWriteIndex < timestamps.querySet->count &&
                     timestamps.endOfPassWriteIndex < timestamps.querySet->count);
        commands.timestamp_pairs.emplace_back(timestamps.beginningOfPassWriteIndex, timestamps.endOfPassWriteIndex);
    }
    return new WGPUComputePassEncoderImpl{1, nullptr, nullptr, false, encoder->operations};
}
void wgpuComputePassEncoderAddRef(WGPUComputePassEncoder value) {
    ++value->refs;
}
void wgpuComputePassEncoderRelease(WGPUComputePassEncoder value) {
    if (!--value->refs)
        delete value;
}
void wgpuComputePassEncoderSetPipeline(WGPUComputePassEncoder pass, WGPUComputePipeline pipeline) {
    BICORE_CHECK(pass != nullptr && !pass->ended && pipeline != nullptr);
    pass->pipeline = pipeline;
    commands.order.push_back('P');
}
void wgpuComputePassEncoderSetBindGroup(WGPUComputePassEncoder pass, uint32_t index, WGPUBindGroup group,
                                        size_t offset_count, const uint32_t *) {
    BICORE_CHECK(pass != nullptr && !pass->ended && index == 0 && group != nullptr && offset_count == 0);
    pass->group = group;
    commands.order.push_back('B');
}
void wgpuComputePassEncoderDispatchWorkgroups(WGPUComputePassEncoder pass, uint32_t x, uint32_t y, uint32_t z) {
    ExternalAllocation external;
    BICORE_CHECK(pass != nullptr && !pass->ended && pass->pipeline != nullptr && pass->group != nullptr);
    commands.dispatches.push_back({pass->pipeline->id, pass->group->id, x, y, z});
    commands.order.push_back('D');
    wgpuBindGroupAddRef(pass->group);
    auto group = wgpu::BindGroup::Acquire(pass->group);
    wgpuComputePipelineAddRef(pass->pipeline);
    auto pipeline = wgpu::ComputePipeline::Acquire(pass->pipeline);
    pass->operations->push_back([group, pipeline] {
        BICORE_CHECK(group && pipeline);
        const auto & uniform = group.Get()->bindings.back();
        BICORE_CHECK(uniform.buffer && !uniform.buffer->destroyed);
        uint32_t value = 0;
        std::memcpy(&value, uniform.buffer->bytes.data() + uniform.offset, sizeof(value));
        commands.observed_params.push_back(value);
        ++commands.executed;
    });
}
void wgpuComputePassEncoderEnd(WGPUComputePassEncoder pass) {
    BICORE_CHECK(pass != nullptr && !pass->ended);
    pass->ended = true;
    ++commands.ends;
    commands.order.push_back('E');
}
void wgpuCommandBufferAddRef(WGPUCommandBuffer value) {
    ++value->refs;
}
void wgpuCommandBufferRelease(WGPUCommandBuffer value) {
    if (!--value->refs)
        delete value;
}
WGPUCommandBuffer wgpuCommandEncoderFinish(WGPUCommandEncoder encoder, const WGPUCommandBufferDescriptor * descriptor) {
    ExternalAllocation external;
    BICORE_CHECK(descriptor == nullptr || descriptor->nextInChain == nullptr);
    return new WGPUCommandBufferImpl{1, encoder->operations};
}
void wgpuQueueSubmit(WGPUQueue queue, size_t count, const WGPUCommandBuffer * buffers) {
    ExternalAllocation external;
    ++commands.submissions;
    for (size_t i = 0; i < count; ++i) {
        const auto operations = buffers[i]->operations;
        queue->pending.push_back([operations] {
            for (auto & operation : *operations)
                operation();
        });
    }
}
WGPUFuture wgpuQueueOnSubmittedWorkDone(WGPUQueue queue, WGPUQueueWorkDoneCallbackInfo info) {
    ExternalAllocation external;
    ++commands.queue_waits;
    const auto id = next_future++;
    const auto target = queue->completed + queue->pending.size();
    futures.emplace(
        id, Future{[=] {
            drain_until(queue, target);
            info.callback(readback.queue_status, {"queue result", WGPU_STRLEN}, info.userdata1, info.userdata2);
        }});
    return {id};
}
void wgpuCommandEncoderResolveQuerySet(WGPUCommandEncoder encoder, WGPUQuerySet queries, uint32_t first, uint32_t count,
                                       WGPUBuffer destination, uint64_t offset) {
    ExternalAllocation external;
    BICORE_CHECK(queries && !queries->destroyed && first <= queries->count && count <= queries->count - first);
    BICORE_CHECK(destination && !destination->destroyed && offset % 256 == 0 &&
                 (destination->usage & WGPUBufferUsage_QueryResolve) != 0);
    BICORE_CHECK(offset <= destination->bytes.size() && count * sizeof(uint64_t) <= destination->bytes.size() - offset);
    commands.resolved_queries.push_back(count);
    wgpuBufferAddRef(destination);
    const auto buffer = wgpu::Buffer::Acquire(destination);
    // Deterministic external timestamp data permits the actual collector's
    // mapping/lifetime path to execute; it represents no GPU elapsed time.
    encoder->operations->push_back([=] {
        BICORE_CHECK(!buffer.Get()->destroyed);
        for (uint32_t index = 0; index < count; ++index) {
            const uint64_t timestamp = first + index;
            std::memcpy(buffer.Get()->bytes.data() + offset + index * sizeof(timestamp), &timestamp, sizeof(timestamp));
        }
    });
}
void wgpuCommandEncoderCopyBufferToBuffer(WGPUCommandEncoder encoder, WGPUBuffer source, uint64_t source_offset,
                                          WGPUBuffer destination, uint64_t destination_offset, uint64_t size) {
    ExternalAllocation external;
    BICORE_CHECK(!source->destroyed && !destination->destroyed && source_offset % 4 == 0 &&
                 destination_offset % 4 == 0 && size % 4 == 0);
    BICORE_CHECK(source_offset <= source->bytes.size() && size <= source->bytes.size() - source_offset);
    BICORE_CHECK(destination_offset <= destination->bytes.size() &&
                 size <= destination->bytes.size() - destination_offset);
    BICORE_CHECK((source->usage & WGPUBufferUsage_CopySrc) != 0 && (destination->usage & WGPUBufferUsage_CopyDst) != 0);
    ++commands.copies;
    wgpuBufferAddRef(source);
    wgpuBufferAddRef(destination);
    const auto src = wgpu::Buffer::Acquire(source), dst = wgpu::Buffer::Acquire(destination);
    encoder->operations->push_back([=] {
        BICORE_CHECK(!src.Get()->destroyed && !dst.Get()->destroyed);
        std::memcpy(dst.Get()->bytes.data() + destination_offset, src.Get()->bytes.data() + source_offset, size);
    });
}
WGPUFuture wgpuBufferMapAsync(WGPUBuffer buffer, WGPUMapMode mode, size_t offset, size_t size,
                              WGPUBufferMapCallbackInfo info) {
    ExternalAllocation external;
    BICORE_CHECK(mode == WGPUMapMode_Read && !buffer->mapped && buffer->queue != nullptr);
    BICORE_CHECK(offset <= buffer->bytes.size() && size <= buffer->bytes.size() - offset);
    BICORE_CHECK((buffer->usage & WGPUBufferUsage_MapRead) != 0);
    ++commands.maps;
    const auto id = next_future++;
    const auto target = buffer->queue->completed + buffer->queue->pending.size();
    futures.emplace(id, Future{[=] {
                        if (readback.before_map)
                            readback.before_map();
                        drain_until(buffer->queue, target);
                        if (readback.map_status == WGPUMapAsyncStatus_Success)
                            buffer->mapped = true;
                        info.callback(readback.map_status, {"map result", WGPU_STRLEN}, info.userdata1, info.userdata2);
                    }});
    return {id};
}
const void * wgpuBufferGetConstMappedRange(WGPUBuffer buffer, size_t offset, size_t size) {
    BICORE_CHECK(buffer->mapped && offset % 8 == 0 && size % 4 == 0);
    BICORE_CHECK(offset <= buffer->bytes.size() && size <= buffer->bytes.size() - offset);
    ++commands.pointer_reads;
    return buffer->bytes.data() + offset;
}
WGPUStatus wgpuBufferReadMappedRange(WGPUBuffer buffer, size_t offset, void * data, size_t size) {
    BICORE_CHECK(buffer->mapped && offset % 8 == 0 && size % 4 == 0);
    BICORE_CHECK(offset <= buffer->bytes.size() && size <= buffer->bytes.size() - offset);
    ++commands.direct_reads;
    if (readback.read_status != WGPUStatus_Success)
        return readback.read_status;
    if (size)
        std::memcpy(data, buffer->bytes.data() + offset, size);
    return WGPUStatus_Success;
}
void wgpuBufferUnmap(WGPUBuffer buffer) {
    BICORE_CHECK(buffer->mapped);
    buffer->mapped = false;
    ++commands.unmaps;
}
void wgpuQuerySetAddRef(WGPUQuerySet value) {
    ++value->refs;
}
void wgpuQuerySetRelease(WGPUQuerySet value) {
    if (!--value->refs)
        delete value;
}
void wgpuQuerySetDestroy(WGPUQuerySet value) {
    BICORE_CHECK(!value->destroyed);
    value->destroyed = true;
}
void wgpuBufferAddRef(WGPUBuffer value) {
    ++value->refs;
}
void wgpuBufferRelease(WGPUBuffer value) {
    if (!--value->refs) {
        ++resources.buffers_released;
        delete value;
    }
}
void wgpuBindGroupLayoutAddRef(WGPUBindGroupLayout value) {
    ++value->refs;
}
void wgpuBindGroupLayoutRelease(WGPUBindGroupLayout value) {
    if (!--value->refs)
        delete value;
}
void wgpuBindGroupAddRef(WGPUBindGroup value) {
    ++value->refs;
}
void wgpuBindGroupRelease(WGPUBindGroup value) {
    if (!--value->refs) {
        for (auto buffer : value->buffers)
            wgpuBufferRelease(buffer);
        --resources.groups_alive;
        delete value;
    }
}
void wgpuSamplerAddRef(WGPUSampler value) {
    ++value->refs;
}
void wgpuSamplerRelease(WGPUSampler value) {
    if (!--value->refs)
        delete value;
}
void wgpuTextureViewAddRef(WGPUTextureView value) {
    ++value->refs;
}
void wgpuTextureViewRelease(WGPUTextureView value) {
    if (!--value->refs)
        delete value;
}
WGPUBindGroupLayout wgpuComputePipelineGetBindGroupLayout(WGPUComputePipeline, uint32_t) {
    ++resources.layouts_created;
    return new WGPUBindGroupLayoutImpl;
}
WGPUBindGroup wgpuDeviceCreateBindGroup(WGPUDevice, const WGPUBindGroupDescriptor * descriptor) {
    ExternalAllocation external;
    ++resources.groups_created;
    if (resources.group_failure == GroupFailure::exception)
        throw std::runtime_error("Recorded bind-group allocation failure");
    if (resources.group_failure == GroupFailure::null_result)
        return nullptr;
    auto * result = new WGPUBindGroupImpl{1, resources.groups_created, {}};
    result->bindings.assign(descriptor->entries, descriptor->entries + descriptor->entryCount);
    ++resources.groups_alive;
    for (size_t i = 0; i < descriptor->entryCount; ++i) {
        if (auto buffer = descriptor->entries[i].buffer) {
            wgpuBufferAddRef(buffer);
            result->buffers.push_back(buffer);
        }
    }
    return result;
}
void wgpuInstanceAddRef(WGPUInstance value) {
    ++value->refs;
}
void wgpuInstanceRelease(WGPUInstance value) {
    if (!--value->refs)
        delete value;
}
void wgpuAdapterAddRef(WGPUAdapter value) {
    ++value->refs;
}
void wgpuAdapterRelease(WGPUAdapter value) {
    if (!--value->refs)
        delete value;
}
void wgpuDeviceAddRef(WGPUDevice value) {
    ++value->refs;
}
void wgpuDeviceRelease(WGPUDevice value) {
    if (!--value->refs) {
        BICORE_CHECK(value->scopes.empty());
        if (value->queue)
            wgpuQueueRelease(value->queue);
        delete value;
    }
}
WGPUBool wgpuDeviceHasFeature(WGPUDevice, WGPUFeatureName) {
    return true;
}
WGPUFuture wgpuAdapterRequestDevice(WGPUAdapter, const WGPUDeviceDescriptor *, WGPURequestDeviceCallbackInfo info) {
    ++device_requests;
    BICORE_CHECK(info.mode == WGPUCallbackMode_WaitAnyOnly);
    last_request_owner = info.userdata2;
    const auto status = request_error ? WGPURequestDeviceStatus_Error : WGPURequestDeviceStatus_Success;
    WGPUDevice device = request_error ? nullptr : new WGPUDeviceImpl;
    const auto id = next_future++;
    futures.emplace(id, Future{[=] {
                        const WGPUStringView message = {"request result", WGPU_STRLEN};
                        info.callback(status, device, message, info.userdata1, info.userdata2);
                    }});
    last_request_future = {id};
    if (request_throw) {
        request_throw = false;
        throw std::runtime_error("request threw after registration");
    }
    return {id};
}
void wgpuShaderModuleAddRef(WGPUShaderModule value) {
    ++value->refs;
}
void wgpuShaderModuleRelease(WGPUShaderModule value) {
    if (!--value->refs)
        delete value;
}
void wgpuPipelineLayoutAddRef(WGPUPipelineLayout value) {
    ++value->refs;
}
void wgpuPipelineLayoutRelease(WGPUPipelineLayout value) {
    if (!--value->refs)
        delete value;
}
void wgpuComputePipelineAddRef(WGPUComputePipeline value) {
    ++value->refs;
}
void wgpuComputePipelineRelease(WGPUComputePipeline value) {
    if (!--value->refs) {
        ++resources.pipelines_released;
        delete value;
    }
}
WGPUShaderModule wgpuDeviceCreateShaderModule(WGPUDevice, const WGPUShaderModuleDescriptor * descriptor) {
    const auto * source = reinterpret_cast<const WGPUShaderSourceWGSL *>(descriptor->nextInChain);
    return new WGPUShaderModuleImpl{1, text(source->code)};
}
WGPUComputePipeline wgpuDeviceCreateComputePipeline(WGPUDevice device,
                                                    const WGPUComputePipelineDescriptor * descriptor) {
    if (inject != WGPUErrorType_NoError) {
        BICORE_CHECK(!device->scopes.empty());
        device->scopes.back().second = inject;
        inject = WGPUErrorType_NoError;
    }
    return new WGPUComputePipelineImpl{1, ++created, descriptor->compute.module->code, text(descriptor->label)};
}
void wgpuDevicePushErrorScope(WGPUDevice device, WGPUErrorFilter filter) {
    ++pushes;
    device->scopes.push_back({filter, WGPUErrorType_NoError});
}
WGPUFuture wgpuDevicePopErrorScope(WGPUDevice device, WGPUPopErrorScopeCallbackInfo info) {
    ++pops;
    BICORE_CHECK(info.mode == WGPUCallbackMode_WaitAnyOnly);
    BICORE_CHECK(!device->scopes.empty());
    const auto error = device->scopes.back().second;
    device->scopes.pop_back();
    const auto id = next_future++;
    futures.emplace(id, Future{[=] {
                        const WGPUStringView message = {"recorded scope", WGPU_STRLEN};
                        info.callback(WGPUPopErrorScopeStatus_Success, error, message, info.userdata1, info.userdata2);
                    }});
    return {id};
}
WGPUWaitStatus wgpuInstanceWaitAny(WGPUInstance, size_t count, WGPUFutureWaitInfo * infos, std::uint64_t) {
    ++waits;
    BICORE_CHECK(count == 1);
    if (wait_throw) {
        wait_throw = false;
        throw std::runtime_error("wait threw");
    }
    if (wait_timeout) {
        wait_timeout = false;
        return WGPUWaitStatus_TimedOut;
    }
    if (wait_error) {
        wait_error = false;
        return WGPUWaitStatus_Error;
    }
    if (wait_without_callback) {
        wait_without_callback = false;
        return WGPUWaitStatus_Success;
    }
    if (on_wait_once) {
        auto hook = std::move(on_wait_once);
        on_wait_once = {};
        hook();
    }
    const auto found = futures.find(infos[0].future.id);
    BICORE_CHECK(found != futures.end());
    const auto item = found->second;
    futures.erase(found);
    infos[0].completed = true;
    item.complete();
    return WGPUWaitStatus_Success;
}
#ifndef BICORE_REAL_GGML
const ggml_type_traits * ggml_get_type_traits(ggml_type type) {
    static const ggml_type_traits q8 = [] {
        ggml_type_traits t{};
        t.type_name = "q8_0";
        return t;
    }();
    static const ggml_type_traits q4 = [] {
        ggml_type_traits t{};
        t.type_name = "q4_0";
        return t;
    }();
    static const ggml_type_traits q4k = [] {
        ggml_type_traits t{};
        t.type_name = "q4_K";
        return t;
    }();
    static const ggml_type_traits q6 = [] {
        ggml_type_traits t{};
        t.type_name = "q6_K";
        return t;
    }();
    static const ggml_type_traits iq3 = [] {
        ggml_type_traits t{};
        t.type_name = "iq3_s";
        return t;
    }();
    if (type == GGML_TYPE_IQ3_S)
        return &iq3;
    static const ggml_type_traits q5 = [] {
        ggml_type_traits t{};
        t.type_name = "q5_1";
        return t;
    }();
    if (type == GGML_TYPE_Q5_1)
        return &q5;
    static const ggml_type_traits iq4 = [] {
        ggml_type_traits t{};
        t.type_name = "iq4_nl";
        return t;
    }();
    if (type == GGML_TYPE_IQ4_NL)
        return &iq4;
    if (type == GGML_TYPE_Q6_K)
        return &q6;
    if (type == GGML_TYPE_Q8_0)
        return &q8;
    if (type == GGML_TYPE_Q4_0)
        return &q4;
    if (type == GGML_TYPE_Q4_K)
        return &q4k;
    throw std::runtime_error("Unexpected recording type");
}
bool ggml_is_quantized(ggml_type type) {
    return type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ4_NL || type == GGML_TYPE_Q5_1 || type == GGML_TYPE_Q4_0 ||
           type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q6_K || type == GGML_TYPE_Q8_0;
}
void ggml_abort(const char *, int, const char *, ...) {
    throw std::runtime_error("Recorded fatal GGML error");
}
#endif
}
