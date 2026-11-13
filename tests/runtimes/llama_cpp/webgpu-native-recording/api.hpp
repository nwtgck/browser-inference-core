#pragma once

// Actual complete shader libraries and pinned SDK C++ bindings. Only the C GPU
// API boundary is replaced; no adapter, native runtime, browser or GPU is used.
#include <webgpu/webgpu_cpp.h>
#include "ggml-impl.h"
#include "pre_wgsl.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
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
#define CEIL_DIV(a, b) (((a) + (b)-1) / (b))

struct WGPUInstanceImpl {
    unsigned refs = 1;
};
struct WGPUAdapterImpl {
    unsigned refs = 1;
};
struct WGPUDeviceImpl {
    unsigned refs = 1;
    std::vector<std::pair<WGPUErrorFilter, WGPUErrorType>> scopes;
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
struct Future {
    std::function<void()> complete;
};
static std::map<std::uint64_t, Future> futures;
static unsigned created = 0, pushes = 0, pops = 0, waits = 0;
static std::uint64_t next_future = 1;
static WGPUErrorType inject = WGPUErrorType_NoError;
static std::function<void()> on_wait_once;
static bool request_error = false, request_throw = false, wait_throw = false, wait_timeout = false,
            wait_without_callback = false;
static unsigned device_requests = 0;
static void * last_request_owner = nullptr;
static WGPUFuture last_request_future{};
static std::string text(WGPUStringView view) {
    return view.data == nullptr         ? ""
           : view.length == WGPU_STRLEN ? std::string(view.data)
                                        : std::string(view.data, view.length);
}
extern "C" {
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
        assert(value->scopes.empty());
        delete value;
    }
}
WGPUBool wgpuDeviceHasFeature(WGPUDevice, WGPUFeatureName) {
    return true;
}
WGPUFuture wgpuAdapterRequestDevice(WGPUAdapter, const WGPUDeviceDescriptor *, WGPURequestDeviceCallbackInfo info) {
    ++device_requests;
    assert(info.mode == WGPUCallbackMode_WaitAnyOnly);
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
    if (!--value->refs)
        delete value;
}
WGPUShaderModule wgpuDeviceCreateShaderModule(WGPUDevice, const WGPUShaderModuleDescriptor * descriptor) {
    const auto * source = reinterpret_cast<const WGPUShaderSourceWGSL *>(descriptor->nextInChain);
    return new WGPUShaderModuleImpl{1, text(source->code)};
}
WGPUComputePipeline wgpuDeviceCreateComputePipeline(WGPUDevice device,
                                                    const WGPUComputePipelineDescriptor * descriptor) {
    if (inject != WGPUErrorType_NoError) {
        assert(!device->scopes.empty());
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
    assert(info.mode == WGPUCallbackMode_WaitAnyOnly);
    assert(!device->scopes.empty());
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
    assert(count == 1);
    if (wait_throw) {
        wait_throw = false;
        throw std::runtime_error("wait threw");
    }
    if (wait_timeout) {
        wait_timeout = false;
        return WGPUWaitStatus_TimedOut;
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
    assert(found != futures.end());
    const auto item = found->second;
    futures.erase(found);
    infos[0].completed = true;
    item.complete();
    return WGPUWaitStatus_Success;
}
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
}
