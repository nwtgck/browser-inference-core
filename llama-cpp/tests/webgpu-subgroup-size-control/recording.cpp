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
    if (type == GGML_TYPE_Q8_0)
        return &q8;
    if (type == GGML_TYPE_Q4_0)
        return &q4;
    if (type == GGML_TYPE_Q4_K)
        return &q4k;
    throw std::runtime_error("Unexpected recording type");
}
void ggml_abort(const char *, int, const char *, ...) {
    throw std::runtime_error("Recorded fatal GGML error");
}
}

// Standard/SDK/ggml/preprocessor declarations are loaded globally first. The
// two complete shader libraries then occupy separate test-only namespaces.
#include OPTIONAL_HEADER
#if RECORD_PROFILE
#include "ggml-webgpu-q8-diagnostic-sha256.hpp"
#include DIAGNOSTIC_HEADER
#endif
namespace baseline {
#include BASE_HEADER
}
#undef GGML_WEBGPU_SHADER_LIB_HPP
namespace candidate {
#include CANDIDATE_HEADER
}

struct Fixture {
    ggml_tensor a{}, x{}, dst{}, ids{};
    Fixture() {
        for (int axis = 0; axis < 4; axis++)
            a.ne[axis] = x.ne[axis] = dst.ne[axis] = 1;
        a.type = GGML_TYPE_Q8_0;
        x.type = dst.type = GGML_TYPE_F32;
        a.ne[0] = x.ne[0] = 6144;
        a.ne[1] = dst.ne[0] = 5120;
    }
    template <class Context> Context context(unsigned minimum, unsigned maximum, bool feature) {
        Context c{};
        c.src0 = &a;
        c.src1 = &x;
        c.dst = &dst;
        c.max_wg_size = 256;
        c.supports_subgroups = true;
        c.min_subgroup_size = minimum;
        c.max_subgroup_size = maximum;
        if constexpr (requires { c.subgroup_size_control_enabled; })
            c.subgroup_size_control_enabled = feature;
        return c;
    }
};
static unsigned checks = 0;
static void require(bool okay, const char * message) {
    ++checks;
    if (!okay)
        throw std::runtime_error(message);
}
static const candidate::ggml_webgpu_mul_mat_vec_shader_decisions &
decisions(const candidate::webgpu_pipeline & pipeline) {
    return *static_cast<const candidate::ggml_webgpu_mul_mat_vec_shader_decisions *>(pipeline.context.get());
}
#include HOST_FRAGMENTS

static Fixture fixture(ggml_type type) {
    Fixture f;
    f.a.type = type;
    f.a.ne[0] = f.x.ne[0] = 2048;
    f.a.ne[1] = f.dst.ne[0] = 4097;
    return f;
}

static std::string without_attribute(std::string code) {
    for (const std::string line : {"enable subgroup_size_control;\n", "@subgroup_size(16)\n"}) {
        const auto at = code.find(line);
        require(at != std::string::npos, "Requested enable/attribute absent");
        code.erase(at, line.size());
    }
    return code;
}

static void write(const std::filesystem::path & path, const std::string & code) {
    std::ofstream output(path);
    output << code;
    require(bool(output), "Emission write failed");
}

int main(int argc, char ** argv) {
    try {
        require(argc == 2, "Missing emission directory");
        const std::filesystem::path output = argv[1];
        std::filesystem::create_directories(output);
        auto instance = wgpu::Instance::Acquire(new WGPUInstanceImpl);
        auto device = wgpu::Device::Acquire(new WGPUDeviceImpl);
        auto state = std::make_shared<ggml_webgpu_optional::DeviceGeneration>();
        baseline::ggml_webgpu_shader_lib original(device);
        candidate::ggml_webgpu_shader_lib patched(device, instance, state);
        std::unordered_map<candidate::ggml_webgpu_mul_mat_vec_pipeline_key, int,
                           candidate::ggml_webgpu_mul_mat_vec_pipeline_key_hash>
            keys;
        for (const auto type : {GGML_TYPE_Q4_0, GGML_TYPE_Q4_K}) {
            auto f = fixture(type);
            const std::string name = type == GGML_TYPE_Q4_0 ? "q4_0" : "q4_K";
            for (unsigned width : {16u, 32u}) {
                const auto before = pushes;
                auto old_context = f.context<baseline::ggml_webgpu_shader_lib_context>(width, width, false);
                auto new_context = f.context<candidate::ggml_webgpu_shader_lib_context>(width, width, true);
                const auto a = original.get_mul_mat_vec_pipeline(old_context, false, false);
                const auto b = patched.get_mul_mat_vec_pipeline(new_context, false, false);
                require(a.pipeline.Get()->code == b.pipeline.Get()->code && a.name == b.name, "Fixed Q4 AUTO changed");
                require(pushes == before, "Fixed Q4 gained optional scopes");
            }
            auto context = f.context<candidate::ggml_webgpu_shader_lib_context>(8, 32, true);
            const auto requested = patched.get_mul_mat_vec_pipeline(context, false, false);
            const auto key = candidate::ggml_webgpu_shader_lib::make_mul_mat_vec_key(context, false, false);
            keys[key] = type;
            require(decisions(requested).logical_subgroup_size == 16 &&
                        decisions(requested).requested_subgroup_size == 16 && decisions(requested).wg_size == 128 &&
                        decisions(requested).outputs_per_wg == 32,
                    "Q4 requested decisions");
            const auto fixed = original.get_mul_mat_vec_pipeline(
                f.context<baseline::ggml_webgpu_shader_lib_context>(16, 16, false), false, false);
            require(without_attribute(requested.pipeline.Get()->code) == fixed.pipeline.Get()->code,
                    "Q4 requested arithmetic changed");
            const auto creates_before = created, pushes_before = pushes, waits_before = waits;
            require(patched.get_mul_mat_vec_pipeline(context, false, false).pipeline.Get() == requested.pipeline.Get(),
                    "Q4 warm identity");
            require(created == creates_before && pushes == pushes_before && waits == waits_before,
                    "Q4 warm compile/scope/wait");
            require(context.min_subgroup_size == 8 && context.max_subgroup_size == 32, "Advertised range changed");
            write(output / (name + "-requested.wgsl"), requested.pipeline.Get()->code);
            auto absent = context;
            absent.subgroup_size_control_enabled = false;
            const auto canonical = patched.get_mul_mat_vec_pipeline(absent, false, false);
            const auto old = original.get_mul_mat_vec_pipeline(
                f.context<baseline::ggml_webgpu_shader_lib_context>(8, 32, false), false, false);
            require(canonical.pipeline.Get()->code == old.pipeline.Get()->code && canonical.name == old.name,
                    "Q4 feature absence changed canonical");
            candidate::ggml_webgpu_shader_lib rejected(device, instance, state);
            inject = WGPUErrorType_Validation;
            const auto fallback = rejected.get_mul_mat_vec_pipeline(context, false, false);
            const auto & d = decisions(fallback);
            require(fallback.pipeline.Get()->code == old.pipeline.Get()->code && fallback.name == old.name,
                    "Q4 fallback dtype/source");
            require(d.requested_subgroup_size == 0 && d.logical_subgroup_size == 0 && d.outputs_per_wg == 4 &&
                        d.wg_size == (type == GGML_TYPE_Q4_0 ? 256u : 128u),
                    "Q4 fallback whole geometry");
            require(production_dispatch(requested, &f.dst, 65535)[0] == 129 &&
                        production_dispatch(fallback, &f.dst, 65535)[0] == 1025,
                    "Q4 actual caller grid handoff");
            const auto rejected_created = created, rejected_pushes = pushes, rejected_waits = waits;
            require(rejected.get_mul_mat_vec_pipeline(context, false, false).pipeline.Get() == fallback.pipeline.Get(),
                    "Q4 failed cache identity");
            require(created == rejected_created && pushes == rejected_pushes && waits == rejected_waits,
                    "Q4 failed key retried");
            write(output / (name + "-canonical.wgsl"), fallback.pipeline.Get()->code);
            if (type == GGML_TYPE_Q4_K) {
                f.a.ne[0] = f.x.ne[0] = 1024;
                require(candidate::ggml_webgpu_shader_lib::make_mul_mat_vec_key(context, false, false) == key,
                        "Q4 requested key unexpectedly depends on K");
                const auto expected = original.get_mul_mat_vec_pipeline(
                    f.context<baseline::ggml_webgpu_shader_lib_context>(8, 32, false), false, false);
                const auto before_create = created, before_push = pushes, before_wait = waits;
                const auto smaller = rejected.get_mul_mat_vec_pipeline(context, false, false);
                require(decisions(smaller).wg_size == 64 && decisions(smaller).outputs_per_wg == 4 &&
                            smaller.pipeline.Get()->code == expected.pipeline.Get()->code,
                        "Rejected Q4 key reused stale canonical geometry");
                require(created == before_create + 1 && pushes == before_push && waits == before_wait,
                        "Changed canonical context retried requested pipeline");
                f.a.ne[0] = f.x.ne[0] = 2048;
                require(rejected.get_mul_mat_vec_pipeline(context, false, false).pipeline.Get() ==
                            fallback.pipeline.Get(),
                        "Q4 canonical geometry did not restore from its own key");
                require(created == before_create + 1 && pushes == before_push && waits == before_wait,
                        "Q4 canonical revisit added compile/scope/wait");
            }
            for (const auto range : {std::pair{0u, 32u}, {16u, 16u}, {32u, 64u}, {4u, 8u}, {32u, 8u}}) {
                auto c = f.context<candidate::ggml_webgpu_shader_lib_context>(range.first, range.second, true);
                require(candidate::ggml_webgpu_q4_requested_subgroup_size(c, false, false) == 0,
                        "Q4 requested range widened");
            }
            for (unsigned which = 0; which < 10; ++which) {
                auto bad = fixture(type);
                auto c = bad.context<candidate::ggml_webgpu_shader_lib_context>(8, 32, true);
                bool overlap = false, mmvq = false;
                switch (which) {
                case 0:
                    bad.a.ne[0] = bad.x.ne[0] = 1023;
                    break;
                case 1:
                    bad.a.ne[0] = bad.x.ne[0] = 4097;
                    break;
                case 2:
                    bad.a.ne[1] = bad.dst.ne[0] = 4095;
                    break;
                case 3:
                    bad.x.ne[1] = bad.dst.ne[1] = 2;
                    break;
                case 4:
                    bad.x.type = GGML_TYPE_F16;
                    break;
                case 5:
                    bad.dst.type = GGML_TYPE_F16;
                    break;
                case 6:
                    c.src2 = &bad.ids;
                    break;
                case 7:
                    c.max_wg_size = 64;
                    break;
                case 8:
                    overlap = true;
                    break;
                case 9:
                    mmvq = true;
                    break;
                }
                require(candidate::ggml_webgpu_q4_requested_subgroup_size(c, mmvq, overlap) == 0,
                        "Q4 ordinary admission widened");
            }
        }
        require(keys.size() == 2, "Q4 dtype/requested cache collision");
        auto q8 = Fixture{};
        const auto fixed_old = original.get_mul_mat_vec_pipeline(
            q8.context<baseline::ggml_webgpu_shader_lib_context>(16, 16, false), false, false);
        const auto fixed_new = patched.get_mul_mat_vec_pipeline(
            q8.context<candidate::ggml_webgpu_shader_lib_context>(16, 16, true), false, false);
        require(fixed_old.pipeline.Get()->code == fixed_new.pipeline.Get()->code && fixed_old.name == fixed_new.name,
                "Composition changed fixed Q8");
        require(futures.empty(), "Pending mock callbacks");
        std::cout << "{\"passed\":true,\"checks\":" << checks << ",\"created\":" << created << ",\"pushes\":" << pushes
                  << ",\"pops\":" << pops << ",\"waits\":" << waits << ",\"gpu\":false}\n";
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
