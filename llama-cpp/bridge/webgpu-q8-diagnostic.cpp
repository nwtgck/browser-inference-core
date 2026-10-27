#include "webgpu-q8-diagnostic.h"
#include <cstddef>
#include <cstdint>
#include <cstring>
#ifdef __EMSCRIPTEN__
#include <emscripten/heap.h>
#endif
#ifndef LCB_WEBGPU_Q8_DIAGNOSTIC
#error "The Q8 diagnostic bridge is restricted to the GPU-profile build"
#endif

static bool q8_diagnostic_span(uint64_t pointer, uint64_t bytes, uint64_t alignment) {
    if (!pointer || pointer > UINTPTR_MAX || pointer % alignment || bytes > UINTPTR_MAX - pointer) { return false; }
#ifdef __EMSCRIPTEN__
    const uint64_t heap = emscripten_get_heap_size();
    if (pointer > heap || bytes > heap - pointer) { return false; }
#endif
    return true;
}

static ggml_backend_webgpu_q8_diagnostic_v1_t q8_diagnostic_provider(uint64_t backend) {
    if (!q8_diagnostic_span(backend, sizeof(void *), alignof(void *))) { return nullptr; }
    auto * native = reinterpret_cast<ggml_backend_t>(uintptr_t(backend));
    auto * device = ggml_backend_get_device(native);
    auto * registry = device ? ggml_backend_dev_backend_reg(device) : nullptr;
    if (!registry || ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_GPU) { return nullptr; }
    const char * name = ggml_backend_reg_name(registry);
    if (!name || std::strcmp(name, "WebGPU") != 0) { return nullptr; }
    return reinterpret_cast<ggml_backend_webgpu_q8_diagnostic_v1_t>(
        ggml_backend_reg_get_proc_address(registry, GGML_BACKEND_WEBGPU_Q8_DIAGNOSTIC_V1));
}

extern "C" uint32_t lcb_webgpu_q8_diagnostic_version(uint64_t backend) {
    return q8_diagnostic_provider(backend) ? 1u : 0u;
}

extern "C" int32_t lcb_webgpu_q8_diagnostic_run_v1(uint64_t backend, uint64_t dst,
                                uint64_t request, uint64_t report, uint32_t report_capacity) {
    using Request = ggml_webgpu_q8_diagnostic_request_v1;
    using Report = ggml_webgpu_q8_diagnostic_report_v1;
    if (report_capacity < sizeof(Report) || !q8_diagnostic_span(report, report_capacity, alignof(Report)) ||
        !q8_diagnostic_span(request, sizeof(Request), alignof(Request)) ||
        !q8_diagnostic_span(dst, sizeof(ggml_tensor), alignof(ggml_tensor))) { return GGML_WEBGPU_Q8_DIAGNOSTIC_BAD_REQUEST_V1; }
    // Copy before calling a provider that may suspend. No heap view is retained.
    const Request req = *reinterpret_cast<const Request *>(uintptr_t(request));
    auto * out = reinterpret_cast<Report *>(uintptr_t(report));
    const auto provider = q8_diagnostic_provider(backend);
    if (!provider) {
        Report result{}; result.abi_version = 1; result.struct_size = sizeof(result);
        result.status = GGML_WEBGPU_Q8_DIAGNOSTIC_UNAVAILABLE_V1;
        result.requested_variant = req.variant; result.phase = req.phase;
        *out = result;
        return GGML_WEBGPU_Q8_DIAGNOSTIC_UNAVAILABLE_V1;
    }
    return provider(reinterpret_cast<ggml_backend_t>(uintptr_t(backend)),
                    reinterpret_cast<ggml_tensor *>(uintptr_t(dst)), &req, out, report_capacity);
}
