#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#define GGML_ASSERT(x)                                                                                                 \
    do {                                                                                                               \
        if (!(x))                                                                                                      \
            throw std::runtime_error("assert: " #x);                                                                   \
    } while (0)
#define GGML_ABORT(...) throw std::runtime_error("runtime abort")
#define GGML_LOG_ERROR(...) ((void)0)
#define GGML_LOG_INFO(...) ((void)0)
#define GGML_UNUSED(x) ((void)(x))
#define WEBGPU_LOG_DEBUG(x) ((void)0)
#define WEBGPU_CPU_PROFILE_TOTAL_START(x) ((void)0)
#define WEBGPU_CPU_PROFILE_TOTAL_END(x, c) ((void)0)
#define GGML_WEBGPU_NAME "WebGPU"
#define GGML_BACKEND_API_VERSION 1
#define WEBGPU_PARAMS_BUF_SIZE_BYTES 128
namespace fixture {
inline std::atomic<int> instances{0}, adapters{0}, devices{0}, queues{0}, memsets{0}, buffers{0}, callbacks{0};
inline bool adapter_wait_error = false, device_wait_error = false, missing_adapter_callback = false,
            missing_device_callback = false;
inline std::vector<std::function<void()>> pending;
inline bool fail_instance = false, fail_adapter = false, fail_device = false, f16 = true, subgroups = true,
            implicit = true, async_callback = false;
} // namespace fixture
namespace wgpu {
using StringView = std::string;
enum class FeatureName {
    ShaderF16,
    Subgroups,
    ImplicitDeviceSynchronization,
    ChromiumExperimentalSubgroupMatrix,
    TimestampQuery
};
enum class WGSLLanguageFeatureName { Packed4x8IntegerDotProduct };
enum class InstanceFeatureName { TimedWaitAny };
enum class CallbackMode { AllowSpontaneous };
enum class WaitStatus { Success, Error };
enum class RequestAdapterStatus { Success, Error };
enum class RequestDeviceStatus { Success, Error };
enum class DeviceLostReason { Destroyed, Unknown };
enum class ErrorType { Validation };
enum class SubgroupMatrixComponentType { F16 };
enum class BufferUsage { CopyDst = 1, Uniform = 2 };
BufferUsage operator|(BufferUsage a, BufferUsage b) {
    return BufferUsage(int(a) | int(b));
}
struct SubgroupMatrixConfig {
    SubgroupMatrixComponentType componentType, resultComponentType;
    uint32_t M, N, K;
};
struct AdapterPropertiesSubgroupMatrixConfigs {
    size_t configCount = 0;
    SubgroupMatrixConfig * configs = nullptr;
};
struct DawnTogglesDescriptor {
    const char * const * enabledToggles = nullptr;
    size_t enabledToggleCount = 0;
    const char * const * disabledToggles = nullptr;
    size_t disabledToggleCount = 0;
};
struct Limits {
    uint32_t maxComputeInvocationsPerWorkgroup = 256, maxComputeWorkgroupsPerDimension = 65535;
    uint64_t maxStorageBufferBindingSize = 128 * 1024 * 1024;
};
struct AdapterInfo {
    void * nextInChain = nullptr;
    StringView vendor = "test", architecture = "fixture", device = "owned", description = "API double";
    uint32_t vendorID = 1, deviceID = 2, subgroupMinSize = 4, subgroupMaxSize = 32;
};
struct RequestAdapterOptions {
    void * nextInChain = nullptr;
};
struct InstanceDescriptor {
    const InstanceFeatureName * requiredFeatures = nullptr;
    size_t requiredFeatureCount = 0;
    void * nextInChain = nullptr;
};
struct Device;
struct DeviceDescriptor {
    Limits * requiredLimits = nullptr;
    const FeatureName * requiredFeatures = nullptr;
    size_t requiredFeatureCount = 0;
    void * nextInChain = nullptr;
    std::function<void(const Device &, DeviceLostReason, StringView)> lost;
    template <class F> void SetDeviceLostCallback(CallbackMode, F f) {
        lost = f;
    }
    template <class F> void SetUncapturedErrorCallback(F) {}
};
struct Root {
    int id;
};
struct AdapterOwner {
    int id;
    std::shared_ptr<Root> root;
};
struct DeviceOwner {
    int id;
    std::shared_ptr<AdapterOwner> adapter;
    std::atomic<bool> lost{false};
    std::function<void(const Device &, DeviceLostReason, StringView)> on_lost;
};
struct Queue {
    std::shared_ptr<DeviceOwner> p;
    int id() const {
        return p ? p->id : 0;
    }
};
struct Device {
    std::shared_ptr<DeviceOwner> p;
    bool operator!=(std::nullptr_t) const {
        return bool(p);
    }
    Queue GetQueue() const {
        ++fixture::queues;
        return {p};
    }
};
struct Buffer {
    std::shared_ptr<DeviceOwner> owner;
};
using Future = std::function<void()>;
struct Adapter {
    std::shared_ptr<AdapterOwner> p;
    bool operator!=(std::nullptr_t) const {
        return bool(p);
    }
    bool HasFeature(FeatureName f) const {
        return p && (f == FeatureName::ShaderF16                       ? fixture::f16
                     : f == FeatureName::Subgroups                     ? fixture::subgroups
                     : f == FeatureName::ImplicitDeviceSynchronization ? fixture::implicit
                                                                       : false);
    }
    void GetLimits(Limits * out) const {
        *out = Limits{};
    }
    void GetInfo(AdapterInfo * out) const {
        out->deviceID = p ? p->id : 0;
    }
    template <class F> Future RequestDevice(const DeviceDescriptor * desc, CallbackMode, F f) const {
        ++fixture::devices;
        bool failed = fixture::fail_device;
        auto a = p;
        auto cb = desc->lost;
        return [failed, a, cb, f] {
            ++fixture::callbacks;
            Device d;
            if (!failed) {
                d.p = std::make_shared<DeviceOwner>();
                d.p->id = fixture::devices;
                d.p->adapter = a;
                d.p->on_lost = cb;
            }
            f(failed ? RequestDeviceStatus::Error : RequestDeviceStatus::Success, d, failed ? "device failure" : "");
        };
    }
};
struct Instance {
    std::shared_ptr<Root> p;
    bool operator!=(std::nullptr_t) const {
        return bool(p);
    }
    bool HasWGSLLanguageFeature(WGSLLanguageFeatureName) const {
        return true;
    }
    template <class F> Future RequestAdapter(const RequestAdapterOptions *, CallbackMode, F f) const {
        int id = ++fixture::adapters;
        bool failed = fixture::fail_adapter;
        auto r = p;
        return [failed, id, r, f] {
            ++fixture::callbacks;
            Adapter a;
            if (!failed)
                a.p = std::make_shared<AdapterOwner>(AdapterOwner{id, r});
            f(failed ? RequestAdapterStatus::Error : RequestAdapterStatus::Success, a, failed ? "adapter failure" : "");
        };
    }
    WaitStatus WaitAny(Future f, uint64_t timeout) const {
        if (timeout != UINT64_MAX)
            throw std::runtime_error("changed wait policy");
        bool device = fixture::devices > 0;
        if (device ? fixture::device_wait_error : fixture::adapter_wait_error) {
            fixture::pending.push_back(f);
            return WaitStatus::Error;
        }
        if (device ? fixture::missing_device_callback : fixture::missing_adapter_callback)
            return WaitStatus::Success;
        if (fixture::async_callback) {
            std::thread t([&] {
                std::this_thread::sleep_for(std::chrono::milliseconds(3));
                f();
            });
            t.join();
        } else
            f();
        return WaitStatus::Success;
    }
};
Instance CreateInstance(const InstanceDescriptor *) {
    int id = ++fixture::instances;
    return fixture::fail_instance ? Instance{} : Instance{std::make_shared<Root>(Root{id})};
}
} // namespace wgpu
struct webgpu_global_context_struct {
    wgpu::Instance instance;
    wgpu::Adapter adapter;
    wgpu::Device device;
    wgpu::Queue queue;
    wgpu::Buffer memset_params_buf;
    uint32_t command_submit_batch_size = 64, max_inflight_batches = UINT32_MAX;
    struct {
        wgpu::Limits limits;
        bool supports_subgroups = false, supports_dot_product = false, supports_subgroup_matrix = false;
        uint32_t sg_mat_m = 0, sg_mat_n = 0, sg_mat_k = 0, min_subgroup_size = 0, max_subgroup_size = 0;
    } capabilities;
    std::string vendor;
};
using webgpu_global_context = std::shared_ptr<webgpu_global_context_struct>;
struct ggml_backend_webgpu_reg_context {
    webgpu_global_context webgpu_global_ctx;
    size_t device_count = 0;
    const char * name = nullptr;
};
struct ggml_backend_webgpu_device_context {
    webgpu_global_context webgpu_global_ctx;
    std::string device_name, device_desc;
};
struct ggml_backend_reg {
    int api_version = 0, iface = 0;
    void * context = nullptr;
};
using ggml_backend_reg_t = ggml_backend_reg *;
struct ggml_backend_device {
    int iface = 0;
    ggml_backend_reg_t reg = nullptr;
    void * context = nullptr;
};
using ggml_backend_dev_t = ggml_backend_device *;
int ggml_backend_webgpu_device_i = 0, ggml_backend_webgpu_reg_i = 0;
uint32_t ggml_backend_webgpu_get_command_submit_batch_size() {
    return 64;
}
uint32_t ggml_backend_webgpu_get_max_inflight_batches() {
    return UINT32_MAX;
}
void ggml_webgpu_init_memset_pipeline(webgpu_global_context & ctx) {
    GGML_ASSERT(ctx->device != nullptr);
    ++fixture::memsets;
}
void ggml_webgpu_create_buffer(wgpu::Device & device, wgpu::Buffer & out, size_t, wgpu::BufferUsage, const char *) {
    out.owner = device.p;
    ++fixture::buffers;
}
