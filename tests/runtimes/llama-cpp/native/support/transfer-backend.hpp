#pragma once

namespace transfer_cases {
enum class BackendKind { host, webgpu };

struct HostAllocation {
    std::vector<uint8_t> storage;
    void * base;
    explicit HostAllocation(size_t size) : storage(size + 255) {
        base = reinterpret_cast<void *>((uintptr_t(storage.data()) + 255) & ~uintptr_t(255));
    }
};

// Ordinary ggml backend/device/registry interfaces are the external test
// boundary. Scheduler, graph allocator, copy helpers and provider remain real.
struct Backend {
    BackendKind kind;
    webgpu_context context;
    ggml_backend_webgpu_context bridge;
    ggml_backend_webgpu_device_context device_bridge;
    ggml_backend backend{};
    ggml_backend_device device{};
    ggml_backend_reg registry{};
    ggml_backend_buffer_type type{};
    unsigned lookups = 0, snapshots = 0, synchronizations = 0, allocations = 0;
    bool offer_snapshot = true;
    std::vector<std::string> lookup_names;
    std::function<void()> complete;
    std::function<ggml_status(ggml_cgraph *)> host_compute;
    std::function<void(ggml_cgraph *)> before_graph;

    static Backend & owner(ggml_backend_dev_t device) {
        BICORE_CHECK(device && device->reg && device->reg->context);
        return *static_cast<Backend *>(device->reg->context);
    }
    static Backend & owner(ggml_backend_t backend) {
        return owner(backend->device);
    }
    explicit Backend(BackendKind selected) : kind(selected) {
        registry.context = this;
        registry.iface.get_proc_address = [](ggml_backend_reg_t value, const char * name) -> void * {
            auto & self = *static_cast<Backend *>(value->context);
            ++self.lookups;
            self.lookup_names.emplace_back(name ? name : "");
            if (self.kind == BackendKind::webgpu && self.offer_snapshot) {
                const auto actual = ggml_backend_webgpu_reg_get_proc_address(value, name);
                if (actual == reinterpret_cast<void *>(ggml_backend_webgpu_copy_host_input_snapshot_v1)) {
                    ggml_backend_copy_host_input_snapshot_v1_t observed =
                        [](ggml_backend_t source, ggml_backend_t destination, const ggml_tensor * input,
                           ggml_tensor * output) {
                            ++owner(destination).snapshots;
                            return ggml_backend_webgpu_copy_host_input_snapshot_v1(source, destination, input, output);
                        };
                    return reinterpret_cast<void *>(observed);
                }
                return actual;
            }
            return nullptr;
        };
        device.reg = &registry;
        device.iface.get_type = kind == BackendKind::webgpu
                                    ? +[](ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_GPU; }
                                    : +[](ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_CPU; };
        device.iface.get_name = kind == BackendKind::webgpu ? +[](ggml_backend_dev_t) { return "recorded-webgpu"; }
                                                            : +[](ggml_backend_dev_t) { return "controlled-host"; };
        device.iface.get_description = device.iface.get_name;
        device.iface.get_buffer_type = [](ggml_backend_dev_t value) { return &owner(value).type; };
        device.iface.supports_buft = [](ggml_backend_dev_t value, ggml_backend_buffer_type_t requested) {
            return requested->device == value;
        };
        device.iface.supports_op = [](ggml_backend_dev_t, const ggml_tensor *) { return true; };
        backend.device = &device;
        backend.iface.graph_compute = [](ggml_backend_t value, ggml_cgraph * graph) {
            auto & self = owner(value);
            if (self.before_graph)
                self.before_graph(graph);
            if (self.kind == BackendKind::webgpu)
                return ggml_backend_webgpu_graph_compute(value, graph);
            BICORE_CHECK(bool(self.host_compute));
            return self.host_compute(graph);
        };
        type.device = &device;
        type.context = this;
        type.iface.get_name = [](ggml_backend_buffer_type_t value) {
            return static_cast<Backend *>(value->context)->kind == BackendKind::webgpu ? "recorded-device-buffer"
                                                                                       : "controlled-host-buffer";
        };
        type.iface.get_alignment = [](ggml_backend_buffer_type_t) -> size_t { return 256; };
        type.iface.get_max_size = [](ggml_backend_buffer_type_t) -> size_t { return 1024 * 1024; };
        type.iface.is_host = [](ggml_backend_buffer_type_t value) {
            return static_cast<Backend *>(value->context)->kind == BackendKind::host;
        };
        type.iface.alloc_buffer = [](ggml_backend_buffer_type_t value, size_t size) {
            auto & self = *static_cast<Backend *>(value->context);
            ++self.allocations;
            if (self.kind == BackendKind::webgpu) {
                wgpu::BufferDescriptor descriptor{};
                descriptor.size = std::max(size_t(4), (size + 3) & ~size_t(3));
                descriptor.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst;
                auto * allocation =
                    new ggml_backend_webgpu_buffer_context(self.context->global_ctx->device.CreateBuffer(&descriptor),
                                                           "scheduler allocation", self.context->global_ctx);
                return ggml_backend_buffer_init(value, ggml_backend_webgpu_buffer_interface, allocation, size);
            }
            ggml_backend_buffer_i api{};
            api.free_buffer = [](ggml_backend_buffer_t buffer) {
                delete static_cast<HostAllocation *>(buffer->context);
            };
            api.get_base = [](ggml_backend_buffer_t buffer) {
                return static_cast<HostAllocation *>(buffer->context)->base;
            };
            api.set_tensor = [](ggml_backend_buffer_t, ggml_tensor * tensor, const void * data, size_t offset,
                                size_t size) {
                std::memcpy(static_cast<uint8_t *>(tensor->data) + offset, data, size);
            };
            api.get_tensor = [](ggml_backend_buffer_t, const ggml_tensor * tensor, void * data, size_t offset,
                                size_t size) {
                std::memcpy(data, static_cast<const uint8_t *>(tensor->data) + offset, size);
            };
            api.memset_tensor = [](ggml_backend_buffer_t, ggml_tensor * tensor, uint8_t byte, size_t offset,
                                   size_t size) {
                std::memset(static_cast<uint8_t *>(tensor->data) + offset, byte, size);
            };
            api.clear = [](ggml_backend_buffer_t buffer, uint8_t byte) {
                std::memset(static_cast<HostAllocation *>(buffer->context)->base, byte, buffer->size);
            };
            return ggml_backend_buffer_init(value, api, new HostAllocation(size), size);
        };
        if (kind == BackendKind::webgpu) {
            context = recording_context();
            auto & global = *context->global_ctx;
            global.capabilities.limits.minStorageBufferOffsetAlignment = 256;
            global.capabilities.limits.maxStorageBufferBindingSize = 1024 * 1024;
            global.capabilities.limits.maxBufferSize = 1024 * 1024;
            global.capabilities.limits.maxComputeInvocationsPerWorkgroup = 256;
            global.capabilities.limits.maxComputeWorkgroupSizeX = 256;
            global.capabilities.limits.maxComputeWorkgroupStorageSize = 32768;
            global.capabilities.limits.maxUniformBuffersPerShaderStage = 12;
            global.capabilities.limits.maxUniformBufferBindingSize = 65536;
            global.capabilities.limits.maxStorageBuffersPerShaderStage = 8;
            global.capabilities.limits.maxBindingsPerBindGroup = 32;
            global.capabilities.limits.maxComputeWorkgroupsPerDimension = 65535;
            context->param_arena.cleanup();
            context->param_arena.init(global.device, 128, 74, 256);
#ifdef GGML_WEBGPU_GPU_PROFILE
            context->batch_compute_passes = false;
            wgpu::BufferDescriptor timestamps{};
            timestamps.size = WEBGPU_MAX_PROFILE_QUERY_COUNT * sizeof(uint64_t);
            timestamps.usage = wgpu::BufferUsage::QueryResolve | wgpu::BufferUsage::CopySrc;
            context->profile_timestamp_dev_buf = global.device.CreateBuffer(&timestamps);
            timestamps.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
            context->profile_timestamp_host_buf = global.device.CreateBuffer(&timestamps);
#endif
            bridge.webgpu_ctx = context;
            device_bridge.webgpu_global_ctx = context->global_ctx;
            device.context = &device_bridge;
            device.iface.supports_op = ggml_backend_webgpu_device_supports_op;
            backend.context = &bridge;
            backend.iface.get_name = ggml_backend_webgpu_name;
            backend.iface.cpy_tensor_async = ggml_backend_webgpu_cpy_tensor_async;
            backend.iface.get_tensor_async = ggml_backend_webgpu_get_tensor_async;
            backend.iface.synchronize = ggml_backend_webgpu_synchronize;
            backend.iface.event_record = ggml_backend_webgpu_event_record;
            backend.iface.event_wait = ggml_backend_webgpu_event_wait;
            device.iface.event_new = ggml_backend_webgpu_device_event_new;
            device.iface.event_free = ggml_backend_webgpu_device_event_free;
            device.iface.event_synchronize = ggml_backend_webgpu_device_event_synchronize;
            type.iface.get_alloc_size = ggml_backend_webgpu_buffer_type_get_alloc_size;
        } else {
            backend.context = this;
            backend.iface.get_name = [](ggml_backend_t) { return "controlled-host"; };
            backend.iface.synchronize = [](ggml_backend_t value) {
                auto & self = owner(value);
                ++self.synchronizations;
                if (self.complete)
                    self.complete();
            };
        }
    }
    ~Backend() {
        if (context) {
            if (std::uncaught_exceptions())
                context->global_ctx->queue.Get()->pending.clear(); // Abort-interception test cleanup only.
            else
                drain_queue(context->global_ctx->queue.Get());
        }
    }
    Backend(const Backend &) = delete;
    Backend & operator=(const Backend &) = delete;
};
using Scheduler = std::unique_ptr<ggml_backend_sched, decltype(&ggml_backend_sched_free)>;
using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
using Memory = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
} // namespace transfer_cases
