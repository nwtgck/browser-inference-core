// Registry and allocator bookkeeping only; production lookup and constructor
// discovery are inserted below, together with the actual provider bodies.
enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t);
struct ggml_backend_reg_i {
    void * (*get_proc_address)(ggml_backend_reg_t, const char *) = nullptr;
};
struct ggml_backend_reg { ggml_backend_reg_i iface; void * context = nullptr; };
// @ACTUAL_REGISTRY_ACCESS@

// The CPU provider's existing extension addresses are never called in these
// tests. Keep its complete lookup body so an unknown version is really absent.
using ggml_backend_set_n_threads_t = void (*)(ggml_backend_t, int);
using ggml_backend_dev_get_extra_bufts_t = ggml_backend_buffer_type_t * (*)(ggml_backend_dev_t);
static void ggml_backend_cpu_set_n_threads(ggml_backend_t, int) { throw std::runtime_error("unexpected CPU extension"); }
static ggml_backend_buffer_type_t * ggml_backend_cpu_device_get_extra_buffers_type(ggml_backend_dev_t) { throw std::runtime_error("unexpected CPU extension"); }
static void ggml_backend_cpu_get_features() { throw std::runtime_error("unexpected CPU extension"); }
static void ggml_backend_cpu_set_abort_callback() { throw std::runtime_error("unexpected CPU extension"); }
static void ggml_numa_init() { throw std::runtime_error("unexpected CPU extension"); }
static void ggml_is_numa() { throw std::runtime_error("unexpected CPU extension"); }
static void ggml_backend_cpu_set_use_ref() { throw std::runtime_error("unexpected CPU extension"); }
static void ggml_threadpool_new() { throw std::runtime_error("unexpected CPU extension"); }
static void ggml_threadpool_free() { throw std::runtime_error("unexpected CPU extension"); }
static void ggml_backend_cpu_set_threadpool() { throw std::runtime_error("unexpected CPU extension"); }
// @ACTUAL_CPU_REGISTRY@
// @ACTUAL_SNAPSHOT_PROVIDER@
// @ACTUAL_WEBGPU_REGISTRY@
// @ACTUAL_GET_ASYNC@
