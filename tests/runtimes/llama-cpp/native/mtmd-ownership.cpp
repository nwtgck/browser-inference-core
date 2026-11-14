#include "mtmd-subcontext.h"
#include "ggml-cpu.h"
#include "support/check.hpp"
#include "support/allocation-fault.hpp"
#include <iostream>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

static int cpu_tag, gpu_tag, scheduler_tag, device_tag, registry_tag, pool_tag;
static bool cpu_live, gpu_live, scheduler_live, pool_live;
static bool alias_backend;
static std::string failure;
static std::vector<std::string> events;
mtmd_logger_state g_logger_state{mtmd_log_callback_default, nullptr};

static ggml_backend_t cpu() {
    return reinterpret_cast<ggml_backend_t>(&cpu_tag);
}
static ggml_backend_t gpu() {
    return reinterpret_cast<ggml_backend_t>(&gpu_tag);
}
static ggml_backend_dev_t device() {
    return reinterpret_cast<ggml_backend_dev_t>(&device_tag);
}
static void point(const char * name) {
    events.push_back(name);
    if (failure == name)
        throw std::runtime_error(name);
}
std::string mtmd_format(const char *, ...) {
    return "controlled failure";
}

static ggml_threadpool_t pool_new(ggml_threadpool_params * params) {
    point("threadpool-new");
    BICORE_CHECK(params->n_threads == 2 && !pool_live);
    if (failure == "threadpool-null")
        return nullptr;
    pool_live = true;
    return reinterpret_cast<ggml_threadpool_t>(&pool_tag);
}
static void pool_free(ggml_threadpool_t pool) {
    BICORE_CHECK(pool == reinterpret_cast<ggml_threadpool_t>(&pool_tag));
    BICORE_CHECK(pool_live && !cpu_live && !gpu_live && !scheduler_live);
    pool_live = false;
    events.push_back("threadpool-free");
}
static void set_pool(ggml_backend_t backend, ggml_threadpool_t pool) {
    BICORE_CHECK(backend == cpu() && pool == reinterpret_cast<ggml_threadpool_t>(&pool_tag));
}

extern "C" {
ggml_backend_dev_t ggml_backend_dev_by_type(enum ggml_backend_dev_type type) {
    return type == GGML_BACKEND_DEVICE_TYPE_CPU ? device() : nullptr;
}
ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t) {
    return reinterpret_cast<ggml_backend_reg_t>(&registry_tag);
}
void * ggml_backend_reg_get_proc_address(ggml_backend_reg_t, const char * name) {
    if (std::string(name) == "ggml_threadpool_new")
        return reinterpret_cast<void *>(pool_new);
    if (std::string(name) == "ggml_threadpool_free")
        return reinterpret_cast<void *>(pool_free);
    if (std::string(name) == "ggml_backend_cpu_set_threadpool")
        return reinterpret_cast<void *>(set_pool);
    return nullptr;
}
ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t) {
    return device();
}
ggml_threadpool_params ggml_threadpool_params_default(int n_threads) {
    ggml_threadpool_params result{};
    result.n_threads = n_threads;
    return result;
}
ggml_backend_t ggml_backend_init_by_type(enum ggml_backend_dev_type, const char *) {
    point("cpu-init");
    if (failure == "cpu-null")
        return nullptr;
    cpu_live = true;
    return cpu();
}
ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t, const char *) {
    point("gpu-init");
    if (failure == "gpu-null")
        return nullptr;
    if (alias_backend)
        return cpu();
    gpu_live = true;
    return gpu();
}
const char * ggml_backend_dev_name(ggml_backend_dev_t) {
    return "controlled";
}
const char * ggml_backend_name(ggml_backend_t) {
    return "controlled";
}
ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t) {
    point("buffer-type");
    return nullptr;
}
void ggml_backend_free(ggml_backend_t value) {
    BICORE_CHECK(!scheduler_live);
    if (value == cpu()) {
        BICORE_CHECK(cpu_live);
        cpu_live = false;
        events.push_back("cpu-free");
    } else if (value == gpu()) {
        BICORE_CHECK(gpu_live);
        gpu_live = false;
        events.push_back("gpu-free");
    } else
        BICORE_CHECK(value == nullptr);
}
ggml_backend_sched_t ggml_backend_sched_new(ggml_backend_t *, ggml_backend_buffer_type_t *, int, size_t, bool, bool) {
    point("scheduler-new");
    scheduler_live = true;
    return reinterpret_cast<ggml_backend_sched_t>(&scheduler_tag);
}
void ggml_backend_sched_free(ggml_backend_sched_t) {
    BICORE_CHECK(scheduler_live);
    scheduler_live = false;
    events.push_back("scheduler-free");
}
void ggml_backend_sched_set_eval_callback(ggml_backend_sched_t, ggml_backend_sched_eval_callback, void *) {
    point("scheduler-callback");
}
}

static bool callback(ggml_tensor *, bool, void *) {
    return true;
}
static void run(const char * fail, bool cpu_only, bool alias = false) {
    events.clear();
    failure = fail;
    alias_backend = alias;
    BICORE_CHECK(!cpu_live && !gpu_live && !scheduler_live);
    bool caught = false;
    try {
        mtmd_backends real(cpu_only ? nullptr : device(), 2, 8192, callback, nullptr);
        BICORE_CHECK(cpu_live && scheduler_live && pool_live && (cpu_only || alias || gpu_live));
    } catch (const std::runtime_error &) {
        caught = true;
    }
    BICORE_CHECK(caught == !failure.empty());
    BICORE_CHECK(!cpu_live && !gpu_live && !scheduler_live && !pool_live);
}

static size_t real_allocation_failures(bool cpu_only) {
    events.clear();
    failure.clear();
    alias_backend = false;
    allocation_fault::state.allocations = 0;
    allocation_fault::state.observing = true;
    { mtmd_backends real(cpu_only ? nullptr : device(), 2, 8192, callback, nullptr); }
    allocation_fault::state.observing = false;
    const size_t count = allocation_fault::state.allocations;
    BICORE_CHECK(count > 0);
    for (size_t i = 0; i < count; ++i) {
        events.clear();
        allocation_fault::state.allocations = 0;
        allocation_fault::state.fail_after = ptrdiff_t(i);
        allocation_fault::state.observing = true;
        bool threw = false;
        try {
            mtmd_backends real(cpu_only ? nullptr : device(), 2, 8192, callback, nullptr);
        } catch (const std::bad_alloc &) {
            threw = true;
        }
        allocation_fault::state.observing = false;
        allocation_fault::state.fail_after = -1;
        BICORE_CHECK(threw);
        BICORE_CHECK(!cpu_live && !gpu_live && !scheduler_live && !pool_live);
    }
    return count;
}
int main() {
    events.reserve(1024);
    run("", true);
    run("", false);
    run("", false, true);
    for (const char * fail : {"threadpool-new", "threadpool-null", "cpu-init", "cpu-null", "gpu-init", "gpu-null",
                              "buffer-type", "scheduler-new", "scheduler-callback"})
        run(fail, false);
    const auto faults = real_allocation_failures(true) + real_allocation_failures(false);
    std::cout << "12 real-constructor/cleanup cases and " << faults << " real allocation faults passed\n";
}
