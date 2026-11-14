#ifdef SDCB_TEST_BROWSER_API
#define __EMSCRIPTEN__ 1
#endif
#include "browser-api.hpp"
#include "ggml-webgpu.cpp"
#include "core/runner_cache.h"
#include "core/ggml_graph_cut.h"
#include <array>
#include <iostream>

namespace {
int allocation_budget = -1;
int & submission_budget = browser_api::state.submission_budget;
auto & counters = browser_api::state;
void reset_counts() {
    counters.reads = counters.writes = counters.submissions = counters.waits = counters.copies = 0;
}
struct Fixture {
    webgpu_global_context global = std::make_shared<webgpu_global_context_struct>();
    ggml_backend_reg registry{};
    ggml_backend_webgpu_device_context device_context{global, "contract", "external API"};
    ggml_backend_device device{};
    ggml_backend_buffer_type type{};
    ggml_backend_webgpu_context backend_context{};
    ggml_backend backend{};
    Fixture() {
        global->instance = wgpu::Instance::Acquire(new WGPUInstanceImpl);
        global->device = wgpu::Device::Acquire(new WGPUDeviceImpl);
        global->queue = wgpu::Queue::Acquire(new WGPUQueueImpl);
        global->capabilities.limits.minStorageBufferOffsetAlignment = 256;
        global->capabilities.limits.minUniformBufferOffsetAlignment = 256;
        global->capabilities.limits.maxStorageBufferBindingSize = 128 * 1024 * 1024;
        global->capabilities.limits.maxComputeInvocationsPerWorkgroup = 256;
        global->capabilities.limits.maxComputeWorkgroupsPerDimension = 65535;
        registry.iface = ggml_backend_webgpu_reg_i;
        // Registry naming is an external discovery boundary for RunnerCache.
        registry.iface.get_name = [](ggml_backend_reg_t) { return "WebGPU"; };
        device.iface = ggml_backend_webgpu_device_i;
        device.reg = &registry;
        device.context = &device_context;
        type = *ggml_backend_webgpu_device_get_buffer_type(&device);
        type.device = &device;
        backend_context.webgpu_ctx = initialize_webgpu_context(&device);
        backend_context.name = "contract";
        backend.iface = ggml_backend_webgpu_i;
        backend.guid = ggml_backend_webgpu_guid();
        backend.context = &backend_context;
        backend.device = &device;
    }
};
struct Ctx {
    ggml_context * p = ggml_init({2 * 1024 * 1024, nullptr, true});
    ~Ctx() {
        ggml_free(p);
    }
};
struct Buf {
    ggml_backend_buffer_t p;
    ~Buf() {
        ggml_backend_buffer_free(p);
    }
};
int checked = 0;
} // namespace
extern "C" ggml_backend_buffer_t __real_ggml_backend_alloc_ctx_tensors(ggml_context *, ggml_backend_t);
extern "C" ggml_backend_buffer_t __wrap_ggml_backend_alloc_ctx_tensors(ggml_context * context, ggml_backend_t backend) {
    if (allocation_budget == 0)
        return nullptr;
    if (allocation_budget > 0)
        --allocation_budget;
    return __real_ggml_backend_alloc_ctx_tensors(context, backend);
}
int main() {
    try {
        Fixture f, other;
        for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16})
            for (bool viewed : {false, true})
                for (bool async : {false, true}) {
                    Ctx ctx;
                    auto backing = ggml_new_tensor_1d(ctx.p, type, 128);
                    auto src = viewed ? ggml_view_1d(ctx.p, backing, 24, 16) : backing;
                    Buf input{ggml_backend_alloc_ctx_tensors(ctx.p, &f.backend)};
                    Ctx target;
                    auto dst = ggml_dup_tensor(target.p, src);
                    Buf output{ggml_backend_alloc_ctx_tensors(target.p, &f.backend)};
                    std::vector<unsigned char> data(ggml_nbytes(backing));
                    for (size_t i = 0; i < data.size(); ++i)
                        data[i] = i % 251;
                    ggml_backend_tensor_set(backing, data.data(), 0, data.size());
                    browser_api::drain();
                    reset_counts();
                    bool ok = async ? ggml_backend_webgpu_cpy_tensor_async(&f.backend, &f.backend, src, dst)
                                    : ggml_backend_webgpu_buffer_cpy_tensor(output.p, src, dst);
                    SDC_CHECK(ok && counters.reads == 0 && counters.writes == 0 && counters.submissions == 1 &&
                              counters.waits == int(!async));
                    if (async) {
                        SDC_CHECK(counters.copies == 0);
                        ggml_backend_synchronize(&f.backend);
                    }
                    SDC_CHECK(counters.copies == 1);
                    auto & actual =
                        static_cast<ggml_backend_webgpu_buffer_context *>(output.p->context)->buffer.Get()->bytes;
                    SDC_CHECK(std::equal(actual.begin(), actual.begin() + ggml_nbytes(dst),
                                         data.begin() + (viewed ? 16 : 0)));
                    ++checked;
                }
        // Refused copies must not enqueue anything; the generic slow path stays available.
        for (int kind = 0; kind < 8; ++kind) {
            Ctx ctx, target;
            auto raw = ggml_new_tensor_1d(ctx.p, GGML_TYPE_F16, 128);
            auto src = ggml_view_1d(ctx.p, raw, kind == 0 ? 3 : 24, kind == 1 ? 2 : 16);
            Buf a{ggml_backend_alloc_ctx_tensors(ctx.p, &f.backend)};
            auto dst = ggml_dup_tensor(target.p, src);
            Buf b{ggml_backend_alloc_ctx_tensors_from_buft(target.p, kind == 2 ? &other.type : &f.type)};
            auto original = *dst;
            if (kind == 3) {
                dst->buffer = a.p;
                dst->data = src->data;
            }
            if (kind == 4)
                dst->data = (void *)(uintptr_t(0x1000) + 2);
            if (kind == 5)
                dst->data = (void *)(uintptr_t(0x1000) + ggml_backend_buffer_get_size(b.p));
            if (kind == 6)
                dst->buffer = nullptr;
            if (kind == 7) {
                dst->nb[1] += 4;
            }
            reset_counts();
            SDC_CHECK(!ggml_backend_webgpu_copy_tensor(src, dst, false));
            SDC_CHECK(counters.submissions == 0 && counters.waits == 0);
            *dst = original;
            ++checked;
        }
        // Actual CachedTensor copy: materialized outputs and CPY/view outputs alike
        // remain on the queue. Preserve source metadata and the view's nonzero offset.
        for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16})
            for (bool view : {false, true}) {
                Ctx ctx;
                auto raw = ggml_new_tensor_1d(ctx.p, type, 128);
                auto src = view ? ggml_view_1d(ctx.p, raw, 32, 16) : raw;
                Buf b{ggml_backend_alloc_ctx_tensors(ctx.p, &f.backend)};
                std::vector<unsigned char> data(ggml_nbytes(raw));
                for (size_t i = 0; i < data.size(); ++i)
                    data[i] = (i * 7) % 251;
                ggml_backend_tensor_set(raw, data.data(), 0, data.size());
                browser_api::drain();
                if (view)
                    src->buffer = nullptr; // GGML resolves this through view_src.
                auto original = *src;
                reset_counts();
                ggml_status status;
                auto entry = sd::CachedTensor::copy(&f.backend, "test", src, status);
                SDC_CHECK(entry && status == GGML_STATUS_SUCCESS && counters.reads == 0 && counters.writes == 0 &&
                          counters.waits == 1 && counters.submissions == 1);
                SDC_CHECK(std::memcmp(&original, src, sizeof(original)) == 0);
                ggml_backend_synchronize(&f.backend);
                auto & actual =
                    static_cast<ggml_backend_webgpu_buffer_context *>(entry->buffer->context)->buffer.Get()->bytes;
                SDC_CHECK(
                    std::equal(actual.begin(), actual.begin() + ggml_nbytes(src), data.begin() + (view ? 16 : 0)));
                ++checked;
            }
        // Batch caching commits only after its queue completion boundary. Eight CPY
        // outputs should have eight device copies but one wait and no host transfers.
        {
            Ctx ctx;
            sd::RunnerCache cache(&f.backend);
            auto graph = ggml_new_graph_custom(ctx.p, 64, false);
            std::array<ggml_tensor *, 8> outputs;
            for (int i = 0; i < 8; ++i) {
                auto raw = ggml_new_tensor_1d(ctx.p, GGML_TYPE_F32, 32);
                outputs[i] = ggml_view_1d(ctx.p, raw, 16, 16);
                ggml_build_forward_expand(graph, outputs[i]);
                cache.stage(std::to_string(i), outputs[i]);
            }
            Buf b{ggml_backend_alloc_ctx_tensors(ctx.p, &f.backend)};
            for (int i = 0; i < 8; ++i) {
                std::vector<float> v(32, float(i + 1));
                ggml_backend_tensor_set(outputs[i]->view_src, v.data(), 0, v.size() * 4);
            }
            browser_api::drain();
            reset_counts();
            SDC_CHECK(cache.capture(graph) == GGML_STATUS_SUCCESS);
            SDC_CHECK(counters.reads == 0 && counters.writes == 0 && counters.submissions == 8 &&
                      counters.copies == 8 && counters.waits == 1);
            cache.graph_end(true);
            for (int i = 0; i < 8; ++i)
                SDC_CHECK(cache.get(std::to_string(i)) != nullptr);
            ++checked;
        }
        // A staged output in another segment and a previously captured output must
        // not create completion callbacks or drain unrelated queued work.
        for (bool already_pending : {false, true}) {
            Ctx ctx;
            sd::RunnerCache cache(&f.backend);
            auto present = ggml_new_tensor_1d(ctx.p, GGML_TYPE_F32, 16);
            auto absent = ggml_new_tensor_1d(ctx.p, GGML_TYPE_F32, 16);
            auto graph = ggml_new_graph_custom(ctx.p, 32, false);
            ggml_build_forward_expand(graph, present);
            Buf buffer{ggml_backend_alloc_ctx_tensors(ctx.p, &f.backend)};
            cache.stage("output", already_pending ? present : absent);
            if (already_pending)
                SDC_CHECK(cache.capture(graph) == GGML_STATUS_SUCCESS);
            bool unrelated_work_completed = false;
            browser_api::state.pending.push_back([&] { unrelated_work_completed = true; });
            reset_counts();
            SDC_CHECK(cache.capture(graph) == GGML_STATUS_SUCCESS);
            SDC_CHECK(counters.waits == 0 && counters.submissions == 0 && counters.copies == 0);
            SDC_CHECK(!unrelated_work_completed);
            cache.graph_end(true);
            SDC_CHECK((cache.get("output") != nullptr) == already_pending);
            browser_api::drain();
            ++checked;
        }
        // A cut with no future consumer (including an empty output list) does not
        // own any pending copy. It must not synchronize unrelated computation.
        for (bool has_output : {false, true}) {
            Ctx ctx;
            sd::GraphCutTensorCache cache(&f.backend);
            auto input = ggml_new_tensor_1d(ctx.p, GGML_TYPE_F32, 16);
            auto output = ggml_cont(ctx.p, input);
            sd::ggml_graph_cut::mark_graph_cut(output, "queue-probe", "unused-output");
            SDC_CHECK(sd::ggml_graph_cut::is_graph_cut_tensor(output));
            auto graph = ggml_new_graph_custom(ctx.p, 32, false);
            ggml_build_forward_expand(graph, output);
            sd::ggml_graph_cut::Segment segment;
            if (has_output)
                segment.output_node_indices.push_back(0);
            bool unrelated_work_completed = false;
            browser_api::state.pending.push_back([&] { unrelated_work_completed = true; });
            reset_counts();
            SDC_CHECK(cache.capture(graph, segment, "empty-cut") == GGML_STATUS_SUCCESS);
            SDC_CHECK(counters.waits == 0 && counters.submissions == 0 && counters.copies == 0);
            SDC_CHECK(!unrelated_work_completed);
            browser_api::drain();
            ++checked;
        }
        // Allocation must finish before ANY async submission, including failures
        // after several destinations were successfully allocated. A queued-copy
        // failure must complete previous submissions before the cache drops owners.
        for (bool allocation_failure : {true, false}) {
            Ctx ctx;
            sd::RunnerCache cache(&f.backend);
            auto graph = ggml_new_graph_custom(ctx.p, 64, false);
            std::array<ggml_tensor *, 4> outputs;
            for (int i = 0; i < 4; ++i) {
                outputs[i] = ggml_new_tensor_1d(ctx.p, GGML_TYPE_F32, 16);
                ggml_build_forward_expand(graph, outputs[i]);
                cache.stage(std::to_string(i), outputs[i]);
            }
            Buf b{ggml_backend_alloc_ctx_tensors(ctx.p, &f.backend)};
            for (int i = 0; i < 4; ++i) {
                std::vector<float> values(16, float(i));
                ggml_backend_tensor_set(outputs[i], values.data(), 0, 64);
            }
            browser_api::drain();
            reset_counts();
            const auto live_before = browser_api::live_buffers;
            if (allocation_failure) {
                allocation_budget = 2;
                SDC_CHECK(cache.capture(graph) == GGML_STATUS_ALLOC_FAILED);
                allocation_budget = -1;
                SDC_CHECK(counters.submissions == 0);
            } else {
                submission_budget = 2;
                bool threw = false;
                try {
                    cache.capture(graph);
                } catch (const std::runtime_error &) {
                    threw = true;
                }
                submission_budget = -1;
                SDC_CHECK(threw && counters.submissions == 2 && counters.copies == 2 && counters.waits == 1);
            }
            SDC_CHECK(browser_api::state.pending.empty());
            cache.graph_end(false);
            SDC_CHECK(cache.empty());
            SDC_CHECK(browser_api::live_buffers == live_before);
            ++checked;
        }
        // Non-contiguous graph-cut spans retain the old host path and their strides.
        {
            Ctx ctx;
            auto raw = ggml_new_tensor_2d(ctx.p, GGML_TYPE_F32, 8, 8);
            auto src = ggml_view_2d(ctx.p, raw, 4, 4, raw->nb[1], 16);
            Buf b{ggml_backend_alloc_ctx_tensors(ctx.p, &f.backend)};
            std::vector<float> values(64);
            for (int i = 0; i < 64; ++i)
                values[i] = float(i);
            ggml_backend_tensor_set(raw, values.data(), 0, values.size() * 4);
            browser_api::drain();
            reset_counts();
            ggml_status status;
            auto entry = sd::CachedTensor::copy(&f.backend, "strided", src, status);
            // The complete backend's host read performs a staging-buffer copy and
            // map wait. The former fixture replaced that external read wholesale.
            SDC_CHECK(entry && counters.reads == 1 && counters.writes == 1 && counters.submissions == 1 &&
                      counters.waits == 1);
            browser_api::drain();
            SDC_CHECK(entry->tensor->nb[1] == src->nb[1]);
            ++checked;
        }
        std::cout << checked << " transfer/cache API contract cases passed; no GPU executed\n";
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
