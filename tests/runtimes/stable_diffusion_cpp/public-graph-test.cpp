// Author tensors through the public C API and the small record-value bridge.
// No model fixture, private runner or additional test export is required.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

extern "C" {
ggml_context * bicore_ggml_init(const ggml_init_params *);
void bicore_ggml_graph_plan(ggml_cplan *, const ggml_cgraph *, int, ggml_threadpool_t);
void bicore_ggml_tallocr_new(ggml_tallocr *, ggml_backend_buffer_t);
}

namespace {
void require(bool condition, const char * message) {
    if (!condition)
        throw std::runtime_error(message);
}
struct Context {
    ggml_context * pointer;
    ~Context() {
        if (pointer)
            ggml_free(pointer);
    }
};
struct Backend {
    ggml_backend_t pointer;
    ~Backend() {
        if (pointer)
            ggml_backend_free(pointer);
    }
};
struct Buffer {
    ggml_backend_buffer_t pointer;
    ~Buffer() {
        if (pointer)
            ggml_backend_buffer_free(pointer);
    }
};
bool abort_compute(void * data) {
    ++*static_cast<int *>(data);
    return true;
}
void expected(const std::array<float, 8> & actual, const std::array<float, 8> & wanted) {
    for (size_t i = 0; i < actual.size(); ++i)
        require(std::abs(actual[i] - wanted[i]) < 1e-5f, "public graph numerical result changed");
}
void run() {
    const ggml_init_params parameters = {4 * 1024 * 1024, nullptr, true};
    Context context{bicore_ggml_init(&parameters)};
    require(context.pointer != nullptr, "context allocation failed");
    Backend backend{ggml_backend_cpu_init()};
    require(backend.pointer != nullptr, "CPU backend unavailable");
    ggml_backend_cpu_set_n_threads(backend.pointer, 1);
    const auto left = ggml_new_tensor_2d(context.pointer, GGML_TYPE_F32, 3, 2);
    const auto right = ggml_new_tensor_2d(context.pointer, GGML_TYPE_F32, 3, 4);
    const auto product = ggml_mul_mat(context.pointer, left, right);
    const auto output = ggml_scale(context.pointer, product, 2.f);
    const auto graph = ggml_new_graph_custom(context.pointer, 128, false);
    ggml_build_forward_expand(graph, output);
    Buffer buffer{ggml_backend_alloc_ctx_tensors(context.pointer, backend.pointer)};
    require(buffer.pointer != nullptr, "tensor allocation failed");
    std::array<float, 6> a{1, 2, 3, 4, 5, 6};
    const std::array<float, 12> b{7, 8, 9, 10, 11, 12, -1, 0, 1, 2, -2, 3};
    std::array<float, 8> values{};
    ggml_backend_tensor_set(left, a.data(), 0, sizeof(a));
    ggml_backend_tensor_set(right, b.data(), 0, sizeof(b));
    require(ggml_backend_graph_compute(backend.pointer, graph) == GGML_STATUS_SUCCESS, "initial graph failed");
    ggml_backend_synchronize(backend.pointer);
    ggml_backend_tensor_get(output, values.data(), 0, sizeof(values));
    expected(values, {100, 244, 136, 334, 4, 4, 14, 32});

    // Reuse the same graph/buffers through the actual record-value bridge.
    a.fill(1);
    ggml_backend_tensor_set(left, a.data(), 0, sizeof(a));
    ggml_cplan plan{};
    bicore_ggml_graph_plan(&plan, graph, 1, nullptr);
    std::vector<uint8_t> work(plan.work_size);
    plan.work_data = work.data();
    require(ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS, "planned graph failed");
    ggml_backend_tensor_get(output, values.data(), 0, sizeof(values));
    expected(values, {48, 48, 66, 66, 0, 0, 6, 6});

    int aborts = 0;
    ggml_backend_cpu_set_abort_callback(backend.pointer, abort_compute, &aborts);
    require(ggml_backend_graph_compute(backend.pointer, graph) == GGML_STATUS_ABORTED, "abort callback was ignored");
    require(aborts > 0, "abort callback was never called");
    ggml_backend_cpu_set_abort_callback(backend.pointer, nullptr, nullptr);
    a.fill(2);
    ggml_backend_tensor_set(left, a.data(), 0, sizeof(a));
    require(ggml_backend_graph_compute(backend.pointer, graph) == GGML_STATUS_SUCCESS,
            "graph did not recover after abort");
    ggml_backend_tensor_get(output, values.data(), 0, sizeof(values));
    expected(values, {96, 96, 132, 132, 0, 0, 12, 12});

    ggml_tallocr allocator{};
    bicore_ggml_tallocr_new(&allocator, buffer.pointer);
    require(allocator.buffer == buffer.pointer && allocator.alignment > 0, "record return bridge changed");
    bool rejected = false;
    try {
        bicore_ggml_graph_plan(nullptr, graph, 1, nullptr);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    require(rejected, "null record return storage was accepted");
}
} // namespace

int main() {
    try {
        run();
        std::puts("public graph ABI: construction, compute, reuse, abort/recovery and records passed");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
