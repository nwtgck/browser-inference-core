// Build and evaluate authored tensors using only the normalized public ABI.
// No model fixture, private runner or additional test export is required.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

extern "C" {
uint64_t sdc_ggml_init(uint64_t);
void sdc_ggml_free(uint64_t);
uint64_t sdc_ggml_new_tensor_2d(uint64_t, int32_t, int64_t, int64_t);
uint64_t sdc_ggml_mul_mat(uint64_t, uint64_t, uint64_t);
uint64_t sdc_ggml_scale(uint64_t, uint64_t, float);
uint64_t sdc_ggml_new_graph_custom(uint64_t, uint64_t, int32_t);
void sdc_ggml_build_forward_expand(uint64_t, uint64_t);
uint64_t sdc_ggml_backend_cpu_init();
void sdc_ggml_backend_free(uint64_t);
void sdc_ggml_backend_cpu_set_n_threads(uint64_t, int32_t);
void sdc_ggml_backend_cpu_set_abort_callback(uint64_t, uint64_t, uint64_t);
uint64_t sdc_ggml_backend_alloc_ctx_tensors(uint64_t, uint64_t);
void sdc_ggml_backend_buffer_free(uint64_t);
void sdc_ggml_backend_tensor_set(uint64_t, uint64_t, uint64_t, uint64_t);
void sdc_ggml_backend_tensor_get(uint64_t, uint64_t, uint64_t, uint64_t);
int32_t sdc_ggml_backend_graph_compute(uint64_t, uint64_t);
void sdc_ggml_backend_synchronize(uint64_t);
void sdc_ggml_graph_plan(uint64_t, uint64_t, int32_t, uint64_t);
int32_t sdc_ggml_graph_compute(uint64_t, uint64_t);
void sdc_ggml_tallocr_new(uint64_t, uint64_t);
}

namespace {
void require(bool condition, const char * message) {
    if (!condition)
        throw std::runtime_error(message);
}
template <class T> uint64_t address(T * pointer) {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(pointer));
}
struct Owned {
    uint64_t pointer;
    void (*release)(uint64_t);
    ~Owned() {
        if (pointer)
            release(pointer);
    }
};
bool abort_compute(void * data) {
    ++*static_cast<int *>(data);
    return true;
}
void expected(const std::array<float, 8> & actual, const std::array<float, 8> & wanted) {
    for (size_t i = 0; i < actual.size(); ++i) {
        require(std::abs(actual[i] - wanted[i]) < 1e-5f, "public graph numerical result changed");
    }
}
void run() {
    ggml_init_params parameters = {4 * 1024 * 1024, nullptr, true};
    Owned context{sdc_ggml_init(address(&parameters)), sdc_ggml_free};
    require(context.pointer != 0, "context allocation failed");
    Owned backend{sdc_ggml_backend_cpu_init(), sdc_ggml_backend_free};
    require(backend.pointer != 0, "CPU backend unavailable");
    sdc_ggml_backend_cpu_set_n_threads(backend.pointer, 1);
    const auto left = sdc_ggml_new_tensor_2d(context.pointer, GGML_TYPE_F32, 3, 2);
    const auto right = sdc_ggml_new_tensor_2d(context.pointer, GGML_TYPE_F32, 3, 4);
    const auto product = sdc_ggml_mul_mat(context.pointer, left, right);
    const auto output = sdc_ggml_scale(context.pointer, product, 2.f);
    const auto graph = sdc_ggml_new_graph_custom(context.pointer, 128, 0);
    sdc_ggml_build_forward_expand(graph, output);
    Owned buffer{sdc_ggml_backend_alloc_ctx_tensors(context.pointer, backend.pointer), sdc_ggml_backend_buffer_free};
    require(buffer.pointer != 0, "tensor allocation failed");
    std::array<float, 6> a{1, 2, 3, 4, 5, 6};
    std::array<float, 12> b{7, 8, 9, 10, 11, 12, -1, 0, 1, 2, -2, 3};
    std::array<float, 8> values{};
    sdc_ggml_backend_tensor_set(left, address(a.data()), 0, sizeof(a));
    sdc_ggml_backend_tensor_set(right, address(b.data()), 0, sizeof(b));
    require(sdc_ggml_backend_graph_compute(backend.pointer, graph) == GGML_STATUS_SUCCESS, "initial graph failed");
    sdc_ggml_backend_synchronize(backend.pointer);
    sdc_ggml_backend_tensor_get(output, address(values.data()), 0, sizeof(values));
    expected(values, {100, 244, 136, 334, 4, 4, 14, 32});

    // Reuse the same graph/buffers with different inputs through the explicit CPU plan.
    a.fill(1);
    sdc_ggml_backend_tensor_set(left, address(a.data()), 0, sizeof(a));
    ggml_cplan plan{};
    sdc_ggml_graph_plan(address(&plan), graph, 1, 0);
    std::vector<uint8_t> work(plan.work_size);
    plan.work_data = work.data();
    require(sdc_ggml_graph_compute(graph, address(&plan)) == GGML_STATUS_SUCCESS, "planned graph failed");
    sdc_ggml_backend_tensor_get(output, address(values.data()), 0, sizeof(values));
    expected(values, {48, 48, 66, 66, 0, 0, 6, 6});

    // A genuine in-compute callback abort is followed by a successful recomputation.
    int aborts = 0;
    sdc_ggml_backend_cpu_set_abort_callback(backend.pointer, address(&abort_compute), address(&aborts));
    require(sdc_ggml_backend_graph_compute(backend.pointer, graph) == GGML_STATUS_ABORTED,
            "abort callback was ignored");
    require(aborts > 0, "abort callback was never called");
    sdc_ggml_backend_cpu_set_abort_callback(backend.pointer, 0, 0);
    a.fill(2);
    sdc_ggml_backend_tensor_set(left, address(a.data()), 0, sizeof(a));
    require(sdc_ggml_backend_graph_compute(backend.pointer, graph) == GGML_STATUS_SUCCESS,
            "graph did not recover after abort");
    sdc_ggml_backend_tensor_get(output, address(values.data()), 0, sizeof(values));
    expected(values, {96, 96, 132, 132, 0, 0, 12, 12});

    ggml_tallocr allocator{};
    sdc_ggml_tallocr_new(address(&allocator), buffer.pointer);
    require(address(allocator.buffer) == buffer.pointer && allocator.alignment > 0, "record return bridge changed");
    bool rejected = false;
    try {
        sdc_ggml_graph_plan(0, graph, 1, 0);
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
