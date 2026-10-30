// Real ggml CPU storage/scheduler for contract and numerical tests. CPU execution
// here does NOT demonstrate WebGPU placement, performance, or model quality.
#include "mtmd-retained-upload.h"
#include "mtmd-retained-graph.h"
#include "mtmd-retained-roles.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "ggml-backend-impl.h" // test-only synchronize counter; not a production adapter

#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static int checks = 0;
static void require(bool value, const char * reason) {
    ++checks;
    if (!value)
        throw std::runtime_error(reason);
}
template <typename F> static void rejected(F f, const char * text) {
    bool caught = false;
    try {
        f();
    } catch (const std::runtime_error & error) {
        caught = true;
        require(std::string(error.what()).find(text) != std::string::npos, "wrong rejection reason");
    }
    require(caught, "expected rejection was not raised");
}
static ggml_context * context() {
    ggml_init_params p{};
    p.mem_size = 4 * 1024 * 1024;
    p.no_alloc = true;
    auto * ctx = ggml_init(p);
    if (!ctx)
        throw std::runtime_error("test metadata allocation failed");
    return ctx;
}
struct fixture {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_context * weights = context();
    ggml_context * activations = context();
    ggml_backend_buffer_t buffer = nullptr;
    ggml_backend_sched_t sched = nullptr;
    fixture() {
        if (!cpu)
            throw std::runtime_error("test CPU backend unavailable");
        ggml_backend_cpu_set_n_threads(cpu, 1);
        sched = ggml_backend_sched_new(&cpu, nullptr, 1, 256, false, false);
        if (!sched)
            throw std::runtime_error("test scheduler allocation failed");
    }
    ~fixture() {
        ggml_backend_sched_free(sched);
        ggml_backend_buffer_free(buffer);
        ggml_free(activations);
        ggml_free(weights);
        ggml_backend_free(cpu);
    }
    void allocate() {
        buffer = ggml_backend_alloc_ctx_tensors(weights, cpu);
        if (!buffer)
            throw std::runtime_error("test weights allocation failed");
        ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    ggml_cgraph * graph(ggml_tensor * result) {
        auto * g = ggml_new_graph_custom(activations, 256, false);
        ggml_build_forward_expand(g, result);
        return g;
    }
};

static void test_reader() {
    lcb_mtmd_retained_reader reader;
    const size_t chunk = lcb_mtmd_retained_reader::chunk_bytes;
    for (size_t size :
         {size_t(0), size_t(1), size_t(2), size_t(3), size_t(131072), chunk - 1, chunk, chunk + 1, chunk * 2 + 7}) {
        std::string raw(size, '\0');
        for (size_t i = 0; i < size; ++i)
            raw[i] = char(i * 19 + (i >> 8));
        std::istringstream input(raw + "NEXT");
        std::string output;
        std::vector<size_t> progress;
        reader.read(
            input, size,
            [&](const uint8_t * data, size_t off, size_t bytes) {
                require(off == output.size(), "raw upload has a gap");
                require(bytes > 0 && bytes <= chunk, "raw upload is not bounded");
                output.append(reinterpret_cast<const char *>(data), bytes);
            },
            [&](size_t consumed) {
                require(consumed <= size, "source progress exceeds tensor");
                if (!progress.empty())
                    require(consumed >= progress.back(), "source progress regressed");
                progress.push_back(consumed);
            });
        require(output == raw, "raw data changed");
        require(input.peek() == 'N', "raw upload crossed tensor boundary");
        require(progress.front() == 0 && progress.back() == size, "start/final checkpoint missing");
        require(reader.scratch_bytes() == (size ? chunk : 0), "wrong explicit scratch size");
    }
    for (size_t size : {size_t(2), chunk, chunk + 2}) {
        std::istringstream input(std::string(size - 1, 'x'));
        size_t submitted = 0;
        rejected(
            [&] {
                reader.read(
                    input, size, [&](const uint8_t *, size_t, size_t n) { submitted += n; }, [](size_t) {});
            },
            "truncated");
        require(submitted < size, "truncated final chunk was uploaded");
    }
    for (size_t size : {size_t(0), size_t(2), chunk + 3}) {
        std::istringstream input(std::string(size, 'x'));
        rejected(
            [&] {
                reader.read(
                    input, size,
                    [](const uint8_t *, size_t, size_t) { throw std::runtime_error("write must not be reached"); },
                    [](size_t) { throw std::runtime_error("cancelled initially"); });
            },
            "cancelled initially");
        require(input.tellg() == 0, "initial cancellation consumed input");
    }
    {
        std::istringstream input("abc");
        bool cancelled = false;
        rejected(
            [&] {
                reader.read(
                    input, 3, [&](const uint8_t *, size_t, size_t) { cancelled = true; },
                    [&](size_t) {
                        if (cancelled)
                            throw std::runtime_error("cancelled in final write");
                    });
            },
            "cancelled in final write");
    }
    {
        std::istringstream input(std::string(chunk + 1, 'x'));
        rejected(
            [&] {
                reader.read(
                    input, chunk + 1, [](const uint8_t *, size_t, size_t) { throw std::runtime_error("write failed"); },
                    [](size_t) {});
            },
            "write failed");
        require(input.tellg() == std::streampos(chunk), "write failure read another chunk");
    }
    require(lcb_mtmd_add_bytes(SIZE_MAX - 1, 1) == SIZE_MAX, "checked addition rejected boundary");
    rejected([] { (void)lcb_mtmd_add_bytes(SIZE_MAX, 1); }, "address space");
    lcb_mtmd_check_retained_backend(MTMD_VISION_BF16_STORAGE_RETAIN_CHECKED, false, false);
    lcb_mtmd_check_retained_backend(MTMD_VISION_BF16_STORAGE_RETAIN_CHECKED, false, true);
    lcb_mtmd_check_retained_backend(MTMD_VISION_BF16_STORAGE_RETAIN_CHECKED, true, true);
    rejected([] { lcb_mtmd_check_retained_backend(MTMD_VISION_BF16_STORAGE_RETAIN_CHECKED, true, false); },
             "unavailable");
    require(lcb_mtmd_retain_bf16(MTMD_VISION_BF16_STORAGE_RETAIN_CHECKED), "retain mode is disabled");
    require(!lcb_mtmd_retain_bf16(MTMD_VISION_BF16_STORAGE_LEGACY_EXPAND), "legacy mode retained BF16");
    lcb_mtmd_check_retained_backend(MTMD_VISION_BF16_STORAGE_LEGACY_EXPAND, true, false);
}

static void test_queue() {
    size_t waits = 0;
    ggml_backend backend{};
    backend.context = &waits;
    backend.iface.synchronize = [](ggml_backend_t b) { ++*static_cast<size_t *>(b->context); };
    using Q = lcb_mtmd_upload_queue;
    {
        Q q(&backend);
        require(!q.reserve(0) && q.pending_bytes() == 0, "empty write was charged");
        require(!q.reserve(Q::budget_bytes - Q::write_overhead), "first write unexpectedly waited");
        require(q.pending_bytes() == Q::budget_bytes, "queue did not include overhead");
        require(q.reserve(1) && waits == 1, "queue window did not synchronize");
        require(q.pending_bytes() == Q::write_overhead + 1, "queue charged twice after waiting");
        q.drain();
        q.drain();
        require(waits == 2 && q.pending_bytes() == 0, "drain not idempotent");
        rejected([&] { q.reserve(Q::budget_bytes - Q::write_overhead + 1); }, "budget");
        rejected([&] { q.reserve(SIZE_MAX); }, "budget");
        require(q.pending_bytes() == 0, "invalid write modified budget");
    }
    require(waits == 2, "empty destructor synchronized");
    rejected(
        [&] {
            Q q(&backend);
            q.reserve(100);
            throw std::runtime_error("write failure");
        },
        "write failure");
    require(waits == 3, "exception destroyed buffer before retiring writes");
    rejected(
        [] {
            Q q(nullptr);
            q.reserve(1);
        },
        "backend");
    // Matches the loader's final sequence: drain, then check cancellation.
    bool cancelled = false;
    backend.context = &cancelled;
    backend.iface.synchronize = [](ggml_backend_t b) { *static_cast<bool *>(b->context) = true; };
    rejected(
        [&] {
            Q q(&backend);
            q.reserve(2);
            q.drain();
            if (cancelled)
                throw std::runtime_error("cancelled while draining");
        },
        "cancelled while draining");
}

static void test_matmul() {
    for (int k : {3, 16, 127})
        for (int cols : {1, 4}) {
            fixture f;
            const int rows = 9;
            ggml_tensor * w = ggml_new_tensor_2d(f.weights, GGML_TYPE_BF16, k, rows);
            ggml_tensor * x = ggml_new_tensor_2d(f.activations, GGML_TYPE_F32, k, cols);
            ggml_set_input(x);
            ggml_tensor * y = ggml_mul_mat(f.activations, w, x);
            ggml_set_output(y);
            auto * g = f.graph(y);
            f.allocate();
            std::vector<ggml_bf16_t> values(k * rows);
            std::vector<float> input(k * cols);
            for (size_t i = 0; i < values.size(); ++i)
                values[i] = ggml_fp32_to_bf16(float(int(i % 7) - 3));
            for (size_t i = 0; i < input.size(); ++i)
                input[i] = float(int(i % 5) - 2);
            // Exercise the SAME bounded raw writer used by the overlay with real
            // tensor_set, including a BF16 tensor with an odd number of elements.
            std::istringstream file(std::string(reinterpret_cast<const char *>(values.data()), values.size() * 2));
            lcb_mtmd_retained_reader reader;
            lcb_mtmd_upload_queue q(f.cpu);
            reader.read(
                file, ggml_nbytes(w),
                [&](const uint8_t * p, size_t off, size_t n) {
                    q.reserve(n);
                    ggml_backend_tensor_set(w, p, off, n);
                },
                [](size_t) {});
            q.drain();
            lcb_mtmd_retained_graph plan;
            plan.prepare(f.weights, g, f.sched, f.cpu, true);
            require(plan.consumer_count() == 1, "matmul not guarded");
            require(ggml_backend_sched_reserve(f.sched, g), "test reserve failed");
            plan.verify_sources();
            require(ggml_backend_sched_get_tensor_backend(f.sched, y) == nullptr,
                    "reserve no longer resets assignments; review guard");
            plan.prepare(f.weights, g, f.sched, f.cpu, true);
            require(ggml_backend_sched_alloc_graph(f.sched, g), "test allocate failed");
            plan.verify_allocated(f.sched);
            require(w->type == GGML_TYPE_BF16 && ggml_nbytes(w) == values.size() * 2, "resident weight expanded");
            ggml_backend_tensor_set(x, input.data(), 0, input.size() * 4);
            require(ggml_backend_sched_graph_compute(f.sched, g) == GGML_STATUS_SUCCESS, "test compute failed");
            std::vector<float> output(rows * cols);
            ggml_backend_tensor_get(y, output.data(), 0, output.size() * 4);
            for (int c = 0; c < cols; ++c)
                for (int r = 0; r < rows; ++r) {
                    double expected = 0;
                    for (int i = 0; i < k; ++i)
                        expected += double(ggml_bf16_to_fp32(values[r * k + i])) * input[c * k + i];
                    require(output[c * rows + r] == expected, "retained matmul differs from independent reference");
                }
            ggml_tensor replacement = *w;
            y->src[0] = &replacement;
            rejected([&] { plan.verify_sources(); }, "copied");
            y->src[0] = w;
            auto * original = x->buffer;
            x->buffer = w->buffer;
            rejected([&] { plan.verify_allocated(f.sched); }, "shares");
            x->buffer = original;
        }
}

static void test_rows() {
    fixture f;
    auto * table = ggml_new_tensor_2d(f.weights, GGML_TYPE_BF16, 7, 5);
    auto * idx = ggml_new_tensor_1d(f.activations, GGML_TYPE_I32, 4);
    ggml_set_input(idx);
    auto * y = ggml_get_rows(f.activations, table, idx);
    ggml_set_output(y);
    auto * g = f.graph(y);
    f.allocate();
    lcb_mtmd_retained_graph plan;
    plan.prepare(f.weights, g, f.sched, f.cpu, true);
    require(plan.consumer_count() == 1, "row selection not guarded");
    require(ggml_backend_sched_alloc_graph(f.sched, g), "rows allocate failed");
    plan.verify_allocated(f.sched);
    std::vector<ggml_bf16_t> values(35);
    for (size_t i = 0; i < values.size(); ++i)
        values[i] = ggml_fp32_to_bf16(float(i));
    const int32_t indices[] = {4, 0, 3, 4};
    ggml_backend_tensor_set(table, values.data(), 0, values.size() * 2);
    ggml_backend_tensor_set(idx, indices, 0, sizeof(indices));
    require(ggml_backend_sched_graph_compute(f.sched, g) == GGML_STATUS_SUCCESS, "rows compute failed");
    float output[28];
    ggml_backend_tensor_get(y, output, 0, sizeof(output));
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 7; ++c) {
            require(output[r * 7 + c] == indices[r] * 7 + c, "rows numerical mismatch");
        }
}

static void test_rejections() {
    // Copies/casts must be rejected BEFORE graph reservation/large allocation.
    for (bool cast : {false, true}) {
        fixture f;
        auto * w = ggml_new_tensor_2d(f.weights, GGML_TYPE_BF16, 16, 16);
        auto * copied = cast ? ggml_cast(f.activations, w, GGML_TYPE_F32)
                             : ggml_cont(f.activations, ggml_transpose(f.activations, w));
        auto * g = f.graph(copied);
        f.allocate();
        lcb_mtmd_retained_graph plan;
        rejected([&] { plan.prepare(f.weights, g, f.sched, f.cpu, true); },
                 cast ? "unsupported operation" : "unsupported weight operand");
    }
    {
        fixture f;
        auto * w = ggml_new_tensor_2d(f.weights, GGML_TYPE_BF16, 16, 16);
        auto * idx = ggml_new_tensor_1d(f.activations, GGML_TYPE_I32, 65536);
        auto * g = f.graph(ggml_get_rows(f.activations, w, idx));
        f.allocate();
        lcb_mtmd_retained_graph plan;
        rejected([&] { plan.prepare(f.weights, g, f.sched, f.cpu, true); }, "one-dimensional");
    }
    {
        fixture f;
        auto * w = ggml_new_tensor_2d(f.weights, GGML_TYPE_BF16, 16, 16);
        auto * x = ggml_new_tensor_2d(f.activations, GGML_TYPE_F32, 16, 1);
        auto * y = ggml_mul_mat(f.activations, w, x);
        auto * g = f.graph(y);
        f.allocate();
        lcb_mtmd_retained_graph plan;
        const size_t saved = w->nb[1];
        w->nb[1] = SIZE_MAX;
        rejected([&] { plan.prepare(f.weights, g, f.sched, f.cpu, true); }, "stride");
        w->nb[1] = saved;
        auto * b = w->buffer;
        w->buffer = nullptr;
        rejected([&] { plan.prepare(f.weights, g, f.sched, f.cpu, true); }, "accessible");
        w->buffer = b;
        const auto old = y->op;
        y->op = GGML_OP_NONE; // test a bad producer without constructing invalid ggml API operands
        rejected([&] { plan.prepare(f.weights, g, f.sched, f.cpu, true); }, "unsupported operation");
        require(ggml_backend_sched_get_tensor_backend(f.sched, y) == nullptr, "failed plan made an assignment");
        y->op = old;
        auto * view = ggml_view_2d(f.activations, w, 16, 1, 32, 0);
        y->src[0] = view;
        view->view_offs = ggml_nbytes(w);
        rejected([&] { plan.prepare(f.weights, g, f.sched, f.cpu, true); }, "exceeds original");
        y->src[0] = w;
        // No-op legacy mode must not inspect even missing contexts.
        plan.prepare(nullptr, nullptr, nullptr, nullptr, false);
        require(plan.consumer_count() == 0, "legacy mode retained old plan");
        rejected([&] { plan.prepare(nullptr, nullptr, nullptr, nullptr, true); }, "missing");
    }
}

static void test_roles() {
    for (projector_type kind : {PROJECTOR_TYPE_MLP, PROJECTOR_TYPE_MLP_NORM, PROJECTOR_TYPE_QWEN2VL,
                                PROJECTOR_TYPE_QWEN25VL, PROJECTOR_TYPE_QWEN3VL, PROJECTOR_TYPE_LFM2,
                                PROJECTOR_TYPE_LFM3VL, PROJECTOR_TYPE_GEMMA3, PROJECTOR_TYPE_GEMMA4V}) {
        fixture f;
        clip_model m{};
        m.proj_type = kind;
        m.layers.resize(1);
        m.layers[0].q_w = ggml_new_tensor_2d(f.weights, GGML_TYPE_BF16, 16, 16);
        lcb_mtmd_check_retained_roles(m, f.weights);
        ++checks;
        // A custom BF16 norm is not made valid by its filename or rank.
        m.layers[0].ln_1_w = ggml_new_tensor_1d(f.weights, GGML_TYPE_BF16, 16);
        rejected([&] { lcb_mtmd_check_retained_roles(m, f.weights); }, "role");
    }
    for (ggml_type type : {GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_BF16}) {
        fixture f;
        clip_model m{};
        m.proj_type = PROJECTOR_TYPE_GEMMA3;
        m.mm_input_proj_w = ggml_new_tensor_2d(f.weights, type, 16, 16);
        if (type == GGML_TYPE_BF16)
            rejected([&] { lcb_mtmd_check_retained_roles(m, f.weights); }, "role");
        else {
            lcb_mtmd_check_retained_roles(m, f.weights);
            ++checks;
        }
    }
    {
        fixture f;
        clip_model m{};
        m.proj_type = PROJECTOR_TYPE_GEMMA4V;
        m.position_embeddings = ggml_new_tensor_3d(f.weights, GGML_TYPE_BF16, 16, 8, 2);
        lcb_mtmd_check_retained_roles(m, f.weights);
        ++checks;
        m.proj_type = PROJECTOR_TYPE_LFM2;
        rejected([&] { lcb_mtmd_check_retained_roles(m, f.weights); }, "role");
    }
}

int main() {
    try {
        test_reader();
        test_queue();
        test_matmul();
        test_rows();
        test_rejections();
        test_roles();
        std::cout << "PASS: " << checks << " checks; real ggml CPU contract/numerical tests, not GPU acceptance\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "FAIL after " << checks << " checks: " << e.what() << '\n';
        return 1;
    }
}
