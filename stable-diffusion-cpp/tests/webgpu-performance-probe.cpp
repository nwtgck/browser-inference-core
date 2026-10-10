// Small synthetic graphs on the requested REAL backend. Optional browser smoke
// runs this on WebGPU; native runs use CPU only. No model files or speed claims.
#include "image-performance-test-utils.h"
#include "core/ggml_extend.h"
#include "core/runner_cache.h"
#include "ggml-backend-impl.h"
#include <cstdio>

namespace {
using namespace image_test;

// Force the bounded branch with small tensors, then use the real backend for
// every allocation, operation support check, shader dispatch and copy. This
// filter is a construction-only limit, not an execution backend.
struct ConstructionBudget {
    ggml_backend_t real;
    ggml_backend_device device{};
    ggml_backend backend;
    explicit ConstructionBudget(ggml_backend_t source) : real(source), backend(*source) {
        device.context = this;
        device.iface.get_name = [](ggml_backend_dev_t) { return "WebGPU-construction-budget"; };
        device.iface.supports_op = [](ggml_backend_dev_t dev, const ggml_tensor* node) {
            auto& self = *static_cast<ConstructionBudget*>(dev->context);
            if ((node->op == GGML_OP_IM2COL || node->op == GGML_OP_MUL_MAT) && ggml_nbytes(node) > 32768) return false;
            return ggml_backend_supports_op(self.real, node);
        };
        backend.device = &device;
    }
};

bool is_compute(const ggml_tensor* node) {
    return node->op != GGML_OP_NONE && node->op != GGML_OP_VIEW && node->op != GGML_OP_RESHAPE &&
           node->op != GGML_OP_PERMUTE && node->op != GGML_OP_TRANSPOSE;
}

std::vector<float> run_graph(ggml_backend_t backend, bool convolution, bool bounded,
                             ggml_type type, bool sink, bool masked) {
    sd::RunnerCache cache(backend);
    std::vector<float> result;
    {
        auto params = context();
        auto computation = context();
        ConstructionBudget budget(backend);
        auto selected = bounded ? &budget.backend : backend;
        ggml_tensor* input;
        ggml_tensor* output;
        Buffer parameters(nullptr, ggml_backend_buffer_free);
        if (convolution) {
            input = ggml_new_tensor_4d(params.get(), GGML_TYPE_F32, 32, 32, 4, 1);
            auto weight = ggml_new_tensor_4d(params.get(), type, 3, 3, 4, 4);
            auto bias = ggml_new_tensor_1d(params.get(), GGML_TYPE_F32, 4);
            parameters.reset(ggml_backend_alloc_ctx_tensors(params.get(), backend));
            require(parameters != nullptr, "convolution parameter allocation");
            write(weight, values(ggml_nelements(weight), 7, 128));
            write(bias, values(4, 3, 128));
            output = ggml_ext_conv_2d(computation.get(), input, weight, bias,
                                       1, 1, 1, 1, 1, 1, false, false, false, 1.f, selected);
        } else {
            constexpr int width = 16, heads = 2, queries = 65, keys = 97;
            input = ggml_new_tensor_3d(params.get(), GGML_TYPE_F32, width, queries, 1);
            auto k = ggml_new_tensor_3d(params.get(), GGML_TYPE_F32, width, keys, 1);
            auto v = ggml_new_tensor_3d(params.get(), GGML_TYPE_F32, width, keys, 1);
            auto sinks = sink ? ggml_new_tensor_1d(params.get(), GGML_TYPE_F32, heads) : nullptr;
            auto mask = masked ? ggml_new_tensor_2d(params.get(), GGML_TYPE_F32, keys, queries) : nullptr;
            parameters.reset(ggml_backend_alloc_ctx_tensors(params.get(), backend));
            require(parameters != nullptr, "attention parameter allocation");
            write(k, values(ggml_nelements(k), 12, 32));
            write(v, values(ggml_nelements(v), 13, 32));
            if (sinks) write(sinks, std::vector<float>(heads, std::log(435.f)));
            if (mask) {
                auto values = image_test::values(ggml_nelements(mask), 3, 8);
                for (size_t i = 0; i < values.size(); ++i) if (i % keys == 3) values[i] = -INFINITY;
                write(mask, values);
            }
            output = ggml_ext_attention_ext(computation.get(), selected, input, k, v, heads, mask,
                                             false, false, 1.f, false, sinks);
        }
        auto graph = image_test::graph(computation.get(), output);
        require((count(graph, GGML_OP_SET) != 0) == bounded, "incorrect bounded-path selection");
        if (bounded) {
            for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                auto node = ggml_graph_node(graph, i);
                require(!is_compute(node) || ggml_backend_supports_op(backend, node), "real backend rejected bounded node");
            }
        }
        auto allocation = allocate(backend, graph);
        for (unsigned iteration : {1u, 2u}) {
            write(input, values(ggml_nelements(input), iteration, 64));
            compute(backend, graph);
            result = read(output);
            cache.stage("output", output);
            require(cache.capture(graph) == GGML_STATUS_SUCCESS, "cache capture");
            cache.graph_end(true);
            close(result, read(cache.get("output")));
        }
    }
    // Source weights, views and graph allocation have been destroyed. The
    // captured output must have independent ownership on the requested backend.
    close(result, read(cache.get("output")));
    return result;
}
// Independent outputs keep every parameterized dispatch observable: unlike a
// long normalization chain, a later operation cannot cancel an earlier mistake.
// SET nodes also stop elementwise fusion from hiding submission-boundary tests.
std::vector<float> run_parameter_batches(ggml_backend_t backend, int branches) {
    auto params = context();
    auto computation = context();
    constexpr int width = 64, rows = 4;
    auto input = ggml_new_tensor_2d(params.get(), GGML_TYPE_F32, width, rows);
    Buffer parameters(ggml_backend_alloc_ctx_tensors(params.get(), backend), ggml_backend_buffer_free);
    require(parameters != nullptr, "parameter batch input allocation");
    auto output = ggml_new_tensor_2d(computation.get(), GGML_TYPE_F32, width, rows * branches);
    for (int i = 0; i < branches; ++i) {
        auto part = ggml_scale_bias(computation.get(), input,
                                    0.5f + float(i + 1) / 257.f, float(i % 17 - 8) / 32.f);
        output = ggml_set_inplace(computation.get(), output, part,
                                  output->nb[1], output->nb[2], output->nb[3],
                                  size_t(i) * rows * output->nb[1]);
    }
    auto graph = image_test::graph(computation.get(), output);
    require(count(graph, GGML_OP_SCALE) == branches && count(graph, GGML_OP_SET) == branches,
            "parameter batch graph lost an observable dispatch");
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto node = ggml_graph_node(graph, i);
        require(!is_compute(node) || ggml_backend_supports_op(backend, node),
                "backend rejected parameter batch node");
    }
    auto allocation = allocate(backend, graph);
    std::vector<float> result;
    for (unsigned seed : {3u, 11u}) {
        write(input, values(width * rows, seed, 64));
        compute(backend, graph);
        auto current = read(output);
        result.insert(result.end(), current.begin(), current.end());
    }
    return result;
}

} // namespace

extern "C" int sdc_test_webgpu_performance(const char* backend_name) {
    try {
        Backend backend(ggml_backend_init_by_name(backend_name, nullptr), ggml_backend_free);
        if (!backend) return -1;
        if (ggml_backend_is_cpu(backend.get())) ggml_backend_cpu_set_n_threads(backend.get(), 1);
        auto reference = cpu();
        for (auto type : {GGML_TYPE_F16, GGML_TYPE_F32}) {
            auto expected = run_graph(reference.get(), true, false, type, false, false);
            auto actual = run_graph(backend.get(), true, true, type, false, false);
            close(expected, actual, 2e-4f, 2e-3f);
        }
        for (bool sink : {false, true}) for (bool masked : {false, true}) {
            auto expected = run_graph(reference.get(), false, false, GGML_TYPE_F32, sink, masked);
            auto actual = run_graph(backend.get(), false, true, GGML_TYPE_F32, sink, masked);
            close(expected, actual, 2e-4f, 2e-3f);
        }
        for (int branches : {1, 32, 33, 97}) {
            auto expected = run_parameter_batches(reference.get(), branches);
            auto actual = run_parameter_batches(backend.get(), branches);
            close(expected, actual, 2e-5f, 2e-4f);
        }
        return 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "WebGPU performance arithmetic/copy probe: %s\n", error.what());
        return -2;
    }
}
