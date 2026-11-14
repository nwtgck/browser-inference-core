// Small synthetic graphs on the requested REAL backend. Optional browser smoke
// runs this on WebGPU; native runs use CPU only. No model files or speed claims.
#include "browser-probes.h"
#include "image-performance-test-utils.h"
#include "core/ggml_extend.h"
#include "core/runner_cache.h"
#include "model/common/rope.hpp"
#include "model/diffusion/qwen_image_2_1.hpp"
#include "ggml-backend-impl.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#ifdef __EMSCRIPTEN__
#include "ggml-webgpu.h"
#endif

namespace {
using namespace image_test;

// Query the real registry while parameter/graph buffers are still alive.
// This is meaningful only on the optional real WebGPU path, not CPU smoke.
void verify_live_registry(ggml_backend_t backend) {
#ifdef __EMSCRIPTEN__
    if (!ggml_backend_is_cpu(backend)) {
        auto device = ggml_backend_get_device(backend);
        auto reg = ggml_backend_dev_backend_reg(device);
        require(reg != nullptr, "backend has no registry");
        for (int i = 0; i < 3; ++i) {
            require(ggml_backend_webgpu_reg() == reg, "registry changed across lookup");
            require(ggml_backend_reg_dev_count(reg) == 1, "registry lost its device count");
            require(ggml_backend_reg_dev_get(reg, 0) == device, "registry replaced the device");
        }
    }
#else
    (void)backend;
#endif
}

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
        device.iface.supports_op = [](ggml_backend_dev_t dev, const ggml_tensor * node) {
            auto & self = *static_cast<ConstructionBudget *>(dev->context);
            if ((node->op == GGML_OP_IM2COL || node->op == GGML_OP_MUL_MAT) && ggml_nbytes(node) > 32768)
                return false;
            return ggml_backend_supports_op(self.real, node);
        };
        backend.device = &device;
    }
};

bool is_compute(const ggml_tensor * node) {
    return node->op != GGML_OP_NONE && node->op != GGML_OP_VIEW && node->op != GGML_OP_RESHAPE &&
           node->op != GGML_OP_PERMUTE && node->op != GGML_OP_TRANSPOSE;
}

std::vector<float> run_graph(ggml_backend_t backend, bool convolution, bool bounded, ggml_type type, bool sink,
                             bool masked) {
    sd::RunnerCache cache(backend);
    std::vector<float> result;
    {
        auto params = context();
        auto computation = context();
        ConstructionBudget budget(backend);
        auto selected = bounded ? &budget.backend : backend;
        auto runner_context = runner(computation.get(), selected);
        ggml_tensor * input;
        ggml_tensor * output;
        Buffer parameters(nullptr, ggml_backend_buffer_free);
        if (convolution) {
            input = ggml_new_tensor_4d(params.get(), GGML_TYPE_F32, 32, 32, 4, 1);
            auto weight = ggml_new_tensor_4d(params.get(), type, 3, 3, 4, 4);
            auto bias = ggml_new_tensor_1d(params.get(), GGML_TYPE_F32, 4);
            parameters.reset(ggml_backend_alloc_ctx_tensors(params.get(), backend));
            require(parameters != nullptr, "convolution parameter allocation");
            write(weight, values(ggml_nelements(weight), 7, 128));
            write(bias, values(4, 3, 128));
            output = ggml_ext_conv_2d(&runner_context, input, weight, bias, 1, 1, 1, 1, 1, 1, false, false, false, 1.f);
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
            if (sinks)
                write(sinks, std::vector<float>(heads, std::log(435.f)));
            if (mask) {
                auto values = image_test::values(ggml_nelements(mask), 3, 8);
                for (size_t i = 0; i < values.size(); ++i)
                    if (i % keys == 3)
                        values[i] = -INFINITY;
                write(mask, values);
            }
            output =
                ggml_ext_attention_ext(&runner_context, input, k, v, heads, mask, false, false, 1.f, nullptr, sinks);
        }
        auto graph = image_test::graph(computation.get(), output);
        require((count(graph, GGML_OP_SET) != 0) == bounded, "incorrect bounded-path selection");
        if (bounded) {
            for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                auto node = ggml_graph_node(graph, i);
                require(!is_compute(node) || ggml_backend_supports_op(backend, node),
                        "real backend rejected bounded node");
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
        auto part = ggml_scale_bias(computation.get(), input, 0.5f + float(i + 1) / 257.f, float(i % 17 - 8) / 32.f);
        output = ggml_set_inplace(computation.get(), output, part, output->nb[1], output->nb[2], output->nb[3],
                                  size_t(i) * rows * output->nb[1]);
    }
    auto graph = image_test::graph(computation.get(), output);
    require(count(graph, GGML_OP_SCALE) == branches && count(graph, GGML_OP_SET) == branches,
            "parameter batch graph lost an observable dispatch");
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto node = ggml_graph_node(graph, i);
        require(!is_compute(node) || ggml_backend_supports_op(backend, node), "backend rejected parameter batch node");
    }
    auto allocation = allocate(backend, graph);
    std::vector<float> result;
    for (unsigned seed : {3u, 11u}) {
        write(input, values(width * rows, seed, 64));
        verify_live_registry(backend);
        compute(backend, graph);
        auto current = read(output);
        result.insert(result.end(), current.begin(), current.end());
    }
    return result;
}

// Exercise the production softmax shader around private-cache boundaries.
// The requested backend must execute the operator, never an implicit fallback.
std::vector<float> run_softmax_rows(ggml_backend_t backend, int width, bool inplace, bool masked, bool sink,
                                    float scale) {
    auto ctx = context();
    constexpr int input_offset = 7, mask_offset = 5, rows = 3, heads = 2;
    constexpr float sentinel = -991.f;
    auto source = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, input_offset + width * rows * heads + 13);
    auto input = ggml_view_3d(ctx.get(), source, width, rows, heads, width * sizeof(float),
                              width * rows * sizeof(float), input_offset * sizeof(float));
    auto mask_source = masked ? ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F16, mask_offset + width * rows + 19) : nullptr;
    auto mask = masked ? ggml_view_2d(ctx.get(), mask_source, width, rows, width * sizeof(ggml_fp16_t),
                                      mask_offset * sizeof(ggml_fp16_t))
                       : nullptr;
    auto sinks = sink ? ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, heads) : nullptr;
    auto before = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 32);
    auto output = inplace ? ggml_soft_max_ext_inplace(ctx.get(), input, mask, scale, 0.f)
                          : ggml_soft_max_ext(ctx.get(), input, mask, scale, 0.f);
    auto after = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 32);
    if (sinks)
        ggml_soft_max_add_sinks(output, sinks);
    require(ggml_backend_supports_op(backend, output), "backend rejected softmax cache probe");
    auto graph = image_test::graph(ctx.get(), output);
    Buffer allocation(ggml_backend_alloc_ctx_tensors(ctx.get(), backend), ggml_backend_buffer_free);
    require(allocation != nullptr, "softmax parameter/output allocation");
    const std::vector<float> sink_values{std::log(435.f), std::log(17.f)};
    if (sinks)
        write(sinks, sink_values);
    std::vector<float> result;
    for (unsigned seed : {0u, 23u}) {
        std::vector<float> source_values(ggml_nelements(source), sentinel);
        std::vector<float> mask_values(masked ? ggml_nelements(mask_source) : 0, sentinel);
        for (int row = 0; row < rows; ++row)
            for (int column = 0; column < width; ++column) {
                const auto i = unsigned(column) + seed;
                // Three independent patterns exercise stable exponentiation,
                // equal logits and cache spill with the same real dispatch.
                const float value = row == 0   ? float(int((i * 37) % 113) - 56) / 19.f
                                    : row == 1 ? float(int(i % 7) - 3) * 50.f
                                               : 0.f;
                for (int head = 0; head < heads; ++head)
                    source_values[input_offset + (head * rows + row) * width + column] = value;
                if (masked)
                    mask_values[mask_offset + row * width + column] =
                        column % 7 == 3 ? -INFINITY : float(int(i % 11) - 5) / 4.f;
            }
        write(source, source_values);
        if (mask_source)
            write(mask_source, mask_values);
        write(before, std::vector<float>(32, sentinel));
        write(after, std::vector<float>(32, sentinel));
        compute(backend, graph);
        auto current = read(output);
        // The oracle uses a scalar double reduction and the authored inputs;
        // it never translates or executes a second copy of the WGSL operator.
        for (int head = 0; head < heads; ++head)
            for (int row = 0; row < rows; ++row) {
                std::vector<double> logits(width);
                double maximum = sink ? sink_values[head] : -INFINITY;
                for (int column = 0; column < width; ++column) {
                    logits[column] = float(source_values[input_offset + (head * rows + row) * width + column] * scale +
                                           (masked ? mask_values[mask_offset + row * width + column] : 0.f));
                    maximum = std::max(maximum, logits[column]);
                }
                double denominator = sink ? std::exp(sink_values[head] - maximum) : 0;
                for (const auto value : logits)
                    denominator += std::exp(value - maximum);
                for (int column = 0; column < width; ++column) {
                    const double expected = std::exp(logits[column] - maximum) / denominator;
                    const float actual = current[(head * rows + row) * width + column];
                    require(std::isfinite(actual) && std::abs(actual - expected) <= 2e-7 + 2e-5 * expected,
                            "softmax scalar oracle/cache boundary");
                }
            }
        const auto observed_source = read(source);
        for (size_t i = 0; i < source_values.size(); ++i)
            if (!inplace || i < input_offset || i >= input_offset + ggml_nelements(input))
                require(observed_source[i] == source_values[i], "softmax source offset/guard");
        if (mask_source) {
            std::vector<ggml_fp16_t> observed(mask_values.size());
            ggml_backend_tensor_get(mask_source, observed.data(), 0, observed.size() * sizeof(ggml_fp16_t));
            for (size_t i = 0; i < observed.size(); ++i)
                require(observed[i] == ggml_fp32_to_fp16(mask_values[i]), "softmax mask/guard was modified");
        }
        for (auto guard : {before, after})
            close(std::vector<float>(32, sentinel), read(guard));
        result.insert(result.end(), current.begin(), current.end());
    }
    return result;
}

std::vector<float> run_shared_rope(ggml_backend_t backend, bool shared) {
    auto params = context(), computation = context();
    auto input = ggml_new_tensor_4d(params.get(), GGML_TYPE_F32, 32, 2, 17, 1);
    auto pe = ggml_new_tensor_4d(params.get(), GGML_TYPE_F32, 2, 2, 16, 17);
    Buffer parameters(ggml_backend_alloc_ctx_tensors(params.get(), backend), ggml_backend_buffer_free);
    require(parameters != nullptr, "RoPE parameter allocation");
    auto runner_context = runner(computation.get(), backend);
    const auto prepared = Rope::prepare_matrix(computation.get(), pe);
    auto apply = [&](ggml_tensor * x) {
        return shared ? Rope::apply_rope_prepared(computation.get(), x, prepared)
                      : Rope::apply_rope(&runner_context, x, pe);
    };
    auto a = apply(input);
    auto b = apply(ggml_scale(computation.get(), input, .75f));
    auto output = ggml_concat(computation.get(), a, b, 1);
    auto graph = image_test::graph(computation.get(), output);
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto node = ggml_graph_node(graph, i);
        require(!is_compute(node) || ggml_backend_supports_op(backend, node), "backend rejected shared RoPE");
    }
    auto allocation = allocate(backend, graph);
    std::vector<float> result;
    for (unsigned seed : {5u, 31u}) {
        write(input, values(ggml_nelements(input), seed, 16));
        write(pe, values(ggml_nelements(pe), seed + 7, 8));
        compute(backend, graph);
        auto current = read(output);
        result.insert(result.end(), current.begin(), current.end());
    }
    return result;
}

// Qwen elementwise probes use real requested-backend allocation and execution.
// The construction wrapper only selects the browser branch for the CPU test.
std::vector<float> run_qwen_elementwise(ggml_backend_t backend, bool optimized, bool packed, int prefix) {
    using Block = Qwen::QwenImage21TransformerBlock;
    auto params = context(), computation = context();
    constexpr int width = 64, tokens = 9, inner = 96;
    auto input = ggml_new_tensor_2d(params.get(), GGML_TYPE_F32, width, tokens);
    auto packed_input = ggml_new_tensor_2d(params.get(), GGML_TYPE_F32, 2 * inner, tokens);
    auto gate = ggml_new_tensor_2d(params.get(), GGML_TYPE_F32, inner, tokens);
    auto up = ggml_new_tensor_2d(params.get(), GGML_TYPE_F32, inner, tokens);
    std::vector<ggml_tensor *> modulation;
    for (int i = 0; i < 4; ++i)
        modulation.push_back(ggml_new_tensor_2d(params.get(), GGML_TYPE_F32, width, 2));
    Buffer parameters(ggml_backend_alloc_ctx_tensors(params.get(), backend), ggml_backend_buffer_free);
    require(parameters != nullptr, "Qwen elementwise parameter allocation");
    ConstructionBudget construction(backend);
    GGMLRunnerContext ctx;
    ctx.ggml_ctx = computation.get();
    ctx.backend = &construction.backend;
    auto prepared = optimized ? Block::prepare_modulation(&ctx, modulation, prefix == 0) : std::nullopt;
    require(!optimized || bool(prepared), "Qwen modulation support on requested backend");
    ggml_tensor * output = nullptr;
    for (int i = 0; i < 4; ++i) {
        auto part = prepared ? Block::modulate_prepared(&ctx, input, prepared->target[i], prepared->prefix[i], prefix)
                             : Block::modulate(&ctx, input, modulation[i], prefix, i % 2 == 1);
        part = ggml_reshape_1d(computation.get(), part, ggml_nelements(part));
        output = output ? ggml_concat(computation.get(), output, part, 0) : part;
    }
    ggml_tensor * activated =
        optimized ? Block::try_webgpu_swiglu(&ctx, packed ? packed_input : gate, packed ? nullptr : up) : nullptr;
    require(!optimized || activated != nullptr, "Qwen GLU support on requested backend");
    if (!activated) {
        if (packed) {
            auto parts = ggml_ext_chunk(&ctx, packed_input, 2, 0);
            activated = ggml_mul(computation.get(), parts[1], ggml_silu(computation.get(), parts[0]));
        } else {
            activated = ggml_mul(computation.get(), up, ggml_silu(computation.get(), gate));
        }
    }
    output = ggml_concat(computation.get(), output,
                         ggml_reshape_1d(computation.get(), activated, ggml_nelements(activated)), 0);
    auto g = image_test::graph(computation.get(), output);
    require(count(g, GGML_OP_GLU) == int(optimized), "Qwen GLU was not selected");
    for (int i = 0; i < ggml_graph_n_nodes(g); ++i) {
        auto n = ggml_graph_node(g, i);
        require(!is_compute(n) || ggml_backend_supports_op(backend, n), "requested backend rejected Qwen probe node");
    }
    auto allocation = allocate(backend, g);
    std::vector<float> result;
    for (unsigned seed : {5u, 23u}) {
        auto original = values(ggml_nelements(input), seed, 16);
        write(input, original);
        write(packed_input, values(ggml_nelements(packed_input), seed + 2, 3));
        write(gate, values(ggml_nelements(gate), seed + 3, 3));
        write(up, values(ggml_nelements(up), seed + 4, 5));
        for (int i = 0; i < 4; ++i)
            write(modulation[i], values(ggml_nelements(modulation[i]), seed + i, 8));
        compute(backend, g);
        auto current = read(output);
        result.insert(result.end(), current.begin(), current.end());
        close(original, read(input));
    }
    return result;
}

// Cross the WebGPU per-axis group limit with small rows, not model weights.
// 257*257 = 66049 rows covers a padding group and different logical heads.
std::vector<float> run_large_rows(ggml_backend_t backend, int kind, bool inplace) {
    auto params = context(), computation = context();
    auto input = ggml_new_tensor_3d(params.get(), GGML_TYPE_F32, 8, 257, 257);
    auto mul = ggml_new_tensor_1d(params.get(), GGML_TYPE_F32, 8);
    auto mask = ggml_new_tensor_2d(params.get(), GGML_TYPE_F32, 8, 257);
    auto sinks = ggml_new_tensor_1d(params.get(), GGML_TYPE_F32, 257);
    Buffer parameters(ggml_backend_alloc_ctx_tensors(params.get(), backend), ggml_backend_buffer_free);
    require(parameters != nullptr, "large-row parameters");
    write(mul, values(8, 11, 16));
    write(mask, values(8 * 257, 7, 8));
    write(sinks, values(257, 19, 8));
    ggml_tensor * output;
    if (kind == 0) {
        output =
            inplace ? ggml_norm_inplace(computation.get(), input, 1e-5f) : ggml_norm(computation.get(), input, 1e-5f);
    } else if (kind == 1 || kind == 2) {
        output = inplace ? ggml_rms_norm_inplace(computation.get(), input, 1e-5f)
                         : ggml_rms_norm(computation.get(), input, 1e-5f);
        if (kind == 2)
            output = ggml_mul(computation.get(), output, mul);
    } else {
        output = inplace ? ggml_soft_max_ext_inplace(computation.get(), input, mask, 0.125f, 0.f)
                         : ggml_soft_max_ext(computation.get(), input, mask, 0.125f, 0.f);
        ggml_soft_max_add_sinks(output, sinks);
    }
    auto g = image_test::graph(computation.get(), output);
    for (int i = 0; i < ggml_graph_n_nodes(g); ++i) {
        auto node = ggml_graph_node(g, i);
        require(!is_compute(node) || ggml_backend_supports_op(backend, node), "large-row backend rejection");
    }
    auto allocation = allocate(backend, g);
    std::vector<float> result;
    for (unsigned seed : {3u, 31u}) {
        write(input, values(ggml_nelements(input), seed, 4));
        verify_live_registry(backend);
        compute(backend, g);
        auto current = read(output);
        result.insert(result.end(), current.begin(), current.end());
    }
    return result;
}

} // namespace

extern "C" int sdc_test_webgpu_performance(const char * backend_name) {
    try {
        Backend backend(std::strcmp(backend_name, "CPU") == 0 ? ggml_backend_cpu_init()
                                                              : ggml_backend_init_by_name(backend_name, nullptr),
                        ggml_backend_free);
        if (!backend)
            return -1;
        if (ggml_backend_is_cpu(backend.get()))
            ggml_backend_cpu_set_n_threads(backend.get(), 1);
        auto reference = cpu();
        for (auto type : {GGML_TYPE_F16, GGML_TYPE_F32}) {
            auto expected = run_graph(reference.get(), true, false, type, false, false);
            auto actual = run_graph(backend.get(), true, true, type, false, false);
            close(expected, actual, 2e-4f, 2e-3f);
        }
        for (bool sink : {false, true})
            for (bool masked : {false, true}) {
                auto expected = run_graph(reference.get(), false, false, GGML_TYPE_F32, sink, masked);
                auto actual = run_graph(backend.get(), false, true, GGML_TYPE_F32, sink, masked);
                close(expected, actual, 2e-4f, 2e-3f);
            }
        for (bool packed : {false, true})
            for (int prefix : {0, 3}) {
                auto expected = run_qwen_elementwise(reference.get(), false, packed, prefix);
                auto actual = run_qwen_elementwise(backend.get(), true, packed, prefix);
                close(expected, actual, 3e-5f, 3e-4f);
            }
        for (int branches : {1, 32, 33, 97}) {
            auto expected = run_parameter_batches(reference.get(), branches);
            auto actual = run_parameter_batches(backend.get(), branches);
            close(expected, actual, 2e-5f, 2e-4f);
        }
        for (int width : {1,   15,  16,  17,   31,   32,   33,   63,   64,   65,    255,   256,   257,
                          511, 512, 513, 1023, 1024, 1025, 4095, 4096, 4097, 16383, 16384, 16385, 32771}) {
            for (bool inplace : {false, true})
                for (bool masked : {false, true})
                    for (bool sink : {false, true})
                        for (float scale : {1.f, .0883883476f}) {
                            auto expected = run_softmax_rows(reference.get(), width, inplace, masked, sink, scale);
                            auto actual = run_softmax_rows(backend.get(), width, inplace, masked, sink, scale);
                            close(expected, actual, 3e-6f, 3e-4f);
                        }
        }
        close(run_shared_rope(reference.get(), false), run_shared_rope(backend.get(), true), 2e-5f, 2e-4f);
        for (int kind = 0; kind < 4; ++kind)
            for (bool inplace : {false, true}) {
                close(run_large_rows(reference.get(), kind, inplace), run_large_rows(backend.get(), kind, inplace),
                      3e-5f, 3e-4f);
            }
        return 1;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "WebGPU performance arithmetic/copy probe: %s\n", error.what());
        return -2;
    }
}
