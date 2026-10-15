// Regression against explicit 512-position padding using the real Anima layer.
// Native arithmetic and capability selection only, not a GPU benchmark.
#include "image-performance-test-utils.h"
#include "model/diffusion/anima.hpp"
#include "ggml-backend-impl.h"
#include <cstdio>
#include <map>

namespace {
using namespace image_test;

struct FlashFilter {
    ggml_backend_t real;
    bool reject;
    int sink_checks = 0;
    ggml_backend_device device = {};
    ggml_backend backend;
    FlashFilter(ggml_backend_t source, bool reject_sinks) : real(source), reject(reject_sinks), backend(*source) {
        device.context = this;
        device.iface.get_name = [](ggml_backend_dev_t) { return "CPU-capability-filter"; };
        device.iface.supports_op = [](ggml_backend_dev_t device, const ggml_tensor* node) {
            auto& self = *static_cast<FlashFilter*>(device->context);
            if (node->op == GGML_OP_FLASH_ATTN_EXT && node->src[4] != nullptr) {
                ++self.sink_checks;
                if (self.reject) return false;
            }
            return ggml_backend_supports_op(self.real, node);
        };
        backend.device = &device;
    }
};

// Eligibility must not assume arbitrary adapters preserve zero K/V positions.
struct UnknownAdapter : WeightAdapter {
    ggml_tensor* patch_weight(ggml_context*, ggml_backend_t, ggml_tensor*, const std::string&) override {
        throw std::runtime_error("unexpected adapter execution");
    }
    ggml_tensor* forward_with_lora(ggml_context*, ggml_backend_t, ggml_tensor*, ggml_tensor*, ggml_tensor*,
                                  const std::string&, ForwardParams) override {
        throw std::runtime_error("unexpected adapter execution");
    }
    ggml_tensor* add_lora_to_output(ggml_context*, ggml_backend_t, ggml_tensor*, ggml_tensor*, ggml_tensor*,
                                   const std::string&, ForwardParams) override {
        throw std::runtime_error("unexpected adapter execution");
    }
    size_t get_extra_graph_size() override { return 0; }
};

ggml_tensor* builtin(ggml_context* ctx) {
    auto tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_name(tensor, "ggml_runner_build_in_tensor:one");
    ggml_set_input(tensor); // Keep the constant alive across repeated graph execution.
    return tensor;
}
void initialize_builtin(ggml_tensor* one) {
    if (one->buffer != nullptr) write(one, {1.f});
}

void run_case(ggml_backend_t cpu, int length, int batch, bool flash, bool reject_flash) {
    constexpr int heads = 2, head_dim = 64, channels = 128, context_dim = 64, queries = 3;
    auto params = context();
    auto ref_ctx = context();
    auto actual_ctx = context();
    auto naive_ctx = context();
    Anima::AnimaAttention layer(channels, context_dim, heads, head_dim);
    layer.init(params.get());
    std::map<std::string, ggml_tensor*> weights;
    layer.get_param_tensors(weights);
    auto x = ggml_new_tensor_3d(params.get(), GGML_TYPE_F32, channels, queries, batch);
    auto text = ggml_new_tensor_3d(params.get(), GGML_TYPE_F32, context_dim, length, batch);
    Buffer data(ggml_backend_alloc_ctx_tensors(params.get(), cpu), ggml_backend_buffer_free);
    require(data != nullptr, "parameter allocation failed");
    unsigned seed = 4;
    for (const auto& [name, weight] : weights) {
        auto v = values(ggml_nelements(weight), seed++, 256.f);
        if (name.find("norm") != std::string::npos) for (auto& item : v) item += 1.f;
        write(weight, v);
    }
    write(text, values(ggml_nelements(text), 21, 64.f));
    auto ref_one = builtin(ref_ctx.get());
    auto actual_one = builtin(actual_ctx.get());
    FlashFilter filter(cpu, reject_flash);
    GGMLRunnerContext ctx;
    ctx.backend = &filter.backend;
    ctx.flash_attn_enabled = flash;
    // If only sink-aware flash is rejected, compare both paths' manual kernels
    // rather than mixing manual-vs-F16-flash rounding into the regression.
    ctx.ggml_ctx = ref_ctx.get();
    ctx.flash_attn_enabled = flash && !reject_flash;
    auto reference_text = Anima::finish_text_context(&ctx, text, heads, false);
    auto reference = layer.forward(&ctx, x, reference_text.values);
    ctx.ggml_ctx = actual_ctx.get();
    ctx.flash_attn_enabled = flash;
    auto actual_text = Anima::finish_text_context(&ctx, text, heads, true);
    auto actual = layer.forward(&ctx, x, actual_text.values, nullptr, nullptr, actual_text.sinks);
    const bool compact = length < Anima::ANIMA_CONTEXT_LENGTH;
    require((actual_text.sinks != nullptr) == compact, "incorrect structural padding policy");
    require(actual_text.values->ne[1] == std::min<int64_t>(length, Anima::ANIMA_CONTEXT_LENGTH), "incorrect live length");
    auto rg = graph(ref_ctx.get(), reference);
    auto ag = graph(actual_ctx.get(), actual);
    const bool actual_flash = flash && (!reject_flash || !compact);
    require(count(ag, GGML_OP_FLASH_ATTN_EXT) == static_cast<int>(actual_flash), "unexpected attention backend fallback");
    int sinks = 0;
    for (int i = 0; i < ggml_graph_n_nodes(ag); ++i) {
        auto node = ggml_graph_node(ag, i);
        if (node->op == GGML_OP_SOFT_MAX && node->src[2] != nullptr) ++sinks;
        if (node->op == GGML_OP_FLASH_ATTN_EXT && node->src[4] != nullptr) ++sinks;
    }
    require(sinks == static_cast<int>(compact), "sink lost or attached to more than one attention");
    if (flash && compact) require(filter.sink_checks > 0, "capability check ran before attaching the sink");
    auto ra = allocate(cpu, rg);
    auto aa = allocate(cpu, ag);
    initialize_builtin(ref_one);
    initialize_builtin(actual_one);
    for (unsigned input : {1u, 2u}) {
        write(x, values(ggml_nelements(x), input, 128.f));
        initialize_builtin(ref_one);
        initialize_builtin(actual_one);
        compute(cpu, rg);
        compute(cpu, ag);
        close(read(reference), read(actual), 2e-5f, 2e-4f);
    }
    if (length == 8 && !flash) {
        ctx.ggml_ctx = naive_ctx.get();
        auto naive = layer.forward(&ctx, x, text);
        auto ng = graph(naive_ctx.get(), naive);
        auto na = allocate(cpu, ng);
        compute(cpu, ng);
        auto expected = read(reference), wrong = read(naive);
        float difference = 0.f;
        for (size_t i = 0; i < expected.size(); ++i) difference = std::max(difference, std::abs(expected[i] - wrong[i]));
        require(difference > 1e-4f, "negative control did not expose incorrect zero deletion");
    }
}

void policy(ggml_backend_t cpu) {
    auto ctx = context();
    builtin(ctx.get());
    GGMLRunnerContext runner;
    runner.backend = cpu;
    runner.ggml_ctx = ctx.get();
    auto values = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 64, 8, 1);
    runner.weight_adapter = std::make_shared<UnknownAdapter>();
    auto result = Anima::finish_text_context(&runner, values, 2, true);
    require(result.sinks == nullptr && result.values->ne[1] == 512, "arbitrary adapter must preserve physical padding");
    runner.weight_adapter.reset();
    result = Anima::finish_text_context(&runner, values, 2, false);
    require(result.sinks == nullptr && result.values->ne[1] == 512, "explicit opt-out ignored");
}
} // namespace

int main() {
    using namespace image_test;
    int cases = 0;
    try {
        auto backend = cpu();
        for (int mode = 0; mode < 3; ++mode) for (int batch : {1, 2}) {
            for (int length : {1, 8, 77, 511, 512, 513}) {
                // The special capability filter only rejects flash WITH sinks.
                // At full length use the normal flash reference too.
                run_case(backend.get(), length, batch, mode != 0, mode == 2 && length < 512);
                ++cases;
            }
        }
        policy(backend.get());
        std::printf("%d Anima attention cases passed (two queries each); padding and negative controls passed\n", cases);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "attention case %d failed: %s\n", cases, error.what());
        return 1;
    }
    return 0;
}
