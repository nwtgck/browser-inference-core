// Actual Qwen methods on deterministic synthetic data. Policy replay checks only
// admission; execution is CPU, not WebGPU throughput or trained-model quality.
#include "image-performance-test-utils.h"
#include "qwen21-block-reference.h"
#include "webgpu-policy-fixture.h"
#include "core/segment_graph_bindings.h"
#include "core/runner_cache.h"
#include <iostream>
#include <map>
#include <set>
#include <unordered_set>
using namespace image_test;
using Block = Qwen::QwenImage21TransformerBlock;
namespace {
struct Policy {
    ggml_backend_device device{};
    ggml_backend backend{};
    ggml_backend_t real;
    ggml_op rejected = GGML_OP_NONE;
    bool named_webgpu = true;
    explicit Policy(ggml_backend_t cpu) : real(cpu) {
        device.context = this;
        device.iface.get_name = [](ggml_backend_dev_t d) {
            return static_cast<Policy *>(d->context)->named_webgpu ? "WebGPU-Qwen-test" : "CPU-policy";
        };
        device.iface.supports_op = [](ggml_backend_dev_t d, const ggml_tensor * t) {
            auto & self = *static_cast<Policy *>(d->context);
            return t->op != self.rejected && webgpu_test_supports(t, {4294967292ull, 32768, true, false});
        };
        backend.device = &device;
    }
};
int cases = 0;

void activation(ggml_backend_t cpu, int width, int rows, int batch, bool packed, bool strided, bool rejected) {
    auto data = context(), before = context(), after = context();
    auto source = ggml_new_tensor_4d(data.get(), GGML_TYPE_F32, width * 2, rows * (strided ? 2 : 1), batch, 1);
    Buffer buffer(ggml_backend_alloc_ctx_tensors(data.get(), cpu), ggml_backend_buffer_free);
    require(buffer != nullptr, "activation input allocation");
    auto part =
        ggml_view_3d(data.get(), source, width * 2, rows, batch, source->nb[1] * (strided ? 2 : 1), source->nb[2], 0);
    auto a = ggml_view_3d(data.get(), part, width, rows, batch, part->nb[1], part->nb[2], 0);
    auto b = ggml_view_3d(data.get(), part, width, rows, batch, part->nb[1], part->nb[2], width * sizeof(float));
    // A packed projection has contiguous rows. The strided case exercises the
    // separate-projection fallback instead of assuming GLU supports every view.
    auto gate = ggml_cont(before.get(), a);
    auto up = ggml_cont(before.get(), b);
    auto expected = ggml_mul(before.get(), up, ggml_silu(before.get(), gate));
    Policy policy(cpu);
    if (rejected)
        policy.rejected = GGML_OP_GLU;
    GGMLRunnerContext ctx;
    ctx.ggml_ctx = after.get();
    ctx.backend = &policy.backend;
    ggml_tensor * actual;
    if (packed) {
        actual = Block::try_webgpu_swiglu(&ctx, part);
    } else {
        actual = Block::try_webgpu_swiglu(&ctx, a, b);
    }
    // Contiguous_1 requires nb[1] == row length in the pinned GLU constructor.
    const bool eligible = packed ? ggml_is_contiguous_1(part) : (ggml_is_contiguous_1(a) && ggml_is_contiguous_1(b));
    require((actual != nullptr) == (eligible && !rejected), "wrong GLU layout/admission guard");
    if (!actual)
        actual = ggml_mul(after.get(), ggml_cont(after.get(), b), ggml_silu(after.get(), ggml_cont(after.get(), a)));
    auto rg = graph(before.get(), expected), ag = graph(after.get(), actual);
    require(count(ag, GGML_OP_GLU) == int(eligible && !rejected), "unexpected GLU selection");
    auto ra = allocate(cpu, rg), aa = allocate(cpu, ag);
    for (unsigned seed : {1u, 17u}) {
        auto values = image_test::values(ggml_nelements(source), seed, 3.f);
        if (values.size() > 8) {
            values[0] = -0.f;
            values[1] = 0.f;
            values[2] = -80.f;
            values[3] = 80.f;
        }
        write(source, values);
        compute(cpu, rg);
        compute(cpu, ag);
        close(read(expected), read(actual), 2e-6f, 2e-5f);
        close(values, read(source)); // Borrowed operands are not modified.
    }
    ++cases;
}

void separated_activation(ggml_backend_t cpu, int width, int rows, int batch, bool reject) {
    auto data = context(), before = context(), after = context();
    auto gate = ggml_new_tensor_3d(data.get(), GGML_TYPE_F32, width, rows, batch);
    auto up = ggml_new_tensor_3d(data.get(), GGML_TYPE_F32, width, rows, batch);
    Buffer buffer(ggml_backend_alloc_ctx_tensors(data.get(), cpu), ggml_backend_buffer_free);
    require(buffer != nullptr, "separate activation allocation");
    Policy p(cpu);
    if (reject)
        p.rejected = GGML_OP_GLU;
    GGMLRunnerContext ctx;
    ctx.ggml_ctx = after.get();
    ctx.backend = &p.backend;
    auto actual = Block::try_webgpu_swiglu(&ctx, gate, up);
    require((actual != nullptr) == !reject, "separate GLU admission");
    if (!actual)
        actual = ggml_mul(after.get(), up, ggml_silu(after.get(), gate));
    auto expected = ggml_mul(before.get(), up, ggml_silu(before.get(), gate));
    auto rg = graph(before.get(), expected), ag = graph(after.get(), actual);
    auto ra = allocate(cpu, rg), aa = allocate(cpu, ag);
    for (unsigned seed : {2u, 21u}) {
        write(gate, values(ggml_nelements(gate), seed, 3));
        write(up, values(ggml_nelements(up), seed + 8, 5));
        compute(cpu, rg);
        compute(cpu, ag);
        close(read(expected), read(actual), 2e-6f, 2e-5f);
    }
    ++cases;
}

void modulation(ggml_backend_t cpu, int width, int tokens, int prefix, int consumers, bool strided, ggml_op reject) {
    auto data = context(), before = context(), after = context();
    auto input = ggml_new_tensor_3d(data.get(), GGML_TYPE_F32, width, tokens * (strided ? 2 : 1), 1);
    std::vector<ggml_tensor *> params;
    for (int i = 0; i < 4; ++i)
        params.push_back(ggml_new_tensor_2d(data.get(), GGML_TYPE_F32, width, 2));
    Buffer buffer(ggml_backend_alloc_ctx_tensors(data.get(), cpu), ggml_backend_buffer_free);
    auto x = strided ? ggml_view_3d(data.get(), input, width, tokens, 1, 2 * input->nb[1], input->nb[2], 0) : input;
    Policy policy(cpu);
    policy.rejected = reject;
    GGMLRunnerContext ctx;
    ctx.ggml_ctx = after.get();
    ctx.backend = &policy.backend;
    auto prepared = Block::prepare_modulation(&ctx, params, prefix == 0);
    require(bool(prepared) == (reject == GGML_OP_NONE), "modulation support rejection lost");
    ggml_tensor * expected = nullptr;
    ggml_tensor * actual = nullptr;
    for (int n = 0; n < consumers; ++n) {
        int i = n % 4;
        auto reference = qwen21_reference::ReferenceQwenImage21TransformerBlock::modulate(before.get(), x, params[i],
                                                                                          prefix, i % 2 == 1);
        auto result = prepared
                          ? Block::modulate_prepared(after.get(), x, prepared->target[i], prepared->prefix[i], prefix)
                          : Block::modulate(after.get(), x, params[i], prefix, i % 2 == 1);
        expected = expected ? ggml_concat(before.get(), expected, reference, 1) : reference;
        actual = actual ? ggml_concat(after.get(), actual, result, 1) : result;
        if (prepared && prefix == 0 && !strided) {
            require(result->op == GGML_OP_MUL && result->view_src == nullptr && result->src[0] == x,
                    "full-span copy removal changed out-of-place MUL");
        }
    }
    auto rg = graph(before.get(), expected), ag = graph(after.get(), actual);
    auto ra = allocate(cpu, rg), aa = allocate(cpu, ag);
    for (unsigned seed : {1u, 11u}) {
        auto original = values(ggml_nelements(input), seed, 8);
        write(input, original);
        for (int i = 0; i < 4; ++i)
            write(params[i], values(ggml_nelements(params[i]), seed + i, 3));
        compute(cpu, rg);
        compute(cpu, ag);
        close(read(expected), read(actual));
        close(original, read(input));
    }
    ++cases;
}

void block(ggml_backend_t cpu, bool packed_weights, bool prefix_reuse, bool shared, bool reject_glu) {
    auto parameters = context(), before = context(), after = context();
    Qwen::QwenImage21Config c;
    c.hidden_size = 32;
    c.context_dim = 32;
    c.head_dim = 8;
    c.intermediate_size = 48;
    c.fused_mlp = packed_weights;
    qwen21_reference::ReferenceQwenImage21TransformerBlock reference(c);
    Block candidate(c);
    reference.init(parameters.get());
    candidate.init(parameters.get());
    std::map<std::string, ggml_tensor *> rw, aw;
    reference.get_param_tensors(rw);
    candidate.get_param_tensors(aw);
    auto input = ggml_new_tensor_3d(parameters.get(), GGML_TYPE_F32, 32, prefix_reuse ? 4 : 9, 1);
    std::vector<ggml_tensor *> mod;
    for (int i = 0; i < 4; ++i)
        mod.push_back(ggml_new_tensor_2d(parameters.get(), GGML_TYPE_F32, 32, 2));
    auto pe = ggml_new_tensor_4d(parameters.get(), GGML_TYPE_F32, 2, 2, 4, input->ne[1]);
    auto mask = ggml_new_tensor_2d(parameters.get(), GGML_TYPE_F32, 5, 5);
    auto pk = ggml_new_tensor_4d(parameters.get(), GGML_TYPE_F32, 8, 5, 4, 1);
    auto pv = ggml_new_tensor_4d(parameters.get(), GGML_TYPE_F32, 8, 4, 5, 1);
    Buffer buffer(ggml_backend_alloc_ctx_tensors(parameters.get(), cpu), ggml_backend_buffer_free);
    require(buffer != nullptr && rw.size() == aw.size(), "block parameters");
    unsigned seed = 5;
    for (auto & [name, tensor] : rw) {
        auto data = values(ggml_nelements(tensor), seed++, 128);
        if (name.find("norm") != std::string::npos)
            for (auto & x : data)
                x += 1.f;
        write(tensor, data);
        write(aw.at(name), data);
    }
    write(pe, values(ggml_nelements(pe), 7, 16));
    write(mask, std::vector<float>(25, 0));
    write(pk, values(ggml_nelements(pk), 9, 32));
    write(pv, values(ggml_nelements(pv), 11, 32));
    auto layout = Qwen::QwenImage21Layout::build(5, {}, {{2, 2}});
    Qwen::QwenImage21PrefixCache cache;
    cache.mode = prefix_reuse ? Qwen::QwenImage21PrefixCache::Mode::REUSE : Qwen::QwenImage21PrefixCache::Mode::NONE;
    cache.prefix_length = 5;
    GGMLRunnerContext rc, ac;
    rc.ggml_ctx = before.get();
    rc.backend = cpu;
    Policy p(cpu);
    if (reject_glu)
        p.rejected = GGML_OP_GLU;
    ac.ggml_ctx = after.get();
    ac.backend = &p.backend;
    auto load = [&](const std::string & n) { return n.back() == 'k' ? pk : pv; };
    rc.get_cache_tensor = load;
    ac.get_cache_tensor = load;
    auto prepared = shared ? Block::prepare_modulation(&ac, mod, prefix_reuse) : std::nullopt;
    auto expected = reference.forward(&rc, input, mod, pe, layout, {mask, nullptr}, cache);
    auto actual = candidate.forward(&ac, input, mod, pe, layout, {mask, nullptr}, cache, nullptr,
                                    prepared ? &*prepared : nullptr);
    auto rg = graph(before.get(), expected), ag = graph(after.get(), actual);
    require(count(ag, GGML_OP_GLU) == int(!reject_glu), "whole block GLU selection");
    auto ra = allocate(cpu, rg), aa = allocate(cpu, ag);
    for (unsigned it : {1u, 9u}) {
        write(input, values(ggml_nelements(input), it, 16));
        for (int i = 0; i < 4; ++i)
            write(mod[i], values(ggml_nelements(mod[i]), i + it, 32));
        compute(cpu, rg);
        compute(cpu, ag);
        close(read(expected), read(actual), 3e-6f, 3e-5f);
    }
    ++cases;
}

void option_parse(ggml_backend_t cpu) {
    for (const auto & argument :
         std::vector<std::pair<const char *, bool>>{{"qwen_image_2_1_webgpu_work=false", false},
                                                    {"qwen_image_2_1_webgpu_work=true", true},
                                                    {"qwen_image_2_1_webgpu_work=invalid", true},
                                                    {"qwen_image_2_1_prefix_cache=false", true}}) {
        Qwen::QwenImage21Runner runner(cpu, {}, "test", nullptr, argument.first);
        require(runner.webgpu_work_enabled == argument.second, "Qwen work option parse");
        ++cases;
    }
}

void guards(ggml_backend_t cpu) {
    auto data = context();
    Policy p(cpu);
    GGMLRunnerContext ctx;
    ctx.ggml_ctx = data.get();
    ctx.backend = &p.backend;
    auto f16 = ggml_new_tensor_2d(data.get(), GGML_TYPE_F16, 16, 2);
    auto f32 = ggml_new_tensor_2d(data.get(), GGML_TYPE_F32, 16, 2);
    require(!Block::try_webgpu_swiglu(&ctx, f16), "F16 activation policy changed");
    require(!Block::try_webgpu_swiglu(&ctx, f32, f16), "mixed-type GLU reached assertion");
    auto odd = ggml_new_tensor_2d(data.get(), GGML_TYPE_F32, 15, 2);
    require(!Block::try_webgpu_swiglu(&ctx, odd), "odd packed gate accepted");
    require(!Block::try_webgpu_swiglu(&ctx, f32, odd), "shape-mismatched GLU reached assertion");
    require(!Block::prepare_modulation(&ctx, {f32}, false), "incomplete modulation accepted");
    require(!Block::prepare_modulation(&ctx, {f16, f16, f16, f16}, false), "F16 modulation policy changed");
    p.named_webgpu = false;
    require(!Block::try_webgpu_swiglu(&ctx, f32), "CPU forced onto fused path");
    require(!Block::prepare_modulation(&ctx, {f32, f32, f32, f32}, false), "CPU modulation policy changed");
    cases += 8;
}

// Execute real graph-cut bindings/cache/allocations, not only a topology plan.
// This deliberately frees each segment allocation before the next one. A shared
// producer cannot accidentally work by retaining the uncut graph's old storage.
std::vector<float> execute_cuts(ggml_backend_t backend, ggml_cgraph * g, ggml_tensor * output,
                                const std::unordered_set<const ggml_tensor *> & weights) {
    auto plan = sd::ggml_graph_cut::build_plan(backend, g, weights, "qwen-work-cut-test");
    require(plan.available && plan.valid && plan.has_cuts && plan.segments.size() > 2,
            "small Qwen graph did not produce usable cuts");
    sd::GraphCutTensorCache cache(backend);
    sd::SegmentGraphBindings bindings(cache, plan, g);
    std::vector<float> result;
    for (size_t i = 0; i < plan.segments.size(); ++i) {
        const auto & segment = plan.segments[i];
        cache.prune(segment.live_cut_names);
        bindings.reset(segment);
        require(bindings.bind_cached_inputs(segment, "qwen-work-cut-test"), "cut input binding");
        ggml_context * raw = nullptr;
        auto sg = sd::ggml_graph_cut::build_segment_graph(g, segment, &raw);
        Context sc(raw, ggml_free);
        require(sg != nullptr, "cut graph construction");
        auto memory = allocate(backend, sg);
        compute(backend, sg);
        if (i + 1 == plan.segments.size())
            result = read(output);
        require(cache.capture(g, segment, "qwen-work-cut-test") == GGML_STATUS_SUCCESS, "cut output capture");
        bindings.restore();
    }
    require(!result.empty(), "cut execution lost final output");
    return result;
}

void cut_model(ggml_backend_t cpu, bool packed, bool reuse) {
    Qwen::QwenImage21Config config;
    config.in_channels = config.out_channels = 8;
    config.hidden_size = config.context_dim = 32;
    config.head_dim = 8;
    config.intermediate_size = 48;
    config.num_layers = 3;
    config.fused_mlp = packed;
    auto data = context();
    Qwen::QwenImage21Model candidate(config);
    qwen21_reference::ReferenceQwenImage21Model reference(config);
    candidate.init(data.get());
    reference.init(data.get());
    std::map<std::string, ggml_tensor *> aw, rw;
    candidate.get_param_tensors(aw);
    reference.get_param_tensors(rw);
    auto x = ggml_new_tensor_4d(data.get(), GGML_TYPE_F32, 2, 2, 8, 1);
    auto time = ggml_new_tensor_1d(data.get(), GGML_TYPE_F32, 1);
    auto text = ggml_new_tensor_3d(data.get(), GGML_TYPE_F32, 32, 5, 1);
    auto pe = ggml_new_tensor_4d(data.get(), GGML_TYPE_F32, 2, 2, 4, reuse ? 4 : 9);
    auto mask = ggml_new_tensor_2d(data.get(), GGML_TYPE_F32, 5, 5);
    auto key = ggml_new_tensor_4d(data.get(), GGML_TYPE_F32, 8, 5, 4, 1);
    auto value = ggml_new_tensor_4d(data.get(), GGML_TYPE_F32, 8, 4, 5, 1);
    Buffer parameters(ggml_backend_alloc_ctx_tensors(data.get(), cpu), ggml_backend_buffer_free);
    require(parameters != nullptr, "small model parameters");
    unsigned wseed = 21;
    for (const auto & [name, t] : aw) {
        auto w = values(ggml_nelements(t), wseed++, 128);
        if (name.find("norm") != std::string::npos)
            for (auto & v : w)
                v += 1.f;
        write(t, w);
        write(rw.at(name), w);
    }
    write(text, values(ggml_nelements(text), 3, 32));
    write(pe, values(ggml_nelements(pe), 5, 16));
    write(mask, std::vector<float>(25, 0));
    write(key, values(ggml_nelements(key), 7, 32));
    write(value, values(ggml_nelements(value), 9, 32));
    auto layout = Qwen::QwenImage21Layout::build(5, {}, {{2, 2}});
    Qwen::QwenImage21PrefixCache cache;
    cache.mode = reuse ? Qwen::QwenImage21PrefixCache::Mode::REUSE : Qwen::QwenImage21PrefixCache::Mode::STORE;
    cache.prefix_length = 5;
    Policy policy(cpu);
    std::vector<float> previous;
    for (unsigned seed : {1u, 19u}) {
        write(x, values(ggml_nelements(x), seed, 16));
        write(time, {float(seed * 17)});
        std::vector<float> expected;
        for (int mode = 0; mode < 4; ++mode) {
            auto gc = context();
            auto one = ggml_new_tensor_1d(gc.get(), GGML_TYPE_F32, 1);
            ggml_set_name(one, "ggml_runner_build_in_tensor:one");
            Buffer constants(ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(cpu), 64),
                             ggml_backend_buffer_free);
            require(constants != nullptr, "constant buffer");
            require(ggml_backend_tensor_alloc(constants.get(), one, ggml_backend_buffer_get_base(constants.get())) ==
                        GGML_STATUS_SUCCESS,
                    "constant tensor binding");
            write(one, {1.f});
            GGMLRunnerContext ctx;
            ctx.ggml_ctx = gc.get();
            ctx.backend = &policy.backend;
            ctx.get_cache_tensor = [&](const std::string & name) { return name.back() == 'k' ? key : value; };
            // Persistent prefix outputs are not required for this cut-lifetime
            // regression; REUSE consumes supplied independent prefix fixtures.
            ctx.cache_tensor = [](const std::string &, ggml_tensor *) {};
            candidate.set_webgpu_work_enabled(mode != 3);
            auto out = mode == 0 ? reference.forward(&ctx, x, time, text, {}, pe, layout, {mask, nullptr}, cache)
                                 : candidate.forward(&ctx, x, time, text, {}, pe, layout, {mask, nullptr}, cache);
            auto g = graph(gc.get(), out, 8192);
            require(count(g, GGML_OP_GLU) == ((mode == 1 || mode == 2) ? config.num_layers : 0),
                    "A/B work control ignored");
            std::vector<float> current;
            if (mode == 2) {
                std::unordered_set<const ggml_tensor *> ws;
                for (auto [_, t] : aw)
                    ws.insert(t);
                current = execute_cuts(cpu, g, out, ws);
            } else {
                auto memory = allocate(cpu, g);
                compute(cpu, g);
                current = read(out);
            }
            if (mode == 0)
                expected = current;
            else if (mode == 3)
                close(expected, current);
            else
                close(expected, current, 2e-5f, 2e-4f);
        }
        require(previous.empty() || previous != expected, "new timestep/input did not affect model");
        previous = expected;
    }
    std::cout << "{\"kind\":\"cut-execution\",\"packed\":" << packed << ",\"reuse\":" << reuse
              << ",\"layers\":3,\"inputs\":2,\"gpu_execution\":false}\n";
    ++cases;
}

void census(ggml_backend_t cpu, int tokens, bool optimized, bool reuse, bool packed) {
    auto params = context(16 * 1024 * 1024), gc = context(32 * 1024 * 1024);
    Policy p(cpu);
    Qwen::QwenImage21Config c;
    c.fused_mlp = packed;
    Qwen::QwenImage21Model model(c);
    qwen21_reference::ReferenceQwenImage21Model reference(c);
    if (optimized)
        model.init(params.get());
    else
        reference.init(params.get());
    auto side = static_cast<int>(std::sqrt(tokens));
    auto x = ggml_new_tensor_4d(params.get(), GGML_TYPE_F32, side, side, 64, 1);
    auto time = ggml_new_tensor_1d(params.get(), GGML_TYPE_F32, 1);
    auto text = ggml_new_tensor_3d(params.get(), GGML_TYPE_F32, 4096, 9, 1);
    auto layout = Qwen::QwenImage21Layout::build(9, {}, {{side, side}});
    auto pe = ggml_new_tensor_4d(params.get(), GGML_TYPE_F32, 2, 2, 64, reuse ? tokens : tokens + 9);
    auto mask = ggml_new_tensor_2d(params.get(), GGML_TYPE_F32, 9, 9);
    GGMLRunnerContext ctx;
    ctx.ggml_ctx = gc.get();
    ctx.backend = &p.backend;
    auto one = ggml_new_tensor_1d(gc.get(), GGML_TYPE_F32, 1);
    ggml_set_name(one, "ggml_runner_build_in_tensor:one");
    Qwen::QwenImage21PrefixCache cache;
    cache.mode = reuse ? Qwen::QwenImage21PrefixCache::Mode::REUSE : Qwen::QwenImage21PrefixCache::Mode::STORE;
    cache.prefix_length = 9;
    std::vector<ggml_tensor *> persisted;
    ctx.cache_tensor = [&](const std::string &, ggml_tensor * tensor) { persisted.push_back(tensor); };
    ctx.get_cache_tensor = [&](const std::string & n) {
        return n.back() == 'k' ? ggml_new_tensor_4d(params.get(), GGML_TYPE_F32, 128, 9, 32, 1)
                               : ggml_new_tensor_4d(params.get(), GGML_TYPE_F32, 128, 32, 9, 1);
    };
    auto output =
        optimized ? model.forward(&ctx, x, time, reuse ? nullptr : text, {}, pe, layout, {mask, nullptr}, cache)
                  : reference.forward(&ctx, x, time, reuse ? nullptr : text, {}, pe, layout, {mask, nullptr}, cache);
    auto g = graph(gc.get(), output, 16384);
    for (auto t : persisted)
        ggml_build_forward_expand(g, t);
    // Both arms keep applied-008 shared RoPE, softmax and WebGPU selection.
    // The immutable reference freezes the actual 008 block/model construction.
    size_t cont_bytes = 0, silu_bytes = 0;
    std::map<std::string, int> ops, unsupported;
    for (int i = 0; i < ggml_graph_n_nodes(g); ++i) {
        auto n = ggml_graph_node(g, i);
        ++ops[ggml_op_name(n->op)];
        if (n->op == GGML_OP_CONT)
            cont_bytes += ggml_nbytes(n);
        if (n->op == GGML_OP_UNARY && ggml_get_unary_op(n) == GGML_UNARY_OP_SILU)
            silu_bytes += ggml_nbytes(n);
        if (!webgpu_test_supports(n, {4294967292ull, 32768, true, false}))
            ++unsupported[ggml_op_name(n->op)];
    }
    require(optimized ? count(g, GGML_OP_GLU) == 32 : count(g, GGML_OP_GLU) == 0, "full model activation count");
    require(unsupported.size() == 1 && unsupported["TIMESTEP_EMBEDDING"] == 1, "introduced WebGPU rejection");
    std::map<std::string, ggml_tensor *> weights;
    if (optimized)
        model.get_param_tensors(weights);
    else
        reference.get_param_tensors(weights);
    std::unordered_set<const ggml_tensor *> ws;
    for (auto [_, t] : weights)
        ws.insert(t);
    auto plan = sd::ggml_graph_cut::build_plan(cpu, g, ws, nullptr);
    require(plan.available && plan.valid && plan.has_cuts, "shared modulation invalidated graph cuts");
    size_t largest_segment = 0;
    for (auto & segment : plan.segments) {
        largest_segment = std::max(largest_segment, segment.compute_buffer_size);
        ggml_context * sc = nullptr;
        require(sd::ggml_graph_cut::build_segment_graph(g, segment, &sc) != nullptr, "cut construction");
        ggml_free(sc);
    }
    std::cout << "{\"kind\":\"graph\",\"tokens\":" << tokens << ",\"optimized\":" << optimized << ",\"reuse\":" << reuse
              << ",\"packed\":" << packed << ",\"nodes\":" << ggml_graph_n_nodes(g) << ",\"cont_bytes\":" << cont_bytes
              << ",\"silu_output_bytes\":" << silu_bytes << ",\"glu\":" << count(g, GGML_OP_GLU)
              << ",\"cont\":" << count(g, GGML_OP_CONT) << ",\"scale\":" << count(g, GGML_OP_SCALE)
              << ",\"unary\":" << count(g, GGML_OP_UNARY) << ",\"cut_segments\":" << plan.segments.size()
              << ",\"direct_reservation_bytes_cpu_plan\":" << plan.compute_buffer_size
              << ",\"largest_segment_bytes_cpu_plan\":" << largest_segment << "}\n";
    ++cases;
}
} // namespace
int main() {
    try {
        auto cpu = image_test::cpu();
        for (int width : {7, 32})
            for (int batch : {1, 2})
                for (bool packed : {false, true})
                    for (bool strided : {false, true})
                        for (bool reject : {false, true})
                            activation(cpu.get(), width, 5, batch, packed, strided, reject);
        for (int width : {7, 32, 48})
            for (int batch : {1, 2})
                for (bool reject : {false, true})
                    separated_activation(cpu.get(), width, 7, batch, reject);
        for (int prefix : {0, 3})
            for (int consumers : {4, 32})
                for (bool strided : {false, true})
                    for (auto reject : {GGML_OP_NONE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_CONT})
                        modulation(cpu.get(), 16, 7, prefix, consumers, strided, reject);
        for (bool packed : {false, true})
            for (bool reuse : {false, true})
                for (bool shared : {false, true})
                    for (bool reject : {false, true})
                        block(cpu.get(), packed, reuse, shared, reject);
        guards(cpu.get());
        option_parse(cpu.get());
        for (bool packed : {false, true})
            for (bool reuse : {false, true})
                cut_model(cpu.get(), packed, reuse);
        for (int tokens : {1024, 4096})
            for (bool reuse : {false, true})
                for (bool packed : {false, true})
                    for (bool opt : {false, true})
                        census(cpu.get(), tokens, opt, reuse, packed);
        std::cout << "{\"cases\":" << cases << ",\"passed\":true,\"gpu_execution\":false}\n";
    } catch (const std::exception & e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
