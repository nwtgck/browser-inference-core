// The real AnimaRunner and RunnerCache with a small, deterministically initialized
// Anima network. No copied caching algorithm or trained checkpoint is used.
#include "image-performance-test-utils.h"
#include "model/diffusion/anima.hpp"
#include <cstdio>
#include <map>
#include <filesystem>
#include <random>
#include "gguf.h"

namespace {
int cache_allocations = 0;
int fail_cache_allocations = 0;
int failed_cache_allocations = 0;
}

#ifdef SDCB_TEST_WRAP_CACHE_ALLOC
extern "C" ggml_backend_buffer_t __real_ggml_backend_alloc_ctx_tensors(ggml_context*, ggml_backend_t);
extern "C" ggml_backend_buffer_t __wrap_ggml_backend_alloc_ctx_tensors(ggml_context* ctx, ggml_backend_t backend) {
    auto tensor = ggml_get_first_tensor(ctx);
    if (tensor != nullptr && std::string(ggml_get_name(tensor)).rfind("anima.context.", 0) == 0) {
        ++cache_allocations;
        if (fail_cache_allocations > 0) {
            --fail_cache_allocations;
            ++failed_cache_allocations;
            return nullptr;
        }
    }
    return __real_ggml_backend_alloc_ctx_tensors(ctx, backend);
}
#endif

namespace {
using namespace image_test;
struct TinyNet : Anima::AnimaNet {
    explicit TinyNet(const Anima::AnimaConfig& config) : AnimaNet(config) {
        // Keep 64-dimensional rotary heads, matching the actual AnimaRunner.
        blocks["llm_adapter"] = std::make_shared<Anima::LLMAdapter>(64, 64, 64, 2, 1);
    }
};
struct TinyRunner : Anima::AnimaRunner {
    std::shared_ptr<ModelManager> manager;
    std::filesystem::path fixture_dir;
    std::map<std::string, ggml_tensor*> tensors;

    static void fill(const std::map<std::string, ggml_tensor*>& tensors, unsigned seed) {
        for (const auto& [name, tensor] : tensors) {
            auto v = values(ggml_nelements(tensor), seed++, 1024.f);
            if (name.find("norm") != std::string::npos && name.find("weight") != std::string::npos) {
                for (auto& item : v) item += 1.f;
            }
            write(tensor, v);
        }
    }
    void load_fixture(ggml_backend_t backend) {
        std::random_device random;
        for (int attempt = 0; attempt < 16 && fixture_dir.empty(); ++attempt) {
            auto candidate = std::filesystem::temp_directory_path() /
                             ("bicore-anima-test-" + std::to_string(random()));
            if (std::filesystem::create_directory(candidate)) fixture_dir = std::move(candidate);
        }
        require(!fixture_dir.empty(), "temporary directory creation failed");
        auto path = (fixture_dir / "weights.gguf").string();
        auto source = context();
        std::map<std::string, ggml_tensor*> copies;
        for (const auto& [name, tensor] : tensors) {
            auto copy = ggml_dup_tensor(source.get(), tensor);
            ggml_set_name(copy, name.c_str());
            copies[name] = copy;
        }
        Buffer storage(ggml_backend_alloc_ctx_tensors(source.get(), backend), ggml_backend_buffer_free);
        require(storage != nullptr, "fixture allocation failed");
        fill(copies, 4);
        std::unique_ptr<gguf_context, decltype(&gguf_free)> file(gguf_init_empty(), gguf_free);
        for (const auto& [name, tensor] : copies) gguf_add_tensor(file.get(), tensor);
        require(gguf_write_to_file(file.get(), path.c_str(), false), "fixture write failed");
        manager = std::make_shared<ModelManager>();
        manager->set_n_threads(1);
        manager->set_enable_mmap(false);
        manager->set_prefetch_disabled(true);
        residency_manager = manager;
        require(manager->add_file(path), "fixture registration failed");
        require(manager->register_param_tensors(ModelComponent::Diffusion, tensors,
                                               ModelManager::ResidencyMode::ParamBackend, backend, backend),
                "parameter registration failed");
        require(manager->load_all_params_eagerly(), "fixture loading failed");
    }
    explicit TinyRunner(ggml_backend_t backend) : AnimaRunner(backend) {
        net = Anima::AnimaNet{};
        free_params_ctx();
        alloc_params_ctx();
        config.hidden_size = 64;
        config.text_embed_dim = 64;
        config.num_heads = 1;
        config.head_dim = 64;
        config.num_layers = 1;
        config.axes_dim = {20, 22, 22};
        net = TinyNet(config);
        net.init(params_ctx);
        net.get_param_tensors(tensors);
        try {
            load_fixture(backend);
        } catch (...) {
            manager.reset();
            if (!fixture_dir.empty()) std::filesystem::remove_all(fixture_dir);
            throw;
        }
    }
    ~TinyRunner() override {
        runner_end();
        manager.reset();
        std::error_code ignored;
        if (!fixture_dir.empty()) std::filesystem::remove_all(fixture_dir, ignored);
    }
    void initialize(unsigned seed) { fill(tensors, seed); }
    ggml_tensor* cached(uint64_t id) {
        auto ctx = get_context();
        return get_cache_tensor_by_name(Anima::context_cache_name(ctx, id));
    }
    bool empty_cache() const { return cache_.empty(); }
};

struct Inputs {
    sd::Tensor<float> image{{4, 4, 16, 1}, values(256, 3, 64.f)};
    sd::Tensor<float> time{{1}, {400.f}};
    sd::Tensor<float> text{{64, 5, 1}, values(320, 9, 128.f)};
    sd::Tensor<int32_t> ids{{3}, {1, 8, 16}};
    sd::Tensor<float> weights{{3}, {1.f, 0.75f, -0.5f}};
    sd::Tensor<float> run(TinyRunner& runner, uint64_t id) {
        DiffusionParams params;
        params.x = &image;
        params.timesteps = &time;
        params.context = &text;
        params.extra = AnimaDiffusionExtra{&ids, &weights, id};
        return runner.compute(1, params);
    }
};

void identifiers() {
    Inputs a, b, c, d;
    AnimaConditionIds ids;
    require(ids.get(&a.text, &a.ids, &a.weights) == 1, "first condition ID");
    require(ids.get(&a.text, &a.ids, &a.weights) == 1, "same condition must keep ID");
    require(ids.get(&b.text, &b.ids, &b.weights) == 2, "negative condition ID");
    require(ids.get(&c.text, &c.ids, &c.weights) == 3, "image-negative condition ID");
    require(ids.get(&d.text, &d.ids, &d.weights) == 0, "bounded condition capacity");
    require(ids.get(&a.text, &a.ids, nullptr) == 0, "token weights are part of identity");
    ids.disable();
    require(ids.get(&a.text, &a.ids, &a.weights) == 0, "extension opt-out must be sticky");
    AnimaConditionIds next;
    require(next.get(&a.text, &a.ids, &a.weights) == 1, "next sampling scope owns fresh IDs");
    require(next.get(nullptr, &a.ids, &a.weights) == 0, "missing context");
    require(next.get(&a.text, nullptr, &a.weights) == 0, "missing target IDs");
    sd::Tensor<int32_t> empty;
    require(next.get(&a.text, &empty, &a.weights) == 0, "empty target IDs");
}

void reuse_and_lifetime(ggml_backend_t backend, bool flash) {
    TinyRunner cached(backend), reference(backend);
    cached.set_flash_attention_enabled(flash);
    reference.set_flash_attention_enabled(flash);
    reference.context_cache_enabled = false;
    Inputs input;
    for (int iteration = 0; iteration < 3; ++iteration) {
        input.image = sd::Tensor<float>({4, 4, 16, 1}, values(256, iteration + 3, 64.f));
        input.time[0] = 400.f - 50.f * iteration;
        auto expected = input.run(reference, 0);
        auto actual = input.run(cached, 1);
        require(!expected.empty() && !actual.empty(), "Anima compute failed");
        close(expected.values(), actual.values());
        auto text = cached.cached(1);
        require(text != nullptr && text->buffer != nullptr && text->view_src == nullptr, "cached tensor lacks independent ownership");
        require(text->ne[0] == 64 && text->ne[1] == 3 && ggml_is_contiguous(text), "wrong compact cached shape");
        require(std::string(text->name).find("anima.context.") == 0, "graph construction renamed a persistent cached tensor");
    }
    auto saved = read(cached.cached(1));
    // A second immutable condition does not overwrite the first one.
    input.text[0] += 0.125f;
    input.ids[0] = 7;
    input.weights[1] = 0.5f;
    auto expected = input.run(reference, 0);
    auto actual = input.run(cached, 2);
    close(expected.values(), actual.values());
    require(cached.cached(2) != nullptr && cached.cached(2) != cached.cached(1), "conditions share a saved tensor");
    close(saved, read(cached.cached(1)));
    cached.runner_end();
    reference.runner_end();
    require(cached.empty_cache(), "sampling end retained condition cache");
    // Same input objects and same ID are safe after a new sampling scope, even
    // when their data and the model weights have changed in place.
    cached.initialize(29);
    reference.initialize(29);
    input.ids[1] = 3;
    input.text[2] -= 0.5f;
    expected = input.run(reference, 0);
    actual = input.run(cached, 1);
    close(expected.values(), actual.values());
    require(cached.cached(1) != nullptr, "new sampling scope did not rebuild cache");
    cached.runner_end();
    actual = input.run(cached, 0);
    close(expected.values(), actual.values());
    require(cached.empty_cache(), "unidentified direct invocation cached condition");
    actual = input.run(cached, 4);
    require(!actual.empty() && cached.empty_cache(), "out-of-range ID was cached");
}

void padding_composition(ggml_backend_t backend) {
    TinyRunner compact(backend), physical(backend);
    physical.net.compact_context_enabled = false;
    Inputs input;
    // AnimaNet forwards the sink only to cross-attention, not the adapter or
    // image self-attention. Compare the complete small model, not just its helper.
    auto expected = input.run(physical, 1);
    auto actual = input.run(compact, 1);
    require(!actual.empty() && !expected.empty(), "small model padding comparison failed");
    close(expected.values(), actual.values(), 2e-5f, 2e-4f);
    actual = input.run(compact, 1);
    close(expected.values(), actual.values(), 2e-5f, 2e-4f);
    // The adapter still evaluates all target tokens before trimming the output.
    compact.runner_end();
    physical.runner_end();
    std::vector<int32_t> ids(513);
    for (size_t i = 0; i < ids.size(); ++i) ids[i] = 1 + i % 37;
    input.ids = sd::Tensor<int32_t>({513}, ids);
    input.weights = sd::Tensor<float>({513}, std::vector<float>(513, 1.f));
    expected = input.run(physical, 1);
    actual = input.run(compact, 1);
    close(expected.values(), actual.values());
    require(compact.cached(1)->ne[1] == 512, "long adapter output not trimmed after computation");
}

void allocation_failure(ggml_backend_t backend) {
#ifdef SDCB_TEST_WRAP_CACHE_ALLOC
    TinyRunner runner(backend), reference(backend);
    reference.context_cache_enabled = false;
    Inputs input;
    auto expected = input.run(reference, 0);
    int before = cache_allocations;
    fail_cache_allocations = 1;
    auto actual = input.run(runner, 1);
    require(!actual.empty(), "allocation fallback did not produce output");
    close(expected.values(), actual.values());
    require(cache_allocations == before + 1 && failed_cache_allocations == 1, "expected one failing cache allocation");
    require(runner.context_cache_disabled && runner.empty_cache(), "failed cache was retained or immediately re-enabled");
    actual = input.run(runner, 1);
    close(expected.values(), actual.values());
    require(cache_allocations == before + 1, "disabled sampling cache was retried again");
    runner.runner_end();
    actual = input.run(runner, 1);
    close(expected.values(), actual.values());
    require(!runner.context_cache_disabled && runner.cached(1) != nullptr, "new sampling scope did not recover cache");
#else
    (void)backend;
    std::puts("cache allocation failure injection unavailable on this platform");
#endif
}

void options(ggml_backend_t backend) {
    // Constructor parsing only: no large parameter buffer is allocated.
    Anima::AnimaRunner disabled(backend, {}, "model.diffusion_model", nullptr,
                                "anima_context_cache=false,anima_compact_context=false");
    require(!disabled.context_cache_enabled && !disabled.net.compact_context_enabled, "Anima option parsing failed");
    GGMLRunnerContext ctx;
    ctx.get_cache_tensor = [](const std::string&) -> ggml_tensor* { return nullptr; };
    ctx.cache_tensor = [](const std::string&, ggml_tensor*) {};
    require(!Anima::context_cache_name(ctx, 1).empty(), "valid cache key rejected");
    require(Anima::context_cache_name(ctx, 0).empty() && Anima::context_cache_name(ctx, 4).empty(), "invalid cache ID accepted");
    auto original = Anima::context_cache_name(ctx, 1);
    ctx.flash_attn_enabled = true;
    require(Anima::context_cache_name(ctx, 1) != original, "flash setting reuses incompatible cache");
    ctx.linear_scale = 1.f;
    require(Anima::context_cache_name(ctx, 1).empty(), "non-default scale unexpectedly cached");
    ctx.linear_scale = 0.f;
    ctx.attn_scale = 1.f;
    require(Anima::context_cache_name(ctx, 1).empty(), "non-default attention scale unexpectedly cached");
}
} // namespace

int main() {
    try {
        sd_set_log_callback([](sd_log_level_t level, const char* text, void*) {
            if (level >= SD_LOG_WARN) std::fputs(text, stderr);
        }, nullptr);
        auto backend = image_test::cpu();
        identifiers();
        options(backend.get());
        reuse_and_lifetime(backend.get(), false);
        reuse_and_lifetime(backend.get(), true);
        padding_composition(backend.get());
        allocation_failure(backend.get());
        std::puts("Anima context ID, whole-runner parity, cache ownership/lifetime, padding and allocation fallback passed (synthetic CPU)");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Anima context regression failed: %s\n", error.what());
        return 1;
    }
    return 0;
}
