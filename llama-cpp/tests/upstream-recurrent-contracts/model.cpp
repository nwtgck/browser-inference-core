// Reuse upstream synthetic weights; do not copy its model implementation.
#define main lcb_unused_architecture_test_main
#include "test-llama-archs.cpp"
#undef main

#include <iostream>
#include <set>

static void require(bool condition, const char * message) {
    if (!condition)
        throw std::runtime_error(message);
}

struct window_capture {
    std::set<int64_t> widths;
    static bool callback(ggml_tensor * tensor, bool ask, void * opaque) {
        // ggml_set_rows keeps its legacy source order: data, indices, destination.
        if (ask && tensor->op == GGML_OP_SET_ROWS && tensor->src[0] && tensor->src[1] &&
            std::strcmp(tensor->src[1]->name, "rs_s_slot_rev") == 0 && tensor->src[0]->ne[0] == 3) {
            static_cast<window_capture *>(opaque)->widths.insert(tensor->src[0]->ne[1]);
        }
        return false; // Metadata only; do not request tensor data or graph slices.
    }
};

static llama_context_ptr context(llama_model * model, window_capture & capture) {
    auto params = llama_context_default_params();
    params.n_ctx = 64;
    params.n_batch = 8;
    params.n_ubatch = 8;
    params.n_seq_max = 1;
    params.n_rs_seq = 3;
    params.n_threads = 1;
    params.n_threads_batch = 1;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    params.cb_eval = window_capture::callback;
    params.cb_eval_user_data = &capture;
    llama_context_ptr result(llama_init_from_model(model, params));
    require(bool(result), "recurrent context creation failed");
    require(llama_n_rs_seq(result.get()) == 3, "rollback slots were not enabled");
    capture.widths.clear();
    return result;
}

static std::vector<float> decode(llama_model * model, llama_context * ctx, const std::vector<llama_token> & tokens,
                                 llama_pos first_pos) {
    common_batch batch(ctx);
    for (size_t i = 0; i < tokens.size(); ++i)
        batch.add(tokens[i], first_pos + static_cast<llama_pos>(i), 0, true);
    require(llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0, "recurrent decode failed");
    const size_t count = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const float * logits = llama_get_logits_ith(ctx, -1);
    std::vector<float> result(logits, logits + count);
    require(std::all_of(result.begin(), result.end(), [](float value) { return std::isfinite(value); }),
            "non-finite recurrent logits");
    return result;
}

int main() {
    try {
        llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
        llama_backend_init();
        init_normal_pool(913, 0.02f);
        auto metadata = get_gguf_ctx(LLM_ARCH_QWEN35, false);
        auto model = get_model(metadata.get(), nullptr, 913, 0.02f, {});
        window_capture longer, shorter;
        auto rollback = context(model.get(), longer);
        auto reference = context(model.get(), shorter);
        decode(model.get(), rollback.get(), {1, 2, 3, 4}, 0);
        require(longer.widths == std::set<int64_t>{4}, "n>=K did not execute the four-window graph");
        auto * memory = llama_get_memory(rollback.get());
        require(llama_memory_seq_pos_max(memory, 0) == 3, "wrong pre-rollback position");
        require(llama_memory_seq_rm(memory, 0, 2, -1), "rollback was rejected");
        require(llama_memory_seq_pos_max(memory, 0) == 1, "rollback did not restore the two-token prefix");
        const auto actual = decode(model.get(), rollback.get(), {5}, 2);
        decode(model.get(), reference.get(), {1, 2}, 0);
        require(shorter.widths == std::set<int64_t>{3}, "n<K did not execute the three-window graph");
        const auto expected = decode(model.get(), reference.get(), {5}, 2);
        const double error = nmse(actual, expected);
        float max_error = 0;
        for (size_t i = 0; i < actual.size(); ++i)
            max_error = std::max(max_error, std::fabs(actual[i] - expected[i]));
        require(error <= 1e-6 && max_error <= 1e-5, "rollback logits differ from the fresh prefix");
        std::cout << "PASS: actual Qwen35 delta-net Kw=3/4 graph and recurrent rollback; nmse=" << error
                  << " max_error=" << max_error << '\n';
        reference.reset();
        rollback.reset();
        model.reset();
        llama_backend_free();
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
