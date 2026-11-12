#include "bridge/model-graph.h"
#include "bridge/model-graph-internal.h"
#include "gguf.h"
#include "llama.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

static int checks = 0;
static void check(bool condition, const char * name) {
    if (!condition)
        throw std::runtime_error(name);
    ++checks;
}
template <typename T> static uint64_t address(T * value) {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(value));
}
static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    std::vector<llama_token> result(2048);
    const int count = llama_tokenize(vocab, text.c_str(), text.size(), result.data(), result.size(), true, true);
    check(count >= 0 && count <= 2048, "tokenize bounded input");
    result.resize(count);
    return result;
}
int main(int argc, char ** argv) {
    if (argc != 2)
        return 2;
    try {
        llama_backend_init();
        auto parameters = llama_model_default_params();
        parameters.n_gpu_layers = 0;
        parameters.load_mode = LLAMA_LOAD_MODE_NONE;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], parameters),
                                                                        llama_model_free);
        check(model != nullptr, "load baseline native model");
        const llama_vocab * baseline = llama_model_get_vocab(model.get());
        std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(gguf_init_from_file(argv[1], {true, nullptr}),
                                                                     gguf_free);
        check(metadata != nullptr, "read fixture metadata");
        gguf_set_val_str(metadata.get(), "general.architecture", "not-registered-in-llama");
        const uint64_t owner = lcb_model_graph_vocab_new(address(metadata.get()));
        check(owner != 0, "load vocabulary independently of architecture");
        struct cleanup {
            uint64_t owner;
            ~cleanup() {
                lcb_model_graph_vocab_free(owner);
            }
        } owner_cleanup{owner};
        const auto * vocab =
            reinterpret_cast<const llama_vocab *>(static_cast<uintptr_t>(lcb_model_graph_vocab_get(owner)));
        check(vocab != nullptr, "expose borrowed vocabulary");
        metadata.reset(); // No metadata pointers may be retained by the owner.
        check(llama_vocab_n_tokens(vocab) == llama_vocab_n_tokens(baseline), "same vocabulary count");
        for (const auto & text : {"hello world", "日本語\nTest", "<s> <unk> </s>"})
            check(tokenize(vocab, text) == tokenize(baseline, text), "tokenization agrees after GGUF release");
        for (int i = 0; i < llama_vocab_n_tokens(vocab); ++i)
            check(llama_vocab_is_eog(vocab, i) == llama_vocab_is_eog(baseline, i), "same end-of-generation metadata");
        const auto native_templates = common_chat_templates_init(model.get(), "");
        const auto external_templates = lcb_model_graph_chat_templates(owner, "");
        check(common_chat_templates_source(native_templates.get()) ==
                  common_chat_templates_source(external_templates.get()),
              "same default template source");
        check(common_chat_templates_was_explicit(native_templates.get()) ==
                  common_chat_templates_was_explicit(external_templates.get()),
              "same explicit-template state");
        common_chat_templates_inputs input;
        common_chat_msg message;
        message.role = "user";
        message.content = "Boundary test";
        input.messages.push_back(message);
        input.add_generation_prompt = true;
        input.use_jinja = true;
        const auto expected = common_chat_templates_apply(native_templates.get(), input);
        const auto actual = common_chat_templates_apply(external_templates.get(), input);
        check(expected.prompt.str() == actual.prompt.str(), "same rendered prompt");
        const int count = llama_vocab_n_tokens(vocab);
        const uint64_t workspace = lcb_model_graph_sampler_workspace_new(count);
        check(workspace != 0, "allocate reusable candidates");
        struct workspace_cleanup {
            uint64_t value;
            ~workspace_cleanup() {
                lcb_model_graph_sampler_workspace_free(value);
            }
        } workspace_owner{workspace};
        std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler(
            llama_sampler_chain_init(llama_sampler_chain_default_params()), llama_sampler_free);
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_k(2));
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_greedy());
        std::vector<float> logits(count, -2.0f);
        logits[13] = 20.0f;
        check(lcb_model_graph_sample(workspace, address(sampler.get()), address(logits.data()), count) == 13,
              "sample token identity after candidate reordering");
        logits[13] = -2.0f;
        logits[37] = 30.0f;
        check(lcb_model_graph_sample(workspace, address(sampler.get()), address(logits.data()), count) == 37,
              "reset reusable candidate state");
        logits[1] = std::numeric_limits<float>::quiet_NaN();
        check(lcb_model_graph_sample(workspace, address(sampler.get()), address(logits.data()), count) ==
                  LLAMA_TOKEN_NULL,
              "reject NaN logits");
        check(std::strlen(reinterpret_cast<const char *>(static_cast<uintptr_t>(lcb_model_graph_last_error()))) > 0,
              "report sampling failure");
        logits[1] = -2.0f;
        check(lcb_model_graph_sample(workspace, address(sampler.get()), address(logits.data()), count - 1) ==
                  LLAMA_TOKEN_NULL,
              "reject count mismatch");
        check(lcb_model_graph_sample(workspace, address(sampler.get()), address(logits.data()), count) == 37,
              "recover after rejected input");
        check(lcb_model_graph_architecture_known(address("llama")) == 1, "recognize actual native architecture");
        check(lcb_model_graph_architecture_known(address("not-registered-in-llama")) == 0,
              "unknown architecture query");
        check(lcb_model_graph_architecture_known(0) == -1, "reject null architecture");
        check(lcb_model_graph_vocab_new(0) == 0, "reject null metadata");
        check(lcb_model_graph_sampler_workspace_new(0) == 0, "reject empty vocabulary");
        std::cout << "native model-graph boundary: " << checks << " checks passed (not a browser test)\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
