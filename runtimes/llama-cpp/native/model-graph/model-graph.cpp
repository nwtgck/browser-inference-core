#include "model-graph.h"
#include "model-graph-internal.h"
#include "gguf.h"
#include "llama.h"
#include "src/llama-arch.h"
#include "src/llama-model-loader.h"
#include "src/llama-vocab.h"

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
// All distribution profiles are single-threaded; thread_local also makes the
// native boundary tests independent without sharing error ownership.
thread_local std::string last_error;
constexpr uint32_t max_vocabulary = 1u << 20;

template <typename T> T * pointer(uint64_t value) {
    if (!value || value > UINTPTR_MAX)
        throw std::runtime_error("Invalid model graph handle");
    return reinterpret_cast<T *>(static_cast<uintptr_t>(value));
}
template <typename T> uint64_t address(T * value) {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(value));
}
template <typename T, typename F> T guarded(T failure, F function) noexcept {
    try {
        last_error.clear();
        return function();
    } catch (const std::exception & error) {
        last_error = error.what();
    } catch (...) {
        last_error = "Unknown model graph boundary failure";
    }
    return failure;
}
std::string text(const gguf_context * metadata, const char * name) {
    const int64_t id = gguf_find_key(metadata, name);
    if (id < 0)
        return {};
    if (gguf_get_kv_type(metadata, id) != GGUF_TYPE_STRING)
        throw std::runtime_error(std::string("Expected GGUF string: ") + name);
    const std::string result = gguf_get_val_str(metadata, id);
    if (result.size() > 1024u * 1024u)
        throw std::runtime_error("Chat metadata exceeds the boundary limit");
    return result;
}
struct vocabulary_owner {
    llama_vocab vocabulary;
    std::string default_template;
    std::string tool_template;
    std::string tts_template;

    explicit vocabulary_owner(gguf_context * metadata)
        : default_template(text(metadata, "tokenizer.chat_template")),
          tool_template(text(metadata, "tokenizer.chat_template.tool_use")),
          tts_template(text(metadata, "tokenizer.chat_template.tts")) {
        const int64_t tokens = gguf_find_key(metadata, "tokenizer.ggml.tokens");
        if (tokens < 0 || gguf_get_kv_type(metadata, tokens) != GGUF_TYPE_ARRAY ||
            gguf_get_arr_type(metadata, tokens) != GGUF_TYPE_STRING || gguf_get_arr_n(metadata, tokens) == 0 ||
            gguf_get_arr_n(metadata, tokens) > max_vocabulary)
            throw std::runtime_error("Missing or unsupported tokenizer vocabulary");
        // The metadata-only loader does not construct a model or own metadata.
        // No dummy architecture is substituted, and no model weights are read.
        std::vector<std::string> splits;
        llama_model_loader loader(metadata, nullptr, nullptr, "", splits, nullptr, LLAMA_LOAD_MODE_NONE, false, true,
                                  false, nullptr, nullptr);
        vocabulary.load(loader, LLM_KV(LLM_ARCH_UNKNOWN));
        if (vocabulary.n_tokens() == 0 || vocabulary.n_tokens() > max_vocabulary)
            throw std::runtime_error("Unsupported vocabulary size");
    }
};
struct sampler_workspace {
    std::vector<llama_token_data> candidates;
    explicit sampler_workspace(uint32_t count) : candidates(count) {}
};
} // namespace

extern "C" uint32_t lcb_model_graph_version(void) {
    return 1;
}
extern "C" uint64_t lcb_model_graph_last_error(void) {
    return address(last_error.c_str());
}
extern "C" int32_t lcb_model_graph_architecture_known(uint64_t name) {
    return guarded<int32_t>(
        -1, [&]() { return llm_arch_from_string(pointer<const char>(name)) != LLM_ARCH_UNKNOWN ? 1 : 0; });
}
extern "C" uint64_t lcb_model_graph_vocab_new(uint64_t metadata) {
    return guarded<uint64_t>(0, [&]() { return address(new vocabulary_owner(pointer<gguf_context>(metadata))); });
}
extern "C" uint64_t lcb_model_graph_vocab_get(uint64_t owner) {
    return guarded<uint64_t>(0, [&]() { return address(&pointer<vocabulary_owner>(owner)->vocabulary); });
}
extern "C" void lcb_model_graph_vocab_free(uint64_t owner) {
    if (owner)
        delete pointer<vocabulary_owner>(owner);
}
extern "C" uint64_t lcb_model_graph_sampler_workspace_new(uint32_t vocabulary_size) {
    return guarded<uint64_t>(0, [&]() {
        if (!vocabulary_size || vocabulary_size > max_vocabulary)
            throw std::runtime_error("Unsupported sampler vocabulary size");
        return address(new sampler_workspace(vocabulary_size));
    });
}
extern "C" void lcb_model_graph_sampler_workspace_free(uint64_t workspace) {
    if (workspace)
        delete pointer<sampler_workspace>(workspace);
}
extern "C" int32_t lcb_model_graph_sample(uint64_t workspace, uint64_t sampler, uint64_t logits, uint32_t count) {
    return guarded<int32_t>(LLAMA_TOKEN_NULL, [&]() {
        auto & candidates = pointer<sampler_workspace>(workspace)->candidates;
        auto * chain = pointer<llama_sampler>(sampler);
        const auto * values = pointer<const float>(logits);
        if (count != candidates.size())
            throw std::runtime_error("Logit count does not match the vocabulary");
        bool finite = false;
        for (uint32_t i = 0; i < count; ++i) {
            if (std::isnan(values[i]) || values[i] == std::numeric_limits<float>::infinity())
                throw std::runtime_error("Invalid model graph logits");
            finite |= std::isfinite(values[i]);
            candidates[i] = {static_cast<llama_token>(i), values[i], 0.0f};
        }
        if (!finite)
            throw std::runtime_error("Model graph has no finite candidate");
        llama_token_data_array values_array = {candidates.data(), candidates.size(), -1, false};
        llama_sampler_apply(chain, &values_array);
        if (values_array.selected < 0 || static_cast<size_t>(values_array.selected) >= values_array.size)
            throw std::runtime_error("Sampler did not select a valid candidate");
        const llama_token token = values_array.data[values_array.selected].id;
        if (token < 0 || static_cast<uint32_t>(token) >= count)
            throw std::runtime_error("Sampler returned an invalid token");
        llama_sampler_accept(chain, token);
        return token;
    });
}

common_chat_templates_ptr lcb_model_graph_chat_templates(uint64_t owner, const std::string & override_source) {
    auto & value = *pointer<vocabulary_owner>(owner);
    // Match the native model entry point: a source override replaces only the
    // default/tool selection, while task metadata remains attached.
    const auto & source = override_source.empty() ? value.default_template : override_source;
    const std::string tools = override_source.empty() ? value.tool_template : "";
    return common_chat_templates_init_from_vocab(&value.vocabulary, source, tools, value.tts_template,
                                                 !source.empty() || !tools.empty());
}
