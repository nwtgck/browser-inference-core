#include "mtmd-subcontext.h"
#include "mtmd-graph.h"
#include "ggml-cpu.h"
#include "support/check.hpp"
#include "support/mtmd-metadata.hpp"
#include <climits>
#include <iostream>
#include <stdexcept>

// The public entry points and context constructor are complete production TUs.
// Small synthetic GGUF inputs pass through the real file loader. The only
// observers are the C GGUF boundary and unused text-model API declarations.
static unsigned loads;
extern "C" {
gguf_context * __real_gguf_init_from_file(const char *, gguf_init_params);
gguf_context * __wrap_gguf_init_from_file(const char * path, gguf_init_params params) {
    ++loads;
    return __real_gguf_init_from_file(path, params);
}
int32_t llama_model_n_embd_inp(const llama_model *) {
    throw std::runtime_error("unexpected text model embedding query");
}
const llama_vocab * llama_model_get_vocab(const llama_model *) {
    throw std::runtime_error("unexpected text model vocabulary query");
}
llama_rope_type llama_model_rope_type(const llama_model *) {
    throw std::runtime_error("unexpected text model rope query");
}
int32_t llama_vocab_n_tokens(const llama_vocab *) {
    throw std::runtime_error("unexpected vocabulary size query");
}
int32_t llama_token_to_piece(const llama_vocab *, llama_token, char *, int32_t, int32_t, bool) {
    throw std::runtime_error("unexpected text token conversion");
}
}
std::unique_ptr<mtmd_graph> mtmd_get_graph_builder(mtmd_subcontext *, const mtmd_image_f32_batch &,
                                                   const mtmd_subctx_encode_params *) {
    throw std::runtime_error("unexpected graph factory reached by entry fixture");
}

int main(int argc, char ** argv) {
    BICORE_CHECK(argc == 2);
    const std::filesystem::path root(argv[1]);
    std::filesystem::create_directories(root);
    const auto input = root / "gen-audio.gguf";
    write_mtmd_fixture(input, false, false, true);
    ggml_backend_register(ggml_backend_cpu_reg());
    auto params = mtmd_context_params_default();
    params.use_gpu = false;
    params.n_threads = 1;
    params.warmup = false;
    BICORE_CHECK(mtmd_vision_bf16_storage_version() == 1);
    BICORE_CHECK(mtmd_context_vision_bf16_storage(nullptr) == MTMD_VISION_BF16_STORAGE_INVALID);
    for (int32_t invalid : {INT_MIN, -2, -1, 2, INT_MAX}) {
        BICORE_CHECK(!mtmd_init_from_file_with_vision_bf16_storage(nullptr, nullptr, params, invalid));
    }
    BICORE_CHECK(loads == 0);
    mtmd_context * nested = nullptr;
    struct CallbackState {
        mtmd_context *& nested;
        const std::filesystem::path & input;
    } callback_state{nested, input};
    params.progress_callback_user_data = &callback_state;
    params.progress_callback = [](float, void * data) {
        auto & state = *static_cast<CallbackState *>(data);
        if (state.nested != nullptr)
            return true;
        auto inner = mtmd_context_params_default();
        inner.use_gpu = false;
        inner.n_threads = 1;
        inner.warmup = false;
        state.nested = mtmd_init_from_file_with_vision_bf16_storage(state.input.c_str(), nullptr, inner,
                                                                    MTMD_VISION_BF16_STORAGE_LEGACY_EXPAND);
        BICORE_CHECK(state.nested != nullptr);
        return true;
    };
    auto * retained = mtmd_init_from_file_with_vision_bf16_storage(input.c_str(), nullptr, params,
                                                                   MTMD_VISION_BF16_STORAGE_RETAIN_CHECKED);
    BICORE_CHECK(retained != nullptr && nested != nullptr && loads == 2);
    BICORE_CHECK(mtmd_context_vision_bf16_storage(retained) == MTMD_VISION_BF16_STORAGE_RETAIN_CHECKED);
    BICORE_CHECK(mtmd_context_vision_bf16_storage(nested) == MTMD_VISION_BF16_STORAGE_LEGACY_EXPAND);
    auto * old = mtmd_init_from_file(input.c_str(), nullptr, params);
    BICORE_CHECK(old != nullptr && mtmd_context_vision_bf16_storage(old) == MTMD_VISION_BF16_STORAGE_LEGACY_EXPAND);
    BICORE_CHECK(!mtmd_init_from_file_with_vision_bf16_storage((root / "missing.gguf").c_str(), nullptr, params, 1));
    auto * again = mtmd_init_from_file_with_vision_bf16_storage(input.c_str(), nullptr, params, 0);
    BICORE_CHECK(again != nullptr && mtmd_context_vision_bf16_storage(again) == MTMD_VISION_BF16_STORAGE_LEGACY_EXPAND);
    BICORE_CHECK(mtmd_context_vision_bf16_storage(retained) == MTMD_VISION_BF16_STORAGE_RETAIN_CHECKED);
    for (auto * context : {again, old, nested, retained})
        mtmd_free(context);
    std::cout << "complete mtmd entry validates raw i32 before loading and preserves reentry policies\n";
}
