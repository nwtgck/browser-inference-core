// Uses upstream's small synthetic architecture fixture. Never downloads weights.
#define main lcb_unused_upstream_architecture_main
#include "test-llama-archs.cpp"
#undef main
#include <filesystem>
#include <iostream>
#include <fstream>
#include "llama-model.h"
extern "C" int32_t lcb_qwen4exp_file_backed_shard(const llama_model * model);
static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}
static void equal_logits(const std::vector<float> & a, const std::vector<float> & b) {
    require(a.size() == b.size() && !a.empty(), "logits size");
    for (size_t i = 0; i < a.size(); ++i) {
        require(std::isfinite(a[i]) && std::isfinite(b[i]), "nonfinite logits");
        require(std::abs(a[i] - b[i]) <= 1e-5f * (1 + std::abs(a[i])), "logits differ");
    }
}
static std::vector<std::string> split_fixture(const std::string & path) {
    ggml_context * raw = nullptr;
    gguf_context_ptr input(gguf_init_from_file(path.c_str(), { false, &raw }));
    ggml_context_ptr data(raw);
    require(bool(input) && bool(data), "read fixture");
    gguf_context_ptr converted(gguf_init_empty());
    gguf_set_kv(converted.get(), input.get());
    for (int64_t i = 0; i < gguf_get_n_tensors(input.get()); ++i) {
        const auto * tensor_name = gguf_get_tensor_name(input.get(), i);
        const auto * tensor = ggml_get_tensor(data.get(), tensor_name);
        require(tensor && tensor->data, "fixture tensor data");
        // Parsed GGUF tensor descriptors are not live backend tensors.
        // Register the materialized ggml tensor in a fresh writer instead.
        gguf_add_tensor(converted.get(), tensor);
    }
    constexpr auto name = "per_layer_token_embd.weight";
    auto * ple = ggml_get_tensor(data.get(), name);
    require(ple != nullptr, "PLE tensor");
    // Use the affected quantization family in the end-to-end graph test too.
    std::vector<float> floats(ggml_nelements(ple));
    if (ple->type == GGML_TYPE_F32) std::memcpy(floats.data(), ple->data, ggml_nbytes(ple));
    else ggml_get_type_traits(ple->type)->to_float(ple->data, floats.data(), floats.size());
    std::vector<uint64_t> quantized((ggml_row_size(GGML_TYPE_IQ4_NL, ple->ne[0]) * ple->ne[1] + 7) / 8);
    ggml_get_type_traits(GGML_TYPE_IQ4_NL)->from_float_ref(floats.data(), quantized.data(), floats.size());
    gguf_set_tensor_type(converted.get(), name, GGML_TYPE_IQ4_NL);
    gguf_set_tensor_data(converted.get(), name, quantized.data());
    require(gguf_write_to_file(converted.get(), path.c_str(), false), "write quantized fixture");
    std::vector<std::string> paths;
    for (uint16_t shard = 0; shard < 3; ++shard) {
        gguf_context_ptr output(gguf_init_empty());
        gguf_set_kv(output.get(), input.get());
        gguf_set_val_u16(output.get(), "split.no", shard);
        gguf_set_val_u16(output.get(), "split.count", 3);
        gguf_set_val_i32(output.get(), "split.tensors.count", gguf_get_n_tensors(input.get()));
        for (int64_t i = 0; i < gguf_get_n_tensors(input.get()); ++i) {
            const char * tensor = gguf_get_tensor_name(input.get(), i);
            const bool is_ple = std::strcmp(tensor, name) == 0;
            if (shard == (is_ple ? 1 : 2)) {
                gguf_add_tensor(output.get(), ggml_get_tensor(data.get(), tensor));
                if (is_ple) {
                    gguf_set_tensor_type(output.get(), tensor, GGML_TYPE_IQ4_NL);
                    gguf_set_tensor_data(output.get(), tensor, quantized.data());
                }
            }
        }
        require(shard != 0 || gguf_get_n_tensors(output.get()) == 0, "metadata-only first shard");
        paths.push_back(path + ".part-" + std::to_string(shard));
        require(gguf_write_to_file(output.get(), paths.back().c_str(), false), "write split fixture");
    }
    return paths;
}
static std::vector<float> continuation(llama_model * model, llama_context * context) {
    common_batch batch(context);
    batch.add(3, 16, 0, true);
    require(llama_process(context, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0, "continuation");
    const auto * logits = llama_get_logits_ith(context, 0);
    return { logits, logits + llama_vocab_n_tokens(llama_model_get_vocab(model)) };
}
int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    try {
        llama_backend_init();
        const std::string path = (std::filesystem::path(argv[1]) / "synthetic-qwen4exp.gguf").string();
        auto metadata = get_gguf_ctx(LLM_ARCH_QWEN4EXP, true);
        {
            auto fixture = get_model_and_ctx(metadata.get(), nullptr, 42, 0.02f, {});
            llama_model_save_to_file(fixture.first.get(), path.c_str());
        }
        const auto shards = split_fixture(path);
        std::vector<const char *> shard_paths;
        for (const auto & item : shards) shard_paths.push_back(item.c_str());
        std::vector<std::pair<llama_model_ptr, llama_context_ptr>> sessions;
        for (int scenario = 0; scenario < 3; ++scenario) {
            const auto lazy = scenario == 0 ? LLAMA_LAZY_MODE_OFF : LLAMA_LAZY_MODE_ON;
            auto mp = llama_model_default_params();
            mp.n_gpu_layers = 0; mp.load_mode = LLAMA_LOAD_MODE_NONE; mp.lazy_mode = lazy;
            llama_model_ptr model(scenario == 2 ? llama_model_load_from_splits(shard_paths.data(), shard_paths.size(), mp)
                                                : llama_model_load_from_file(path.c_str(), mp));
            require(bool(model), "model load");
            if (!sessions.empty()) {
                require(llama_model_size(model.get()) == llama_model_size(sessions[0].first.get()), "logical model bytes must include the backing table");
                require(llama_model_n_params(model.get()) == llama_model_n_params(sessions[0].first.get()), "logical parameter count must include the backing table");
            }
            require(lcb_qwen4exp_file_backed_shard(model.get()) == (scenario == 2 ? 1 : scenario == 1 ? 0 : -1), "retained shard query");
            auto cp = llama_context_default_params();
            cp.n_ctx = 128; cp.n_batch = 32; cp.n_ubatch = 8; cp.n_threads = cp.n_threads_batch = 1;
            llama_context_ptr context(llama_init_from_model(model.get(), cp));
            require(bool(context), "context load");
            sessions.emplace_back(std::move(model), std::move(context));
        }
        // Exercise prefill microbatches, EOS resets, repeated hash rows and
        // graph reuse at unchanged batch width, then mixed embedding inputs.
        const std::vector<llama_token> tokens{1,2,2,0,3,4,4,5,0,6,7,1,2,0,5,7};
        auto & a = sessions[0];
        for (size_t scenario = 1; scenario < sessions.size(); ++scenario) {
        auto & b = sessions[scenario];
        llama_memory_clear(llama_get_memory(a.second.get()), true);
        equal_logits(get_logits(a.first.get(), a.second.get(), tokens),
                     get_logits(b.first.get(), b.second.get(), tokens));
        for (int iteration = 0; iteration < 2; ++iteration) {
            llama_memory_clear(llama_get_memory(a.second.get()), true);
            llama_memory_clear(llama_get_memory(b.second.get()), true);
            equal_logits(get_logits(a.first.get(), a.second.get(), tokens),
                         get_logits(b.first.get(), b.second.get(), tokens));
        }
        std::vector<float> embeddings(size_t(llama_model_n_embd_inp(a.first.get())) * (tokens.size()/4), 0.1f);
        for (bool mixed : {false, true}) {
            std::vector<float> la, lb;
            require(get_logits_mixed(a.first.get(), a.second.get(), tokens, embeddings, mixed, la) == 0, "resident mixed decode");
            require(get_logits_mixed(b.first.get(), b.second.get(), tokens, embeddings, mixed, lb) == 0, "file-backed mixed decode");
            equal_logits(la, lb);
        }
        // Save/restore n-gram predecessor and recurrent history, then reuse
        // the same one-token graph. No reader cursor is part of model state.
        llama_memory_clear(llama_get_memory(a.second.get()), true);
        llama_memory_clear(llama_get_memory(b.second.get()), true);
        equal_logits(get_logits(a.first.get(), a.second.get(), tokens),
                     get_logits(b.first.get(), b.second.get(), tokens));
        std::vector<uint8_t> state(llama_state_get_size(b.second.get()));
        const auto used = llama_state_get_data(b.second.get(), state.data(), state.size());
        require(used > 0 && used <= state.size(), "state capture");
        const auto expected = continuation(a.first.get(), a.second.get());
        equal_logits(expected, continuation(b.first.get(), b.second.get()));
        require(llama_state_set_data(b.second.get(), state.data(), used) == used, "state restore");
        equal_logits(expected, continuation(b.first.get(), b.second.get()));
        }
        // Read failure aborts before output publication. After restoring the
        // source and explicitly clearing history, no half-filled input survives.
        auto & streaming = sessions[2];
        std::ifstream input(shards[1], std::ios::binary);
        std::vector<char> saved((std::istreambuf_iterator<char>(input)), {});
        input.close();
        std::filesystem::resize_file(shards[1], 1);
        llama_memory_clear(llama_get_memory(streaming.second.get()), true);
        bool failed = false;
        try { get_logits(streaming.first.get(), streaming.second.get(), tokens); }
        catch (const std::exception &) { failed = true; }
        require(failed, "truncated retained shard should abort decode");
        {
            std::ofstream output(shards[1], std::ios::binary | std::ios::trunc);
            output.write(saved.data(), saved.size());
            require(bool(output), "restore retained shard");
        }
        llama_memory_clear(llama_get_memory(a.second.get()), true);
        llama_memory_clear(llama_get_memory(streaming.second.get()), true);
        equal_logits(get_logits(a.first.get(), a.second.get(), tokens),
                     get_logits(streaming.first.get(), streaming.second.get(), tokens));
        const auto export_path = path + ".incomplete-export.gguf";
        std::filesystem::remove(export_path);
        bool export_rejected = false;
        try { llama_model_save_to_file(streaming.first.get(), export_path.c_str()); }
        catch (const std::exception &) { export_rejected = true; }
        require(export_rejected && !std::filesystem::exists(export_path), "streaming export must fail before creating a file");
        // A global opt-in must not turn Gemma4 into an mmap-dependent model on
        // the same browser mounts. Keep its NONE load behavior resident.
        const auto gemma_path = path + ".gemma4.gguf";
        {
            auto gemma_metadata = get_gguf_ctx(LLM_ARCH_GEMMA4, false);
            // The generic upstream fixture still writes a legacy scalar here;
            // Gemma4 requires a per-layer boolean pattern.
            const auto layers = gguf_get_val_u32(gemma_metadata.get(), gguf_find_key(gemma_metadata.get(), "gemma4.block_count"));
            std::vector<uint8_t> sliding(layers, 1); sliding.back() = 0;
            gguf_set_arr_data(gemma_metadata.get(), "gemma4.attention.sliding_window_pattern", GGUF_TYPE_BOOL, sliding.data(), sliding.size());
            auto fixture = get_model_and_ctx(gemma_metadata.get(), nullptr, 42, 0.02f, {});
            // Use the complete fixture metadata and live tensor map: the
            // generic upstream saver omits some Gemma4 keys and tensors.
            gguf_context_ptr complete(gguf_init_empty());
            gguf_set_kv(complete.get(), gemma_metadata.get());
            for (const auto & item : llama_internal_get_tensor_map(fixture.first.get())) {
                if (gguf_find_tensor(complete.get(), item.first.c_str()) < 0) {
                    gguf_add_tensor(complete.get(), item.second);
                }
            }
            require(gguf_write_to_file(complete.get(), gemma_path.c_str(), false), "write complete Gemma4 fixture");
        }
        for (auto lazy : {LLAMA_LAZY_MODE_OFF, LLAMA_LAZY_MODE_ON}) {
            auto params = llama_model_default_params();
            params.n_gpu_layers = 0; params.load_mode = LLAMA_LOAD_MODE_NONE; params.lazy_mode = lazy;
            llama_model_ptr control(llama_model_load_from_file(gemma_path.c_str(), params));
            require(bool(control) && lcb_qwen4exp_file_backed_shard(control.get()) == -1, "non-Qwen4Exp control remains resident");
        }
        // Whole tensor integrity checking must not silently become row-only.
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 0; mp.load_mode = LLAMA_LOAD_MODE_NONE;
        mp.lazy_mode = LLAMA_LAZY_MODE_ON; mp.check_tensors = true;
        llama_model_ptr rejected(llama_model_load_from_file(path.c_str(), mp));
        require(!rejected, "check_tensors must reject streaming");
        sessions.clear();
        llama_backend_free();
        std::cout << "PASS synthetic Qwen4Exp resident/file-backed logits, prefill, EOS, graph reuse, mixed batches, IQ4_NL, metadata-only first of three shards, state restore, read-failure recovery and whole-tensor validation guard\n";
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
