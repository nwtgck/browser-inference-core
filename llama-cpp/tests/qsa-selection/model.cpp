// Two separately built executables run the same saved fixture. Do not compare
// two modes that accidentally share the same compiled optimization.
#define main lcb_unused_architecture_test_main
#include "test-llama-archs.cpp"
#undef main
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <chrono>

static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

static void generate(const std::string & path, ggml_type quant) {
    auto metadata = get_gguf_ctx(LLM_ARCH_QWEN4EXP, true);
    llama_model_saver ms(LLM_ARCH_QWEN4EXP, metadata.get());
    ms.add_kv(LLM_KV_CONTEXT_LENGTH, uint32_t(1024));
    ms.add_kv(LLM_KV_BLOCK_COUNT, uint32_t(4)); // two attention layers, not just one
    ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS, std::vector<uint32_t>(4, 4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, uint32_t(4));
    // 64 selected pools. Upstream pads to 64 then 128 as the real count grows,
    // so a small fixture exercises both branches and a live threshold crossing.
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K, uint32_t(256));
    {
        auto fixture = get_model_and_ctx(metadata.get(), nullptr, 913, 0.02f, {});
        llama_model_save_to_file(fixture.first.get(), path.c_str());
    }
    ggml_context * raw = nullptr;
    gguf_context_ptr input(gguf_init_from_file(path.c_str(), {false, &raw}));
    ggml_context_ptr data(raw);
    require(bool(input) && bool(data), "read synthetic model");
    gguf_context_ptr output(gguf_init_empty());
    gguf_set_kv(output.get(), input.get());
    std::vector<std::vector<uint64_t>> storage;
    size_t quantized_count = 0;
    for (int64_t i = 0; i < gguf_get_n_tensors(input.get()); ++i) {
        const char * name = gguf_get_tensor_name(input.get(), i);
        const auto * tensor = ggml_get_tensor(data.get(), name);
        require(tensor && tensor->data, "tensor data");
        gguf_add_tensor(output.get(), tensor);
        const bool ple = std::strcmp(name, "per_layer_token_embd.weight") == 0;
        const bool matrix = tensor->ne[1] > 1 && tensor->ne[0] % 256 == 0 &&
                            std::strstr(name, "norm") == nullptr;
        if (!ple && (!matrix || quant == GGML_TYPE_F16)) continue;
        const auto type = ple ? GGML_TYPE_IQ4_NL : quant;
        std::vector<float> floats(ggml_nelements(tensor));
        if (tensor->type == GGML_TYPE_F32) std::memcpy(floats.data(), tensor->data, floats.size()*sizeof(float));
        else ggml_get_type_traits(tensor->type)->to_float(tensor->data, floats.data(), floats.size());
        const auto bytes = ggml_row_size(type, tensor->ne[0]) * (floats.size()/tensor->ne[0]);
        storage.emplace_back((bytes + 7)/8);
        ggml_get_type_traits(type)->from_float_ref(floats.data(), storage.back().data(), floats.size());
        gguf_set_tensor_type(output.get(), name, type);
        gguf_set_tensor_data(output.get(), name, storage.back().data());
        if (!ple) ++quantized_count;
    }
    require(quant == GGML_TYPE_F16 || quantized_count > 10, "quantized matrix coverage");
    require(gguf_write_to_file(output.get(), path.c_str(), false), "save quantized fixture");
    std::cout << "fixture=" << ggml_type_name(quant) << " quantized_matrices=" << quantized_count << '\n';
}

struct capture {
    std::map<std::string, size_t> ops;
    std::map<int64_t, size_t> pools;
    size_t top_k_selections = 0;
    size_t query_projections = 0;
    size_t raw_key_projections = 0;
    size_t selection_masks = 0;
    static bool callback(ggml_tensor * t, bool ask, void * opaque) {
        if (!ask) return true;
        auto & self = *static_cast<capture *>(opaque);
        self.ops[ggml_op_name(t->op)]++;
        if (t->op == GGML_OP_TOP_K && std::strncmp(t->name, "indexer_top_k-", 14) == 0) ++self.top_k_selections;
        if (t->op == GGML_OP_LIGHTNING_INDEXER) self.pools[t->ne[0]]++;
        if (std::strncmp(t->name, "indexer_q-", 10) == 0) ++self.query_projections;
        if (std::strncmp(t->name, "indexer_k_raw-", 14) == 0) ++self.raw_key_projections;
        if (std::strncmp(t->name, "indexer_sel-", 12) == 0) ++self.selection_masks;
        return false; // metadata only; never request a node's completion
    }
    void record(std::ostream & out, const std::string & phase) {
        out << phase << ',' << ops["LIGHTNING_INDEXER"] << ',' << top_k_selections << ','
            << query_projections << ',' << raw_key_projections << ',' << selection_masks << ',' << pools[64] << ',' << pools[128] << '\n';
        ops.clear(); pools.clear(); top_k_selections = query_projections = raw_key_projections = selection_masks = 0;
    }
};

static void trace(const std::string & path, const std::filesystem::path & directory, bool observe) {
    std::filesystem::create_directories(directory);
    std::ofstream values(directory / "logits.f32", std::ios::binary);
    std::ofstream counts(directory / "nodes.csv");
    counts << "phase,indexer,top_k,query,raw_key,selection_mask,pools64,pools128\n";
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0; mp.load_mode = LLAMA_LOAD_MODE_NONE; mp.lazy_mode = LLAMA_LAZY_MODE_ON;
    llama_model_ptr model(llama_model_load_from_file(path.c_str(), mp));
    require(bool(model), "load fixture");
    capture seen;
    auto cp = llama_context_default_params();
    cp.n_ctx = 1024; cp.n_seq_max = 2; cp.n_batch = 128; cp.n_ubatch = 32;
    cp.n_threads = cp.n_threads_batch = 1;
    if (observe) { cp.cb_eval = capture::callback; cp.cb_eval_user_data = &seen; }
    llama_context_ptr context(llama_init_from_model(model.get(), cp));
    require(bool(context), "create context");
    seen.record(counts, "reserve");
    const size_t vocabulary = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    auto write = [&](const float * data, size_t size) {
        require(data != nullptr, "missing logits");
        for (size_t i = 0; i < size; ++i) require(std::isfinite(data[i]), "nonfinite logits");
        values.write(reinterpret_cast<const char *>(data), size*sizeof(float));
        require(bool(values), "trace write");
    };
    auto decode = [&](int from, int to, int sequence, const std::string & phase) {
        common_batch batch(context.get());
        for (int pos = from; pos < to; ++pos) {
            // Include EOS and repeated n-grams without using model-generated
            // tokens: later inputs must be identical in the two implementations.
            const llama_token token = pos % 127 == 0 ? 0 : 1 + (pos*37 + 13 + sequence*11) % 127;
            batch.add(token, pos, sequence, true);
        }
        require(llama_process(context.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0, "decode fixture");
        for (int i = 0; i < to - from; ++i) write(llama_get_logits_ith(context.get(), i), vocabulary);
        seen.record(counts, phase);
    };
    const auto started = std::chrono::steady_clock::now();
    decode(0, 8, 0, "short-prefill");
    for (int pos = 8; pos < 32; ++pos) decode(pos, pos+1, 0, "short-reused");
    for (int pos = 32; pos < 224; pos += 32) decode(pos, pos+32, 0, "prefill");
    decode(224, 248, 0, "before-threshold");
    std::vector<uint8_t> state(llama_state_get_size(context.get()));
    const auto used = llama_state_get_data(context.get(), state.data(), state.size());
    require(used > 0 && used <= state.size(), "state capture");
    for (int pos = 248; pos < 272; ++pos) decode(pos, pos+1, 0, "cross-threshold");
    require(llama_state_set_data(context.get(), state.data(), used) == used, "state restore");
    for (int pos = 248; pos < 272; ++pos) decode(pos, pos+1, 0, "restore-cross-threshold");
    decode(272, 304, 0, "ranked-prefill");
    auto * memory = llama_get_memory(context.get());
    llama_memory_clear(memory, true);
    decode(0, 32, 0, "shared-prefix");
    llama_memory_seq_cp(memory, 0, 1, -1, -1);
    decode(32, 36, 0, "sequence-zero-divergence");
    decode(32, 36, 1, "sequence-one-divergence");
    require(llama_memory_seq_rm(memory, 1, -1, -1), "remove secondary sequence");
    decode(36, 40, 0, "after-sequence-removal");
    const std::vector<llama_token> tokens{1,2,2,0,3,4,4,5,0,6,7,1,2,0,5,7};
    std::vector<float> embeddings(size_t(llama_model_n_embd_inp(model.get())) * (tokens.size()/4), 0.1f);
    for (bool mixed : {false, true}) {
        std::vector<float> result;
        require(get_logits_mixed(model.get(), context.get(), tokens, embeddings, mixed, result) == 0, "mixed input");
        write(result.data(), result.size()); seen.record(counts, mixed ? "mixed" : "embedding-chunks");
    }
    // Simultaneous sequences exercise the batched visibility/layout contract,
    // not only two sequence IDs decoded in separate calls.
    // Keep earlier cases on the default cache layout; exercise unified-cache
    // batched sequences separately. PLE forbids multiple IDs on one input token,
    // so share the prefix via the supported state-copy API instead.
    context.reset();
    cp.kv_unified = true;
    context.reset(llama_init_from_model(model.get(), cp));
    require(bool(context), "create unified context");
    seen.record(counts, "parallel-reserve");
    {
        common_batch batch(context.get());
        for (int pos = 0; pos < 32; ++pos) batch.add(1 + pos % 127, pos, 0, true);
        require(llama_process(context.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0, "unified prefix");
        for (int i = 0; i < 32; ++i) write(llama_get_logits_ith(context.get(), i), vocabulary);
        llama_memory_seq_cp(llama_get_memory(context.get()), 0, 1, -1, -1);
        seen.record(counts, "unified-prefix");
    }
    {
        common_batch batch(context.get());
        for (int sequence = 0; sequence < 2; ++sequence) {
            for (int pos = 32; pos < 36; ++pos) batch.add(1 + (pos*3 + sequence*7) % 127, pos, sequence, true);
        }
        require(llama_process(context.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0, "parallel sequences");
        for (int i = 0; i < 8; ++i) write(llama_get_logits_ith(context.get(), i), vocabulary);
        seen.record(counts, "parallel-sequences");
    }
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    std::cout << "PASS trace seconds=" << elapsed << " observation=" << observe << '\n';
}

int main(int argc, char ** argv) {
    if (argc != 4) { std::cerr << "usage: model generate|trace|benchmark MODEL PATH_OR_Q3_K_OR_Q4_K_OR_F16\n"; return 2; }
    try {
        llama_backend_init();
        const std::string command(argv[1]);
        if (command == "generate") {
            const std::string quant(argv[3]);
            require(quant == "Q3_K" || quant == "Q4_K" || quant == "F16", "quantization argument");
            generate(argv[2], quant == "Q3_K" ? GGML_TYPE_Q3_K : quant == "Q4_K" ? GGML_TYPE_Q4_K : GGML_TYPE_F16);
        } else {
            require(command == "trace" || command == "benchmark", "command argument");
            trace(argv[2], argv[3], command == "trace");
        }
        llama_backend_free();
    } catch (const std::exception & e) { std::cerr << "FAIL " << e.what() << '\n'; return 1; }
}
