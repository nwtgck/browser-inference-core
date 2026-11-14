// Actual upstream GGUF selection plus the production reader; no model graph.
#include "llama-model-loader.h"
#include "file-backed-rows.h"
#include "gguf.h"
#include <cstdio>
#include <filesystem>
#include <memory>

static void check(bool ok, const char * message) {
    if (!ok)
        throw std::runtime_error(message);
}

static void write_shard(const std::string & path, uint16_t index, const char * name, bool ple) {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> data(ggml_init({64 * 1024, nullptr, false}), ggml_free);
    std::unique_ptr<gguf_context, decltype(&gguf_free)> meta(gguf_init_empty(), gguf_free);
    check(bool(data) && bool(meta), "fixture contexts");
    auto * tensor = ggml_new_tensor_2d(data.get(), GGML_TYPE_F32, ple ? 2 : 1, ple ? 3 : 1);
    ggml_set_name(tensor, name);
    for (int64_t i = 0; i < ggml_nelements(tensor); ++i)
        static_cast<float *>(tensor->data)[i] = float(i + 1);
    gguf_set_val_str(meta.get(), "general.architecture", "qwen4exp");
    gguf_set_val_u16(meta.get(), "split.no", index);
    gguf_set_val_u16(meta.get(), "split.count", 3);
    gguf_set_val_i32(meta.get(), "split.tensors.count", 3);
    gguf_add_tensor(meta.get(), tensor);
    check(gguf_write_to_file(meta.get(), path.c_str(), false), "write authored GGUF");
}

int main(int argc, char ** argv) {
    if (argc != 2)
        return 2;
    try {
        const std::string directory = argv[1];
        std::filesystem::create_directories(directory);
        std::vector<std::string> paths;
        for (size_t i = 0; i < 3; ++i) {
            char name[64];
            std::snprintf(name, sizeof(name), "/model-%05zu-of-00003.gguf", i + 1);
            paths.push_back(directory + name);
        }
        paths.push_back(directory + "/replacement.gguf");
        write_shard(paths[0], 0, "a.weight", false);
        write_shard(paths[1], 1, "per_layer_token_embd.weight", true);
        write_shard(paths[2], 2, "b.weight", false);
        write_shard(paths[3], 1, "per_layer_token_embd.weight", true);
        for (int scenario = 0; scenario < 2; ++scenario) {
            if (scenario == 1) {
                std::filesystem::remove(paths[1]);
                std::swap(paths[1], paths[3]);
            }
            const auto supplied = paths;
            llama_model_loader loader(nullptr, nullptr, nullptr, paths.front(), paths, nullptr, LLAMA_LOAD_MODE_NONE,
                                      false, false, false, nullptr, nullptr);
            const auto * weight = loader.get_weight("per_layer_token_embd.weight");
            check(weight && loader.files.size() == 3 && weight->idx == 2, "actual compacted loader file index");
            const auto & file = loader.files.at(weight->idx);
            lcb_file_backed_rows reader(file->file_id(), file->size(), weight->offs, weight->tensor->type, 2, 3, 0);
            reader.resolve_shard(supplied);
            const int32_t expected = scenario == 0 ? 3 : 1;
            check(reader.shard() == expected && reader.shard() != weight->idx, "original supplied input index");
            std::vector<float> output;
            const int32_t indices[] = {2, 0, 2};
            reader.gather(indices, 3, output);
            check(output == std::vector<float>({5, 6, 1, 2, 5, 6}), "selected replacement file and tensor offset");
            std::printf("PASS loader scenario=%d actual_file_index=%u supplied_index=%d files=%zu\n", scenario,
                        unsigned(weight->idx), reader.shard(), loader.files.size());
        }
        std::filesystem::remove_all(directory);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
