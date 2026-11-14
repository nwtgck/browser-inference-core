#include "mtmd-model.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "support/check.hpp"
#include "support/mtmd-metadata.hpp"
#include <filesystem>
#include <iostream>
#include <stdexcept>

int main(int argc, char ** argv) {
    BICORE_CHECK(argc == 2);
    const std::filesystem::path root(argv[1]);
    std::filesystem::create_directories(root);
    ggml_backend_register(ggml_backend_cpu_reg());
    const auto buft = ggml_backend_cpu_buffer_type();
    for (const auto flags : {std::pair{true, false}, std::pair{false, true}, std::pair{true, true}}) {
        const auto path = root / (std::to_string(flags.first) + std::to_string(flags.second) + ".gguf");
        write_mtmd_fixture(path, flags.first, flags.second, false);
        for (const bool gpu_requested : {false, true}) {
            unsigned progress = 0;
            mtmd_model_params params;
            params.vision_bf16_storage = MTMD_VISION_BF16_STORAGE_RETAIN_CHECKED;
            params.gpu_requested = gpu_requested;
            params.no_alloc = true;
            params.progress_callback_user_data = &progress;
            params.progress_callback = [](float, void * data) {
                ++*static_cast<unsigned *>(data);
                return true;
            };
            bool rejected = false;
            try {
                const auto models = mtmd_model_load(path.c_str(), params, buft);
                BICORE_CHECK(bool(models.v) == flags.first && bool(models.a) == flags.second && !models.gen_a);
                if (models.v) {
                    BICORE_CHECK(models.v->vision_bf16_storage == MTMD_VISION_BF16_STORAGE_RETAIN_CHECKED);
                    BICORE_CHECK(models.v->modality == MTMD_MODALITY_VISION && models.v->buf);
                }
                if (models.a) {
                    BICORE_CHECK(models.a->vision_bf16_storage == MTMD_VISION_BF16_STORAGE_LEGACY_EXPAND);
                    BICORE_CHECK(models.a->modality == MTMD_MODALITY_AUDIO && models.a->buf);
                }
                BICORE_CHECK(progress > 0);
            } catch (const std::runtime_error & error) {
                rejected = true;
                BICORE_CHECK(std::string(error.what()) ==
                             "retain-checked BF16: requested WebGPU backend is unavailable");
                BICORE_CHECK(progress == 0);
            }
            BICORE_CHECK(rejected == (flags.first && gpu_requested));
        }
    }
    std::cout << "real GGUF loader preserves audio policy and rejects unavailable vision retention before weights\n";
}
