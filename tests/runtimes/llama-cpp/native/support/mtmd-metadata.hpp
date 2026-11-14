#pragma once
#include "mtmd-model.h"
#include "gguf.h"
#include "check.hpp"
#include <cstring>
#include <filesystem>

// Small synthetic GGUF files exercise actual loading, never model inference.
inline void write_mtmd_fixture(const std::filesystem::path & path, bool vision, bool audio, bool gen_audio) {
    gguf_context_ptr file(gguf_init_empty());
    gguf_set_val_bool(file.get(), KEY_HAS_VISION_ENC, vision);
    gguf_set_val_bool(file.get(), KEY_HAS_AUDIO_ENC, audio);
    gguf_set_val_bool(file.get(), KEY_HAS_GEN_AUDIO_ENC, gen_audio);
    gguf_set_val_str(file.get(), KEY_GEN_AUDIO_PROJ_TYPE, "mlp");
    gguf_set_val_str(file.get(), KEY_VISION_PROJ_TYPE, "mlp");
    gguf_set_val_str(file.get(), KEY_AUDIO_PROJ_TYPE, "qwen2a");
    for (const auto * prefix : {"vision", "audio", "gen.audio"}) {
        for (const auto * key : {KEY_N_EMBD, KEY_N_FF, KEY_PROJ_DIM}) {
            gguf_set_val_u32(file.get(), mtmd_format(key, prefix).c_str(), 4);
        }
        gguf_set_val_u32(file.get(), mtmd_format(KEY_N_HEAD, prefix).c_str(), 1);
        gguf_set_val_u32(file.get(), mtmd_format(KEY_N_BLOCK, prefix).c_str(), 0);
        gguf_set_val_f32(file.get(), mtmd_format(KEY_LAYER_NORM_EPS, prefix).c_str(), 1e-5f);
    }
    gguf_set_val_u32(file.get(), KEY_IMAGE_SIZE, 2);
    gguf_set_val_u32(file.get(), KEY_PATCH_SIZE, 2);
    gguf_set_val_u32(file.get(), KEY_A_NUM_MEL_BINS, 4);
    const float normalization[]{0.5f, 0.5f, 0.5f};
    gguf_set_arr_data(file.get(), KEY_IMAGE_MEAN, GGUF_TYPE_FLOAT32, normalization, 3);
    gguf_set_arr_data(file.get(), KEY_IMAGE_STD, GGUF_TYPE_FLOAT32, normalization, 3);
    ggml_context_ptr tensors(ggml_init({64 * 1024, nullptr, false}));
    auto tensor = [&](const std::string & name) {
        auto * value = ggml_new_tensor_2d(tensors.get(), GGML_TYPE_F32, 4, 4);
        ggml_set_name(value, name.c_str());
        std::memset(value->data, 0, ggml_nbytes(value));
        gguf_add_tensor(file.get(), value);
    };
    if (vision) {
        tensor(TN_PATCH_EMBD);
    }
    if (vision || gen_audio) {
        for (int layer : {0, 2}) {
            tensor(mtmd_format(TN_LLAVA_PROJ, layer, "weight"));
            tensor(mtmd_format(TN_LLAVA_PROJ, layer, "bias"));
        }
    }
    if (audio) {
        for (int layer : {1, 2}) {
            tensor(mtmd_format(TN_CONV1D, layer, "weight"));
            tensor(mtmd_format(TN_CONV1D, layer, "bias"));
        }
        tensor(mtmd_format(TN_MM_AUDIO_FC, "weight"));
        tensor(mtmd_format(TN_MM_AUDIO_FC, "bias"));
    }
    BICORE_CHECK(gguf_write_to_file(file.get(), path.c_str(), false));
}
