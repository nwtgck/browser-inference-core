#pragma once

#include <stdint.h>

// Test-variant C declarations are shared by the definitions, native consumers
// and compiler-derived ABI inspection. They are absent from browser variants.
#ifdef __cplusplus
extern "C" {
#endif
uint64_t sdc_test_gguf_offset(const char * path);
uint32_t sdc_test_gguf_value(const char * path);
void sdc_test_callbacks(void);
uint64_t sdc_test_safetensors_offset(const char * path);
uint32_t sdc_test_safetensors_value(const char * path);
uint32_t sdc_test_model_tensor_count(const char * path);
uint64_t sdc_test_gguf_shape(const char * path, const char * tensor_name, int dimension);
int sdc_test_gguf_original_shape_values(const char * path);
int sdc_test_qwen_timestep(const char * backend_name);
int sdc_test_bf16_weights(const char * backend_name);
int sdc_test_graph_walk(void);
int sdc_test_conv3d_bias(const char * backend_name);
int sdc_test_webgpu_performance(const char * backend_name);
#ifdef __cplusplus
}

// A typed registration of suspension policy, without repeating signatures or
// assembling wrappers. sizeof also diagnoses a removed/renamed declaration.
enum class BicoreSuspendingTests : uint64_t {
    BICORE_sdc_test_qwen_timestep = sizeof(&sdc_test_qwen_timestep),
    BICORE_sdc_test_bf16_weights = sizeof(&sdc_test_bf16_weights),
    BICORE_sdc_test_conv3d_bias = sizeof(&sdc_test_conv3d_bias),
    BICORE_sdc_test_webgpu_performance = sizeof(&sdc_test_webgpu_performance),
};
#endif
