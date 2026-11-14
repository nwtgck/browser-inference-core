#pragma once

#include <stdint.h>

struct llama_model;

#ifdef __cplusplus
extern "C" {
#endif

uint32_t bicore_file_backed_rows_version(void);
int32_t bicore_model_file_backed_shard(const struct llama_model * model);

#ifdef __cplusplus
}
#endif
