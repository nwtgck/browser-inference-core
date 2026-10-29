#pragma once

#include "mtmd.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Values are stable. INVALID is returned only by a query on a null context.
enum mtmd_vision_bf16_storage {
    MTMD_VISION_BF16_STORAGE_INVALID = -1,
    MTMD_VISION_BF16_STORAGE_LEGACY_EXPAND = 0,
    MTMD_VISION_BF16_STORAGE_RETAIN_CHECKED = 1,
};

MTMD_API uint32_t mtmd_vision_bf16_storage_version(void);
// Select before loading. Unknown i32 values fail before file I/O. The old init
// entry retains legacy behavior. An explicit CPU load keeps upstream storage;
// its success is not evidence of checked WebGPU placement.
// Audio-only loads keep their original behavior in every profile.
MTMD_API struct mtmd_context * mtmd_init_from_file_with_vision_bf16_storage(
    const char * path, const struct llama_model * text_model,
    const struct mtmd_context_params params, int32_t storage);
// Requested, immutable policy only; not actual backend/weight-placement status.
MTMD_API enum mtmd_vision_bf16_storage mtmd_context_vision_bf16_storage(
    const struct mtmd_context * context);

#ifdef __cplusplus
}
#endif
