#pragma once
#include "mtmd-vision-bf16.h"
#include <stdexcept>

inline constexpr bool lcb_mtmd_retain_bf16(mtmd_vision_bf16_storage storage) {
    return storage == MTMD_VISION_BF16_STORAGE_RETAIN_CHECKED;
}

// An explicitly requested GPU run must not silently become CPU execution when
// the experimental mode is used. An explicitly CPU-only load stays unchanged.
inline void lcb_mtmd_check_retained_backend(mtmd_vision_bf16_storage storage, bool gpu_requested, bool webgpu_available) {
    if (lcb_mtmd_retain_bf16(storage) && gpu_requested && !webgpu_available) {
        throw std::runtime_error("retain-checked BF16: requested WebGPU backend is unavailable");
    }
}
