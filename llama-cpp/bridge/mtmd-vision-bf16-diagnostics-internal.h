#pragma once

#include "mtmd-vision-bf16-diagnostics.h"
#include <stdexcept>

// Only preupload role-review refusals use this type. Malformed layouts, I/O,
// allocation, cancellation, graph and device failures remain ordinary errors.
class lcb_mtmd_retention_refused : public std::runtime_error {
  public:
    const int32_t reason;
    lcb_mtmd_retention_refused(int32_t value, const char * message) : std::runtime_error(message), reason(value) {}
};

inline mtmd_vision_bf16_load_report lcb_mtmd_new_load_report() {
    return {1,
            sizeof(mtmd_vision_bf16_load_report),
            MTMD_VISION_BF16_LOAD_FAILED,
            MTMD_VISION_BF16_REASON_OTHER,
            MTMD_VISION_BF16_PHASE_UNKNOWN,
            MTMD_VISION_BF16_CLEANUP_UNKNOWN};
}
inline mtmd_vision_bf16_statistics lcb_mtmd_empty_statistics(int32_t state) {
    return {1, sizeof(mtmd_vision_bf16_statistics), state, 0, 0, 0, 0, 0};
}
inline mtmd_vision_bf16_statistics lcb_mtmd_weight_statistics(bool retained, uint64_t count, uint64_t source) {
    if (source > UINT64_MAX / 2)
        throw std::overflow_error("BF16 diagnostic size overflow");
    auto result = lcb_mtmd_empty_statistics(source == 0 ? MTMD_VISION_BF16_EFFECTIVE_NO_BF16
                                            : retained  ? MTMD_VISION_BF16_EFFECTIVE_RETAINED
                                                        : MTMD_VISION_BF16_EFFECTIVE_EXPANDED);
    result.tensor_count = count;
    result.source_bf16_bytes = source;
    result.retained_bf16_bytes = retained ? source : 0;
    result.expanded_source_bf16_bytes = retained ? 0 : source;
    result.expanded_f32_bytes = retained ? 0 : source * 2;
    return result;
}
