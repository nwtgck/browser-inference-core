#pragma once

#include "mtmd-vision-bf16-diagnostics.h"
#include <stdexcept>

// Used only by metadata role checks before vision storage allocation/upload.
// Layout, backend, graph, cancellation and general failures keep their types.
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
