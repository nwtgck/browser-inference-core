#pragma once

#include "mtmd.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Family review is not a guarantee about a particular file, graph or device.
// UNREVIEWED remains visible as review work, rather than implying incompatibility.
enum mtmd_vision_bf16_candidate {
    MTMD_VISION_BF16_CANDIDATE_UNKNOWN = 0,
    MTMD_VISION_BF16_CANDIDATE_REVIEWED = 1,
    MTMD_VISION_BF16_CANDIDATE_LEGACY_FIRST = 2,
    MTMD_VISION_BF16_CANDIDATE_NOT_APPLICABLE = 3,
    MTMD_VISION_BF16_CANDIDATE_UNREVIEWED = 4,
};
enum mtmd_vision_bf16_load_status {
    MTMD_VISION_BF16_LOAD_FAILED = 0,
    MTMD_VISION_BF16_LOAD_SUCCEEDED = 1,
    MTMD_VISION_BF16_LOAD_RETENTION_REFUSED = 2,
};
enum mtmd_vision_bf16_load_reason {
    MTMD_VISION_BF16_REASON_NONE = 0,
    MTMD_VISION_BF16_REASON_OTHER = 1,
    MTMD_VISION_BF16_REASON_PROJECTOR_UNREVIEWED = 2,
    MTMD_VISION_BF16_REASON_WEIGHT_UNREVIEWED = 3,
    MTMD_VISION_BF16_REASON_INVALID_MODE = 4,
};
enum mtmd_vision_bf16_load_phase {
    MTMD_VISION_BF16_PHASE_UNKNOWN = 0,
    MTMD_VISION_BF16_PHASE_PREUPLOAD = 1,
    MTMD_VISION_BF16_PHASE_CONTEXT = 2,
    MTMD_VISION_BF16_PHASE_COMPLETE = 3,
};
enum mtmd_vision_bf16_cleanup {
    MTMD_VISION_BF16_CLEANUP_UNKNOWN = 0,
    MTMD_VISION_BF16_CLEANUP_COMPLETE = 1,
    MTMD_VISION_BF16_CLEANUP_NOT_REQUIRED = 2,
};
struct mtmd_vision_bf16_load_report {
    uint32_t version;
    uint32_t size;
    int32_t status;
    int32_t reason;
    int32_t phase;
    int32_t cleanup;
};

MTMD_API uint32_t mtmd_vision_bf16_diagnostics_version(void);
// No file I/O, weight allocation or GPU work. Names follow the upstream metadata
// precedence. has_vision is 0/1; other values or unknown names are UNKNOWN.
MTMD_API int32_t mtmd_vision_bf16_classify_metadata(int32_t has_vision, const char * common_projector,
                                                    const char * vision_projector);
// Exactly one strict load. Invalid report storage is rejected before file I/O
// without touching caller memory. Only dedicated preupload review refusals are
// retryable; ordinary I/O, allocation, cancellation and graph failures are not.
MTMD_API struct mtmd_context *
mtmd_init_from_file_with_vision_bf16_report(const char * path, const struct llama_model * text_model,
                                            const struct mtmd_context_params params, int32_t storage,
                                            struct mtmd_vision_bf16_load_report * report, uint32_t report_size);

#ifdef __cplusplus
}
#endif
