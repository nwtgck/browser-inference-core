#pragma once

#include "mtmd.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// A candidate is a reviewed projector family, NOT a guarantee about a GGUF,
// device, tensor role, or image graph. No file I/O or weight allocation occurs.
enum mtmd_vision_bf16_candidate {
    MTMD_VISION_BF16_CANDIDATE_UNKNOWN = 0,
    MTMD_VISION_BF16_CANDIDATE_REVIEWED = 1,
    MTMD_VISION_BF16_CANDIDATE_LEGACY_FIRST = 2,
    MTMD_VISION_BF16_CANDIDATE_NOT_APPLICABLE = 3,
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
enum mtmd_vision_bf16_effective_storage {
    MTMD_VISION_BF16_EFFECTIVE_UNAVAILABLE = 0,
    MTMD_VISION_BF16_EFFECTIVE_NOT_APPLICABLE = 1,
    MTMD_VISION_BF16_EFFECTIVE_UPSTREAM = 2,
    MTMD_VISION_BF16_EFFECTIVE_RETAINED = 3,
    MTMD_VISION_BF16_EFFECTIVE_EXPANDED = 4,
    MTMD_VISION_BF16_EFFECTIVE_NO_BF16 = 5,
};
// Logical payload of ORIGINAL BF16 vision weights only, not allocator capacity,
// physical GPU memory, audio weights, F32 weights, or language-model weights.
// Counts are valid only for RETAINED / EXPANDED / NO_BF16. Other states mean
// missing/not-applicable evidence, NOT a measured zero. No graph readback occurs.
struct mtmd_vision_bf16_statistics {
    uint32_t version;
    uint32_t size;
    int32_t effective_storage;
    uint64_t tensor_count;
    uint64_t source_bf16_bytes;
    uint64_t retained_bf16_bytes;
    uint64_t expanded_source_bf16_bytes;
    uint64_t expanded_f32_bytes;
};

MTMD_API uint32_t mtmd_vision_bf16_diagnostics_version(void);
// has_vision: 0=false, 1=true, any other value=unknown. Null/empty names mean
// absent metadata. Unknown/overlong names are not an instruction to retain.
MTMD_API int32_t mtmd_vision_bf16_classify_metadata(int32_t has_vision, const char * common_projector,
                                                    const char * vision_projector);
// One strict initialization only. No fallback or global last-error state.
// report_size must exactly match the generated record size. A null/wrong-size
// report returns null BEFORE file I/O and leaves caller memory untouched.
MTMD_API struct mtmd_context *
mtmd_init_from_file_with_vision_bf16_report(const char * path, const struct llama_model * text_model,
                                            const struct mtmd_context_params params, int32_t storage,
                                            struct mtmd_vision_bf16_load_report * report, uint32_t report_size);
// Only a successful context can publish statistics. Copies a fixed record;
// never scans weights, allocates GPU memory, synchronizes, or reads tensor data.
// Returns 1 on success, 0 on null/wrong-size arguments (output untouched).
MTMD_API int32_t mtmd_context_vision_bf16_statistics(const struct mtmd_context * context,
                                                     struct mtmd_vision_bf16_statistics * statistics,
                                                     uint32_t statistics_size);

#ifdef __cplusplus
}
#endif
