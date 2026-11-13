#include "mtmd-vision-bf16-classification.h"
#include "mtmd-vision-bf16-diagnostics-internal.h"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

static void require(bool value, const char * message) {
    if (!value)
        throw std::runtime_error(message);
}

int main() {
    require(lcb_mtmd_classify_metadata(0, "gemma4uv", nullptr) == MTMD_VISION_BF16_CANDIDATE_NOT_APPLICABLE,
            "no vision is not applicable");
    for (int32_t flag : {-1, 2, std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max()})
        require(lcb_mtmd_classify_metadata(flag, "gemma4uv", nullptr) == MTMD_VISION_BF16_CANDIDATE_UNKNOWN,
                "invalid vision flag must be unknown");
    require(lcb_mtmd_classify_metadata(1, nullptr, nullptr) == MTMD_VISION_BF16_CANDIDATE_UNKNOWN, "missing names");
    require(lcb_mtmd_classify_metadata(1, "", "gemma4uv") == MTMD_VISION_BF16_CANDIDATE_REVIEWED, "vision fallback");
    require(lcb_mtmd_classify_metadata(1, "unrecognized", "gemma4uv") == MTMD_VISION_BF16_CANDIDATE_UNKNOWN,
            "common name precedence must match upstream");
    require(lcb_mtmd_classify_metadata(1, std::string(128, 'x').c_str(), "gemma4uv") ==
                MTMD_VISION_BF16_CANDIDATE_UNKNOWN,
            "bounded name");
    size_t reviewed = 0;
    for (const auto & entry : PROJECTOR_TYPE_NAMES) {
        const auto resolved = entry.first == PROJECTOR_TYPE_QWEN25O ? PROJECTOR_TYPE_QWEN25VL : entry.first;
        require(lcb_mtmd_classify_metadata(1, entry.second.c_str(), nullptr) == lcb_mtmd_review_projector(resolved),
                "all upstream names and vision aliases");
        reviewed += lcb_mtmd_review_projector(entry.first) == MTMD_VISION_BF16_CANDIDATE_REVIEWED;
    }
    require(reviewed == 9, "review inventory changed: review the corresponding role guard");
    // MLP_NORM is a loader-resolved subtype and has no standalone GGUF name.
    require(lcb_mtmd_review_projector(PROJECTOR_TYPE_MLP_NORM) == MTMD_VISION_BF16_CANDIDATE_REVIEWED,
            "review the loader-resolved normalized MLP subtype");
    const auto initial = lcb_mtmd_new_load_report();
    require(initial.version == 1 && initial.size == sizeof(initial) && initial.status == MTMD_VISION_BF16_LOAD_FAILED &&
                initial.cleanup == MTMD_VISION_BF16_CLEANUP_UNKNOWN,
            "failure and cleanup are conservative by default");
    const auto retained = lcb_mtmd_weight_statistics(true, 1, 29491200);
    require(retained.effective_storage == MTMD_VISION_BF16_EFFECTIVE_RETAINED && retained.tensor_count == 1 &&
                retained.source_bf16_bytes == 29491200 && retained.retained_bf16_bytes == 29491200 &&
                retained.expanded_source_bf16_bytes == 0 && retained.expanded_f32_bytes == 0,
            "Gemma4UV image projection retained bytes");
    const auto expanded = lcb_mtmd_weight_statistics(false, 1, 29491200);
    require(expanded.effective_storage == MTMD_VISION_BF16_EFFECTIVE_EXPANDED &&
                expanded.expanded_source_bf16_bytes == 29491200 && expanded.expanded_f32_bytes == 58982400 &&
                expanded.retained_bf16_bytes == 0,
            "expanded bytes are not GPU physical allocation");
    const auto none = lcb_mtmd_weight_statistics(true, 0, 0);
    require(none.effective_storage == MTMD_VISION_BF16_EFFECTIVE_NO_BF16 && none.tensor_count == 0,
            "observed no-BF16 is distinct from unobserved");
    require(lcb_mtmd_empty_statistics(MTMD_VISION_BF16_EFFECTIVE_UPSTREAM).effective_storage ==
                MTMD_VISION_BF16_EFFECTIVE_UPSTREAM,
            "unmeasured upstream storage");
    bool rejected = false;
    try {
        (void)lcb_mtmd_weight_statistics(false, 1, std::numeric_limits<uint64_t>::max());
    } catch (const std::overflow_error &) {
        rejected = true;
    }
    require(rejected, "no overflowing diagnostic counters");
    try {
        throw lcb_mtmd_retention_refused(MTMD_VISION_BF16_REASON_WEIGHT_UNREVIEWED, "fixed refusal");
    } catch (const lcb_mtmd_retention_refused & error) {
        require(error.reason == MTMD_VISION_BF16_REASON_WEIGHT_UNREVIEWED, "typed reason");
    }
    std::cout << "all BF16 diagnostic and classification contracts passed; upstream names="
              << PROJECTOR_TYPE_NAMES.size() << '\n';
}
