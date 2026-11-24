#include "mtmd-vision-bf16-classification.h"
#include "support/check.hpp"
#include <iostream>
#include <string>

// These assertions protect the established semantic decisions, not just a case
// count. Enum exhaustiveness alone cannot diagnose a mistaken classification.
static_assert(lcb_mtmd_review_projector(PROJECTOR_TYPE_MUSE_GLIMMER) == MTMD_VISION_BF16_CANDIDATE_REVIEWED);
static_assert(lcb_mtmd_review_projector(PROJECTOR_TYPE_GEMMA4UV) == MTMD_VISION_BF16_CANDIDATE_REVIEWED);
static_assert(lcb_mtmd_review_projector(PROJECTOR_TYPE_LLAMA4) == MTMD_VISION_BF16_CANDIDATE_UNREVIEWED);
static_assert(lcb_mtmd_review_projector(PROJECTOR_TYPE_LFM3A_GEN) == MTMD_VISION_BF16_CANDIDATE_NOT_APPLICABLE);
static_assert(lcb_mtmd_review_projector(PROJECTOR_TYPE_MOSHI) == MTMD_VISION_BF16_CANDIDATE_NOT_APPLICABLE);
static_assert(lcb_mtmd_review_projector(PROJECTOR_TYPE_MOSHI_GEN) == MTMD_VISION_BF16_CANDIDATE_NOT_APPLICABLE);

int main() {
    for (auto type : {PROJECTOR_TYPE_MLP, PROJECTOR_TYPE_QWEN2VL, PROJECTOR_TYPE_QWEN25VL, PROJECTOR_TYPE_QWEN3VL,
                      PROJECTOR_TYPE_GEMMA3, PROJECTOR_TYPE_GEMMA4V, PROJECTOR_TYPE_GEMMA4UV, PROJECTOR_TYPE_LFM2,
                      PROJECTOR_TYPE_LFM3VL, PROJECTOR_TYPE_MUSE_GLIMMER}) {
        // Use the real upstream spelling, and fail if a reviewed name vanishes.
        const auto & name = PROJECTOR_TYPE_NAMES.at(type);
        BICORE_CHECK(lcb_mtmd_classify_metadata(1, name.c_str(), nullptr) == MTMD_VISION_BF16_CANDIDATE_REVIEWED);
    }
    BICORE_CHECK(lcb_mtmd_classify_metadata(1, "muse-glimmer", nullptr) == MTMD_VISION_BF16_CANDIDATE_REVIEWED);
    BICORE_CHECK(lcb_mtmd_classify_metadata(1, nullptr, "gemma4uv") == MTMD_VISION_BF16_CANDIDATE_REVIEWED);
    BICORE_CHECK(lcb_mtmd_classify_metadata(1, "llama4", "muse-glimmer") == MTMD_VISION_BF16_CANDIDATE_UNREVIEWED);
    BICORE_CHECK(lcb_mtmd_classify_metadata(1, "not-a-real-projector", "muse-glimmer") ==
                 MTMD_VISION_BF16_CANDIDATE_UNKNOWN);
    BICORE_CHECK(lcb_mtmd_classify_metadata(1, nullptr, nullptr) == MTMD_VISION_BF16_CANDIDATE_UNKNOWN);
    BICORE_CHECK(lcb_mtmd_classify_metadata(-1, "muse-glimmer", nullptr) == MTMD_VISION_BF16_CANDIDATE_UNKNOWN);
    BICORE_CHECK(lcb_mtmd_classify_metadata(2, "muse-glimmer", nullptr) == MTMD_VISION_BF16_CANDIDATE_UNKNOWN);
    BICORE_CHECK(lcb_mtmd_classify_metadata(0, "muse-glimmer", nullptr) == MTMD_VISION_BF16_CANDIDATE_NOT_APPLICABLE);
    const std::string long_name(128, 'a');
    BICORE_CHECK(lcb_mtmd_classify_metadata(1, long_name.c_str(), nullptr) == MTMD_VISION_BF16_CANDIDATE_UNKNOWN);
    std::cout << "metadata-only BF16 classification preserves reviewed, pending, audio and unknown cases\n";
}
