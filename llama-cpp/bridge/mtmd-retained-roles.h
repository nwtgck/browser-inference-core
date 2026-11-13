#pragma once

#include "clip-model.h"
#include "ggml.h"
#include "mtmd-vision-bf16-classification.h"
#include "mtmd-vision-bf16-diagnostics-internal.h"

#include <stdexcept>
#include <unordered_set>

// Deliberately coupled to the reviewed vision model roles. Do not infer roles
// from filename suffixes or accept every rank-2 tensor: a position table may be
// interpolated, and Gemma 3's input projection is transposed/CONT'ed. Reject such
// custom BF16 files BEFORE their graph builders can assert on an invalid type.
// A second guard checks the actual graph and scheduler on every shape.
inline void lcb_mtmd_check_retained_roles(const clip_model & model, ggml_context * weights) {
    bool has_bf16 = false;
    for (ggml_tensor * t = ggml_get_first_tensor(weights); t; t = ggml_get_next_tensor(weights, t)) {
        if (t->type != GGML_TYPE_BF16)
            continue;
        // Malformed layouts must not become a retryable unreviewed-projector
        // result, even when the family has not yet been classified for retain.
        if (t->ne[0] <= 0 || t->ne[1] <= 0 || t->ne[2] <= 0 || t->ne[3] <= 0 || !ggml_is_contiguous(t))
            throw std::runtime_error("retain-checked BF16: unsupported weight role or layout");
        has_bf16 = true;
    }
    if (!has_bf16)
        return;
    if (lcb_mtmd_review_projector(model.proj_type) != MTMD_VISION_BF16_CANDIDATE_REVIEWED) {
        throw lcb_mtmd_retention_refused(MTMD_VISION_BF16_REASON_PROJECTOR_UNREVIEWED,
                                         "retain-checked BF16: projector roles have not been reviewed");
    }
    std::unordered_set<const ggml_tensor *> matrices;
    switch (model.proj_type) {
    case PROJECTOR_TYPE_MLP:
        matrices.insert(model.mm_0_w);
        matrices.insert(model.mm_2_w);
        break;
    case PROJECTOR_TYPE_MLP_NORM:
        matrices.insert(model.mm_0_w);
        matrices.insert(model.mm_3_w);
        break;
    case PROJECTOR_TYPE_QWEN2VL:
    case PROJECTOR_TYPE_QWEN25VL:
    case PROJECTOR_TYPE_QWEN3VL:
    case PROJECTOR_TYPE_LFM3VL:
        matrices.insert(model.mm_0_w);
        matrices.insert(model.mm_1_w);
        break;
    case PROJECTOR_TYPE_LFM2:
        matrices.insert(model.mm_1_w);
        matrices.insert(model.mm_2_w);
        break;
    case PROJECTOR_TYPE_GEMMA3:
        // The standard converter already stores the transposed projection
        // as F16. Leave it that way; a custom BF16 version is not accepted.
        break;
    case PROJECTOR_TYPE_GEMMA4UV:
        // Unified vision has no transformer layers. Only its direct image
        // input projection is reviewed for BF16 retention.
        matrices.insert(model.mm_input_proj_w);
        break;
    case PROJECTOR_TYPE_GEMMA4V:
        matrices.insert(model.mm_input_proj_w);
        break;
    default:
        throw std::logic_error("retain-checked BF16: reviewed projector has no role implementation");
    }
    if (model.proj_type != PROJECTOR_TYPE_GEMMA4UV) {
        for (const clip_layer & layer : model.layers) {
            for (const ggml_tensor * t : {layer.q_w, layer.k_w, layer.v_w, layer.o_w, layer.qkv_w, layer.ff_up_w,
                                          layer.ff_down_w, layer.ff_gate_w}) {
                matrices.insert(t);
            }
            if (model.proj_type == PROJECTOR_TYPE_QWEN3VL) {
                matrices.insert(layer.deepstack_fc1_w);
                matrices.insert(layer.deepstack_fc2_w);
            }
            if (model.proj_type == PROJECTOR_TYPE_LFM3VL) {
                matrices.insert(layer.conv2d_in_proj_w);
                matrices.insert(layer.conv2d_out_proj_w);
            }
        }
    }
    for (ggml_tensor * t = ggml_get_first_tensor(weights); t; t = ggml_get_next_tensor(weights, t)) {
        if (t->type != GGML_TYPE_BF16)
            continue;
        const bool matrix_role = matrices.count(t) != 0;
        const bool lookup_role = model.proj_type == PROJECTOR_TYPE_GEMMA4V && t == model.position_embeddings;
        if ((matrix_role && (t->ne[2] != 1 || t->ne[3] != 1)) || (lookup_role && (t->ne[2] != 2 || t->ne[3] != 1))) {
            // A known role with a malformed shape is NOT an unreviewed role.
            throw std::runtime_error("retain-checked BF16: unsupported weight role or layout");
        }
        if (!matrix_role && !lookup_role) {
            throw lcb_mtmd_retention_refused(MTMD_VISION_BF16_REASON_WEIGHT_UNREVIEWED,
                                             "retain-checked BF16: unsupported weight role or layout");
        }
    }
}
