#pragma once
#include "mtmd-vision-bf16-policy.h"
#include "clip.h"
#include "mtmd-vision-bf16-diagnostics-internal.h"

clip_init_result lcb_clip_init_with_vision_bf16_storage(const char * path, clip_context_params params,
                                                        mtmd_vision_bf16_storage storage,
                                                        mtmd_vision_bf16_load_report * report = nullptr);

mtmd_vision_bf16_statistics lcb_clip_vision_bf16_statistics(const clip_ctx * context);
