#include "stable-diffusion.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "ggml-cpu.h"
#ifdef SDCB_VARIANT_TEST
#include "browser-probes.h"
#endif
#include <stdint.h>
#include <stddef.h>
#include <errno.h>

// These setters are synchronous leaves even while generation is suspended.
// Type-check the real declarations without generating compatibility wrappers.
namespace {
[[maybe_unused]] constexpr void (*sync_preview_callback)(sd_preview_cb_t, enum preview_t, int, bool, bool,
                                                         void *) = &sd_set_preview_callback;
[[maybe_unused]] constexpr void (*sync_cancel_generation)(sd_ctx_t *, enum sd_cancel_mode_t) = &sd_cancel_generation;
} // namespace

#define BICORE_CONSTANT(name) BICORE_##name = (name)
enum class BicoreMacroConstants : int64_t {
    BICORE_CONSTANT(EINVAL),
    BICORE_CONSTANT(EIO),
    BICORE_CONSTANT(EOVERFLOW),
    BICORE_CONSTANT(EROFS),
    BICORE_CONSTANT(ENOENT),
    BICORE_CONSTANT(SEEK_SET),
    BICORE_CONSTANT(SEEK_CUR),
    BICORE_CONSTANT(SEEK_END),
    BICORE_CONSTANT(GGUF_VERSION),
    BICORE_CONSTANT(GGUF_DEFAULT_ALIGNMENT),
    BICORE_CONSTANT(GGML_MAX_DIMS),
    BICORE_CONSTANT(GGML_MAX_PARAMS),
    BICORE_CONSTANT(GGML_MAX_SRC),
    BICORE_CONSTANT(GGML_MAX_N_THREADS),
    BICORE_CONSTANT(GGML_MAX_OP_PARAMS),
    BICORE_CONSTANT(GGML_MAX_NAME),
    BICORE_CONSTANT(GGML_TENSOR_SIZE),
    BICORE_CONSTANT(GGML_DEFAULT_N_THREADS),
    BICORE_CONSTANT(GGML_DEFAULT_GRAPH_SIZE),
    BICORE_CONSTANT(GGML_MEM_ALIGN),
};
#undef BICORE_CONSTANT

#define BICORE_RECORD(name) BICORE_##name = sizeof(struct name)
#define BICORE_TYPE(name) BICORE_##name = sizeof(name)
enum class BicoreRecordLayouts : uint64_t {
    BICORE_RECORD(ggml_backend_dev_caps),
    BICORE_RECORD(ggml_backend_dev_props),
    BICORE_RECORD(ggml_backend_feature),
    BICORE_RECORD(ggml_backend_graph_copy),
    BICORE_RECORD(ggml_backend_meta_split_state),
    BICORE_TYPE(ggml_bf16_t),
    BICORE_RECORD(ggml_cplan),
    BICORE_RECORD(ggml_init_params),
    BICORE_RECORD(ggml_tallocr),
    BICORE_RECORD(ggml_tensor),
    BICORE_RECORD(ggml_threadpool_params),
    BICORE_RECORD(ggml_type_traits),
    BICORE_RECORD(ggml_type_traits_cpu),
    BICORE_RECORD(gguf_init_params),
    BICORE_TYPE(sd_adetailer_params_t),
    BICORE_TYPE(sd_audio_t),
    BICORE_TYPE(sd_cache_params_t),
    BICORE_TYPE(sd_ctx_params_t),
    BICORE_TYPE(sd_embedding_t),
    BICORE_TYPE(sd_generation_progress_t),
    BICORE_TYPE(sd_guidance_params_t),
    BICORE_TYPE(sd_hires_params_t),
    BICORE_TYPE(sd_image_generation_result_t),
    BICORE_TYPE(sd_image_preprocess_params_t),
    BICORE_TYPE(sd_image_result_t),
    BICORE_TYPE(sd_image_t),
    BICORE_TYPE(sd_img_gen_params_t),
    BICORE_TYPE(sd_lora_t),
    BICORE_TYPE(sd_memory_info_t),
    BICORE_TYPE(sd_pm_params_t),
    BICORE_TYPE(sd_preview_info_t),
    BICORE_TYPE(sd_pulid_params_t),
    BICORE_TYPE(sd_ref_video_t),
    BICORE_TYPE(sd_runtime_info_t),
    BICORE_TYPE(sd_sample_params_t),
    BICORE_TYPE(sd_slg_params_t),
    BICORE_TYPE(sd_tiling_params_t),
    BICORE_TYPE(sd_vid_gen_params_t),
    BICORE_TYPE(sd_video_generation_result_t),
    BICORE_TYPE(sd_video_upscale_params_t),
    BICORE_TYPE(sd_video_upscaler_ctx_params_t),
};
#undef BICORE_TYPE
#undef BICORE_RECORD

// These versions describe this owned browser interop implementation. Behavioral
// validation remains independent; a declaration alone is not an inference test.
enum class BicoreCapabilities : uint64_t {
    ggufFileOffsetBits = 64,
    safetensorsFileOffsetBits = 64,
    callerOwnedRandomAccess = 1,
    upstreamApi = 1,
    ggufShards = 1,
    ggufOriginalShapes = 1,
    bf16WeightRetention = 1,
    modelIoMask = 7,
};

#ifndef BICORE_ABI_EXTRACT
#include "fingerprint.h"
extern "C" {
uint32_t bicore_abi_version(void) {
    return 3;
}
uint32_t bicore_pointer_bytes(void) {
    return sizeof(void *);
}
const char * bicore_schema_hash(void) {
    return BICORE_ABI_FINGERPRINT;
}
uint32_t bicore_model_io_capabilities(void) {
    return static_cast<uint32_t>(BicoreCapabilities::modelIoMask);
}
}
#endif
