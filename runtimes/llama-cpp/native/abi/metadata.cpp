#include "llama.h"
#include "gguf.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "mtmd-vision-bf16.h"
#include "model-graph.h"
#include "metadata.h"
#include "webgpu-q8-diagnostic.h"
#include <stdint.h>
#include <stddef.h>
#include <errno.h>

// These reviewed leaves may run inside a synchronous native callback. Ordinary
// typed registrations diagnose header signature changes without new wrappers.
namespace {
[[maybe_unused]] constexpr const char * (*sync_op_desc)(const ggml_tensor *) = &ggml_op_desc;
[[maybe_unused]] constexpr const char * (*sync_buffer_name)(ggml_backend_buffer_t) = &ggml_backend_buffer_name;
[[maybe_unused]] constexpr bool (*sync_buffer_is_host)(ggml_backend_buffer_t) = &ggml_backend_buffer_is_host;
[[maybe_unused]] constexpr size_t (*sync_device_count)(void) = &ggml_backend_dev_count;
[[maybe_unused]] constexpr ggml_backend_dev_t (*sync_device_get)(size_t) = &ggml_backend_dev_get;
[[maybe_unused]] constexpr const char * (*sync_device_name)(ggml_backend_dev_t) = &ggml_backend_dev_name;
[[maybe_unused]] constexpr bool (*sync_device_supports_op)(ggml_backend_dev_t,
                                                           const ggml_tensor *) = &ggml_backend_dev_supports_op;
} // namespace

// This ordinary declaration asks the target compiler to evaluate public macros.
// Record layout and enum values come from the same target's compiler metadata.
#define BICORE_CONSTANT(name) BICORE_##name = (name)
enum class BicoreMacroConstants : int64_t {
    BICORE_CONSTANT(LLAMA_DEFAULT_SEED),
    BICORE_CONSTANT(LLAMA_TOKEN_NULL),
    BICORE_CONSTANT(LLAMA_DRY_MAX_SEQ_BREAKERS),
    BICORE_CONSTANT(LLAMA_FILE_MAGIC_GGLA),
    BICORE_CONSTANT(LLAMA_FILE_MAGIC_GGSN),
    BICORE_CONSTANT(LLAMA_FILE_MAGIC_GGSQ),
    BICORE_CONSTANT(LLAMA_SESSION_MAGIC),
    BICORE_CONSTANT(LLAMA_SESSION_VERSION),
    BICORE_CONSTANT(LLAMA_STATE_SEQ_MAGIC),
    BICORE_CONSTANT(LLAMA_STATE_SEQ_VERSION),
    BICORE_CONSTANT(LLAMA_STATE_SEQ_FLAGS_NONE),
    BICORE_CONSTANT(LLAMA_STATE_SEQ_FLAGS_SWA_ONLY),
    BICORE_CONSTANT(LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY),
    BICORE_CONSTANT(LLAMA_STATE_SEQ_FLAGS_ON_DEVICE),
    BICORE_CONSTANT(GGUF_VERSION),
    BICORE_CONSTANT(GGUF_DEFAULT_ALIGNMENT),
    BICORE_CONSTANT(GGML_FILE_MAGIC),
    BICORE_CONSTANT(GGML_FILE_VERSION),
    BICORE_CONSTANT(GGML_QNT_VERSION),
    BICORE_CONSTANT(GGML_QNT_VERSION_FACTOR),
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
    BICORE_CONSTANT(GGML_EXIT_SUCCESS),
    BICORE_CONSTANT(GGML_EXIT_ABORTED),
    BICORE_CONSTANT(GGML_ROPE_TYPE_NORMAL),
    BICORE_CONSTANT(GGML_ROPE_TYPE_NEOX),
    BICORE_CONSTANT(GGML_ROPE_TYPE_MROPE),
    BICORE_CONSTANT(GGML_ROPE_TYPE_VISION),
    BICORE_CONSTANT(GGML_ROPE_TYPE_IMROPE),
    BICORE_CONSTANT(GGML_MROPE_SECTIONS),
    BICORE_CONSTANT(GGML_N_TASKS_MAX),
    BICORE_CONSTANT(GGML_BACKEND_META_MAX_DEVICES),
    BICORE_CONSTANT(EINVAL),
    BICORE_CONSTANT(EIO),
    BICORE_CONSTANT(EOVERFLOW),
    BICORE_CONSTANT(EROFS),
    BICORE_CONSTANT(ENOENT),
    BICORE_CONSTANT(SEEK_SET),
    BICORE_CONSTANT(SEEK_CUR),
    BICORE_CONSTANT(SEEK_END),
};
#undef BICORE_CONSTANT

// Naming a public type is enough to ask Clang for its actual target layout.
// The binding reader checks this finite registration against the public AST.
#define BICORE_RECORD(name) BICORE_##name = sizeof(struct name)
enum class BicoreRecordLayouts : uint64_t {
    BICORE_RECORD(ggml_backend_dev_caps),
    BICORE_RECORD(ggml_backend_dev_props),
    BICORE_RECORD(ggml_backend_feature),
    BICORE_RECORD(ggml_backend_graph_copy),
    BICORE_RECORD(ggml_backend_meta_split_state),
    BICORE_ggml_bf16_t = sizeof(ggml_bf16_t),
    BICORE_RECORD(ggml_cplan),
    BICORE_RECORD(ggml_init_params),
    BICORE_RECORD(ggml_opt_optimizer_params),
    BICORE_RECORD(ggml_opt_params),
    BICORE_RECORD(ggml_tallocr),
    BICORE_RECORD(ggml_tensor),
    BICORE_RECORD(ggml_threadpool_params),
    BICORE_RECORD(ggml_type_traits),
    BICORE_RECORD(ggml_type_traits_cpu),
    BICORE_RECORD(gguf_init_params),
    BICORE_RECORD(llama_batch),
    BICORE_RECORD(llama_chat_message),
    BICORE_RECORD(llama_context_params),
    BICORE_RECORD(llama_embd),
    BICORE_RECORD(llama_logit_bias),
    BICORE_RECORD(llama_model_imatrix_data),
    BICORE_RECORD(llama_model_kv_override),
    BICORE_RECORD(llama_model_params),
    BICORE_RECORD(llama_model_quantize_params),
    BICORE_RECORD(llama_model_tensor_buft_override),
    BICORE_RECORD(llama_model_tensor_override),
    BICORE_RECORD(llama_opt_params),
    BICORE_RECORD(llama_perf_context_data),
    BICORE_RECORD(llama_perf_sampler_data),
    BICORE_RECORD(llama_sampler),
    BICORE_RECORD(llama_sampler_chain_params),
    BICORE_RECORD(llama_sampler_data),
    BICORE_RECORD(llama_sampler_i),
    BICORE_RECORD(llama_sampler_seq_config),
    BICORE_RECORD(llama_token_data),
    BICORE_RECORD(llama_token_data_array),
    BICORE_RECORD(mtmd_audio_stream_params),
    BICORE_RECORD(mtmd_caps),
    BICORE_RECORD(mtmd_context_params),
    BICORE_RECORD(mtmd_decoder_pos),
    BICORE_RECORD(mtmd_gen_audio_info),
    BICORE_RECORD(mtmd_gen_inp),
    BICORE_RECORD(mtmd_gen_out),
    BICORE_RECORD(mtmd_gen_state_params),
    BICORE_RECORD(mtmd_helper_bitmap_wrapper),
    BICORE_RECORD(mtmd_helper_cap),
    BICORE_RECORD(mtmd_helper_embd_batch),
    BICORE_RECORD(mtmd_helper_gen_audio_inp),
    BICORE_RECORD(mtmd_helper_init_opt),
    BICORE_RECORD(mtmd_helper_video_info),
    BICORE_RECORD(mtmd_helper_video_init_params),
    BICORE_RECORD(mtmd_helper_video_src),
    BICORE_RECORD(mtmd_input_part),
    BICORE_RECORD(mtmd_input_text),
    BICORE_RECORD(mtmd_video_config),
    BICORE_RECORD(ggml_webgpu_q8_diagnostic_request_v1),
    BICORE_RECORD(ggml_webgpu_q8_diagnostic_report_v1),
};
#undef BICORE_RECORD

#ifndef BICORE_ABI_EXTRACT
#include "fingerprint.h"
#ifdef LCB_FILE_BACKED_PLE
extern "C" int32_t lcb_qwen4exp_file_backed_shard(const llama_model * model);
#endif
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
uint64_t lcb_model_graph_schema_hash(void) {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(BICORE_ABI_FINGERPRINT));
}
#ifdef LCB_WEBGPU_Q8_DIAGNOSTIC
uint64_t lcb_webgpu_q8_diagnostic_schema_hash(void) {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(BICORE_ABI_FINGERPRINT));
}
#endif
uint32_t bicore_file_backed_rows_version(void) {
#ifdef LCB_FILE_BACKED_PLE
    return 1;
#else
    return 0;
#endif
}
int32_t bicore_model_file_backed_shard(const llama_model * model) {
#ifdef LCB_FILE_BACKED_PLE
    return model ? lcb_qwen4exp_file_backed_shard(model) : -1;
#else
    (void)model;
    return -1;
#endif
}
}
#endif
