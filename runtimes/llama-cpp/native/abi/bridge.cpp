#include "llama.h"
#include "gguf.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-opt.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "mtmd-vision-bf16.h"
#include <stdexcept>

namespace {
template <class T> T & record(T * value) {
    if (!value)
        throw std::invalid_argument("null record storage");
    return *value;
}
} // namespace

// Only aggregate-by-value boundaries live here. Scalar/pointer APIs use their
// original C exports; these ordinary definitions are checked against upstream.
extern "C" {
void bicore_ggml_backend_graph_copy(struct ggml_backend_graph_copy * out, ggml_backend_t backend,
                                    struct ggml_cgraph * graph) {
    record(out) = ggml_backend_graph_copy(backend, graph);
}
void bicore_ggml_backend_graph_copy_free(const struct ggml_backend_graph_copy * copy) {
    ggml_backend_graph_copy_free(record(copy));
}
float bicore_ggml_bf16_to_fp32(const ggml_bf16_t * value) {
    return ggml_bf16_to_fp32(record(value));
}
void bicore_ggml_fp32_to_bf16(ggml_bf16_t * out, float value) {
    record(out) = ggml_fp32_to_bf16(value);
}
void bicore_ggml_graph_plan(struct ggml_cplan * out, const struct ggml_cgraph * graph, int threads,
                            struct ggml_threadpool * pool) {
    record(out) = ggml_graph_plan(graph, threads, pool);
}
struct ggml_context * bicore_ggml_init(const struct ggml_init_params * params) {
    return ggml_init(record(params));
}
void bicore_ggml_opt_default_params(struct ggml_opt_params * out, ggml_backend_sched_t scheduler,
                                    enum ggml_opt_loss_type loss) {
    record(out) = ggml_opt_default_params(scheduler, loss);
}
void bicore_ggml_opt_get_constant_optimizer_params(struct ggml_opt_optimizer_params * out, void * data) {
    record(out) = ggml_opt_get_constant_optimizer_params(data);
}
void bicore_ggml_opt_get_default_optimizer_params(struct ggml_opt_optimizer_params * out, void * data) {
    record(out) = ggml_opt_get_default_optimizer_params(data);
}
ggml_opt_context_t bicore_ggml_opt_init(const struct ggml_opt_params * params) {
    return ggml_opt_init(record(params));
}
void bicore_ggml_tallocr_new(struct ggml_tallocr * out, ggml_backend_buffer_t buffer) {
    record(out) = ggml_tallocr_new(buffer);
}
void bicore_ggml_threadpool_params_default(struct ggml_threadpool_params * out, int threads) {
    record(out) = ggml_threadpool_params_default(threads);
}
struct gguf_context * bicore_gguf_init_from_buffer(const void * data, size_t size,
                                                   const struct gguf_init_params * params) {
    return gguf_init_from_buffer(data, size, record(params));
}
struct gguf_context * bicore_gguf_init_from_callback(gguf_reader_callback_t callback, void * data, size_t chunk,
                                                     uint64_t maximum, const struct gguf_init_params * params) {
    return gguf_init_from_callback(callback, data, chunk, maximum, record(params));
}
struct gguf_context * bicore_gguf_init_from_file(const char * path, const struct gguf_init_params * params) {
    return gguf_init_from_file(path, record(params));
}
struct gguf_context * bicore_gguf_init_from_file_ptr(FILE * file, const struct gguf_init_params * params) {
    return gguf_init_from_file_ptr(file, record(params));
}
int32_t bicore_llama_batch_ext_add_embd(struct llama_batch_ext * batch, llama_seq_id sequence,
                                        const struct llama_embd * embedding) {
    return llama_batch_ext_add_embd(batch, sequence, record(embedding));
}
bool bicore_llama_batch_ext_set_embd_state(struct llama_batch_ext * batch, int32_t index,
                                           const struct llama_embd * embedding) {
    return llama_batch_ext_set_embd_state(batch, index, record(embedding));
}
bool bicore_llama_batch_ext_set_embd_token(struct llama_batch_ext * batch, int32_t index,
                                           const struct llama_embd * embedding) {
    return llama_batch_ext_set_embd_token(batch, index, record(embedding));
}
void bicore_llama_batch_free(const struct llama_batch * batch) {
    llama_batch_free(record(batch));
}
void bicore_llama_batch_get_one(struct llama_batch * out, llama_token * tokens, int32_t count) {
    record(out) = llama_batch_get_one(tokens, count);
}
void bicore_llama_batch_init(struct llama_batch * out, int32_t tokens, int32_t embeddings, int32_t sequences) {
    record(out) = llama_batch_init(tokens, embeddings, sequences);
}
void bicore_llama_context_default_params(struct llama_context_params * out) {
    record(out) = llama_context_default_params();
}
int32_t bicore_llama_decode(struct llama_context * context, const struct llama_batch * batch) {
    return llama_decode(context, record(batch));
}
int32_t bicore_llama_encode(struct llama_context * context, const struct llama_batch * batch) {
    return llama_encode(context, record(batch));
}
struct llama_context * bicore_llama_init_from_model(struct llama_model * model,
                                                    const struct llama_context_params * params) {
    return llama_init_from_model(model, record(params));
}
void bicore_llama_model_default_params(struct llama_model_params * out) {
    record(out) = llama_model_default_params();
}
struct llama_model * bicore_llama_model_init_from_user(struct gguf_context * metadata,
                                                       llama_model_set_tensor_data_t set_data, void * data,
                                                       const struct llama_model_params * params) {
    return llama_model_init_from_user(metadata, set_data, data, record(params));
}
struct llama_model * bicore_llama_model_load_from_file(const char * path, const struct llama_model_params * params) {
    return llama_model_load_from_file(path, record(params));
}
struct llama_model * bicore_llama_model_load_from_file_ptr(FILE * file, const struct llama_model_params * params) {
    return llama_model_load_from_file_ptr(file, record(params));
}
struct llama_model * bicore_llama_model_load_from_splits(const char ** paths, size_t count,
                                                         const struct llama_model_params * params) {
    return llama_model_load_from_splits(paths, count, record(params));
}
void bicore_llama_model_quantize_default_params(struct llama_model_quantize_params * out) {
    record(out) = llama_model_quantize_default_params();
}
void bicore_llama_opt_init(struct llama_context * context, struct llama_model * model,
                           const struct llama_opt_params * params) {
    llama_opt_init(context, model, record(params));
}
void bicore_llama_perf_context(struct llama_perf_context_data * out, const struct llama_context * context) {
    record(out) = llama_perf_context(context);
}
void bicore_llama_perf_sampler(struct llama_perf_sampler_data * out, const struct llama_sampler * sampler) {
    record(out) = llama_perf_sampler(sampler);
}
void bicore_llama_sampler_chain_default_params(struct llama_sampler_chain_params * out) {
    record(out) = llama_sampler_chain_default_params();
}
struct llama_sampler * bicore_llama_sampler_chain_init(const struct llama_sampler_chain_params * params) {
    return llama_sampler_chain_init(record(params));
}
mtmd_audio_stream * bicore_mtmd_audio_stream_init(mtmd_context * context,
                                                  const struct mtmd_audio_stream_params * params) {
    return mtmd_audio_stream_init(context, record(params));
}
void bicore_mtmd_audio_stream_params_default(struct mtmd_audio_stream_params * out) {
    record(out) = mtmd_audio_stream_params_default();
}
void bicore_mtmd_context_params_default(struct mtmd_context_params * out) {
    record(out) = mtmd_context_params_default();
}
void bicore_mtmd_gen_audio_get_info(struct mtmd_gen_audio_info * out, const mtmd_context * context) {
    record(out) = mtmd_gen_audio_get_info(context);
}
void bicore_mtmd_gen_inp_default(struct mtmd_gen_inp * out, const mtmd_context * context) {
    record(out) = mtmd_gen_inp_default(context);
}
mtmd_gen_state * bicore_mtmd_gen_state_init(mtmd_context * context, const struct mtmd_gen_state_params * params) {
    return mtmd_gen_state_init(context, record(params));
}
void bicore_mtmd_gen_state_params_default(struct mtmd_gen_state_params * out) {
    record(out) = mtmd_gen_state_params_default();
}
void bicore_mtmd_get_cap_from_file(struct mtmd_caps * out, const char * path) {
    record(out) = mtmd_get_cap_from_file(path);
}
void bicore_mtmd_get_video_config(struct mtmd_video_config * out, const mtmd_context * context, int32_t width,
                                  int32_t height, int32_t frames, float fps) {
    record(out) = mtmd_get_video_config(context, width, height, frames, fps);
}
void bicore_mtmd_helper_bitmap_init_from_buf(struct mtmd_helper_bitmap_wrapper * out, const mtmd_context * context,
                                             const unsigned char * data, size_t size, bool audio,
                                             const struct mtmd_helper_init_opt * options) {
    record(out) = mtmd_helper_bitmap_init_from_buf(context, data, size, audio, record(options));
}
void bicore_mtmd_helper_bitmap_init_from_file(struct mtmd_helper_bitmap_wrapper * out, const mtmd_context * context,
                                              const char * path, bool audio,
                                              const struct mtmd_helper_init_opt * options) {
    record(out) = mtmd_helper_bitmap_init_from_file(context, path, audio, record(options));
}
void bicore_mtmd_helper_gen_audio_inp_default(struct mtmd_helper_gen_audio_inp * out) {
    record(out) = mtmd_helper_gen_audio_inp_default();
}
void bicore_mtmd_helper_get_capabilities(struct mtmd_helper_cap * out, const struct llama_model * model,
                                         const struct mtmd_context * context) {
    record(out) = mtmd_helper_get_capabilities(model, context);
}
void bicore_mtmd_helper_init_opt_default(struct mtmd_helper_init_opt * out) {
    record(out) = mtmd_helper_init_opt_default();
}
void bicore_mtmd_helper_video_init_params_default(struct mtmd_helper_video_init_params * out) {
    record(out) = mtmd_helper_video_init_params_default();
}
void bicore_mtmd_image_tokens_get_decoder_pos(struct mtmd_decoder_pos * out, const mtmd_image_tokens * tokens,
                                              llama_pos position, size_t index) {
    record(out) = mtmd_image_tokens_get_decoder_pos(tokens, position, index);
}
mtmd_context * bicore_mtmd_init_from_file(const char * path, const struct llama_model * model,
                                          const struct mtmd_context_params * params) {
    return mtmd_init_from_file(path, model, record(params));
}
struct mtmd_context * bicore_mtmd_init_from_file_with_vision_bf16_storage(const char * path,
                                                                          const struct llama_model * model,
                                                                          const struct mtmd_context_params * params,
                                                                          int32_t storage) {
    return mtmd_init_from_file_with_vision_bf16_storage(path, model, record(params), storage);
}
}
