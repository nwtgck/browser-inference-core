#include "stable-diffusion.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include <stdexcept>

namespace {
template <class T> T & record(T * value) {
    if (!value)
        throw std::invalid_argument("null record storage");
    return *value;
}
} // namespace

extern "C" {
bool bicore_adetail_image(adetailer_ctx_t * adetailer, sd_ctx_t * context, const sd_image_t * image,
                          const sd_adetailer_params_t * options, const sd_img_gen_params_t * inpaint,
                          sd_image_generation_result_t ** output) {
    return adetail_image(adetailer, context, record(image), options, inpaint, output);
}
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
bool bicore_preprocess_canny(const sd_image_t * image, float high, float low, float weak, float strong, bool inverse) {
    return preprocess_canny(record(image), high, low, weak, strong, inverse);
}
void bicore_sd_get_preview_info(sd_preview_info_t * out) {
    record(out) = sd_get_preview_info();
}
bool bicore_upscale(upscaler_ctx_t * context, const sd_image_t * image, uint32_t factor,
                    sd_image_generation_result_t ** output) {
    return upscale(context, record(image), factor, output);
}
}
