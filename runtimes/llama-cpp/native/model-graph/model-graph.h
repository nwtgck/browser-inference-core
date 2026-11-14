#ifndef LCB_MODEL_GRAPH_H
#define LCB_MODEL_GRAPH_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Optional, versioned leaf capability. No application policy or graph interpreter.
 * Handles/pointers always cross the boundary as uint64_t on wasm32 and wasm64.
 * These calls are synchronous and must not overlap a suspended core call. */
uint32_t lcb_model_graph_version(void);
uint64_t lcb_model_graph_schema_hash(void);
uint64_t lcb_model_graph_last_error(void);
int32_t lcb_model_graph_architecture_known(uint64_t name);
/* GGUF is borrowed only during this call. The returned owner owns its vocab
 * and copied chat metadata. Its vocabulary must outlive samplers/sessions. */
uint64_t lcb_model_graph_vocab_new(uint64_t metadata);
uint64_t lcb_model_graph_vocab_get(uint64_t owner);
void lcb_model_graph_vocab_free(uint64_t owner);
uint64_t lcb_model_graph_sampler_workspace_new(uint32_t vocabulary_size);
void lcb_model_graph_sampler_workspace_free(uint64_t workspace);
/* Builds candidates, applies and accepts the existing CPU sampler, exactly
 * once. A negative return denotes an error, described by last_error(). */
int32_t lcb_model_graph_sample(uint64_t workspace, uint64_t sampler, uint64_t logits, uint32_t count);

#ifdef __cplusplus
}
#endif
#endif
