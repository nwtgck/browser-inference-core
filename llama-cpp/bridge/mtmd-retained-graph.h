#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

// Opt-in guard for original BF16 vision weights, not a generic ggml validator.
// It uses public tensor/backend/scheduler interfaces and does not replace model
// graph builders, arithmetic, shaders, or the scheduler. Failures are explicit;
// the caller must NOT retry with whole-model expansion or a CPU fallback.
class lcb_mtmd_retained_graph {
public:
    void prepare(ggml_context * weights, ggml_cgraph * graph,
                 ggml_backend_sched_t scheduler, ggml_backend_t target, bool enabled) {
        consumers.clear();
        backend = target;
        if (!enabled) return;
        if (!weights || !graph || !scheduler || !target) {
            fail("missing graph planning context");
        }
        std::unordered_set<const ggml_tensor *> roots;
        for (ggml_tensor * t = ggml_get_first_tensor(weights); t; t = ggml_get_next_tensor(weights, t)) {
            if (t->type == GGML_TYPE_BF16) roots.insert(t);
        }
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            ggml_tensor * node = ggml_graph_node(graph, i);
            bool uses_weight = false;
            for (const ggml_tensor * src : node->src) {
                uses_weight = uses_weight || is_weight(src, roots);
            }
            if (!uses_weight) continue;
            if (is_metadata_view(node->op)) {
                // A metadata operation must actually remain a view of storage.
                if (!is_weight(node, roots)) fail("weight view lost its original storage");
                continue;
            }
            if (!is_weight(node->src[0], roots) || node->src[0]->type != GGML_TYPE_BF16 ||
                node->type != GGML_TYPE_F32) {
                fail("unsupported weight operand or result type");
            }
            for (int j = 1; j < GGML_MAX_SRC; ++j) {
                if (is_weight(node->src[j], roots)) fail("BF16 weight used outside the left operand");
            }
            ggml_tensor * weight = node->src[0];
            ggml_tensor * input = node->src[1];
            checked_span(weight, 2);
            checked_span(node, 4);
            if (!ggml_is_contiguous(weight)) fail("non-contiguous retained weight");
            const ggml_tensor * root = weight->view_src ? weight->view_src : weight;
            const uint64_t root_bytes = checked_span(root, 2);
            const uint64_t weight_bytes = checked_span(weight, 2);
            if (weight->view_offs > root_bytes || weight_bytes > root_bytes - weight->view_offs) {
                fail("weight view exceeds original storage");
            }
            if (!input) fail("missing input");
            if (node->op == GGML_OP_MUL_MAT) {
                if (input->type != GGML_TYPE_F32) fail("matmul input must be F32");
                checked_span(input, 4);
                if (weight->ne[0] != input->ne[0] ||
                    input->ne[2] % weight->ne[2] != 0 || input->ne[3] % weight->ne[3] != 0 ||
                    node->ne[0] != weight->ne[1] || node->ne[1] != input->ne[1] ||
                    node->ne[2] != input->ne[2] || node->ne[3] != input->ne[3]) {
                    fail("incompatible matmul shape");
                }
                // Conservative subset within baseline WebGPU dispatch limits.
                // No knowledge of a device's larger limit or chosen tile needed.
                const uint64_t batches = uint64_t(input->ne[2]) * uint64_t(input->ne[3]);
                if (input->ne[1] > 65535 || batches > 65535 ||
                    uint64_t(weight->ne[1]) > 65535 / batches) {
                    fail("matmul exceeds reviewed dispatch envelope");
                }
            } else if (node->op == GGML_OP_GET_ROWS) {
                if (input->type != GGML_TYPE_I32) fail("row indices must be I32");
                checked_span(input, 4);
                // Upstream BF16 GET_ROWS uses one invocation per output row and
                // only its X coordinate. Avoid 2D dispatch altogether here. This
                // deliberately tighter-than-device bound is not a shader fix.
                const uint64_t rows = uint64_t(node->ne[1]) * uint64_t(node->ne[2]) * uint64_t(node->ne[3]);
                if (rows > 65535) fail("row selection exceeds reviewed one-dimensional dispatch");
            } else {
                // Includes CONT/DUP/CPY/cast, interpolation and weight-side
                // arithmetic: do not hide a full-size weight copy in the graph.
                fail("unsupported operation on retained weight");
            }
            if (!root->buffer || !ggml_backend_supports_buft(target, ggml_backend_buffer_get_type(root->buffer))) {
                fail("retained storage is not accessible by the target backend");
            }
            if (!ggml_backend_supports_op(target, node)) fail("backend rejects retained weight operation");
            consumers.push_back({node, weight});
        }
        // Only commit assignments after validating the complete graph. A late
        // rejection must not leave half a plan attached to the scheduler.
        for (const consumer & item : consumers) {
            ggml_backend_sched_set_tensor_backend(scheduler, item.node, target);
        }
    }

    // reserve() resets scheduler assignments. Check only retained input identity
    // there; allocation-time checks below also inspect actual assigned backends.
    void verify_sources() const {
        for (const consumer & item : consumers) {
            if (item.node->src[0] != item.weight) fail("scheduler copied retained weight");
        }
    }

    void verify_allocated(ggml_backend_sched_t scheduler) const {
        verify_sources();
        for (const consumer & item : consumers) {
            if (ggml_backend_sched_get_tensor_backend(scheduler, item.node) != backend) {
                fail("retained weight operation left the target backend");
            }
            const ggml_backend_buffer_t weights = buffer_of(item.weight);
            const ggml_backend_buffer_t input = buffer_of(item.node->src[1]);
            const ggml_backend_buffer_t output = buffer_of(item.node);
            if (!weights || !input || !output) fail("missing allocated tensor buffer");
            // The mixed BF16/F32 scalar overlap shader is not assumed valid.
            // Separate ggml buffer objects are a conservative physical-storage
            // boundary for the pinned WebGPU allocator; no pointer arithmetic or
            // unreviewed aligned-binding overlap calculation is substituted.
            if (weights == input || weights == output) fail("retained weight shares an operand buffer");
        }
    }

    size_t consumer_count() const { return consumers.size(); }

private:
    struct consumer { ggml_tensor * node; ggml_tensor * weight; };
    std::vector<consumer> consumers;
    ggml_backend_t backend = nullptr;

    [[noreturn]] static void fail(const char * reason) {
        throw std::runtime_error(std::string("retain-checked BF16: ") + reason);
    }

    static bool is_metadata_view(ggml_op op) {
        return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE ||
               op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
    }

    static bool is_weight(const ggml_tensor * tensor, const std::unordered_set<const ggml_tensor *> & roots) {
        // ggml views refer directly to their original storage (no arbitrary
        // source traversal: a matmul output is an activation, not a weight).
        return tensor && (roots.count(tensor) || (tensor->view_src && roots.count(tensor->view_src)));
    }

    static ggml_backend_buffer_t buffer_of(const ggml_tensor * tensor) {
        if (!tensor) return nullptr;
        return tensor->buffer ? tensor->buffer : tensor->view_src ? tensor->view_src->buffer : nullptr;
    }

    static uint64_t checked_span(const ggml_tensor * tensor, uint64_t scalar_bytes) {
        if (!tensor || tensor->nb[0] != scalar_bytes) fail("unsupported inner stride");
        uint64_t elements = 1;
        uint64_t span = scalar_bytes;
        const uint64_t limit = std::numeric_limits<uint32_t>::max();
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            if (tensor->ne[i] <= 0 || uint64_t(tensor->ne[i]) > limit ||
                tensor->nb[i] == 0 || tensor->nb[i] % scalar_bytes != 0 || tensor->nb[i] > limit) {
                fail("unsupported tensor shape or stride");
            }
            const uint64_t count = uint64_t(tensor->ne[i]);
            if (elements > limit / count) fail("tensor element count exceeds reviewed range");
            elements *= count;
            if (count - 1 > (limit - span) / tensor->nb[i]) fail("tensor byte span exceeds reviewed range");
            span += (count - 1) * tensor->nb[i];
        }
        return span;
    }
};
