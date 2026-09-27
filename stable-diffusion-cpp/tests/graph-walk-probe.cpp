// Test variants only: real GGML graph construction, no backend or model weights.
#include "ggml.h"
#include "ggml-impl.h"
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
    void check(bool condition, const char* message) {
        if (!condition) throw std::runtime_error(message);
    }

    using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;

    Context context(size_t tensors, size_t graph_size) {
        const size_t bytes = ggml_tensor_overhead() * tensors + ggml_graph_overhead_custom(graph_size, false) + 4096;
        Context result(ggml_init({bytes, nullptr, true}), ggml_free);
        check(result != nullptr, "graph probe context allocation");
        return result;
    }

    int uses(ggml_cgraph* graph, ggml_tensor* tensor) {
        const size_t position = ggml_hash_find(&graph->visited_hash_set, tensor);
        check(position != GGML_HASHSET_FULL && ggml_bitset_get(graph->visited_hash_set.used, position), "missing visited tensor");
        return graph->use_counts[position];
    }

    void check_order(ggml_cgraph_eval_order order) {
        auto owner = context(16, 32);
        auto ctx = owner.get();
        auto x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        auto y = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        auto z = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        auto parameter = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        ggml_set_param(parameter);
        auto shared = ggml_add(ctx, x, parameter);
        auto left = ggml_mul(ctx, shared, y);
        auto right = ggml_add(ctx, shared, z);
        auto output = ggml_add(ctx, left, right);
        auto graph = ggml_new_graph_custom(ctx, 32, false);
        graph->order = order;
        ggml_build_forward_expand(graph, output);
        const bool reversed = order == GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT;
        const std::vector<ggml_tensor*> nodes = {parameter, shared, reversed ? right : left, reversed ? left : right, output};
        const std::vector<ggml_tensor*> leaves = reversed ? std::vector<ggml_tensor*>{z, x, y} : std::vector<ggml_tensor*>{x, y, z};
        check(graph->n_nodes == int(nodes.size()) && graph->n_leafs == int(leaves.size()), "DAG node/leaf counts");
        for (size_t i = 0; i < nodes.size(); ++i) {
            check(graph->nodes[i] == nodes[i], "postorder node ordering");
            char name[32]; std::snprintf(name, sizeof(name), "node_%zu", i);
            check(std::strcmp(nodes[i]->name, name) == 0, "stable automatic node naming");
        }
        for (size_t i = 0; i < leaves.size(); ++i) {
            check(graph->leafs[i] == leaves[i], "leaf ordering");
            char name[32]; std::snprintf(name, sizeof(name), "leaf_%zu", i);
            check(std::strcmp(leaves[i]->name, name) == 0, "stable automatic leaf naming");
        }
        for (auto node : {x, y, z, parameter, left, right}) check(uses(graph, node) == 1, "single operand use count");
        check(uses(graph, shared) == 2 && uses(graph, output) == 0, "shared DAG use counts");
        ggml_build_forward_expand(graph, output);
        check(graph->n_nodes == 5 && uses(graph, shared) == 2, "repeated expansion is idempotent");
        auto next = ggml_add(ctx, output, shared);
        ggml_build_forward_expand(graph, next);
        check(graph->n_nodes == 6 && graph->nodes[5] == next, "incremental graph expansion");
        check(uses(graph, shared) == 3 && uses(graph, output) == 1, "new edges counted once");
    }

    void check_deep_selection() {
        constexpr int depth = 32768;
        auto owner = context(depth + 4, depth + 8);
        auto ctx = owner.get();
        auto source = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        auto output = source;
        std::vector<ggml_tensor*> chain;
        chain.reserve(depth);
        for (int i = 0; i < depth; ++i) {
            output = ggml_scale(ctx, output, 1.0f);
            chain.push_back(output);
        }
        auto alternative = ggml_scale(ctx, source, 2.0f);
        ggml_tensor* branches[] = {output, alternative};
        auto graph = ggml_new_graph_custom(ctx, depth + 8, false);
        check(ggml_build_forward_select(graph, branches, 2, 1) == alternative, "selected branch identity");
        check(graph->n_nodes == depth + 1 && graph->n_leafs == 1, "deep first traversal");
        for (int i = 0; i < depth; ++i) {
            check(graph->nodes[i] == chain[i], "deep postorder");
            check((chain[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0, "unselected branch stays excluded");
            check(uses(graph, chain[i]) == (i == depth - 1 ? 0 : 1), "deep edge counts");
        }
        check((alternative->flags & GGML_TENSOR_FLAG_COMPUTE) != 0, "selected branch computes");
        // Revisit the entire chain to propagate compute, the second formerly
        // recursive path. No topology or use counts may change on this visit.
        ggml_build_forward_expand(graph, output);
        check(graph->n_nodes == depth + 1 && uses(graph, source) == 2, "compute revisit preserves edges");
        for (int i = 0; i < depth; ++i) {
            check((chain[i]->flags & GGML_TENSOR_FLAG_COMPUTE) != 0, "deep compute propagation");
            check(uses(graph, chain[i]) == (i == depth - 1 ? 0 : 1), "compute revisit use counts");
        }
    }
}

extern "C" int sdc_test_graph_walk() {
    try {
        check_order(GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT);
        check_order(GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT);
        check_deep_selection();
        return 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "graph-walk probe: %s\n", error.what());
        return 0;
    }
}
