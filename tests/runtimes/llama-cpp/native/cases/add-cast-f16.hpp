#pragma once

#include "ggml-alloc.h"

namespace add_cast_f16_cases {

enum class Guard { none, add_view, input_cast_view, input_cast_output };

struct Fixture {
    transfer_cases::Backend gpu{transfer_cases::BackendKind::webgpu};
    transfer_cases::Memory memory{ggml_init({1024 * 1024, nullptr, true}), ggml_free};
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator{nullptr, ggml_gallocr_free};
    ggml_tensor * bias = nullptr;
    ggml_tensor * mask = nullptr;
    ggml_tensor * cast = nullptr;
    ggml_tensor * add = nullptr;
    ggml_tensor * output = nullptr;
    ggml_cgraph * graph = nullptr;
    int begin = -1, end = -1;

    Fixture(int64_t keys, int64_t queries, bool half_mask, Guard guard = Guard::none) {
        constexpr int64_t heads = 24;
        BICORE_CHECK(memory);
        auto * ctx = memory.get();
        auto & limits = gpu.context->global_ctx->capabilities.limits;
        limits.maxBufferSize = limits.maxStorageBufferBindingSize = 128 * 1024 * 1024;
        limits.maxStorageBuffersPerShaderStage = 8;
        limits.maxUniformBuffersPerShaderStage = 1;
        limits.maxBindingsPerBindGroup = 12;
        limits.maxUniformBufferBindingSize = 65536;
        gpu.type.iface.get_max_size = [](ggml_backend_buffer_type_t) -> size_t { return 128 * 1024 * 1024; };
        gpu.type.iface.get_name = ggml_backend_webgpu_buffer_type_get_name;

        // Preserve the producer/consumer layout that controls real graph arena reuse.
        auto * table = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, heads, 32, queries);
        auto * indices = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, keys, queries);
        ggml_set_input(table);
        ggml_set_input(indices);
        auto * gathered = ggml_get_rows(ctx, table, indices);
        auto * shaped = ggml_reshape_4d(ctx, gathered, heads, keys, queries, 1);
        bias = ggml_cont(ctx, ggml_permute(ctx, shaped, 2, 0, 1, 3));

        mask = ggml_new_tensor_4d(ctx, half_mask ? GGML_TYPE_F16 : GGML_TYPE_F32, keys, queries, 1, 1);
        ggml_set_input(mask);
        cast = half_mask ? ggml_cast(ctx, mask, GGML_TYPE_F32) : mask;
        add = ggml_add(ctx, bias, cast);
        output = ggml_cast(ctx, add, GGML_TYPE_F16);
        if (guard == Guard::input_cast_output) {
            ggml_set_output(cast);
        }

        auto * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, queries, heads, 1);
        auto * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 128, keys, 4, 1);
        auto * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 128, keys, 4, 1);
        ggml_set_input(q);
        ggml_set_input(k);
        ggml_set_input(v);
        auto * sink = ggml_fill(ctx, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, heads), 0.0f);
        auto * attention = ggml_flash_attn_ext(ctx, q, k, v, output, 1.0f / std::sqrt(128.0f), 0.0f, 0.0f);
        ggml_flash_attn_ext_add_sinks(attention, sink);
        ggml_set_output(attention);
        graph = ggml_new_graph_custom(ctx, 128, false);
        ggml_build_forward_expand(graph, attention);
        if (guard == Guard::add_view || guard == Guard::input_cast_view) {
            auto * raw = ggml_view_tensor(ctx, guard == Guard::add_view ? add : cast);
            auto * reader = ggml_dup(ctx, raw);
            ggml_set_output(reader);
            ggml_build_forward_expand(graph, reader);
            BICORE_CHECK(ggml_hash_contains(&graph->visited_hash_set, raw));
        }
        allocator.reset(ggml_gallocr_new(&gpu.type));
        BICORE_CHECK(allocator && ggml_gallocr_alloc_graph(allocator.get(), graph));
        for (int i = 0; i < graph->n_nodes; ++i) {
            if (graph->nodes[i] == add) {
                begin = i - int(half_mask);
                end = i + 2;
                BICORE_CHECK(graph->nodes[i + 1] == output);
            }
        }
        BICORE_CHECK(begin >= 0 && end <= graph->n_nodes);
        if (half_mask) {
            BICORE_CHECK(graph->nodes[begin] == cast);
        }
    }

    void compute(int start, int stop) {
        auto view = ggml_graph_view(graph, start, stop);
        commands = {};
        BICORE_CHECK(ggml_backend_graph_compute_async(&gpu.backend, &view) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(&gpu.backend);
        BICORE_CHECK(commands.executed == commands.dispatches.size());
#ifdef GGML_WEBGPU_GPU_PROFILE
        BICORE_CHECK(commands.timestamp_pairs.size() == commands.dispatches.size());
#endif
    }

    const webgpu_bind_group_cache::slot & packet(const RecordedDispatch & dispatch) {
        const auto & slots = gpu.context->bind_group_cache.slots;
        const auto slot = std::find_if(slots.begin(), slots.end(), [&](const auto & item) {
            return item.group && item.group.Get()->id == dispatch.group;
        });
        BICORE_CHECK(slot != slots.end() && slot->pipeline.Get()->id == dispatch.pipeline);
        return *slot;
    }
};

static void all() {
    for (const bool half : {false, true}) {
        Fixture covered(256, 3, half);
        covered.compute(covered.begin, covered.end);
        BICORE_CHECK(commands.dispatches.size() == 1);
        const auto & slot = covered.packet(commands.dispatches.front());
        BICORE_CHECK(slot.pipeline.Get()->name == (half ? "add_cast_f16_mask_f16" : "add_cast_f16"));
        BICORE_CHECK(slot.bindings.size() == 4);
        const auto * arena = slot.bindings[0].buffer.Get();
        for (int i = 0; i < 3; ++i) {
            const auto & b = slot.bindings[i];
            BICORE_CHECK(b.buffer.Get() == arena);
            for (int j = 0; j < i; ++j) {
                const auto & a = slot.bindings[j];
                BICORE_CHECK(a.offset + a.size <= b.offset || b.offset + b.size <= a.offset);
            }
        }

        Fixture alias(256, 3, half);
        alias.output->buffer = alias.bias->buffer;
        alias.output->data = alias.bias->data;
        alias.compute(alias.begin, alias.end);
        BICORE_CHECK(commands.dispatches.size() > 1);
    }
    for (const auto guard : {Guard::add_view, Guard::input_cast_view, Guard::input_cast_output}) {
        Fixture escape(256, 3, true, guard);
        escape.compute(escape.begin, escape.end);
        BICORE_CHECK(commands.dispatches.size() > 1);
        for (const auto & dispatch : commands.dispatches) {
            BICORE_CHECK(escape.packet(dispatch).pipeline.Get()->name != "add_cast_f16_mask_f16");
        }
        if (guard == Guard::add_view) {
            BICORE_CHECK(commands.dispatches.size() == 3);
        }
    }
}

} // namespace add_cast_f16_cases
