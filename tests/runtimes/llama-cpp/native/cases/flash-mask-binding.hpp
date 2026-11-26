#pragma once

namespace flash_mask_binding_cases {

inline void indexed_decode_blocks() {
    struct Shape {
        int64_t queries, keys, streams;
        bool fused;
    };
    for (const auto [queries, keys, streams, fused] :
         {Shape{1, 32, 1, true}, Shape{1, 256, 2, true}, Shape{4, 256, 1, false}, Shape{1, 34, 1, false}}) {
        transfer_cases::Backend gpu(transfer_cases::BackendKind::webgpu);
        auto & caps = gpu.context->global_ctx->capabilities;
        caps.supports_subgroups = true;
        caps.min_subgroup_size = caps.max_subgroup_size = 16;
        gpu.type.iface.get_max_size = ggml_backend_webgpu_buffer_type_get_max_size;
        gpu.context->param_arena.cleanup();
        gpu.context->param_arena.init(gpu.context->global_ctx->device, WEBGPU_PARAMS_BUF_SIZE_BYTES, 8, 256);
        transfer_cases::Memory graph_memory(ggml_init({128 * 1024, nullptr, true}), ggml_free);
        auto * ctx = graph_memory.get();
        constexpr int64_t dim = 128, heads = 6, kv_heads = 2, buckets = 65;
        auto * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim, queries, heads, streams);
        auto * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, dim, keys, kv_heads, streams);
        auto * v = ggml_dup_tensor(ctx, k);
        auto * raw = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, keys, queries, 1, streams);
        auto * table = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, buckets, heads, queries * streams);
        auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, keys, queries * streams);
        for (auto * input : {q, k, v, raw, table, ids}) {
            ggml_set_input(input);
        }
        auto * out = ggml_flash_attn_ext(ctx, q, k, v, raw, 0.125f, 0.0f, 0.0f);
        ggml_flash_attn_ext_add_indexed_bias(out, table, ids);
        out->op_params[5] = GGML_WEBGPU_FLASH_INDEXED_HALF_BIAS;
        ggml_set_output(out);
        auto * graph = ggml_new_graph_custom(ctx, 16, false);
        ggml_build_forward_expand(graph, out);
        const std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator(ggml_gallocr_new(&gpu.type),
                                                                                    ggml_gallocr_free);
        BICORE_CHECK(allocator && ggml_gallocr_alloc_graph(allocator.get(), graph));
        BICORE_CHECK(ggml_backend_dev_supports_op(&gpu.device, out));
        commands = {};
        BICORE_CHECK(ggml_backend_graph_compute_async(&gpu.backend, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(&gpu.backend);
        const auto & slots = gpu.context->bind_group_cache.slots;
        BICORE_CHECK(slots.size() == commands.executed);
        BICORE_CHECK(slots.size() == (fused ? 2u : 3u) + (keys > 64 ? 1u : 0u));
        const auto & producer = slots[0];
        BICORE_CHECK(producer.pipeline.Get()->name == (fused ? "indexed_vec_mask_block" : "indexed_vec_mask"));
        if (!fused) {
            BICORE_CHECK(slots[1].pipeline.Get()->name == "flash_attn_vec_blk_kvt32_wg32");
        }
        const auto & written = fused ? producer.bindings[4] : slots[1].bindings[1];
        const auto & split = slots[fused ? 1 : 2];
        const auto read = std::find_if(split.bindings.begin(), split.bindings.end(), [&](const auto & entry) {
            return entry.buffer.Get() == written.buffer.Get() && entry.offset == written.offset &&
                   entry.size == written.size;
        });
        BICORE_CHECK(read != split.bindings.end());
        BICORE_CHECK(written.buffer.Get() == producer.bindings[3].buffer.Get());
        BICORE_CHECK(written.offset + written.size <= producer.bindings[3].offset);
    }
}

inline void all() {
    indexed_decode_blocks();
    transfer_cases::Backend gpu(transfer_cases::BackendKind::webgpu);
    auto & caps = gpu.context->global_ctx->capabilities;
    caps.supports_subgroups = true;
    caps.min_subgroup_size = caps.max_subgroup_size = 16;
    auto & limits = caps.limits;
    constexpr uint64_t allocation_limit = 8 * 1024 * 1024;
    limits.maxStorageBufferBindingSize = limits.maxBufferSize = allocation_limit;
    const size_t alignment = limits.minStorageBufferOffsetAlignment;
    BICORE_CHECK(alignment >= 8);
    const size_t prefix = alignment / 2;
    gpu.type.iface.get_max_size = ggml_backend_webgpu_buffer_type_get_max_size;

    transfer_cases::Memory ctx(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
    constexpr int64_t head_dim = 128, query_count = 32, kv_count = 512, query_heads = 24, kv_heads = 2;
    auto * q = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, head_dim, query_count, query_heads, 1);
    auto * k = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, head_dim, kv_count, kv_heads, 1);
    auto * v = ggml_dup_tensor(ctx.get(), k);
    constexpr size_t row = kv_count * sizeof(ggml_fp16_t);
    constexpr size_t plane = row * query_count;
    constexpr size_t bytes = plane * query_heads;
    BICORE_CHECK(ggml_nbytes(q) < bytes && ggml_nbytes(k) < bytes && ggml_nbytes(v) < bytes);
    const auto new_mask_root = [&](ggml_context * memory) {
        return ggml_new_tensor_1d(memory, GGML_TYPE_F16, (bytes + prefix) / sizeof(ggml_fp16_t));
    };
    const auto mask_view = [&](ggml_context * memory, ggml_tensor * root, size_t offset) {
        return ggml_view_4d(memory, root, kv_count, query_count, query_heads, 1, row, plane, bytes, offset);
    };
    const auto flash = [&](ggml_tensor * mask) {
        BICORE_CHECK(ggml_nbytes(mask) == bytes);
        return ggml_flash_attn_ext(ctx.get(), q, k, v, mask, 0.125f, 0.0f, 0.0f);
    };
    auto * root = new_mask_root(ctx.get());
    auto * direct = flash(mask_view(ctx.get(), root, 0));
    auto * shifted = flash(mask_view(ctx.get(), root, prefix));
    for (auto * tensor : {q, k, v, root}) {
        ggml_set_input(tensor);
    }
    ggml_set_output(direct);
    ggml_set_output(shifted);
    auto * graph = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(graph, direct);
    ggml_build_forward_expand(graph, shifted);

    const auto verify_webgpu_views = [&] {
        limits.maxStorageBufferBindingSize = bytes;
        BICORE_CHECK(ggml_backend_dev_supports_op(&gpu.device, direct));
        BICORE_CHECK(!ggml_backend_dev_supports_op(&gpu.device, shifted));
        limits.maxStorageBufferBindingSize = bytes + prefix - 1;
        BICORE_CHECK(!ggml_backend_dev_supports_op(&gpu.device, shifted));
        limits.maxStorageBufferBindingSize = bytes + prefix;
        BICORE_CHECK(ggml_backend_dev_supports_op(&gpu.device, shifted));
        limits.maxBufferSize = bytes + prefix - 1;
        BICORE_CHECK(!ggml_backend_dev_supports_op(&gpu.device, shifted));
        limits.maxStorageBufferBindingSize = limits.maxBufferSize = allocation_limit;
    };
    verify_webgpu_views();
    const std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator(ggml_gallocr_new(&gpu.type), ggml_gallocr_free);
    BICORE_CHECK(allocator && ggml_gallocr_alloc_graph(allocator.get(), graph));
    BICORE_CHECK(ggml_webgpu_tensor_binding_size(direct->src[3], alignment) == bytes);
    BICORE_CHECK(ggml_webgpu_tensor_binding_size(shifted->src[3], alignment) == bytes + prefix);
    verify_webgpu_views();

    transfer_cases::Backend host(transfer_cases::BackendKind::host);
    transfer_cases::Memory host_context(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
    auto * host_view = mask_view(host_context.get(), new_mask_root(host_context.get()), prefix);
    const transfer_cases::Buffer host_buffer(
        ggml_backend_alloc_ctx_tensors_from_buft(host_context.get(), &host.type), ggml_backend_buffer_free);
    BICORE_CHECK(host_buffer && host_view->buffer == host_buffer.get());
    auto * copied = flash(host_view);
    limits.maxStorageBufferBindingSize = bytes;
    BICORE_CHECK(ggml_backend_dev_supports_op(&gpu.device, copied));
    limits.maxStorageBufferBindingSize = bytes - 1;
    BICORE_CHECK(!ggml_backend_dev_supports_op(&gpu.device, copied));

    const auto * odd_mask = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F16, 3);
    limits.maxStorageBufferBindingSize = ggml_nbytes(odd_mask);
    BICORE_CHECK(!ggml_webgpu_flash_mask_binding_fits(gpu.context->global_ctx, odd_mask));
    limits.maxStorageBufferBindingSize = 8;
    BICORE_CHECK(ggml_webgpu_flash_mask_binding_fits(gpu.context->global_ctx, odd_mask));
}

} // namespace flash_mask_binding_cases
