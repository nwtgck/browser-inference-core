#pragma once

namespace dense_q4_swiglu_cases {

enum class Guard {
    ordinary,
    observed_projection,
    raw_view,
    final_view,
    live_alias,
    shape,
    subgroup_range,
    cold_raw_view,
    rejection
};

static void graph(uint32_t subgroup, Guard guard, int64_t input_width = 6656, int64_t m = 19968) {
    transfer_cases::Backend gpu(transfer_cases::BackendKind::webgpu);
    auto & global = *gpu.context->global_ctx;
    auto & caps = global.capabilities;
    caps.supports_subgroups = true;
    caps.min_subgroup_size = guard == Guard::subgroup_range ? 16 : subgroup;
    caps.max_subgroup_size = subgroup;
    caps.limits.maxStorageBufferBindingSize = 128 * 1024 * 1024;
    caps.limits.maxBufferSize = 256 * 1024 * 1024;
    gpu.type.iface.get_max_size = [](ggml_backend_buffer_type_t) -> size_t { return 128 * 1024 * 1024; };
    auto & arena = gpu.context->param_arena;
    arena.cleanup();
    arena.init(global.device, WEBGPU_PARAMS_BUF_SIZE_BYTES, 74, 256);

    const int64_t k = guard == Guard::shape ? input_width - 256 : input_width;
    const bool live_alias = guard == Guard::live_alias;
    const auto weights = transfer_cases::Memory(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
    const auto memory = transfer_cases::Memory(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
    BICORE_CHECK(weights && memory);
    auto * w = weights.get();
    auto * ctx = memory.get();

    // A complete Q4_K block of nonzero prefix exercises the quantized rebase.
    auto * gate_storage = ggml_new_tensor_2d(w, GGML_TYPE_Q4_K, k, m + 1);
    const size_t prefix = ggml_type_size(GGML_TYPE_Q4_K);
    auto * gate_weight = ggml_view_2d(w, gate_storage, k, m, gate_storage->nb[1], prefix);
    auto * up_weight = ggml_new_tensor_2d(w, GGML_TYPE_Q4_K, k, m);
    auto * input = ggml_new_tensor_1d(w, GGML_TYPE_F32, live_alias ? m : k);
    const auto persistent =
        transfer_cases::Buffer(ggml_backend_alloc_ctx_tensors_from_buft(w, &gpu.type), ggml_backend_buffer_free);
    BICORE_CHECK(persistent);
    const auto allocator =
        std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)>(ggml_gallocr_new(&gpu.type), ggml_gallocr_free);

    auto * producer = ggml_dup(ctx, input);
    auto * x = live_alias ? ggml_view_1d(ctx, producer, k, 0) : producer;
    auto * gate = ggml_mul_mat(ctx, gate_weight, x);
    auto * up = ggml_mul_mat(ctx, up_weight, x);
    auto * y = ggml_swiglu_split(ctx, gate, up);
    auto * consumer = ggml_dup(ctx, y);
    ggml_set_output(consumer);
    if (guard == Guard::observed_projection) {
        ggml_set_output(gate);
    }
    auto * layout = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(layout, consumer);
    const int stop = ggml_graph_n_nodes(layout);
    const auto append_reader = [&](ggml_tensor * root) {
        auto * view = ggml_view_tensor(ctx, root);
        BICORE_CHECK(view->view_src == root && !view->src[0]);
        auto * observer = ggml_dup(ctx, view);
        ggml_set_output(observer);
        ggml_build_forward_expand(layout, observer);
    };
    if (guard == Guard::raw_view || guard == Guard::final_view) {
        append_reader(guard == Guard::raw_view ? gate : y);
    }

    BICORE_CHECK(allocator && ggml_gallocr_alloc_graph(allocator.get(), layout));
    BICORE_CHECK(ggml_webgpu_tensor_buf(x).Get() == ggml_webgpu_tensor_buf(y).Get());
    const size_t x_begin = ggml_webgpu_tensor_offset(x);
    const size_t y_begin = ggml_webgpu_tensor_offset(y);
    BICORE_CHECK((x_begin < y_begin + ggml_nbytes(y) && y_begin < x_begin + ggml_nbytes(x)) == live_alias);

    const auto gate_node = std::find(layout->nodes, layout->nodes + stop, gate);
    BICORE_CHECK(gate_node != layout->nodes + stop);
    const int start = int(gate_node - layout->nodes);
    auto slice = ggml_graph_view(layout, start, stop);
    BICORE_CHECK(slice.n_nodes == 4 && slice.nodes[0] == gate && slice.nodes[1] == up && slice.nodes[2] == y);
    const auto append_late_reader = [&](ggml_tensor * root) {
        append_reader(root);
        // The slice still ends at consumer. The new reader has no src edge
        // to its raw storage root, so it does not change GGML use counts.
        const int indices[] = {0, 1, 2};
        const int outputs[] = {2};
        const ggml_op ops[] = {GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU};
        BICORE_CHECK(ggml_graph_n_nodes(layout) > stop && slice.n_nodes == 4 &&
                     ggml_can_fuse_subgraph_ext(&slice, indices, 3, ops, outputs, 1) &&
                     webgpu_fusion_view_sources(slice).contains(root));
    };

    const auto run = [&](bool fused) {
        commands = {};
        BICORE_CHECK(ggml_backend_graph_compute_async(&gpu.backend, &slice) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(&gpu.backend);
        const size_t expected = fused ? 2 : 4;
        BICORE_CHECK(commands.dispatches.size() == expected && commands.executed == expected);
#ifdef GGML_WEBGPU_GPU_PROFILE
        BICORE_CHECK(commands.timestamp_pairs.size() == expected);
#endif
        const auto pipeline_at = [&](size_t index) -> const webgpu_bind_group_cache::slot & {
            const auto & dispatch = commands.dispatches[index];
            const auto & slots = gpu.context->bind_group_cache.slots;
            const auto slot = std::find_if(slots.begin(), slots.end(), [&](const auto & entry) {
                return entry.group && entry.group.Get()->id == dispatch.group;
            });
            BICORE_CHECK(slot != slots.end());
            return *slot;
        };
        const auto & first = pipeline_at(0);
        if (!fused) {
            BICORE_CHECK(first.pipeline.Get()->name.starts_with("mul_mat_vec_q4_K_f32_") &&
                         pipeline_at(1).pipeline.Get()->name.starts_with("mul_mat_vec_q4_K_f32_") &&
                         pipeline_at(2).pipeline.Get()->name == "glu_swiglu_f32_split");
            return;
        }
        const auto & dispatch = commands.dispatches.front();
        const uint32_t rows_per_workgroup = subgroup == 16 ? 32 : 16;
        BICORE_CHECK(first.pipeline.Get()->name == "dense_q4_swiglu_r4_sg" + std::to_string(subgroup) &&
                     dispatch.x == uint32_t(m) / rows_per_workgroup && dispatch.y == 1 && dispatch.z == 1);
        BICORE_CHECK(first.bindings.size() == 5);
        const ggml_tensor * tensors[] = {gate_weight, x, up_weight, y};
        uint32_t offsets[4] = {};
        for (size_t i = 0; i < std::size(tensors); ++i) {
            const auto * tensor = tensors[i];
            const auto & entry = first.bindings[i];
            BICORE_CHECK(entry.binding_index == i && entry.buffer.Get() == ggml_webgpu_tensor_buf(tensor).Get() &&
                         entry.offset == ggml_webgpu_tensor_align_offset(gpu.context, tensor) &&
                         entry.size == ggml_webgpu_tensor_binding_size(gpu.context, tensor));
            offsets[i] = ggml_webgpu_tensor_misalignment(gpu.context, tensor) / ggml_type_size(tensor->type);
        }
        const auto & uniform = first.bindings.back();
        BICORE_CHECK(uniform.binding_index == 4 && uniform.size == WEBGPU_PARAMS_BUF_SIZE_BYTES);
        std::array<uint32_t, 20> params;
        std::memcpy(params.data(), uniform.buffer.Get()->bytes.data() + uniform.offset, sizeof(params));
        const uint32_t width = uint32_t(k);
        const uint32_t stride = width / 256;
        const uint32_t weight_blocks = stride * uint32_t(m);
        const std::array<uint32_t, 20> expected_params = {
            offsets[0],
            offsets[1],
            offsets[3],
            uint32_t(m),
            1,
            width,
            stride,
            width,
            weight_blocks,
            width,
            weight_blocks,
            width,
            1,
            1,
            1,
            1,
            offsets[2],
            0,
            0,
            0,
        };
        BICORE_CHECK(params == expected_params && offsets[0] == 1);
    };

    const auto before = pushes;
    if (guard == Guard::rejection) {
        webgpu_fusion_view_sources views(slice);
        BICORE_CHECK(!views.contains(x));
        on_wait_once = [&] { append_late_reader(x); };
        inject = WGPUErrorType_Validation;
        int consumed = 1;
        commands = {};
        BICORE_CHECK(!webgpu_dense_q4_swiglu::encode(gpu.context, slice, 0, views, consumed));
        BICORE_CHECK(commands.dispatches.empty() && consumed == 1 && views.contains(x));
        BICORE_CHECK(!on_wait_once && pushes == before + 3 && inject == WGPUErrorType_NoError);
        // The new reader observes an input, so the next eligible graph still
        // reaches the cached rejection.
        run(false);
        BICORE_CHECK(pushes == before + 3);
    } else if (guard == Guard::cold_raw_view) {
        on_wait_once = [&] { append_late_reader(gate); };
        run(false);
        BICORE_CHECK(!on_wait_once && pushes == before + 3);
        run(false);
        BICORE_CHECK(pushes == before + 3);
    } else {
        const bool fused = guard == Guard::ordinary || guard == Guard::final_view;
        run(fused);
        BICORE_CHECK(pushes == before + (fused ? 3 : 0));
        if (guard == Guard::ordinary) {
            const auto created_after_cold = created;
            run(true);
            BICORE_CHECK(pushes == before + 3 && created == created_after_cold);
        }
    }
    BICORE_CHECK(futures.empty());
}

static void all() {
    graph(16, Guard::ordinary);
    for (const auto guard :
         {Guard::ordinary, Guard::observed_projection, Guard::raw_view, Guard::final_view, Guard::live_alias,
          Guard::shape, Guard::subgroup_range, Guard::cold_raw_view, Guard::rejection}) {
        graph(32, guard);
    }
    graph(16, Guard::ordinary, 2048, 10752);
    for (const auto guard : {Guard::ordinary, Guard::shape, Guard::cold_raw_view, Guard::subgroup_range}) {
        graph(32, guard, 2048, 10752);
    }
}

} // namespace dense_q4_swiglu_cases
