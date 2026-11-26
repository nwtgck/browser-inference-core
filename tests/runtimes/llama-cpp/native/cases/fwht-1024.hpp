#pragma once

namespace fwht_1024_cases {

enum class Route {
    plain,
    reshape,
    no_hint,
    matvec,
    strided,
    alias,
    accumulator_precision,
    source_precision,
    subgroup_matrix,
    invocations_limit,
    x_limit,
    shared_limit,
    grid_limit,
};

static void graph(Route route) {
    transfer_cases::Backend gpu(transfer_cases::BackendKind::webgpu);
    auto & global = *gpu.context->global_ctx;
    auto & caps = global.capabilities;
    auto & limits = caps.limits;
    limits.maxBufferSize = limits.maxStorageBufferBindingSize = 16 * 1024 * 1024;
    gpu.type.iface.get_max_size = [](ggml_backend_buffer_type_t) -> size_t { return 16 * 1024 * 1024; };
    gpu.context->param_arena.cleanup();
    gpu.context->param_arena.init(global.device, WEBGPU_PARAMS_BUF_SIZE_BYTES, 74, 256);

    constexpr int64_t width = 1024;
    const int64_t columns = route == Route::matvec       ? 4
                            : route == Route::reshape    ? 6
                            : route == Route::grid_limit ? 33
                                                         : 5;
    const auto memory = transfer_cases::Memory(ggml_init({256 * 1024, nullptr, true}), ggml_free);
    BICORE_CHECK(memory);
    auto * ctx = memory.get();
    auto * rotation = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, width);
    ggml_tensor * input = nullptr;
    constexpr size_t prefix = 4 * sizeof(float);
    if (route == Route::reshape) {
        auto * bank = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, width * columns + prefix / sizeof(float));
        auto * selected = ggml_view_1d(ctx, bank, width * columns, prefix);
        input = ggml_reshape_2d(ctx, selected, width, columns);
        BICORE_CHECK(input->view_src == bank && input->view_offs == prefix);
    } else if (route == Route::strided) {
        auto * bank = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width + 4, columns);
        input = ggml_view_2d(ctx, bank, width, columns, bank->nb[1], 0);
        BICORE_CHECK(!ggml_is_contiguous(input));
    } else {
        input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, columns);
    }
    auto * output = ggml_mul_mat(ctx, rotation, input);
    if (route != Route::no_hint) {
        ggml_mul_mat_set_hint(output, GGML_HINT_SRC0_IS_HADAMARD);
    }
    if (route == Route::accumulator_precision) {
        BICORE_CHECK(ggml_prec_set_acc(output, GGML_PREC_F32));
    }
    if (route == Route::source_precision) {
        BICORE_CHECK(ggml_prec_set_src(output, GGML_PREC_F16, 1));
    }
    auto * full = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(full, output);
    const auto allocated =
        transfer_cases::Buffer(ggml_backend_alloc_ctx_tensors_from_buft(ctx, &gpu.type), ggml_backend_buffer_free);
    BICORE_CHECK(allocated && !output->view_src);
    BICORE_CHECK(ggml_webgpu_tensor_buf(input).Get() == ggml_webgpu_tensor_buf(output).Get());
    BICORE_CHECK(!ggml_webgpu_tensor_binding_overlap(gpu.context->global_ctx, input, output));
    if (route == Route::alias) {
        // The ordinary matmul is only recorded on this synthetic alias.
        output->data = input->data;
        BICORE_CHECK(ggml_webgpu_tensor_binding_overlap(gpu.context->global_ctx, input, output));
    }

    limits.maxComputeInvocationsPerWorkgroup = route == Route::invocations_limit ? 127 : 128;
    limits.maxComputeWorkgroupSizeX = route == Route::x_limit ? 127 : 128;
    limits.maxComputeWorkgroupStorageSize = route == Route::shared_limit ? 4095 : 4096;
    if (route == Route::grid_limit) {
        limits.maxComputeWorkgroupsPerDimension = 32;
    }
    if (route == Route::subgroup_matrix) {
        caps.supports_subgroups = caps.supports_subgroup_matrix = true;
        caps.sg_mat_m = caps.sg_mat_n = caps.sg_mat_k = 8;
        caps.min_subgroup_size = caps.max_subgroup_size = 32;
    }

    auto slice = ggml_graph_view(full, ggml_graph_n_nodes(full) - 1, ggml_graph_n_nodes(full));
    BICORE_CHECK(slice.n_nodes == 1 && slice.nodes[0] == output);
    commands = {};
    BICORE_CHECK(ggml_backend_graph_compute_async(&gpu.backend, &slice) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(&gpu.backend);
    BICORE_CHECK(commands.dispatches.size() == 1 && commands.executed == 1);
#ifdef GGML_WEBGPU_GPU_PROFILE
    BICORE_CHECK(commands.timestamp_pairs.size() == 1);
#endif
    const auto & dispatch = commands.dispatches.front();
    const auto & slots = gpu.context->bind_group_cache.slots;
    const auto packet = std::find_if(slots.begin(), slots.end(), [&](const auto & entry) {
        return entry.group && entry.group.Get()->id == dispatch.group;
    });
    BICORE_CHECK(packet != slots.end());
    const std::string_view label = packet->pipeline.Get()->name;
    const bool transformed = route == Route::plain || route == Route::reshape;
    if (!transformed) {
        const std::string_view fallback = route == Route::matvec            ? "mul_mat_vec_"
                                          : route == Route::subgroup_matrix ? "mul_mat_subgroup_matrix_f32_f32"
                                                                            : "mul_mat_reg_tile_f32_f32";
        BICORE_CHECK(label.starts_with(fallback));
        if (route == Route::grid_limit) {
            BICORE_CHECK(dispatch.x == 32 && dispatch.y == 2);
        }
        return;
    }

    BICORE_CHECK(label == "mul_mat_fwht_1024_f32_half_input_wg128");
    BICORE_CHECK(dispatch.x == uint32_t(columns) && dispatch.y == 1 && dispatch.z == 1);
    BICORE_CHECK(packet->bindings.size() == 3);
    const auto & src = packet->bindings[0];
    const auto & dst = packet->bindings[1];
    const auto & uniform = packet->bindings[2];
    const size_t expected_prefix = route == Route::reshape ? prefix : 0;
    const size_t payload_bytes = width * columns * sizeof(float);
    BICORE_CHECK(src.binding_index == 0 && src.buffer.Get() == ggml_webgpu_tensor_buf(input).Get());
    BICORE_CHECK(dst.binding_index == 1 && dst.buffer.Get() == src.buffer.Get());
    BICORE_CHECK(src.offset == ggml_webgpu_tensor_offset(input) - expected_prefix);
    BICORE_CHECK(dst.offset == ggml_webgpu_tensor_offset(output));
    BICORE_CHECK(src.size == payload_bytes + expected_prefix && dst.size == payload_bytes);
    BICORE_CHECK(src.offset + src.size <= dst.offset || dst.offset + dst.size <= src.offset);
    BICORE_CHECK(uniform.binding_index == 2 && uniform.size == WEBGPU_PARAMS_BUF_SIZE_BYTES);
    std::array<uint32_t, 2> params{};
    std::memcpy(params.data(), uniform.buffer.Get()->bytes.data() + uniform.offset, sizeof(params));
    BICORE_CHECK(params[0] == expected_prefix / sizeof(float) && params[1] == 0);
}

static void all() {
    for (const auto route : {Route::plain, Route::reshape, Route::no_hint, Route::matvec, Route::strided, Route::alias,
                             Route::accumulator_precision, Route::source_precision, Route::subgroup_matrix,
                             Route::invocations_limit, Route::x_limit, Route::shared_limit, Route::grid_limit}) {
        graph(route);
    }
}

} // namespace fwht_1024_cases
