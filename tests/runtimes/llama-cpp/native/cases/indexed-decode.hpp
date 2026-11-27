namespace indexed_decode_cases {

static void check_bounds(ggml_webgpu_shader_lib_context candidate) {
    const auto accepted = [&](const auto & value) { return ggml_webgpu_id_subgroup_size(value) != 0; };
    BICORE_CHECK(accepted(candidate));
    auto invalid = candidate;
    invalid.indexed_prefix_bytes[0] = 1;
    BICORE_CHECK(!accepted(invalid));
    invalid = candidate;
    invalid.indexed_binding_bytes[0]--;
    BICORE_CHECK(!accepted(invalid));
    invalid = candidate;
    invalid.indexed_buffer_remaining[1]--;
    BICORE_CHECK(!accepted(invalid));
    invalid = candidate;
    invalid.max_workgroups_per_dim = 1;
    BICORE_CHECK(!accepted(invalid));
    invalid = candidate;
    invalid.max_wg_size = 127;
    BICORE_CHECK(!accepted(invalid));
    invalid = candidate;
    auto copied = *candidate.src0;
    copied.nb[2] -= copied.nb[0];
    invalid.src0 = &copied;
    BICORE_CHECK(!accepted(invalid));
    copied = *candidate.src0;
    copied.ne[0] -= ggml_blck_size(copied.type);
    auto activations = *candidate.src1;
    activations.ne[0] = copied.ne[0];
    invalid.src1 = &activations;
    BICORE_CHECK(!accepted(invalid));
    if (candidate.src0->type == GGML_TYPE_IQ3_S)
        return;
    invalid = candidate;
    copied = *candidate.src0;
    copied.ne[2] = copied.ne[0] == 2560 ? 512 : 128;
    const size_t full_weights = ggml_nbytes(&copied);
    invalid.src0 = &copied;
    invalid.indexed_binding_bytes[0] = full_weights;
    invalid.indexed_buffer_remaining[0] = full_weights;
    invalid.max_storage_binding_bytes = full_weights;
    BICORE_CHECK(accepted(invalid));
    invalid.max_storage_binding_bytes--;
    BICORE_CHECK(!accepted(invalid));
}

static void check_provider(ggml_type quant, int64_t k, int64_t input_rows, uint32_t sg_min, uint32_t sg_max,
                           bool independent, int64_t rows = 2304, int64_t selected = 8) {
    transfer_cases::Backend gpu(transfer_cases::BackendKind::webgpu);
    constexpr auto max_buffer = 128u * 1024 * 1024;
    auto & caps = gpu.context->global_ctx->capabilities;
    caps.supports_subgroups = true;
    caps.min_subgroup_size = sg_min;
    caps.max_subgroup_size = sg_max;
    caps.limits.maxStorageBufferBindingSize = max_buffer;
    caps.limits.maxBufferSize = max_buffer;
    gpu.type.iface.get_max_size = [](ggml_backend_buffer_type_t) -> size_t { return max_buffer; };
    transfer_cases::Memory memory(ggml_init({256 * 1024, nullptr, true}), ggml_free);
    BICORE_CHECK(memory);
    const int64_t experts = 16;
    auto * weights = ggml_new_tensor_3d(memory.get(), quant, k, rows, experts);
    auto * input = ggml_new_tensor_3d(memory.get(), GGML_TYPE_F32, k, input_rows, 1);
    auto * ids = ggml_new_tensor_2d(memory.get(), GGML_TYPE_I32, selected, 1);
    auto * result = ggml_mul_mat_id(memory.get(), weights, input, ids);
    const std::array<ggml_tensor *, 4> tensors{weights, input, ids, result};
    std::vector<transfer_cases::Buffer> storage;
    ggml_webgpu_shader_lib_context decision{};
    decision.src0 = weights;
    decision.src1 = input;
    decision.src2 = ids;
    decision.dst = result;
    decision.supports_subgroups = true;
    decision.max_wg_size = caps.limits.maxComputeInvocationsPerWorkgroup;
    decision.max_workgroups_per_dim = caps.limits.maxComputeWorkgroupsPerDimension;
    decision.max_storage_binding_bytes = caps.limits.maxStorageBufferBindingSize;
    decision.min_subgroup_size = sg_min;
    decision.max_subgroup_size = sg_max;
    for (size_t i = 0; i < tensors.size(); ++i) {
        auto * tensor = tensors[i];
        const size_t bytes = ggml_nbytes(tensor);
        const size_t allocation_bytes = ggml_backend_buft_get_alloc_size(&gpu.type, tensor);
        storage.emplace_back(ggml_backend_buft_alloc_buffer(&gpu.type, allocation_bytes), ggml_backend_buffer_free);
        BICORE_CHECK(storage.back());
        BICORE_CHECK(ggml_backend_tensor_alloc(storage.back().get(), tensor,
                                               ggml_backend_buffer_get_base(storage.back().get())) ==
                     GGML_STATUS_SUCCESS);
        decision.indexed_binding_bytes[i] = bytes;
        decision.indexed_buffer_remaining[i] = bytes;
    }
    if (independent) {
        check_bounds(decision);
    } else {
        BICORE_CHECK(ggml_webgpu_id_subgroup_size(decision) == 0);
    }
    auto * graph = ggml_new_graph_custom(memory.get(), 16, false);
    ggml_build_forward_expand(graph, result);
    commands = {};
    BICORE_CHECK(ggml_backend_graph_compute_async(&gpu.backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(&gpu.backend);
    BICORE_CHECK(commands.dispatches.size() == 1);
    const auto & dispatch = commands.dispatches.front();
    const uint32_t rows_per_group = independent ? 4 * (128 / sg_max) : 4;
    BICORE_CHECK(dispatch.x == CEIL_DIV(rows, rows_per_group) * selected && dispatch.y == 1 && dispatch.z == 1);
    const auto & cached = gpu.context->bind_group_cache.slots.front();
    const auto & pipeline = *cached.pipeline.Get();
    BICORE_CHECK((pipeline.name.find("_independent_sg" + std::to_string(sg_max)) != std::string::npos) == independent);
    BICORE_CHECK(cached.bindings.size() == 5);
    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto & view = cached.bindings[i];
        BICORE_CHECK(view.offset == 0 && view.size == ggml_nbytes(tensors[i]));
    }
}

static void all() {
    for (const auto quant : {GGML_TYPE_IQ4_NL, GGML_TYPE_NVFP4, GGML_TYPE_Q8_0, GGML_TYPE_Q5_K}) {
        for (const int64_t k : {1152, 2304}) {
            if (quant == GGML_TYPE_Q5_K && k == 1152) {
                continue;
            }
            const int64_t input_rows = k == 1152 ? 8 : 1;
            check_provider(quant, k, input_rows, 16, 16, true);
            check_provider(quant, k, input_rows, 32, 32, true);
            check_provider(quant, k, input_rows, 8, 32, true);
            check_provider(quant, k, input_rows, 4, 32, false);
            check_provider(quant, k, input_rows, 64, 64, false);
        }
    }
    for (uint32_t sg : {16, 32, 64})
        check_provider(GGML_TYPE_IQ3_S, 2048, 1, sg, sg, true);
    check_provider(GGML_TYPE_IQ3_S, 2048, 1, 8, 32, true);
    check_provider(GGML_TYPE_IQ3_S, 2048, 1, 8, 64, false);
    for (const auto quant : {GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS}) {
        check_provider(quant, 2304, 1, 16, 16, true);
        check_provider(quant, 2304, 1, 32, 32, true);
        check_provider(quant, 2304, 1, 8, 16, true);
        check_provider(quant, 2304, 1, 8, 32, true);
        check_provider(quant, 2304, 1, 4, 32, false);
        check_provider(quant, 2304, 1, 64, 64, false);
        check_provider(quant, 2560, 1, 16, 16, true, 640, 10);
        check_provider(quant, 2560, 1, 32, 32, true, 640, 10);
        check_provider(quant, 2560, 1, 16, 32, true, 640, 10);
    }
}
} // namespace indexed_decode_cases
