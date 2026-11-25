#pragma once

namespace loop_controller_cases {

enum class Guard {
    ordinary,
    weight_prefix,
    scalar_weight,
    observed_projection,
    intermediate_view,
    final_view,
    partial_overlap,
    shared_limit,
    prefill
};

static void graph(bool natural, bool subgroup, Guard guard) {
    transfer_cases::Backend gpu(transfer_cases::BackendKind::webgpu);
    auto & caps = gpu.context->global_ctx->capabilities;
    caps.supports_subgroups = subgroup;
    caps.min_subgroup_size = caps.max_subgroup_size = subgroup ? 32 : 0;
    caps.limits.maxBufferSize = caps.limits.maxStorageBufferBindingSize = 4 * 1024 * 1024;
    gpu.type.iface.get_max_size = [](ggml_backend_buffer_type_t) -> size_t { return 4 * 1024 * 1024; };
    const auto memory = transfer_cases::Memory(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
    BICORE_CHECK(memory);
    auto * ctx = memory.get();
    constexpr int64_t width = 2304;
    constexpr int64_t gate_width = 64;
    const int64_t tokens = guard == Guard::prefill ? 2 : 1;
    const size_t prefix = guard == Guard::weight_prefix ? 16 : guard == Guard::scalar_weight ? 4 : 0;
    auto * current_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, tokens);
    auto * message_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, tokens);
    auto * down = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2 * width, gate_width);
    auto * forget_storage = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, gate_width, width + (prefix != 0));
    auto * forget_weight =
        prefix ? ggml_view_2d(ctx, forget_storage, gate_width, width, forget_storage->nb[1], prefix) : forget_storage;
    auto * gain_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, gate_width, width);
    for (auto * input : {current_input, message_input, down, forget_storage, gain_weight})
        ggml_set_input(input);
    auto * cur = ggml_dup(ctx, current_input);
    auto * message = ggml_dup(ctx, message_input);
    auto * joined = ggml_concat(ctx, ggml_rms_norm(ctx, cur, 1e-6f), ggml_rms_norm(ctx, message, 1e-6f), 0);
    auto * gate = ggml_mul_mat(ctx, down, joined);
    auto * forget_projection = ggml_mul_mat(ctx, forget_weight, gate);
    auto * forget = ggml_tanh(ctx, forget_projection);
    auto * residual = ggml_sub(ctx, cur, ggml_mul(ctx, forget, cur));
    auto * gain_projection = ggml_mul_mat(ctx, gain_weight, gate);
    auto * output = ggml_add(ctx, residual, ggml_mul(ctx, ggml_tanh(ctx, gain_projection), message));
    auto * consumer = ggml_dup(ctx, output);
    ggml_set_output(consumer);
    if (guard == Guard::observed_projection)
        ggml_set_output(forget_projection);
    auto * layout = ggml_new_graph_custom(ctx, 128, false);
    ggml_build_forward_expand(layout, consumer);
    const int controller_end = ggml_graph_n_nodes(layout);

    const bool view_reader = guard == Guard::intermediate_view || guard == Guard::final_view;
    if (view_reader) {
        auto * source = guard == Guard::intermediate_view ? forget : output;
        auto * alias = ggml_view_tensor(ctx, source);
        BICORE_CHECK(alias->view_src == source && !alias->src[0]);
        auto * reader = ggml_dup(ctx, alias);
        ggml_set_output(reader);
        ggml_build_forward_expand(layout, reader);
    }
    const auto allocator = std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)>(
        natural ? ggml_gallocr_new(&gpu.type) : nullptr, ggml_gallocr_free);
    const auto storage = transfer_cases::Buffer(
        natural ? nullptr : ggml_backend_alloc_ctx_tensors_from_buft(ctx, &gpu.type), ggml_backend_buffer_free);
    BICORE_CHECK(natural ? allocator && ggml_gallocr_alloc_graph(allocator.get(), layout) : bool(storage));
    auto * const destination = output->data;
    if (guard == Guard::partial_overlap) {
        BICORE_CHECK(output->data == cur->data);
        output->data = static_cast<char *>(output->data) + sizeof(float);
    }
    if (guard == Guard::shared_limit)
        caps.limits.maxComputeWorkgroupStorageSize = 8191;

    const bool fused = guard == Guard::ordinary || guard == Guard::weight_prefix || guard == Guard::final_view;
    const auto pipeline = gpu.context->shader_lib->get_loop_controller_pipeline(natural, subgroup);
    const auto run = [&](ggml_cgraph * graph, unsigned extra) {
        commands = {};
        BICORE_CHECK(ggml_backend_graph_compute_async(&gpu.backend, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(&gpu.backend);
        const unsigned expected = (fused ? 8u : 15u) + extra;
        BICORE_CHECK(commands.dispatches.size() == expected && commands.executed == expected);
        const auto & dispatch = commands.dispatches[6];
        BICORE_CHECK((dispatch.pipeline == pipeline.pipeline.Get()->id) == fused);
        if (!fused)
            return;

        BICORE_CHECK(dispatch.x == 576 && dispatch.y == 1 && dispatch.z == 1);
        const auto & packet = gpu.context->bind_group_cache.slots[6].bindings;
        BICORE_CHECK(packet.size() == (natural ? 6u : 7u));
        BICORE_CHECK((output->data == cur->data && output->buffer == cur->buffer) == natural);
        const ggml_tensor * operands[] = {forget_weight, gain_weight, gate, cur, message, output};
        const uint64_t alignment = caps.limits.minStorageBufferOffsetAlignment;
        std::array<uint32_t, 6> expected_params{};
        for (size_t i = 0; i < std::size(operands); ++i) {
            const auto * tensor = operands[i];
            const uint64_t base = ggml_webgpu_tensor_offset(tensor);
            const uint64_t offset = base & ~(alignment - 1);
            expected_params[i] = uint32_t((base - offset) / (i < 3 ? 16 : 4));
            if (i < packet.size() - 1) {
                BICORE_CHECK(packet[i].binding_index == i &&
                             packet[i].buffer.Get() == ggml_webgpu_tensor_buf(tensor).Get());
                BICORE_CHECK(packet[i].offset == offset && packet[i].size == base - offset + ggml_nbytes(tensor));
                for (size_t j = 0; j < i; ++j) {
                    BICORE_CHECK(
                        packet[i].buffer.Get() != packet[j].buffer.Get() ||
                        rms_scale_disjoint(packet[i].offset, packet[i].size, packet[j].offset, packet[j].size));
                }
            }
        }
        const auto & uniform = packet.back();
        std::array<uint32_t, 6> params{};
        BICORE_CHECK(uniform.size >= sizeof(params));
        std::memcpy(params.data(), uniform.buffer.Get()->bytes.data() + uniform.offset, sizeof(params));
        BICORE_CHECK(params == expected_params && params[0] == prefix / 16);
    };
    if (view_reader) {
        auto slice = ggml_graph_view(layout, 0, controller_end);
        run(&slice, 0);
    }
    run(layout, unsigned(view_reader));
    output->data = destination;

    if (guard == Guard::ordinary && natural && subgroup) {
        auto * weight_buffer = static_cast<ggml_backend_webgpu_buffer_context *>(forget_weight->buffer->context);
        const auto owner = weight_buffer->global_ctx;
        weight_buffer->global_ctx = std::make_shared<webgpu_global_context_struct>();
        int consumed = 1;
        webgpu_fusion_view_sources views(*layout);
        const auto projection = std::find(layout->nodes, layout->nodes + layout->n_nodes, forget_projection);
        const auto rejected =
            ggml_webgpu_loop_controller(gpu.context, *layout, int(projection - layout->nodes), views, consumed);
        weight_buffer->global_ctx = owner;
        BICORE_CHECK(!rejected && consumed == 1);
    }
}

static void all() {
    graph(false, true, Guard::ordinary);
    graph(true, false, Guard::ordinary);
    for (auto guard :
         {Guard::ordinary, Guard::weight_prefix, Guard::scalar_weight, Guard::observed_projection,
          Guard::intermediate_view, Guard::final_view, Guard::partial_overlap, Guard::shared_limit, Guard::prefill}) {
        graph(true, true, guard);
    }
}

} // namespace loop_controller_cases
