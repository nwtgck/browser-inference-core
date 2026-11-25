#pragma once

namespace fusion_cases {
enum class Family { scale_unary, rms_scale };
enum class Guard { ordinary, exposed_intermediate, external_consumer };

static void graph(Family family, Guard guard, bool final_scale = false) {
    transfer_cases::Backend gpu(transfer_cases::BackendKind::webgpu);
    const auto memory = transfer_cases::Memory(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
    BICORE_CHECK(memory);
    auto * input = ggml_new_tensor_2d(memory.get(), GGML_TYPE_F32, 64, 2);
    ggml_set_input(input);
    auto * intermediate = family == Family::scale_unary ? ggml_scale(memory.get(), input, 0.5f)
                                                        : ggml_rms_norm(memory.get(), input, 1e-5f);
    auto * output = family == Family::scale_unary ? ggml_silu(memory.get(), intermediate)
                                                  : ggml_scale(memory.get(), intermediate, 0.5f);
    if (final_scale)
        output = ggml_scale(memory.get(), output, 0.25f);
    ggml_set_output(output);
    if (guard == Guard::exposed_intermediate)
        ggml_set_output(intermediate);
    auto * graph = ggml_new_graph_custom(memory.get(), 64, false);
    ggml_build_forward_expand(graph, output);
    if (guard == Guard::external_consumer) {
        auto * consumer = ggml_scale(memory.get(), intermediate, 0.75f);
        ggml_set_output(consumer);
        ggml_build_forward_expand(graph, consumer);
    }
    const auto allocator =
        std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)>(ggml_gallocr_new(&gpu.type), ggml_gallocr_free);
    BICORE_CHECK(allocator && ggml_gallocr_alloc_graph(allocator.get(), graph));
    const bool fused = guard == Guard::ordinary;
    const auto dispatches = fused ? 1u : unsigned(graph->n_nodes);
    commands = {};
    readback = {};
    BICORE_CHECK(ggml_backend_graph_compute_async(&gpu.backend, graph) == GGML_STATUS_SUCCESS);
    BICORE_CHECK(commands.dispatches.size() == dispatches);
    ggml_backend_synchronize(&gpu.backend);
    if (!fused)
        return;
    const auto pipeline = commands.dispatches[0].pipeline;
    const auto created = ::created, groups = resources.groups_created;
    const float coefficient = 2.5f;
    auto * scale = family == Family::scale_unary ? intermediate : output;
    std::memcpy(scale->op_params, &coefficient, sizeof(coefficient));
    commands = {};
    BICORE_CHECK(ggml_backend_graph_compute_async(&gpu.backend, graph) == GGML_STATUS_SUCCESS);
    BICORE_CHECK(commands.dispatches.size() == 1 && commands.dispatches[0].pipeline == pipeline &&
                 ::created == created && resources.groups_created == groups);
    std::array<uint8_t, sizeof(float)> bytes;
    std::memcpy(bytes.data(), &coefficient, bytes.size());
    bool uploaded = false;
    for (const auto & write : commands.writes)
        for (size_t offset = 0; offset + bytes.size() <= write.bytes.size(); offset += sizeof(float))
            uploaded |= std::memcmp(write.bytes.data() + offset, bytes.data(), bytes.size()) == 0;
    BICORE_CHECK(uploaded);
    ggml_backend_synchronize(&gpu.backend);
    if (final_scale)
        return;

    // Check a rejected candidate before invoking any fallback. Invalid partial
    // overlap must neither skip nodes nor create pipelines/encode commands.
    const auto original_data = output->data;
    const auto original_buffer = output->buffer;
    output->data = reinterpret_cast<void *>(uintptr_t(input->data) + 4);
    output->buffer = input->buffer;
    commands = {};
    int consumed = 1;
    const auto before = ::created;
    const auto result = family == Family::scale_unary ? ggml_webgpu_scale_unary(gpu.context, graph, 0, consumed)
                                                      : ggml_webgpu_rms_scale(gpu.context, graph, 0, consumed);
    output->data = original_data;
    output->buffer = original_buffer;
    BICORE_CHECK(!result && consumed == 1 && ::created == before && commands.dispatches.empty());
}

static void l2_clip_graph(int64_t width, int64_t rows, bool inplace, Guard guard, bool expected_fusion) {
    transfer_cases::Backend gpu(transfer_cases::BackendKind::webgpu);
    const auto memory = transfer_cases::Memory(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
    BICORE_CHECK(memory);
    auto * input = ggml_new_tensor_2d(memory.get(), GGML_TYPE_F32, width, rows);
    ggml_set_input(input);
    auto * square = ggml_sqr(memory.get(), input);
    auto * sum = ggml_sum_rows(memory.get(), square);
    auto * norm = ggml_sqrt(memory.get(), sum);
    auto * denominator = ggml_scale_bias(memory.get(), norm, 1.0f, 1e-6f);
    auto * threshold = ggml_fill(memory.get(), denominator, 0.75f);
    auto * factor = ggml_clamp(memory.get(), ggml_div(memory.get(), threshold, denominator), 0.0f, 1.0f);
    auto * output = ggml_mul(memory.get(), input, factor);
    ggml_set_output(output);
    if (guard == Guard::exposed_intermediate)
        ggml_set_output(denominator);
    auto * graph = ggml_new_graph_custom(memory.get(), 64, false);
    ggml_build_forward_expand(graph, output);
    if (guard == Guard::external_consumer) {
        auto * consumer = ggml_scale(memory.get(), denominator, 2.0f);
        ggml_set_output(consumer);
        ggml_build_forward_expand(graph, consumer);
    }
    const auto allocator = std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)>(
        inplace ? ggml_gallocr_new(&gpu.type) : nullptr, ggml_gallocr_free);
    const auto buffer =
        transfer_cases::Buffer(inplace ? nullptr : ggml_backend_alloc_ctx_tensors_from_buft(memory.get(), &gpu.type),
                               ggml_backend_buffer_free);
    BICORE_CHECK(inplace ? allocator && ggml_gallocr_alloc_graph(allocator.get(), graph) : bool(buffer));
    BICORE_CHECK(input->buffer == output->buffer && (input->data == output->data) == inplace);

    commands = {};
    readback = {};
    BICORE_CHECK(ggml_backend_graph_compute_async(&gpu.backend, graph) == GGML_STATUS_SUCCESS);
    BICORE_CHECK(commands.dispatches.size() == (expected_fusion ? 1u : unsigned(graph->n_nodes)));
    ggml_backend_synchronize(&gpu.backend);
    if (!expected_fusion)
        return;

    const auto pipeline = gpu.context->shader_lib->get_l2_clip_pipeline(inplace);
    BICORE_CHECK(pipeline.name == (inplace ? "l2_clip_f32_inplace" : "l2_clip_f32"));
    BICORE_CHECK(commands.dispatches[0].pipeline == pipeline.pipeline.Get()->id);
    BICORE_CHECK(commands.dispatches[0].x == uint32_t(rows) && commands.dispatches[0].y == 1);

    if (inplace) {
        const auto original_data = output->data;
        output->data = reinterpret_cast<void *>(uintptr_t(input->data) + sizeof(float));
        int consumed = 1;
        commands = {};
        const auto rejected = ggml_webgpu_l2_clip(gpu.context, graph, 0, consumed);
        output->data = original_data;
        BICORE_CHECK(!rejected && consumed == 1 && commands.dispatches.empty());
    }
}

static void all() {
    for (const auto family : {Family::scale_unary, Family::rms_scale})
        for (const auto guard : {Guard::ordinary, Guard::exposed_intermediate, Guard::external_consumer})
            graph(family, guard);
    graph(Family::scale_unary, Guard::ordinary, true);
    l2_clip_graph(128, 24, true, Guard::ordinary, true);
    l2_clip_graph(128, 8, false, Guard::ordinary, true);
    l2_clip_graph(16, 24, false, Guard::ordinary, true);
    l2_clip_graph(128, 12, false, Guard::ordinary, false);
    l2_clip_graph(128, 24, false, Guard::exposed_intermediate, false);
    l2_clip_graph(128, 24, false, Guard::external_consumer, false);
}
} // namespace fusion_cases
