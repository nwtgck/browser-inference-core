#pragma once

namespace fusion_cases {
enum class Family { scale_unary, scale_tanh, rms_scale };
enum class Guard { ordinary, exposed_intermediate, external_consumer, raw_intermediate, raw_output };
enum class RouterDenominator { clamp, epsilon, other_epsilon, other_scale, exposed };
enum class RouterAlias { none, sum, result };

static void graph(Family family, Guard guard, bool final_scale = false) {
    transfer_cases::Backend gpu(transfer_cases::BackendKind::webgpu);
    const auto memory = transfer_cases::Memory(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
    BICORE_CHECK(memory);
    const bool tanh = family == Family::scale_tanh;
    const bool unary = family != Family::rms_scale;
    auto * input = ggml_new_tensor_2d(memory.get(), GGML_TYPE_F32, tanh ? 202048 : 64, tanh ? 1 : 2);
    ggml_set_input(input);
    auto * intermediate =
        unary ? ggml_scale(memory.get(), input, tanh ? 0.05f : 0.5f) : ggml_rms_norm(memory.get(), input, 1e-5f);
    auto * output =
        tanh ? ggml_tanh(memory.get(), intermediate)
             : (unary ? ggml_silu(memory.get(), intermediate) : ggml_scale(memory.get(), intermediate, 0.5f));
    if (final_scale)
        output = ggml_scale(memory.get(), output, tanh ? 20.0f : 0.25f);
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
    if (tanh) {
        const auto & dispatch = commands.dispatches[0];
        BICORE_CHECK(dispatch.x == 1579 && dispatch.y == 1);
        const auto & shader = *gpu.context->bind_group_cache.slots[0].pipeline.Get();
        BICORE_CHECK(shader.name.starts_with(final_scale ? "scale_tanh_scale" : "scale_tanh"));
    }
    const auto created = ::created, groups = resources.groups_created;
    const float coefficient = 2.5f;
    auto * scale = unary ? intermediate : output;
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
    webgpu_fusion_view_sources views(*graph);
    const auto result = unary ? ggml_webgpu_scale_unary(gpu.context, graph, 0, views, consumed)
                              : ggml_webgpu_rms_scale(gpu.context, graph, 0, views, consumed);
    output->data = original_data;
    output->buffer = original_buffer;
    BICORE_CHECK(!result && consumed == 1 && ::created == before && commands.dispatches.empty());
}

static void l2_clip_graph(int64_t width, int64_t heads, int64_t tokens, bool inplace, Guard guard,
                          bool expected_fusion) {
    transfer_cases::Backend gpu(transfer_cases::BackendKind::webgpu);
    constexpr size_t fixture_bytes = 8 * 1024 * 1024;
    auto & limits = gpu.context->global_ctx->capabilities.limits;
    limits.maxStorageBufferBindingSize = fixture_bytes;
    limits.maxBufferSize = fixture_bytes;
    gpu.type.iface.get_max_size = [](ggml_backend_buffer_type_t) { return fixture_bytes; };
    const auto memory = transfer_cases::Memory(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
    BICORE_CHECK(memory);
    auto * input = ggml_new_tensor_3d(memory.get(), GGML_TYPE_F32, width, heads, tokens);
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
    const int clip_end = ggml_graph_n_nodes(graph);
    const bool raw_view = guard == Guard::raw_intermediate || guard == Guard::raw_output;
    if (guard == Guard::external_consumer) {
        auto * consumer = ggml_scale(memory.get(), denominator, 2.0f);
        ggml_set_output(consumer);
        ggml_build_forward_expand(graph, consumer);
    } else if (raw_view) {
        auto * source = guard == Guard::raw_intermediate ? denominator : output;
        auto * view = ggml_view_tensor(memory.get(), source);
        BICORE_CHECK(view->view_src == source && !view->src[0]);
        auto * consumer = ggml_dup(memory.get(), view);
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

    const auto run = [&](ggml_cgraph * slice, unsigned count) {
        commands = {};
        readback = {};
        BICORE_CHECK(ggml_backend_graph_compute_async(&gpu.backend, slice) == GGML_STATUS_SUCCESS);
        BICORE_CHECK(commands.dispatches.size() == count);
        ggml_backend_synchronize(&gpu.backend);
    };
    const auto clip_dispatches = expected_fusion ? 1u : unsigned(clip_end);
    if (raw_view) {
        // The scheduler may send the reader to a later slice.
        auto slice = ggml_graph_view(graph, 0, clip_end);
        run(&slice, clip_dispatches);
    }
    run(graph, clip_dispatches + unsigned(ggml_graph_n_nodes(graph) - clip_end));
    if (!expected_fusion)
        return;

    const auto pipeline = gpu.context->shader_lib->get_l2_clip_pipeline(inplace);
    BICORE_CHECK(pipeline.name == (inplace ? "l2_clip_f32_inplace" : "l2_clip_f32"));
    BICORE_CHECK(commands.dispatches[0].pipeline == pipeline.pipeline.Get()->id);
    BICORE_CHECK(commands.dispatches[0].x == uint32_t(heads * tokens) && commands.dispatches[0].y == 1);

    if (inplace) {
        const auto original_data = output->data;
        output->data = reinterpret_cast<void *>(uintptr_t(input->data) + sizeof(float));
        int consumed = 1;
        commands = {};
        webgpu_fusion_view_sources views(*graph);
        const auto rejected = ggml_webgpu_l2_clip(gpu.context, graph, 0, views, consumed);
        output->data = original_data;
        BICORE_CHECK(!rejected && consumed == 1 && commands.dispatches.empty());
    }
}

static void router_tail_graph(bool natural, RouterDenominator denominator, RouterAlias alias = RouterAlias::none) {
    transfer_cases::Backend gpu(transfer_cases::BackendKind::webgpu);
    auto & limits = gpu.context->global_ctx->capabilities.limits;
    limits.maxComputeInvocationsPerWorkgroup = 1024;
    limits.maxComputeWorkgroupSizeX = 1024;
    const auto memory = transfer_cases::Memory(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
    BICORE_CHECK(memory);
    auto * logits = ggml_new_tensor_2d(memory.get(), GGML_TYPE_F32, 128, 1);
    auto * bias = ggml_new_tensor_1d(memory.get(), GGML_TYPE_F32, 128);
    ggml_set_input(logits);
    ggml_set_input(bias);
    auto * probs = ggml_sigmoid(memory.get(), logits);
    auto * ids = ggml_argsort_top_k(memory.get(), ggml_add(memory.get(), probs, bias), 8);
    auto * gathered = ggml_get_rows(memory.get(), ggml_reshape_3d(memory.get(), probs, 1, 128, 1), ids);
    auto * flat = ggml_reshape_2d(memory.get(), gathered, 8, 1);
    auto * sum = ggml_sum_rows(memory.get(), flat);
    const bool clamp = denominator == RouterDenominator::clamp;
    const float scale = denominator == RouterDenominator::other_scale ? 2.0f : 1.0f;
    const float epsilon = denominator == RouterDenominator::other_epsilon ? 1e-10f : 1e-20f;
    auto * normalizer =
        clamp ? ggml_clamp(memory.get(), sum, 0x1p-14f, INFINITY) : ggml_scale_bias(memory.get(), sum, scale, epsilon);
    if (denominator == RouterDenominator::exposed)
        ggml_set_output(normalizer);
    auto * div = ggml_div(memory.get(), flat, normalizer);
    auto * output = ggml_reshape_3d(memory.get(), div, 1, 8, 1);
    auto * weights_consumer = ggml_scale(memory.get(), output, 2.0f);
    auto * ids_consumer = ggml_cont(memory.get(), ids);
    ggml_set_output(weights_consumer);
    ggml_set_output(ids_consumer);
    auto * graph = ggml_new_graph_custom(memory.get(), 64, false);
    ggml_build_forward_expand(graph, weights_consumer);
    ggml_build_forward_expand(graph, ids_consumer);
    const int stop = graph->n_nodes;
    if (alias != RouterAlias::none) {
        auto * root = alias == RouterAlias::sum ? sum : div;
        auto * view = ggml_view_tensor(memory.get(), root);
        BICORE_CHECK(view->view_src == root && !view->src[0]);
        auto * reader = ggml_dup(memory.get(), view);
        ggml_set_output(reader);
        ggml_build_forward_expand(graph, reader);
    }
    const auto allocator = std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)>(
        natural ? ggml_gallocr_new(&gpu.type) : nullptr, ggml_gallocr_free);
    const auto storage =
        transfer_cases::Buffer(natural ? nullptr : ggml_backend_alloc_ctx_tensors_from_buft(memory.get(), &gpu.type),
                               ggml_backend_buffer_free);
    BICORE_CHECK(natural ? allocator && ggml_gallocr_alloc_graph(allocator.get(), graph) : bool(storage));

    const bool fused = alias != RouterAlias::sum && (clamp || denominator == RouterDenominator::epsilon);
    const unsigned dispatches = fused ? 6u : 9u;
    const auto compute = [&](ggml_cgraph * view, unsigned expected) {
        commands = {};
        readback = {};
        BICORE_CHECK(ggml_backend_graph_compute_async(&gpu.backend, view) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(&gpu.backend);
        BICORE_CHECK(commands.dispatches.size() == expected);
    };
    if (alias != RouterAlias::none) {
        auto slice = ggml_graph_view(graph, 0, stop);
        compute(&slice, dispatches);
    }
    compute(graph, dispatches + unsigned(alias != RouterAlias::none));
    if (!fused)
        return;

    // The three upstream operations still encode separately. Natural allocation
    // reuses the probabilities for the weights, exercising the overlap path.
    const auto pipeline = gpu.context->shader_lib->get_router_tail_pipeline(1024, natural, !clamp);
    const auto sibling = gpu.context->shader_lib->get_router_tail_pipeline(1024, natural, clamp);
    const auto & dispatch = commands.dispatches[3];
    BICORE_CHECK(pipeline.pipeline.Get() && sibling.pipeline.Get() &&
                 pipeline.pipeline.Get() != sibling.pipeline.Get());
    BICORE_CHECK(dispatch.pipeline == pipeline.pipeline.Get()->id && dispatch.x == 1 && dispatch.y == 1 &&
                 dispatch.z == 1);
    if (!clamp) {
        const auto & arena = gpu.context->param_arena;
        const size_t offset = 3 * arena.slot_stride + 8 * sizeof(uint32_t);
        const std::array expected{scale, epsilon};
        const auto upload = std::find_if(commands.writes.begin(), commands.writes.end(), [&](const auto & write) {
            return write.buffer == arena.buffer.Get() && write.offset == 0 &&
                   write.bytes.size() >= offset + sizeof(expected);
        });
        BICORE_CHECK(upload != commands.writes.end());
        BICORE_CHECK(std::memcmp(upload->bytes.data() + offset, expected.data(), sizeof(expected)) == 0);
    }
}

static void rms_vec4() {
    struct Case {
        int64_t width, rows;
        size_t prefix, padding;
        bool inplace, affine;
        uint32_t invocation_limit;
        std::string_view pipeline;
    };
    for (const auto & test : {
             Case{2304, 1, 0, 0, false, false, 256, "rms_norm_src_f32_dst_f32"},
             Case{4096, 1, 0, 0, false, false, 256, "rms_norm_src_f32_dst_f32_vec4"},
             Case{128, 8, 16, 16, false, false, 256, "rms_norm_src_f32_dst_f32_vec4"},
             Case{128, 8, 16, 0, true, false, 256, "rms_norm_src_f32_dst_f32_vec4_inplace"},
             Case{128, 8, 4, 16, false, false, 256, "rms_norm_src_f32_dst_f32"},
             Case{128, 8, 16, 4, false, false, 256, "rms_norm_src_f32_dst_f32"},
             Case{127, 8, 0, 0, false, false, 256, "rms_norm_src_f32_dst_f32"},
             Case{128, 8, 0, 0, false, false, 64, "rms_norm_src_f32_dst_f32"},
             Case{128, 8, 0, 0, false, true, 256, "rms_norm_scale_f32_vec4"},
             Case{128, 8, 4, 0, false, true, 256, "rms_norm_scale_f32"},
         }) {
        transfer_cases::Backend gpu(transfer_cases::BackendKind::webgpu);
        gpu.context->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup = test.invocation_limit;
        const transfer_cases::Memory memory(ggml_init({32 * 1024, nullptr, true}), ggml_free);
        const size_t stride = test.width * sizeof(float) + test.padding;
        auto * allocation =
            ggml_new_tensor_1d(memory.get(), GGML_TYPE_F32, (stride * test.rows + test.prefix) / sizeof(float));
        ggml_set_input(allocation);
        auto * input = ggml_view_2d(memory.get(), allocation, test.width, test.rows, stride, test.prefix);
        auto * rms = test.inplace ? ggml_rms_norm_inplace(memory.get(), input, 1e-6f)
                                  : ggml_rms_norm(memory.get(), input, 1e-6f);
        auto * output = test.affine ? ggml_scale_bias(memory.get(), rms, 0.75f, -0.125f) : rms;
        ggml_set_output(output);
        auto * graph = ggml_new_graph_custom(memory.get(), 16, false);
        ggml_build_forward_expand(graph, output);
        const transfer_cases::Buffer buffer(ggml_backend_alloc_ctx_tensors_from_buft(memory.get(), &gpu.type),
                                            ggml_backend_buffer_free);
        BICORE_CHECK(buffer && ggml_backend_dev_supports_op(&gpu.device, rms));
        commands = {};
        BICORE_CHECK(ggml_backend_graph_compute_async(&gpu.backend, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(&gpu.backend);
        const auto & slots = gpu.context->bind_group_cache.slots;
        BICORE_CHECK(slots.size() == 1 && commands.executed == 1);
        BICORE_CHECK(slots.front().pipeline.Get()->name == test.pipeline);
        BICORE_CHECK(slots.front().bindings.size() == (test.inplace ? 2u : 3u));
        BICORE_CHECK(commands.dispatches.front().x == test.rows && commands.dispatches.front().y == 1);
    }
}

static void rms_concat() {
    enum class Input { direct, view, unaligned_view, alias, computed };
    enum class Use { ordinary, output, raw_view, different_axis };
    struct Case {
        Input input;
        Use use;
        int64_t rows;
        bool use_graph_allocator, fused;
    };
    for (const auto & test : {
             Case{Input::direct, Use::ordinary, 1, false, true},
             Case{Input::view, Use::ordinary, 1, true, true},
             Case{Input::view, Use::ordinary, 24, true, true},
             Case{Input::unaligned_view, Use::ordinary, 1, true, false},
             Case{Input::alias, Use::ordinary, 1, false, false},
             Case{Input::computed, Use::ordinary, 1, false, false},
             Case{Input::direct, Use::different_axis, 1, false, false},
             Case{Input::direct, Use::output, 1, true, false},
             Case{Input::direct, Use::raw_view, 1, true, false},
         }) {
        transfer_cases::Backend gpu(transfer_cases::BackendKind::webgpu);
        auto & limits = gpu.context->global_ctx->capabilities.limits;
        limits.maxComputeWorkgroupsPerDimension = test.rows == 24 ? 7 : 64;
        const transfer_cases::Memory memory(ggml_init({128 * 1024, nullptr, true}), ggml_free);
        auto * ctx = memory.get();
        constexpr int64_t width = 2304;
        auto * left = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, test.rows);
        ggml_set_input(left);
        auto * root = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width + 16, test.rows);
        ggml_set_input(root);
        ggml_tensor * right = left;
        if (test.input != Input::alias) {
            const size_t offset = test.input == Input::unaligned_view ? 4u : 16u;
            right = test.input == Input::view || test.input == Input::unaligned_view
                        ? ggml_view_2d(ctx, root, width, test.rows, root->nb[1], offset)
                        : ggml_dup_tensor(ctx, left);
            if (test.input == Input::computed) {
                right = ggml_scale(ctx, left, 3.25f);
            } else if (!right->view_src) {
                ggml_set_input(right);
            }
        }
        // Keep both original inputs live across the concat in the allocator cases.
        if (test.use_graph_allocator) {
            ggml_set_output(left);
            ggml_set_output(right->view_src ? root : right);
        }
        auto * a = ggml_rms_norm(ctx, left, 1e-6f);
        auto * b = ggml_rms_norm(ctx, right, 1e-4f);
        auto * concat = ggml_concat(ctx, a, b, test.use == Use::different_axis ? 1 : 0);
        ggml_set_output(concat);
        if (test.use == Use::output) {
            ggml_set_output(a);
        }
        auto * graph = ggml_new_graph_custom(ctx, 32, false);
        ggml_build_forward_expand(graph, concat);
        if (test.input == Input::view) {
            BICORE_CHECK(ggml_graph_n_nodes(graph) == 4 && ggml_graph_node(graph, 1) == right);
        }
        if (test.use == Use::raw_view) {
            auto * observer = ggml_dup(ctx, ggml_view_tensor(ctx, a));
            ggml_set_output(observer);
            ggml_build_forward_expand(graph, observer);
        }
        const auto allocator = std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)>(
            test.use_graph_allocator ? ggml_gallocr_new(&gpu.type) : nullptr, ggml_gallocr_free);
        const transfer_cases::Buffer buffer(
            test.use_graph_allocator ? nullptr : ggml_backend_alloc_ctx_tensors_from_buft(ctx, &gpu.type),
            ggml_backend_buffer_free);
        BICORE_CHECK(test.use_graph_allocator ? allocator && ggml_gallocr_alloc_graph(allocator.get(), graph)
                                              : bool(buffer));
        commands = {};
        BICORE_CHECK(ggml_backend_graph_compute_async(&gpu.backend, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(&gpu.backend);
        const auto & slots = gpu.context->bind_group_cache.slots;
        const auto fused = std::count_if(slots.begin(), slots.end(), [](const auto & slot) {
            return slot.pipeline.Get()->name == "rms_norm_concat_f32_vec4";
        });
        const auto extra = unsigned(test.use == Use::raw_view) + unsigned(test.input == Input::computed);
        BICORE_CHECK(fused == test.fused && commands.executed == (test.fused ? 1u : 3u + extra));
        if (test.fused) {
            BICORE_CHECK(slots[0].bindings.size() == 4u);
            if (test.rows == 24) {
                BICORE_CHECK(commands.dispatches[0].x == 7 && commands.dispatches[0].y == 7);
            } else {
                BICORE_CHECK(commands.dispatches[0].x == 2 * test.rows && commands.dispatches[0].y == 1);
            }
            const auto & binding = slots[0].bindings.back();
            std::array<uint32_t, 10> words;
            std::memcpy(words.data(), binding.buffer.Get()->bytes.data() + binding.offset, sizeof(words));
            BICORE_CHECK(words[6] == width && words[7] == test.rows);
            BICORE_CHECK(words[8] == std::bit_cast<uint32_t>(1e-6f) && words[9] == std::bit_cast<uint32_t>(1e-4f));
        }
    }
}

static void all() {
    rms_concat();
    rms_vec4();
    for (const auto family : {Family::scale_unary, Family::scale_tanh, Family::rms_scale})
        for (const auto guard : {Guard::ordinary, Guard::exposed_intermediate, Guard::external_consumer})
            graph(family, guard);
    graph(Family::scale_unary, Guard::ordinary, true);
    graph(Family::scale_tanh, Guard::ordinary, true);
    l2_clip_graph(128, 24, 1, true, Guard::ordinary, true);
    l2_clip_graph(128, 8, 1, false, Guard::ordinary, true);
    l2_clip_graph(16, 24, 1, false, Guard::ordinary, true);
    l2_clip_graph(128, 12, 1, false, Guard::ordinary, false);
    l2_clip_graph(128, 24, 1, false, Guard::exposed_intermediate, false);
    l2_clip_graph(128, 24, 1, false, Guard::external_consumer, false);
    l2_clip_graph(128, 24, 128, true, Guard::ordinary, true);
    l2_clip_graph(128, 8, 53, false, Guard::ordinary, true);
    l2_clip_graph(16, 24, 2, false, Guard::ordinary, true);
    l2_clip_graph(128, 24, 129, false, Guard::ordinary, false);
    l2_clip_graph(128, 24, 2, false, Guard::raw_intermediate, false);
    l2_clip_graph(128, 24, 2, false, Guard::raw_output, true);
    for (bool natural : {false, true}) {
        for (const auto denominator :
             {RouterDenominator::clamp, RouterDenominator::epsilon, RouterDenominator::other_epsilon,
              RouterDenominator::other_scale, RouterDenominator::exposed})
            router_tail_graph(natural, denominator);
        router_tail_graph(natural, RouterDenominator::clamp, RouterAlias::sum);
        router_tail_graph(natural, RouterDenominator::epsilon, RouterAlias::sum);
        router_tail_graph(natural, RouterDenominator::epsilon, RouterAlias::result);
    }
}
} // namespace fusion_cases
