#pragma once

namespace q4_geglu_cases {

enum class Guard {
    ordinary,
    reversed,
    observed_projection,
    raw_view,
    final_view,
    live_alias,
    shape,
    mixed_quant,
    prefill,
    activation,
    glu_swapped,
    glu_params,
    gate_precision,
    up_precision,
    packed_dot,
    no_subgroups,
    subgroup_range,
    requested_feature,
    requested_range,
    storage_limit,
    binding_limit,
    uniform_count,
    uniform_size,
    workgroup_size,
    workgroup_invocations,
    workgroups_limit,
    cold_raw_view,
    cold_mode,
    cold_buffer,
    rejection,
    rejection_both,
};

// Protect actual graph dispatch and binding ownership. A future kernel or
// matcher change must preserve the ordinary fallback for each rejected case.
static void graph(uint32_t subgroup, int64_t k, int64_t m, Guard guard, bool requested = false) {
    transfer_cases::Backend gpu(transfer_cases::BackendKind::webgpu);
    auto & global = *gpu.context->global_ctx;
    auto & caps = global.capabilities;
    const bool packed_dot = guard == Guard::packed_dot;
    if (packed_dot) {
        caps.supports_dot_product = true;
        global.vendor = "intel";
    }
    caps.supports_subgroups = guard != Guard::no_subgroups;
    caps.min_subgroup_size = requested ? 8 : guard == Guard::subgroup_range ? 16 : subgroup;
    caps.max_subgroup_size = requested ? 32 : subgroup;
    caps.subgroup_size_control_enabled = requested && guard != Guard::requested_feature;
    if (guard == Guard::requested_range) {
        caps.min_subgroup_size = 32;
        caps.max_subgroup_size = 64;
    }
    caps.limits.maxStorageBufferBindingSize = 128 * 1024 * 1024;
    caps.limits.maxBufferSize = 256 * 1024 * 1024;
    gpu.type.iface.get_max_size = [](ggml_backend_buffer_type_t) -> size_t { return 128 * 1024 * 1024; };
    auto & arena = gpu.context->param_arena;
    arena.cleanup();
    arena.init(global.device, WEBGPU_PARAMS_BUF_SIZE_BYTES, 74, 256);

    if (guard == Guard::shape) {
        k = 2048;
    }
    const int64_t tokens = guard == Guard::prefill ? 2 : 1;
    const bool live_alias = guard == Guard::live_alias;
    const auto weights = transfer_cases::Memory(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
    const auto memory = transfer_cases::Memory(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
    BICORE_CHECK(weights && memory);
    auto * w = weights.get();
    auto * ctx = memory.get();

    // Two Q4_0 blocks supply a word-aligned, nonzero quantized binding prefix.
    auto * gate_storage = ggml_new_tensor_2d(w, GGML_TYPE_Q4_0, k, m + 1);
    const size_t prefix = 2 * ggml_type_size(GGML_TYPE_Q4_0);
    auto * gate_weight = ggml_view_2d(w, gate_storage, k, m, gate_storage->nb[1], prefix);
    const ggml_type up_type = guard == Guard::mixed_quant ? GGML_TYPE_Q4_K : GGML_TYPE_Q4_0;
    auto * up_weight = ggml_new_tensor_2d(w, up_type, k, m);
    auto * input = ggml_new_tensor_2d(w, GGML_TYPE_F32, live_alias ? m : k, tokens);
    const auto persistent =
        transfer_cases::Buffer(ggml_backend_alloc_ctx_tensors_from_buft(w, &gpu.type), ggml_backend_buffer_free);
    BICORE_CHECK(persistent);
    const auto allocator =
        std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)>(ggml_gallocr_new(&gpu.type), ggml_gallocr_free);

    auto * producer = ggml_dup(ctx, input);
    auto * x = live_alias ? ggml_view_1d(ctx, producer, k, 0) : producer;
    auto * gate = ggml_mul_mat(ctx, gate_weight, x);
    auto * up = ggml_mul_mat(ctx, up_weight, x);
    auto * y = guard == Guard::activation ? ggml_swiglu_split(ctx, gate, up) : ggml_geglu_split(ctx, gate, up);
    if (guard == Guard::glu_swapped) {
        y->op_params[1] = 1;
    }
    if (guard == Guard::glu_params) {
        y->op_params[2] = 1;
    }
    if (guard == Guard::gate_precision) {
        BICORE_CHECK(ggml_prec_set_acc(gate, GGML_PREC_F32));
    }
    if (guard == Guard::up_precision) {
        BICORE_CHECK(ggml_prec_set_src(up, GGML_PREC_F32, 1));
    }
    auto * consumer = ggml_dup(ctx, y);
    ggml_set_output(consumer);
    if (guard == Guard::observed_projection) {
        ggml_set_output(gate);
    }
    auto * layout = ggml_new_graph_custom(ctx, 32, false);
    const bool reversed = guard == Guard::reversed;
    if (reversed) {
        // Materialize the gate weight's view before the up projection so the
        // reversed projections stay adjacent, as required by this fusion.
        ggml_build_forward_expand(layout, gate_weight);
        ggml_build_forward_expand(layout, up);
    }
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

    const auto first = std::find(layout->nodes, layout->nodes + stop, reversed ? up : gate);
    BICORE_CHECK(first != layout->nodes + stop);
    auto slice = ggml_graph_view(layout, int(first - layout->nodes), stop);
    BICORE_CHECK(slice.n_nodes == 4 && slice.nodes[0] == (reversed ? up : gate) &&
                 slice.nodes[1] == (reversed ? gate : up) && slice.nodes[2] == y);
    const auto append_late_reader = [&](ggml_tensor * root) {
        append_reader(root);
        const int indices[] = {0, 1, 2};
        const int outputs[] = {2};
        const ggml_op ops[] = {GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU};
        BICORE_CHECK(ggml_graph_n_nodes(layout) > stop && slice.n_nodes == 4 &&
                     ggml_can_fuse_subgraph_ext(&slice, indices, 3, ops, outputs, 1) &&
                     webgpu_fusion_view_sources(slice).contains(root));
    };
    const auto waits_before_graph_completions = [] {
        // A public graph run synchronizes its queue (and profile readback).
        // Those waits are distinct from optional-pipeline creation waits.
        return waits - commands.queue_waits - commands.maps;
    };

    const auto run = [&](bool fused, bool requested_variant) {
        commands = {};
        BICORE_CHECK(ggml_backend_graph_compute_async(&gpu.backend, &slice) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(&gpu.backend);
        const size_t expected = fused ? 2 : (packed_dot ? 6 : 4);
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
            BICORE_CHECK(!first.pipeline.Get()->name.starts_with("q4_0_geglu_") &&
                         pipeline_at(packed_dot ? 4 : 2).pipeline.Get()->name ==
                             (guard == Guard::activation ? "glu_swiglu_f32_split" : "glu_geglu_f32_split"));
            if (requested && (guard == Guard::rejection || guard == Guard::rejection_both)) {
                auto * projection = slice.nodes[0];
                auto ordinary =
                    ggml_webgpu_mat_context(gpu.context, projection->src[0], projection->src[1], projection);
                if (guard == Guard::rejection_both) {
                    ordinary.subgroup_size_control_enabled = false;
                }
                const auto selected = gpu.context->shader_lib->get_mul_mat_vec_pipeline(ordinary, false, false);
                const auto * decisions =
                    static_cast<const ggml_webgpu_mul_mat_vec_shader_decisions *>(selected.context.get());
                BICORE_CHECK(first.pipeline.Get() == selected.pipeline.Get() && decisions &&
                             decisions->requested_subgroup_size == (guard == Guard::rejection ? 16 : 0) &&
                             decisions->logical_subgroup_size == (guard == Guard::rejection ? 16 : 0));
            }
            return;
        }
        const auto & dispatch = commands.dispatches.front();
        const uint32_t rows_per_workgroup = 128 / subgroup * 4;
        BICORE_CHECK(first.pipeline.Get()->name ==
                         "q4_0_geglu_r4_sg" + std::to_string(subgroup) + (requested_variant ? "_requested16" : "") &&
                     dispatch.x == (m + rows_per_workgroup - 1) / rows_per_workgroup && dispatch.y == 1 &&
                     dispatch.z == 1);
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
        std::array<uint32_t, 8> params;
        std::memcpy(params.data(), uniform.buffer.Get()->bytes.data() + uniform.offset, sizeof(params));
        const std::array<uint32_t, 8> expected_params = {offsets[0],  offsets[1],  offsets[2],       offsets[3],
                                                         uint32_t(k), uint32_t(m), uint32_t(k / 32), 0};
        BICORE_CHECK(params == expected_params && offsets[0] == 2);
    };

    const auto before = pushes;
    const auto original_caps = caps;
    bool insufficient_limits = true;
    switch (guard) {
    case Guard::storage_limit:
        caps.limits.maxStorageBuffersPerShaderStage = 3;
        break;
    case Guard::binding_limit:
        caps.limits.maxBindingsPerBindGroup = 4;
        break;
    case Guard::uniform_count:
        caps.limits.maxUniformBuffersPerShaderStage = 0;
        break;
    case Guard::uniform_size:
        caps.limits.maxUniformBufferBindingSize = arena.slot_size - 1;
        break;
    case Guard::workgroup_size:
        caps.limits.maxComputeWorkgroupSizeX = 64;
        break;
    case Guard::workgroup_invocations:
        caps.limits.maxComputeInvocationsPerWorkgroup = 64;
        break;
    case Guard::workgroups_limit:
        caps.limits.maxComputeWorkgroupsPerDimension = 1;
        break;
    default:
        insufficient_limits = false;
        break;
    }
    if (insufficient_limits) {
        // Synthetic limits may also make ordinary shaders invalid. Check the
        // optional encoder's refusal before any factory or recording, then
        // restore the real fixture limits and verify no cache was poisoned.
        webgpu_fusion_view_sources views(slice);
        const auto created_before = created;
        int consumed = 1;
        commands = {};
        BICORE_CHECK(!webgpu_q4_geglu::encode(gpu.context, slice, 0, views, consumed));
        BICORE_CHECK(commands.dispatches.empty() && consumed == 1 && pushes == before && created == created_before);
        caps = original_caps;
        run(true, requested);
        BICORE_CHECK(pushes == before + 3);
    } else if (guard == Guard::rejection || guard == Guard::rejection_both) {
        webgpu_fusion_view_sources views(slice);
        BICORE_CHECK(!views.contains(x));
        on_wait_once = [&] {
            append_late_reader(x);
            if (guard == Guard::rejection_both) {
                inject = WGPUErrorType_Validation;
            }
        };
        inject = WGPUErrorType_Validation;
        int consumed = 1;
        commands = {};
        BICORE_CHECK(!webgpu_q4_geglu::encode(gpu.context, slice, 0, views, consumed));
        BICORE_CHECK(commands.dispatches.empty() && consumed == 1 && views.contains(x));
        BICORE_CHECK(!on_wait_once && pushes == before + 3 &&
                     inject == (guard == Guard::rejection_both ? WGPUErrorType_Validation : WGPUErrorType_NoError));
        run(false, false);
        // Rejected fusion still lets the ordinary projection choose its own
        // requested-width pipeline and fallback; the two caches are separate.
        BICORE_CHECK(pushes == before + (requested ? 6 : 3));
        const auto warm = std::tuple{created, pushes, waits};
        run(false, false);
        BICORE_CHECK((std::tuple{created, pushes, waits_before_graph_completions()} == warm));
        if (requested) {
            caps.min_subgroup_size = caps.max_subgroup_size = subgroup;
            caps.subgroup_size_control_enabled = false;
            run(true, false);
            BICORE_CHECK(pushes == before + 9);
            caps = original_caps;
            run(false, false);
            BICORE_CHECK(pushes == before + 9);
        }
    } else if (guard == Guard::cold_raw_view) {
        on_wait_once = [&] { append_late_reader(gate); };
        run(false, false);
        BICORE_CHECK(!on_wait_once && pushes == before + (requested ? 6 : 3));
        run(false, false);
        BICORE_CHECK(pushes == before + (requested ? 6 : 3));
    } else if (guard == Guard::cold_mode) {
        BICORE_CHECK(requested && subgroup == 16);
        on_wait_once = [&] {
            caps.min_subgroup_size = caps.max_subgroup_size = subgroup;
            caps.subgroup_size_control_enabled = false;
        };
        webgpu_fusion_view_sources views(slice);
        int consumed = 1;
        commands = {};
        BICORE_CHECK(!webgpu_q4_geglu::encode(gpu.context, slice, 0, views, consumed));
        BICORE_CHECK(!on_wait_once && commands.dispatches.empty() && consumed == 1 && pushes == before + 3);
        run(true, false);
        BICORE_CHECK(pushes == before + 6);
        caps = original_caps;
        const auto warm = std::tuple{created, pushes, waits};
        run(true, true);
        BICORE_CHECK((std::tuple{created, pushes, waits_before_graph_completions()} == warm));
    } else if (guard == Guard::cold_buffer) {
        const auto buffer = ggml_webgpu_tensor_buf(up_weight);
        const auto usage = buffer.Get()->usage;
        on_wait_once = [&] { buffer.Get()->usage = WGPUBufferUsage_CopyDst; };
        webgpu_fusion_view_sources views(slice);
        int consumed = 1;
        commands = {};
        BICORE_CHECK(!webgpu_q4_geglu::encode(gpu.context, slice, 0, views, consumed));
        BICORE_CHECK(!on_wait_once && commands.dispatches.empty() && consumed == 1 && pushes == before + 3);
        buffer.Get()->usage = usage;
        const auto factory_warm = std::pair{pushes, waits};
        run(true, requested);
        BICORE_CHECK((std::pair{pushes, waits_before_graph_completions()} == factory_warm));
        const auto warm = std::tuple{created, pushes, waits};
        run(true, requested);
        BICORE_CHECK((std::tuple{created, pushes, waits_before_graph_completions()} == warm));
    } else {
        const bool fused = guard == Guard::ordinary || guard == Guard::reversed || guard == Guard::final_view;
        run(fused, requested);
        BICORE_CHECK(pushes == before + (fused ? 3 : 0));
        if (guard == Guard::ordinary) {
            const auto warm = std::tuple{created, pushes, waits};
            run(true, requested);
            BICORE_CHECK((std::tuple{created, pushes, waits_before_graph_completions()} == warm));
        }
    }
    BICORE_CHECK(futures.empty());
}

static void lifecycle() {
    for (const bool requested : {false, true}) {
        auto ctx = recording_context();
        auto & library = *ctx->shader_lib;
        const ggml_webgpu_q4_geglu_pipeline_key key{16, requested};
        const ggml_webgpu_q4_geglu_pipeline_key other{16, !requested};
        const auto before = std::tuple{created, pushes, pops, waits};
        on_wait_once = [&] {
            const auto issuing = std::tuple{created, pushes, pops, waits};
            bool made = true;
            BICORE_CHECK(!library.get_q4_geglu_pipeline(key, made).pipeline && !made);
            BICORE_CHECK((std::tuple{created, pushes, pops, waits} == issuing));
        };
        bool made = false;
        const auto ready = library.get_q4_geglu_pipeline(key, made);
        BICORE_CHECK(!on_wait_once && made && ready.pipeline);
        BICORE_CHECK(
            (std::tuple{created, pushes, pops, waits} == std::tuple{std::get<0>(before) + 1, std::get<1>(before) + 3,
                                                                    std::get<2>(before) + 3, std::get<3>(before) + 3}));
        inject = WGPUErrorType_Validation;
        const auto declined = library.get_q4_geglu_pipeline(other, made);
        BICORE_CHECK(made && !declined.pipeline && inject == WGPUErrorType_NoError);
        const auto warm = std::tuple{created, pushes, pops, waits};
        BICORE_CHECK(library.get_q4_geglu_pipeline(key, made).pipeline.Get() == ready.pipeline.Get() && !made);
        BICORE_CHECK(!library.get_q4_geglu_pipeline(other, made).pipeline && !made);
        BICORE_CHECK((std::tuple{created, pushes, pops, waits} == warm));
        // Both a ready hit and a rejected hit must still reject a lost device.
        ctx->global_ctx->optional_device->control.lost.store(true);
        for (const auto checked : {key, other}) {
            bool aborted = false;
            try {
                (void)library.get_q4_geglu_pipeline(checked, made);
            } catch (const std::runtime_error &) {
                aborted = true;
            }
            BICORE_CHECK(aborted);
        }
        BICORE_CHECK((std::tuple{created, pushes, pops, waits} == warm) && futures.empty());

        auto cold = recording_context();
        on_wait_once = [&] { cold->global_ctx->optional_device->control.lost.store(true); };
        bool aborted = false;
        try {
            (void)cold->shader_lib->get_q4_geglu_pipeline(key, made);
        } catch (const std::runtime_error &) {
            aborted = true;
        }
        BICORE_CHECK(aborted && !on_wait_once && cold->global_ctx->optional_device->control.terminal &&
                     futures.empty());
        BICORE_CHECK(
            (std::tuple{created, pushes, pops, waits} ==
             std::tuple{std::get<0>(warm) + 1, std::get<1>(warm) + 3, std::get<2>(warm) + 3, std::get<3>(warm) + 3}));
    }
}

static void all() {
    lifecycle();
    for (uint32_t subgroup : {16u, 32u}) {
        graph(subgroup, 1536, 6144, Guard::ordinary);
        graph(subgroup, 1536, 12288, Guard::ordinary);
        graph(subgroup, 2560, 10240, Guard::ordinary);
    }
    graph(16, 1536, 6144, Guard::ordinary, true);
    graph(16, 1536, 12288, Guard::ordinary, true);
    graph(16, 2560, 10240, Guard::ordinary, true);
    for (const auto guard :
         {Guard::reversed, Guard::final_view, Guard::no_subgroups, Guard::requested_feature, Guard::requested_range,
          Guard::cold_raw_view, Guard::cold_mode, Guard::cold_buffer, Guard::rejection, Guard::rejection_both}) {
        graph(16, 1536, 6144, guard, true);
    }
    for (const auto guard : {Guard::reversed,         Guard::observed_projection,
                             Guard::raw_view,         Guard::final_view,
                             Guard::live_alias,       Guard::shape,
                             Guard::mixed_quant,      Guard::prefill,
                             Guard::activation,       Guard::glu_swapped,
                             Guard::glu_params,       Guard::gate_precision,
                             Guard::up_precision,     Guard::packed_dot,
                             Guard::no_subgroups,     Guard::subgroup_range,
                             Guard::storage_limit,    Guard::binding_limit,
                             Guard::uniform_count,    Guard::uniform_size,
                             Guard::workgroup_size,   Guard::workgroup_invocations,
                             Guard::workgroups_limit, Guard::cold_raw_view,
                             Guard::rejection}) {
        graph(32, 1536, 6144, guard);
    }
}

} // namespace q4_geglu_cases
