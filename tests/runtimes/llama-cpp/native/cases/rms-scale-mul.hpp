#pragma once

namespace rms_scale_mul_cases {

enum class Guard { ordinary, rms_output, scale_output, rms_reader, scale_reader, raw_rms, raw_scale, raw_output };

struct Fixture {
    transfer_cases::Backend gpu{transfer_cases::BackendKind::webgpu};
    transfer_cases::Memory memory{ggml_init({1024 * 1024, nullptr, true}), ggml_free};
    transfer_cases::Buffer buffer{nullptr, ggml_backend_buffer_free};
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator{nullptr, ggml_gallocr_free};
    ggml_tensor * source = nullptr;
    ggml_tensor * weight = nullptr;
    ggml_tensor * rms = nullptr;
    ggml_tensor * scale = nullptr;
    ggml_tensor * output = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_cgraph slice{};

    Fixture(int64_t width, int64_t rows, bool natural = false, size_t source_prefix = 0, size_t weight_prefix = 0,
            Guard guard = Guard::ordinary, int weight_rows = 1, bool scalar_weight = false, bool reversed = false) {
        BICORE_CHECK(memory);
        auto * ctx = memory.get();
        auto & limits = gpu.context->global_ctx->capabilities.limits;
        limits.maxStorageBufferBindingSize = limits.maxBufferSize = 8 * 1024 * 1024;
        limits.maxComputeWorkgroupsPerDimension = 4;
        gpu.type.iface.get_max_size = [](ggml_backend_buffer_type_t) -> size_t { return 8 * 1024 * 1024; };
        gpu.context->param_arena.cleanup();
        gpu.context->param_arena.init(gpu.context->global_ctx->device, WEBGPU_PARAMS_BUF_SIZE_BYTES, 74, 256);
        const auto input = [&](int64_t columns, int64_t count, size_t prefix) {
            auto * root = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, columns, count + int64_t(prefix != 0));
            ggml_set_input(root);
            if (prefix)
                return ggml_view_2d(ctx, root, columns, count, size_t(columns) * sizeof(float), prefix);
            return root;
        };
        auto * original = input(width, rows, source_prefix);
        source = natural ? ggml_dup(ctx, original) : original;
        weight = input(scalar_weight ? 1 : width, weight_rows, weight_prefix);
        rms = ggml_rms_norm(ctx, source, 1e-6f);
        scale = ggml_scale_bias(ctx, rms, 1.0f / std::sqrt(float(width)), 0.125f);
        output = reversed ? ggml_mul(ctx, weight, scale) : ggml_mul(ctx, scale, weight);
        ggml_set_output(output);
        if (guard == Guard::rms_output)
            ggml_set_output(rms);
        if (guard == Guard::scale_output)
            ggml_set_output(scale);
        graph = ggml_new_graph_custom(ctx, 128, false);
        // Input views precede the adjacent normalization chain in a model graph.
        ggml_build_forward_expand(graph, weight);
        ggml_build_forward_expand(graph, output);
        const auto stop = graph->n_nodes;
        if (guard == Guard::rms_reader || guard == Guard::scale_reader || guard == Guard::raw_rms ||
            guard == Guard::raw_scale || guard == Guard::raw_output) {
            auto * observed = guard == Guard::rms_reader || guard == Guard::raw_rms ? rms
                              : guard == Guard::raw_output                          ? output
                                                                                    : scale;
            append_reader(observed, guard == Guard::raw_rms || guard == Guard::raw_scale || guard == Guard::raw_output);
        }
        // Leave a checked allocation tail for shifted-output admission probes.
        (void)ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
        if (natural) {
            allocator.reset(ggml_gallocr_new(&gpu.type));
            BICORE_CHECK(allocator && ggml_gallocr_alloc_graph(allocator.get(), graph));
            BICORE_CHECK(ggml_webgpu_tensor_equal(source, output));
        } else {
            buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx, &gpu.type));
            BICORE_CHECK(buffer);
        }
        const auto first = std::find(graph->nodes, graph->nodes + stop, rms);
        BICORE_CHECK(first != graph->nodes + stop);
        slice = ggml_graph_view(graph, int(first - graph->nodes), stop);
        BICORE_CHECK(slice.n_nodes == 3 && slice.nodes[1] == scale && slice.nodes[2] == output);
    }

    void append_reader(ggml_tensor * observed, bool raw) {
        if (raw) {
            observed = ggml_view_tensor(memory.get(), observed);
            BICORE_CHECK(observed->view_src && !observed->src[0]);
        }
        auto * reader = ggml_dup(memory.get(), observed);
        ggml_set_output(reader);
        ggml_build_forward_expand(graph, reader);
    }

    void compute(size_t expected) {
        commands = {};
        readback = {};
        BICORE_CHECK(ggml_backend_graph_compute_async(&gpu.backend, &slice) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(&gpu.backend);
        BICORE_CHECK(commands.dispatches.size() == expected && commands.executed == expected);
#ifdef GGML_WEBGPU_GPU_PROFILE
        BICORE_CHECK(commands.timestamp_pairs.size() == expected);
#endif
    }

    const webgpu_bind_group_cache::slot & packet() {
        const auto & dispatch = commands.dispatches.front();
        const auto & slots = gpu.context->bind_group_cache.slots;
        const auto found = std::find_if(slots.begin(), slots.end(), [&](const auto & item) {
            return item.group && item.group.Get()->id == dispatch.group;
        });
        BICORE_CHECK(found != slots.end() && found->pipeline.Get()->id == dispatch.pipeline);
        return *found;
    }

    void reject() {
        webgpu_fusion_view_sources views(slice);
        const auto before = std::tuple{created, pushes, gpu.allocations};
        int consumed = 1;
        commands = {};
        BICORE_CHECK(!webgpu_rms_scale_mul::encode(gpu.context, slice, 0, views, consumed));
        BICORE_CHECK(consumed == 1 && commands.dispatches.empty() &&
                     (std::tuple{created, pushes, gpu.allocations} == before));
    }
};

static void admitted(int64_t width, int64_t rows, bool natural, size_t source_prefix, size_t weight_prefix) {
    Fixture f(width, rows, natural, source_prefix, weight_prefix);
    const bool vec4 = (rows >= 8 || width >= 4096) && width % 4 == 0 && source_prefix % 16 == 0;
    const auto made_before = created;
    f.compute(1);
    const auto & packet = f.packet();
    const auto name = std::string("rms_norm_scale_mul_f32") + (vec4 ? "_vec4" : "") + (natural ? "_inplace" : "");
    BICORE_CHECK(packet.pipeline.Get()->name == name && created == made_before + 1);
    const auto dispatch = commands.dispatches.front();
    BICORE_CHECK(uint64_t(dispatch.x) * dispatch.y >= uint64_t(rows) && dispatch.x <= 4 && dispatch.y <= 4 &&
                 dispatch.z == 1);
    const size_t storage_count = natural ? 2 : 3;
    const std::array<const ggml_tensor *, 3> tensors =
        natural ? std::array<const ggml_tensor *, 3>{f.source, f.weight, nullptr}
                : std::array<const ggml_tensor *, 3>{f.source, f.output, f.weight};
    BICORE_CHECK(packet.bindings.size() == storage_count + 1);
    for (size_t i = 0; i < storage_count; ++i) {
        const auto & binding = packet.bindings[i];
        BICORE_CHECK(binding.binding_index == i && binding.buffer.Get() == ggml_webgpu_tensor_buf(tensors[i]).Get() &&
                     binding.offset == ggml_webgpu_tensor_align_offset(f.gpu.context, tensors[i]) &&
                     binding.size == ggml_webgpu_tensor_binding_size(f.gpu.context, tensors[i]));
    }
    const auto & uniform = packet.bindings.back();
    std::array<uint32_t, 16> params{};
    const auto read_params = [&] {
        std::memcpy(params.data(), uniform.buffer.Get()->bytes.data() + uniform.offset, sizeof(params));
    };
    read_params();
    BICORE_CHECK(params[0] == ggml_webgpu_tensor_misalignment(f.gpu.context, f.source) / 4 &&
                 params[1] == ggml_webgpu_tensor_misalignment(f.gpu.context, f.output) / 4 &&
                 params[15] == ggml_webgpu_tensor_misalignment(f.gpu.context, f.weight) / 4 &&
                 params[8] == uint32_t(width) && params[9] == uint32_t(rows) && params[10] == 1 && params[11] == 1);
    for (size_t axis = 1; axis < 4; ++axis)
        BICORE_CHECK(params[1 + axis] == f.source->nb[axis] / 4 && params[4 + axis] == f.output->nb[axis] / 4);
    const auto warm = std::tuple{created, pushes, resources.groups_created, f.gpu.allocations};
    const float epsilon = 0.01f;
    const std::array affine{-0.25f, -0.75f};
    ggml_set_op_params(f.rms, &epsilon, sizeof(epsilon));
    ggml_set_op_params(f.scale, affine.data(), sizeof(affine));
    const auto size_queries = resources.buffer_size_queries, usage_queries = resources.buffer_usage_queries;
    f.compute(1);
    BICORE_CHECK(commands.dispatches.front().pipeline == dispatch.pipeline &&
                 (std::tuple{created, pushes, resources.groups_created, f.gpu.allocations} == warm));
    // All four checked tensors share one graph arena; each physical buffer is
    // queried once per warm plan, without a persistent metadata cache.
    BICORE_CHECK(resources.buffer_size_queries == size_queries + 1 &&
                 resources.buffer_usage_queries == usage_queries + 1);
    read_params();
    BICORE_CHECK(params[12] == ggml_webgpu_u32_from_f32(epsilon) && params[13] == ggml_webgpu_u32_from_f32(affine[0]) &&
                 params[14] == ggml_webgpu_u32_from_f32(affine[1]));
    bool made = true;
    const auto cached = f.gpu.context->shader_lib->get_rms_scale_mul_pipeline(natural, vec4, made);
    BICORE_CHECK(!made && cached.pipeline.Get()->id == dispatch.pipeline);
    const auto ordinary = vec4 ? f.gpu.context->shader_lib->get_rms_vec4_pipeline(natural, true)
                               : f.gpu.context->shader_lib->get_row_norm_scale_pipeline(GGML_OP_RMS_NORM, natural);
    BICORE_CHECK(ordinary.pipeline.Get() != cached.pipeline.Get());
}

static void guards() {
    for (auto guard : {Guard::rms_output, Guard::scale_output, Guard::rms_reader, Guard::scale_reader, Guard::raw_rms,
                       Guard::raw_scale, Guard::raw_output}) {
        Fixture f(2816, 1, false, 0, 0, guard);
        const bool rms_visible = guard == Guard::rms_output || guard == Guard::rms_reader || guard == Guard::raw_rms;
        f.compute(guard == Guard::raw_output ? 1 : rms_visible ? 3 : 2);
    }
    Fixture row_weights(2816, 8, false, 0, 0, Guard::ordinary, 8);
    row_weights.compute(2);
    Fixture scalar_weight(2816, 1, false, 0, 0, Guard::ordinary, 1, true);
    scalar_weight.compute(2);
    Fixture reversed(2816, 1, false, 0, 0, Guard::ordinary, 1, false, true);
    reversed.compute(2);
    Fixture unaligned_weight(2816, 8, false, 0, 4);
    unaligned_weight.compute(2);
    BICORE_CHECK(unaligned_weight.packet().pipeline.Get()->name == "rms_norm_scale_f32_vec4");

    Fixture f(2816, 1);
    const auto saved_output = *f.output, saved_weight = *f.weight, saved_source = *f.source;
    f.output->buffer = f.source->buffer;
    f.output->data = reinterpret_cast<void *>(uintptr_t(f.source->data) + sizeof(float));
    f.reject();
    *f.output = saved_output;
    f.weight->buffer = f.scale->buffer;
    f.weight->data = f.scale->data;
    f.reject();
    *f.weight = saved_weight;
    f.source->nb[1] += sizeof(float);
    f.reject();
    *f.source = saved_source;
    f.weight->type = GGML_TYPE_F16;
    f.reject();
    *f.weight = saved_weight;
    f.weight->view_src = f.scale;
    f.weight->data = f.scale->data;
    f.reject();
    *f.weight = saved_weight;
    f.output->view_src = f.source;
    f.output->data = f.source->data;
    f.reject();
    *f.output = saved_output;
    auto & limits = f.gpu.context->global_ctx->capabilities.limits;
    const auto saved_limits = limits;
    for (int change = 0; change < 8; ++change) {
        switch (change) {
        case 0:
            limits.maxStorageBuffersPerShaderStage = 2;
            break;
        case 1:
            limits.maxBindingsPerBindGroup = 3;
            break;
        case 2:
            limits.maxComputeInvocationsPerWorkgroup = 127;
            break;
        case 3:
            limits.maxComputeWorkgroupSizeX = 127;
            break;
        case 4:
            limits.maxComputeWorkgroupStorageSize = 1023;
            break;
        case 5:
            limits.maxUniformBufferBindingSize = uint32_t(f.gpu.context->param_arena.slot_size - 1);
            break;
        case 6:
            limits.maxStorageBufferBindingSize = 2816 * sizeof(float) - 1;
            break;
        case 7:
            limits.maxComputeWorkgroupsPerDimension = 0;
            break;
        }
        f.reject();
        limits = saved_limits;
    }
    f.compute(1);
    // A warm pipeline must not retain an earlier wrapper or physical size.
    const auto physical = ggml_webgpu_tensor_buf(f.weight);
    const auto physical_size = physical.Get()->logical_size;
    physical.Get()->logical_size = 4;
    f.reject();
    physical.Get()->logical_size = physical_size;
    const auto wrapper_size = f.weight->buffer->size;
    f.weight->buffer->size = 4;
    f.reject();
    f.weight->buffer->size = wrapper_size;
    f.compute(1);
    Fixture prefix_overlap(7, 1);
    prefix_overlap.weight->data = reinterpret_cast<void *>(uintptr_t(prefix_overlap.source->data) + 7 * sizeof(float));
    prefix_overlap.reject(); // Payloads are disjoint, but their writable binding prefixes overlap.

    Fixture wrapper_alias(2816, 1);
    auto wrapper = *wrapper_alias.source->buffer;
    wrapper_alias.output->buffer = &wrapper;
    wrapper_alias.output->data = wrapper_alias.source->data;
    wrapper_alias.compute(1);
    BICORE_CHECK(wrapper_alias.packet().pipeline.Get()->name == "rms_norm_scale_mul_f32_inplace");
}

static void cold_revalidation() {
    for (int change = 0; change < 8; ++change) {
        Fixture f(2816, change == 2 ? 8 : 1);
        webgpu_fusion_view_sources views(f.slice);
        BICORE_CHECK(!views.contains(f.rms) && !views.contains(f.scale));
        const auto output = *f.output;
        const auto buffer = ggml_webgpu_tensor_buf(f.weight);
        const auto usage = buffer.Get()->usage;
        const auto physical_size = buffer.Get()->logical_size;
        on_wait_once = [&] {
            switch (change) {
            case 0:
                f.append_reader(f.rms, true);
                break;
            case 1:
                f.append_reader(f.scale, true);
                break;
            case 2:
                f.scale->data = reinterpret_cast<void *>(uintptr_t(f.scale->data) + sizeof(float));
                break;
            case 3:
                buffer.Get()->usage = WGPUBufferUsage_CopyDst;
                break;
            case 4:
                f.output->buffer = f.source->buffer;
                f.output->data = f.source->data;
                break;
            case 5:
                f.weight->ne[0] = 1;
                break;
            case 6:
                f.gpu.context->global_ctx->capabilities.limits.maxComputeWorkgroupSizeX = 127;
                break;
            case 7:
                buffer.Get()->logical_size = 4;
                break;
            }
        };
        int consumed = 1;
        commands = {};
        BICORE_CHECK(!webgpu_rms_scale_mul::encode(f.gpu.context, f.slice, 0, views, consumed));
        BICORE_CHECK(!on_wait_once && consumed == 1 && commands.dispatches.empty());
        if (change < 2)
            BICORE_CHECK(views.contains(change == 0 ? f.rms : f.scale));
        buffer.Get()->usage = usage;
        buffer.Get()->logical_size = physical_size;
        *f.output = output;
    }

    Fixture rejected(2816, 1);
    webgpu_fusion_view_sources views(rejected.slice);
    BICORE_CHECK(!views.contains(rejected.scale));
    inject = WGPUErrorType_Validation;
    on_wait_once = [&] { rejected.append_reader(rejected.scale, true); };
    commands = {};
    int consumed = 1;
    BICORE_CHECK(!webgpu_rms_scale_mul::encode(rejected.gpu.context, rejected.slice, 0, views, consumed));
    BICORE_CHECK(!on_wait_once && consumed == 1 && commands.dispatches.empty() && views.contains(rejected.scale));
    rejected.compute(2);
    const auto attempts = pushes;
    rejected.compute(2);
    BICORE_CHECK(pushes == attempts && futures.empty());
}

static void lifecycle() {
    auto ctx = recording_context();
    std::array<WGPUComputePipeline, 4> handles{};
    for (const bool vec4 : {false, true}) {
        for (const bool inplace : {false, true}) {
            on_wait_once = [&] {
                const auto issuing = std::tuple{created, pushes, pops, waits};
                bool nested_made = true;
                BICORE_CHECK(!ctx->shader_lib->get_rms_scale_mul_pipeline(inplace, vec4, nested_made).pipeline &&
                             !nested_made);
                BICORE_CHECK((std::tuple{created, pushes, pops, waits} == issuing));
            };
            bool made = true;
            const auto pipeline = ctx->shader_lib->get_rms_scale_mul_pipeline(inplace, vec4, made);
            BICORE_CHECK(!on_wait_once && made && pipeline.pipeline);
            const auto id = unsigned(inplace) + 2u * unsigned(vec4);
            handles[id] = pipeline.pipeline.Get();
            for (unsigned i = 0; i < id; ++i)
                BICORE_CHECK(handles[i] != handles[id]);
            const auto warm = std::tuple{created, pushes, pops, waits};
            BICORE_CHECK(ctx->shader_lib->get_rms_scale_mul_pipeline(inplace, vec4, made).pipeline.Get() ==
                             handles[id] &&
                         !made);
            BICORE_CHECK((std::tuple{created, pushes, pops, waits} == warm));
        }
    }
    auto rejected = recording_context();
    bool made = false;
    inject = WGPUErrorType_Validation;
    BICORE_CHECK(!rejected->shader_lib->get_rms_scale_mul_pipeline(false, false, made).pipeline && made);
    const auto attempts = pushes;
    BICORE_CHECK(!rejected->shader_lib->get_rms_scale_mul_pipeline(false, false, made).pipeline && !made &&
                 pushes == attempts);
    BICORE_CHECK(rejected->shader_lib->get_rms_scale_mul_pipeline(false, true, made).pipeline && made);
    for (const auto & device : {ctx, rejected}) {
        device->global_ctx->optional_device->control.lost.store(true);
        for (const bool vec4 : {false, true}) {
            bool aborted = false;
            try {
                (void)device->shader_lib->get_rms_scale_mul_pipeline(false, vec4, made);
            } catch (const std::runtime_error &) {
                aborted = true;
            }
            BICORE_CHECK(aborted);
        }
    }
    auto cold = recording_context();
    on_wait_once = [&] { cold->global_ctx->optional_device->control.lost.store(true); };
    bool aborted = false;
    try {
        (void)cold->shader_lib->get_rms_scale_mul_pipeline(false, false, made);
    } catch (const std::runtime_error &) {
        aborted = true;
    }
    BICORE_CHECK(aborted && !on_wait_once && cold->global_ctx->optional_device->control.terminal && futures.empty());
}

static void higher_dimensions() {
    Fixture f(2816, 8);
    for (auto * tensor : {f.source, f.rms, f.scale, f.output}) {
        tensor->ne[1] = tensor->ne[2] = tensor->ne[3] = 2;
        tensor->nb[2] = tensor->nb[1] * 2;
        tensor->nb[3] = tensor->nb[2] * 2;
    }
    f.compute(1);
    BICORE_CHECK(f.packet().pipeline.Get()->name == "rms_norm_scale_mul_f32_vec4");
    const auto & uniform = f.packet().bindings.back();
    std::array<uint32_t, 16> params{};
    std::memcpy(params.data(), uniform.buffer.Get()->bytes.data() + uniform.offset, sizeof(params));
    BICORE_CHECK(params[9] == 2 && params[10] == 2 && params[11] == 2 && params[3] == 2816 * 2 &&
                 params[4] == 2816 * 4 && params[6] == 2816 * 2 && params[7] == 2816 * 4);
}

static void all() {
    for (int64_t width : {1536, 2560, 2816}) {
        admitted(width, 1, true, 0, 0);
        admitted(width, 9, false, 0, 16);
    }
    for (int64_t width : {7, 127, 128, 129, 4096})
        admitted(width, 1, false, 0, 0);
    admitted(2816, 8, false, 4, 20);
    admitted(2816, 8, true, 0, 0);
    guards();
    cold_revalidation();
    lifecycle();
    higher_dimensions();
}

} // namespace rms_scale_mul_cases
