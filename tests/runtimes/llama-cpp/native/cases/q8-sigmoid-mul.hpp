#pragma once

namespace q8_sigmoid_mul_cases {

enum class Use {
    separate,
    in_place,
    raw_matvec_reader,
    raw_sigmoid_reader,
    observed_matvec,
    split,
    varying_subgroups,
    wrong_shape,
    explicit_precision,
    partial_alias,
    input_stride
};

struct Projection {
    ggml_type type;
    int64_t input;
    int64_t output;
};

static void graph(const Projection & projection, uint32_t subgroup, Use use) {
    transfer_cases::Backend gpu(transfer_cases::BackendKind::webgpu);
    auto & global = *gpu.context->global_ctx;
    auto & caps = global.capabilities;
    caps.supports_subgroups = true;
    caps.min_subgroup_size = use == Use::varying_subgroups ? 16 : subgroup;
    caps.max_subgroup_size = subgroup;
    caps.limits.maxStorageBufferBindingSize = 64 * 1024 * 1024;
    caps.limits.maxBufferSize = 64 * 1024 * 1024;
    gpu.type.iface.get_max_size = [](ggml_backend_buffer_type_t) -> size_t { return 64 * 1024 * 1024; };
    gpu.context->param_arena.cleanup();
    gpu.context->param_arena.init(global.device, WEBGPU_PARAMS_BUF_SIZE_BYTES, 74, 256);

    const auto storage = transfer_cases::Memory(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
    const auto memory = transfer_cases::Memory(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
    BICORE_CHECK(storage && memory);
    auto * persistent = storage.get();
    auto * ctx = memory.get();
    const int64_t k = projection.input;
    const int64_t m = projection.output + (use == Use::wrong_shape ? 128 : 0);
    auto * weight_root = ggml_new_tensor_2d(persistent, projection.type, k, m + 1);
    // Preserve nonzero quantized rebasing for both 34-byte and 144-byte blocks.
    const size_t prefix = 2 * ggml_type_size(projection.type);
    auto * weight = ggml_view_2d(persistent, weight_root, k, m, weight_root->nb[1], prefix);
    auto * input = ggml_new_tensor_1d(persistent, GGML_TYPE_F32, k);
    auto * attention_input = ggml_new_tensor_2d(persistent, GGML_TYPE_F32, 128, m / 128);
    const auto allocated = transfer_cases::Buffer(ggml_backend_alloc_ctx_tensors_from_buft(persistent, &gpu.type),
                                                  ggml_backend_buffer_free);
    BICORE_CHECK(allocated);

    auto * attention = ggml_dup(ctx, attention_input);
    auto * value = ggml_reshape_1d(ctx, attention, m);
    auto * matvec = ggml_mul_mat(ctx, weight, input);
    if (use == Use::explicit_precision) {
        BICORE_CHECK(ggml_prec_set_acc(matvec, GGML_PREC_F32));
    }
    auto * sigmoid = ggml_sigmoid(ctx, matvec);
    auto * output = ggml_mul(ctx, value, sigmoid);
    auto * layout = ggml_new_graph_custom(ctx, 32, false);
    ggml_set_output(output);
    if (use == Use::observed_matvec) {
        ggml_set_output(matvec);
    }
    ggml_build_forward_expand(layout, output);
    const int stop = ggml_graph_n_nodes(layout);
    if (use == Use::raw_matvec_reader || use == Use::raw_sigmoid_reader) {
        auto * root = use == Use::raw_matvec_reader ? matvec : sigmoid;
        auto * view = ggml_view_tensor(ctx, root);
        BICORE_CHECK(view->view_src == root && !view->src[0]);
        auto * reader = ggml_dup(ctx, view);
        ggml_set_output(reader);
        ggml_build_forward_expand(layout, reader);
    }
    const auto allocator =
        std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)>(ggml_gallocr_new(&gpu.type), ggml_gallocr_free);
    BICORE_CHECK(allocator && ggml_gallocr_alloc_graph(allocator.get(), layout));
    BICORE_CHECK(value->view_src == attention && !output->view_src);
    BICORE_CHECK(ggml_webgpu_tensor_buf(output).Get() == ggml_webgpu_tensor_buf(value).Get() &&
                 output->data != value->data);
    if (use == Use::in_place) {
        // Exercise the same physical reuse that GGML can assign to MUL.
        output->data = value->data;
    } else if (use == Use::partial_alias) {
        output->data = static_cast<char *>(value->data) + sizeof(float);
    } else if (use == Use::input_stride) {
        input->nb[1] += sizeof(float);
    }

    const auto first_matvec = std::find(layout->nodes, layout->nodes + stop, matvec);
    BICORE_CHECK(first_matvec != layout->nodes + stop);
    const int start = int(first_matvec - layout->nodes);
    auto slice = ggml_graph_view(layout, start, stop);
    BICORE_CHECK(slice.n_nodes == 3 && slice.nodes[0] == matvec && slice.nodes[1] == sigmoid &&
                 slice.nodes[2] == output);
    const auto run = [&](ggml_cgraph & graph) {
        BICORE_CHECK(ggml_backend_graph_compute_async(&gpu.backend, &graph) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(&gpu.backend);
    };
    commands = {};
    if (use == Use::split) {
        auto matvec_only = ggml_graph_view(layout, start, start + 1);
        auto remainder = ggml_graph_view(layout, start + 1, stop);
        run(matvec_only);
        run(remainder);
    } else {
        run(slice);
    }
    const bool fused = use == Use::separate || use == Use::in_place;
    const size_t expected = fused ? 1 : 3;
    BICORE_CHECK(commands.dispatches.size() == expected && commands.executed == expected);
#ifdef GGML_WEBGPU_GPU_PROFILE
    BICORE_CHECK(commands.timestamp_pairs.size() == expected);
#endif
    if (!fused) {
        return;
    }
    const auto & dispatch = commands.dispatches.front();
    const auto & slots = gpu.context->bind_group_cache.slots;
    const auto first = std::find_if(slots.begin(), slots.end(), [&](const auto & entry) {
        return entry.group && entry.group.Get()->id == dispatch.group;
    });
    BICORE_CHECK(first != slots.end());
    const bool in_place = use == Use::in_place;
    BICORE_CHECK(first->pipeline.Get()->name.ends_with(in_place ? "_sigmoid_mul_in_place" : "_sigmoid_mul"));
    BICORE_CHECK(dispatch.x == uint32_t(m) / (4 * 128 / subgroup) && dispatch.y == 1 && dispatch.z == 1);
    const ggml_tensor * tensors[] = {weight, input, output, value};
    const size_t storage_count = in_place ? 3 : 4;
    BICORE_CHECK(first->bindings.size() == storage_count + 1);
    for (size_t i = 0; i < storage_count; ++i) {
        const auto & binding = first->bindings[i];
        BICORE_CHECK(binding.binding_index == i && binding.buffer.Get() == ggml_webgpu_tensor_buf(tensors[i]).Get() &&
                     binding.offset == ggml_webgpu_tensor_align_offset(gpu.context, tensors[i]) &&
                     binding.size == ggml_webgpu_tensor_binding_size(gpu.context, tensors[i]));
    }
    const auto & uniform = first->bindings.back();
    BICORE_CHECK(uniform.binding_index == storage_count && uniform.size == WEBGPU_PARAMS_BUF_SIZE_BYTES);
    std::array<uint32_t, 17> params{};
    std::memcpy(params.data(), uniform.buffer.Get()->bytes.data() + uniform.offset, sizeof(params));
    const uint32_t ku = uint32_t(k), mu = uint32_t(m);
    const uint32_t stride = ku / ggml_blck_size(projection.type);
    const std::array<uint32_t, 17> expected_params = {2,  0,           0,  mu, 1, ku, stride, ku, stride * mu,
                                                      ku, stride * mu, ku, 1,  1, 1,  1,      0};
    BICORE_CHECK(params == expected_params);

    const auto pipelines_after_cold = created;
    const auto groups_after_cold = resources.groups_created;
    commands = {};
    run(slice);
    BICORE_CHECK(commands.dispatches.size() == 1 && created == pipelines_after_cold &&
                 resources.groups_created == groups_after_cold);
}

static void all() {
    for (const Projection projection :
         {Projection{GGML_TYPE_Q8_0, 2304, 3072}, Projection{GGML_TYPE_Q4_K, 6656, 4096},
          Projection{GGML_TYPE_Q8_0, 6656, 4096}}) {
        graph(projection, 16, Use::separate);
        graph(projection, 16, Use::in_place);
        for (const auto use : {Use::separate, Use::in_place, Use::raw_matvec_reader, Use::raw_sigmoid_reader,
                               Use::observed_matvec, Use::split, Use::varying_subgroups, Use::wrong_shape,
                               Use::explicit_precision, Use::partial_alias, Use::input_stride}) {
            graph(projection, 32, use);
        }
    }
}

} // namespace q8_sigmoid_mul_cases
