#pragma once

namespace rms_mul_add_cases {

enum class Output { separate, norm_input, residual };
enum class Reader { none, rms_output, mul_output, rms_consumer, mul_consumer, raw_rms, raw_mul };

static void limits(transfer_cases::Backend & gpu) {
    auto & caps = gpu.context->global_ctx->capabilities.limits;
    caps.maxStorageBufferBindingSize = 64 * 1024 * 1024;
    caps.maxBufferSize = 64 * 1024 * 1024;
    gpu.type.iface.get_max_size = [](ggml_backend_buffer_type_t) -> size_t { return 64 * 1024 * 1024; };
    gpu.context->param_arena.cleanup();
    gpu.context->param_arena.init(gpu.context->global_ctx->device, WEBGPU_PARAMS_BUF_SIZE_BYTES, 74, 256);
}

struct Graph {
    transfer_cases::Backend gpu{transfer_cases::BackendKind::webgpu};
    transfer_cases::Memory persistent{ggml_init({1024 * 1024, nullptr, true}), ggml_free};
    transfer_cases::Memory memory{ggml_init({1024 * 1024, nullptr, true}), ggml_free};
    transfer_cases::Buffer storage{nullptr, ggml_backend_buffer_free};
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator{nullptr, ggml_gallocr_free};
    ggml_tensor *source = nullptr, *weight = nullptr, *residual = nullptr, *rms = nullptr, *mul = nullptr,
                *output = nullptr;
    ggml_cgraph * graph = nullptr;
    int first = 0, stop = 0;

    Graph(int64_t rows, Output mode, Reader reader = Reader::none, bool swapped = false, int64_t width = 6656) {
        limits(gpu);
        BICORE_CHECK(memory && persistent);
        auto * input_root = ggml_new_tensor_1d(persistent.get(), GGML_TYPE_F32, width * rows + 4);
        auto * input = ggml_view_2d(persistent.get(), input_root, width, rows, width * sizeof(float), 16);
        auto * weight_root = ggml_new_tensor_1d(persistent.get(), GGML_TYPE_F32, width + 8);
        weight = ggml_view_1d(persistent.get(), weight_root, width, 32);
        auto * residual_root = ggml_new_tensor_1d(persistent.get(), GGML_TYPE_F32, width * rows + 12);
        residual = ggml_view_2d(persistent.get(), residual_root, width, rows, width * sizeof(float), 48);
        storage.reset(ggml_backend_alloc_ctx_tensors_from_buft(persistent.get(), &gpu.type));
        BICORE_CHECK(storage);
        source = mode == Output::norm_input ? ggml_dup(memory.get(), input) : input;
        rms = ggml_rms_norm(memory.get(), source, 1e-8f);
        mul = swapped ? ggml_mul(memory.get(), weight, rms) : ggml_mul(memory.get(), rms, weight);
        output = swapped ? ggml_add(memory.get(), residual, mul) : ggml_add(memory.get(), mul, residual);
        if (reader == Reader::rms_output)
            ggml_set_output(rms);
        if (reader == Reader::mul_output)
            ggml_set_output(mul);
        ggml_set_output(output);
        graph = ggml_new_graph_custom(memory.get(), 32, false);
        // Visit persistent input views before the chain, as the model does
        // through its earlier projections and residual dependencies.
        ggml_build_forward_expand(graph, weight);
        ggml_build_forward_expand(graph, residual);
        ggml_build_forward_expand(graph, output);
        stop = graph->n_nodes;
        if (reader == Reader::rms_consumer || reader == Reader::mul_consumer || reader == Reader::raw_rms ||
            reader == Reader::raw_mul) {
            auto * observed = reader == Reader::rms_consumer || reader == Reader::raw_rms ? rms : mul;
            if (reader == Reader::raw_rms || reader == Reader::raw_mul) {
                observed = ggml_view_tensor(memory.get(), observed);
                BICORE_CHECK(!observed->src[0] && observed->view_src);
            }
            auto * consumer = ggml_dup(memory.get(), observed);
            ggml_set_output(consumer);
            ggml_build_forward_expand(graph, consumer);
        }
        allocator.reset(ggml_gallocr_new(&gpu.type));
        BICORE_CHECK(allocator && ggml_gallocr_alloc_graph(allocator.get(), graph));
        const auto found = std::find(graph->nodes, graph->nodes + stop, rms);
        BICORE_CHECK(found != graph->nodes + stop);
        first = int(found - graph->nodes);
        BICORE_CHECK(stop - first == 3);
        if (mode == Output::residual) {
            output->buffer = residual->buffer;
            output->data = residual->data;
        } else if (mode == Output::norm_input && reader == Reader::none) {
            BICORE_CHECK(ggml_webgpu_tensor_equal(source, output));
        }
    }

    void run(ggml_cgraph & slice) {
        BICORE_CHECK(ggml_backend_graph_compute_async(&gpu.backend, &slice) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(&gpu.backend);
    }

    void reject() {
        auto slice = ggml_graph_view(graph, first, stop);
        webgpu_fusion_view_sources views(slice);
        commands = {};
        int consumed = 1;
        const auto before = created;
        const auto encoded = webgpu_rms_mul_add::encode(gpu.context, slice, 0, views, consumed);
        BICORE_CHECK(!encoded && consumed == 1 && created == before && commands.dispatches.empty());
    }
};

static void fused(int64_t rows, Output mode, uint32_t workgroup, bool swapped,
                  const std::filesystem::path & shader_directory, uint32_t width = 6656) {
    Graph test(rows, mode, Reader::none, swapped, width);
    auto & caps = test.gpu.context->global_ctx->capabilities.limits;
    caps.maxComputeInvocationsPerWorkgroup = workgroup;
    caps.maxComputeWorkgroupSizeX = workgroup;
    if (rows == 129)
        caps.maxComputeWorkgroupsPerDimension = 16;
    auto slice = ggml_graph_view(test.graph, test.first, test.stop);
    commands = {};
    test.run(slice);
    BICORE_CHECK(commands.dispatches.size() == 1 && commands.executed == 1);
#ifdef GGML_WEBGPU_GPU_PROFILE
    BICORE_CHECK(commands.timestamp_pairs.size() == 1);
#endif
    const auto & dispatch = commands.dispatches.front();
    const auto & slots = test.gpu.context->bind_group_cache.slots;
    const auto found = std::find_if(slots.begin(), slots.end(), [&](const auto & slot) {
        return slot.group && slot.group.Get()->id == dispatch.group;
    });
    BICORE_CHECK(found != slots.end());
    const auto & packet = *found;
    const std::string suffix = mode == Output::norm_input ? "_norm_inplace"
                               : mode == Output::residual ? "_residual_inplace"
                                                          : "";
    BICORE_CHECK(packet.pipeline.Get()->name == "RMS_NORM_MUL_ADD" + suffix);
    BICORE_CHECK(dispatch.x == (rows == 129 ? 15u : uint32_t(rows)) && dispatch.y == (rows == 129 ? 9u : 1u) &&
                 dispatch.z == 1);
    const ggml_tensor * tensors[] = {test.source, test.weight, test.residual, test.output};
    const size_t storage_count = mode == Output::separate ? 4 : 3;
    BICORE_CHECK(packet.bindings.size() == storage_count + 1);
    for (size_t i = 0; i < storage_count; ++i) {
        const auto & binding = packet.bindings[i];
        BICORE_CHECK(binding.binding_index == i && binding.buffer.Get() == ggml_webgpu_tensor_buf(tensors[i]).Get() &&
                     binding.offset == ggml_webgpu_tensor_align_offset(test.gpu.context, tensors[i]) &&
                     binding.size == ggml_webgpu_tensor_binding_size(test.gpu.context, tensors[i]));
    }
    const auto & uniform = packet.bindings.back();
    BICORE_CHECK(uniform.binding_index == storage_count && uniform.size == WEBGPU_PARAMS_BUF_SIZE_BYTES);
    std::array<uint32_t, 25> actual;
    std::memcpy(actual.data(), uniform.buffer.Get()->bytes.data() + uniform.offset, sizeof(actual));
    const uint32_t length = width * uint32_t(rows);
    const std::array<uint32_t, 25> expected = {mode == Output::norm_input ? 0u : 4u,
                                               8,
                                               mode == Output::residual ? 12u : 0u,
                                               width,
                                               length,
                                               length,
                                               width,
                                               width,
                                               width,
                                               width,
                                               length,
                                               length,
                                               width,
                                               1,
                                               1,
                                               1,
                                               width,
                                               uint32_t(rows),
                                               1,
                                               1,
                                               std::bit_cast<uint32_t>(1e-8f),
                                               12,
                                               width,
                                               length,
                                               length};
    BICORE_CHECK(actual == expected);

    const auto pipeline_count = created;
    const auto group_count = resources.groups_created;
    const float changed_epsilon = 2e-8f;
    std::memcpy(test.rms->op_params, &changed_epsilon, sizeof(float));
    commands = {};
    test.run(slice);
    BICORE_CHECK(commands.dispatches.size() == 1 && created == pipeline_count &&
                 resources.groups_created == group_count);
    uint32_t uploaded;
    std::memcpy(&uploaded, uniform.buffer.Get()->bytes.data() + uniform.offset + 20 * sizeof(uint32_t),
                sizeof(uploaded));
    BICORE_CHECK(uploaded == std::bit_cast<uint32_t>(changed_epsilon));

    if (!shader_directory.empty()) {
        std::filesystem::create_directories(shader_directory);
        const auto write = [&](const std::string & name, const std::string & code) {
            std::ofstream file(shader_directory / (name + "-wg" + std::to_string(workgroup) + ".wgsl"));
            file << code;
            file.flush();
            BICORE_CHECK(bool(file));
        };
        write("fused" + suffix, packet.pipeline.Get()->code);
        if (mode == Output::separate) {
            ggml_backend_buffer weighted_buffer{}, added_buffer{};
            auto weighted = *test.mul;
            weighted.buffer = &weighted_buffer;
            auto output = *test.output;
            output.buffer = &added_buffer;
            ggml_webgpu_shader_lib_context context{};
            context.src0 = test.source;
            context.src1 = test.weight;
            context.dst = &weighted;
            context.max_wg_size = workgroup;
            write("weighted",
                  test.gpu.context->shader_lib->get_rms_norm_mul_pipeline(context, false).pipeline.Get()->code);
            context.src0 = &weighted;
            context.src1 = test.residual;
            context.dst = &output;
            write("add", test.gpu.context->shader_lib->get_binary_pipeline(context, false).pipeline.Get()->code);
        }
    }
}

static void graph_guards() {
    for (const auto reader : {Reader::rms_output, Reader::mul_output, Reader::rms_consumer, Reader::mul_consumer,
                              Reader::raw_rms, Reader::raw_mul}) {
        Graph test(1, Output::separate, reader);
        test.reject();
        auto slice = ggml_graph_view(test.graph, test.first, test.stop);
        commands = {};
        test.run(slice);
        const bool blocks_pair =
            reader == Reader::rms_output || reader == Reader::rms_consumer || reader == Reader::raw_rms;
        BICORE_CHECK(commands.dispatches.size() == (blocks_pair ? 3 : 2));
    }
    Graph test(1, Output::separate);
    auto prefix = ggml_graph_view(test.graph, test.first, test.first + 2);
    auto remainder = ggml_graph_view(test.graph, test.first + 2, test.stop);
    commands = {};
    test.run(prefix);
    test.run(remainder);
    BICORE_CHECK(commands.dispatches.size() == 2);
}

static void binding_guards() {
    const auto reject = [](const auto & alter) {
        Graph test(1, Output::separate);
        alter(test);
        test.reject();
    };
    reject([](Graph & test) { test.source->nb[1] += 4; });
    reject([](Graph & test) { test.weight->nb[1] += 4; });
    reject([](Graph & test) { test.residual->nb[1] += 4; });
    reject([](Graph & test) { test.output->nb[1] += 4; });
    reject([](Graph & test) { test.weight->type = GGML_TYPE_F16; });
    reject([](Graph & test) { test.mul->op_params[0] = 1; });
    reject([](Graph & test) { test.rms->src[1] = test.residual; });
    reject([](Graph & test) {
        test.output->buffer = test.source->buffer;
        test.output->data = static_cast<char *>(test.source->data) + 4;
    });
    reject([](Graph & test) {
        test.output->buffer = test.residual->buffer;
        test.output->data = static_cast<char *>(test.residual->data) + 4;
    });
    reject([](Graph & test) {
        test.output->buffer = test.weight->buffer;
        test.output->data = test.weight->data;
    });
    reject([](Graph & test) {
        test.rms->buffer = test.residual->buffer;
        test.rms->data = test.residual->data;
    });
    reject([](Graph & test) {
        test.mul->buffer = test.source->buffer;
        test.mul->data = test.source->data;
    });
    reject([](Graph & test) {
        test.residual->buffer = test.source->buffer;
        test.residual->data = test.source->data;
        test.residual->view_src = test.source->view_src;
        test.residual->view_offs = test.source->view_offs;
    });
    for (const bool partial_mul : {false, true}) {
        Graph test(1, Output::norm_input);
        // Use the larger persistent buffer, whose gap after the input fits
        // the shifted intermediate. This must exercise alias admission,
        // rather than fail early because a binding extends past capacity.
        auto * original_input = test.source->src[0];
        for (auto * tensor : {test.source, test.output}) {
            tensor->buffer = original_input->buffer;
            tensor->data = original_input->data;
        }
        auto * partial = partial_mul ? test.mul : test.rms;
        partial->buffer = original_input->buffer;
        partial->data = static_cast<char *>(original_input->data) + sizeof(float);
        using namespace webgpu_checked_tensor_binding;
        std::array<buffer_facts, 6> facts{};
        binding source, discarded;
        BICORE_CHECK(get_binding(test.gpu.context, test.source, facts, source) &&
                     get_binding(test.gpu.context, partial, facts, discarded) && bindings_overlap(source, discarded) &&
                     !webgpu_rms_mul_add::same_binding(source, discarded));
        test.reject();
    }
    reject([](Graph & test) { test.gpu.context->global_ctx->capabilities.limits.maxStorageBuffersPerShaderStage = 3; });
    reject([](Graph & test) { test.gpu.context->global_ctx->capabilities.limits.maxBindingsPerBindGroup = 4; });
    reject(
        [](Graph & test) { test.gpu.context->global_ctx->capabilities.limits.maxComputeWorkgroupStorageSize = 1023; });
    reject([](Graph & test) { test.gpu.context->global_ctx->capabilities.limits.maxStorageBufferBindingSize = 26623; });
    reject([](Graph & test) {
        auto & caps = test.gpu.context->global_ctx->capabilities.limits;
        caps.maxComputeInvocationsPerWorkgroup = 192;
    });
    Graph wrong_width(1, Output::separate, Reader::none, false, 6652);
    wrong_width.reject();
}

static void pipeline_cache() {
    transfer_cases::Backend gpu(transfer_cases::BackendKind::webgpu);
    using Mode = ggml_webgpu_shader_lib::rms_mul_add_output;
    std::vector<WGPUComputePipeline> identities;
    for (uint32_t workgroup : {128u, 256u}) {
        for (const auto mode : {Mode::separate, Mode::norm_input, Mode::residual}) {
            const auto pipeline = gpu.context->shader_lib->get_rms_norm_mul_add_pipeline(workgroup, mode);
            BICORE_CHECK(pipeline.pipeline &&
                         std::find(identities.begin(), identities.end(), pipeline.pipeline.Get()) == identities.end());
            identities.push_back(pipeline.pipeline.Get());
        }
    }
    const auto cold = created;
    size_t index = 0;
    for (uint32_t workgroup : {128u, 256u})
        for (const auto mode : {Mode::separate, Mode::norm_input, Mode::residual})
            BICORE_CHECK(gpu.context->shader_lib->get_rms_norm_mul_add_pipeline(workgroup, mode).pipeline.Get() ==
                         identities[index++]);
    BICORE_CHECK(created == cold);
}

static void muse_census(int64_t tokens) {
    transfer_cases::Backend gpu(transfer_cases::BackendKind::webgpu);
    limits(gpu);
    const auto memory = transfer_cases::Memory(ggml_init({4 * 1024 * 1024, nullptr, true}), ggml_free);
    const auto persistent = transfer_cases::Memory(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
    auto * input = ggml_new_tensor_2d(persistent.get(), GGML_TYPE_F32, 6656, tokens);
    auto * ids = ggml_new_tensor_1d(persistent.get(), GGML_TYPE_I32, 1);
    auto * current = input;
    std::vector<ggml_tensor *> normalizations;
    for (int layer = 0; layer < 52; ++layer) {
        for (int branch = 0; branch < 2; ++branch) {
            auto * residual = current;
            // Preserve the actual post-norm dependencies; DUP stands in for
            // the branch projection. This is a topology test, not inference.
            auto * source = ggml_dup(memory.get(), current);
            auto * weight = ggml_new_tensor_1d(persistent.get(), GGML_TYPE_F32, 6656);
            auto * rms = ggml_rms_norm(memory.get(), source, 1e-8f);
            normalizations.push_back(rms);
            auto * weighted = ggml_mul(memory.get(), rms, weight);
            if (layer == 51 && branch == 0) {
                weighted = ggml_get_rows(memory.get(), weighted, ids);
                residual = ggml_get_rows(memory.get(), residual, ids);
            }
            current = ggml_add(memory.get(), weighted, residual);
        }
    }
    ggml_set_output(current);
    auto * graph = ggml_new_graph_custom(memory.get(), 1024, false);
    ggml_build_forward_expand(graph, current);
    const auto storage = transfer_cases::Buffer(ggml_backend_alloc_ctx_tensors_from_buft(persistent.get(), &gpu.type),
                                                ggml_backend_buffer_free);
    const auto allocator =
        std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)>(ggml_gallocr_new(&gpu.type), ggml_gallocr_free);
    BICORE_CHECK(storage && allocator && ggml_gallocr_alloc_graph(allocator.get(), graph));
    using Mode = ggml_webgpu_shader_lib::rms_mul_add_output;
    const auto expected_pipeline =
        gpu.context->shader_lib->get_rms_norm_mul_add_pipeline(256, Mode::norm_input).pipeline.Get()->id;
    commands = {};
    BICORE_CHECK(ggml_backend_graph_compute_async(&gpu.backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(&gpu.backend);
    const auto fused_count =
        std::count_if(commands.dispatches.begin(), commands.dispatches.end(),
                      [&](const auto & dispatch) { return dispatch.pipeline == expected_pipeline; });
    BICORE_CHECK(normalizations.size() == 104 && fused_count == 103 && commands.dispatches.size() == 211);
    std::cout << "RMS_POST_NORM_CENSUS tokens=" << tokens << " chains=104 fused=103 dispatches=211\n";
}

static void optional_failure_and_replan() {
    Graph rejected(1, Output::separate);
    auto rejected_slice = ggml_graph_view(rejected.graph, rejected.first, rejected.stop);
    inject = WGPUErrorType_Validation;
    commands = {};
    rejected.run(rejected_slice);
    BICORE_CHECK(commands.dispatches.size() == 2 && commands.executed == 2);
    const auto attempted = pushes;
    commands = {};
    rejected.run(rejected_slice);
    BICORE_CHECK(commands.dispatches.size() == 2 && pushes == attempted);

    Graph changed(1, Output::separate);
    auto changed_slice = ggml_graph_view(changed.graph, changed.first, changed.stop);
    webgpu_fusion_view_sources views(changed_slice);
    const auto columns = changed.weight->ne[0];
    on_wait_once = [&] { changed.weight->ne[0] = 1; };
    int consumed = 1;
    commands = {};
    BICORE_CHECK(!webgpu_rms_mul_add::encode(changed.gpu.context, changed_slice, 0, views, consumed));
    BICORE_CHECK(!on_wait_once && consumed == 1 && commands.dispatches.empty());
    changed.weight->ne[0] = columns;
    changed.run(changed_slice);
    BICORE_CHECK(commands.dispatches.size() == 1 && commands.executed == 1);
}

static void all(const std::filesystem::path & shader_directory) {
    for (const auto workgroup : {128u, 256u})
        for (const auto mode : {Output::separate, Output::norm_input, Output::residual})
            fused(1, mode, workgroup, false, shader_directory);
    fused(1, Output::separate, 256, true, {});
    fused(8, Output::norm_input, 256, false, {});
    fused(129, Output::separate, 256, false, {});
    for (uint32_t width : {1536u, 2560u, 2816u}) {
        fused(1, Output::norm_input, 256, false, {}, width);
    }
    optional_failure_and_replan();
    graph_guards();
    binding_guards();
    pipeline_cache();
    muse_census(1);
    muse_census(128);
}

} // namespace rms_mul_add_cases
