#pragma once

namespace indexed_weight_window_cases {

struct Shape {
    ggml_type type;
    int64_t k, m, experts, selected, input_rows;
    uint64_t first_experts;
    uint32_t extra_bindings;
    ggml_prec accumulator = GGML_PREC_UNDEFINED;
};

constexpr Shape q4_gate{GGML_TYPE_Q4_0, 2816, 1408, 128, 8, 1, 43, 2};
constexpr Shape q4_down{GGML_TYPE_Q4_0, 704, 2816, 128, 8, 8, 64, 1};
constexpr Shape mxfp4{GGML_TYPE_MXFP4, 2880, 2880, 32, 4, 4, 16, 1};
constexpr Shape iq4_gate{GGML_TYPE_IQ4_NL, 2304, 2304, 128, 8, 1, 43, 2};
constexpr Shape iq4_down{GGML_TYPE_IQ4_NL, 1152, 2304, 128, 8, 8, 64, 1};
constexpr Shape nvfp4_gate{GGML_TYPE_NVFP4, 2304, 2304, 128, 8, 1, 43, 2, GGML_PREC_BF16};
constexpr Shape nvfp4_down{GGML_TYPE_NVFP4, 1152, 2304, 128, 8, 8, 64, 1, GGML_PREC_F32};
constexpr uint64_t alignment = 256;
constexpr uint64_t storage_cap = 128 * 1024 * 1024;

static void check_encoded(const Shape & shape, int64_t tokens, bool subgroups = true, bool indirect = false,
                          uint32_t workgroup_storage = 32768) {
    transfer_cases::Backend gpu(transfer_cases::BackendKind::webgpu);
    gpu.type.iface.get_name = ggml_backend_webgpu_buffer_type_get_name;
    transfer_cases::Memory memory(ggml_init({128 * 1024, nullptr, true}), ggml_free);
    BICORE_CHECK(memory);
    auto & global = *gpu.context->global_ctx;
    auto & caps = global.capabilities;
    auto & limits = caps.limits;
    limits.maxBufferSize = 512 * 1024 * 1024;
    limits.maxStorageBufferBindingSize = storage_cap;
    limits.maxComputeWorkgroupStorageSize = workgroup_storage;
    caps.supports_subgroups = subgroups;
    caps.min_subgroup_size = 8;
    caps.max_subgroup_size = 32;
    using phase = webgpu_global_context_struct::indexed_live_grid_phase;
    if (indirect) {
        wgpu::BufferDescriptor descriptor{};
        descriptor.size = 3 * sizeof(uint32_t);
        descriptor.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::Indirect;
        global.indexed_live_grid_args = global.device.CreateBuffer(&descriptor);
    }
    global.indexed_live_grid_state.store(indirect ? phase::ready : phase::rejected);

    auto * weights = ggml_new_tensor_3d(memory.get(), shape.type, shape.k, shape.m, shape.experts);
    auto * input = ggml_new_tensor_3d(memory.get(), GGML_TYPE_F32, shape.k, shape.input_rows, tokens);
    auto * ids = ggml_new_tensor_2d(memory.get(), GGML_TYPE_I32, shape.selected, tokens);
    auto * output = ggml_mul_mat_id(memory.get(), weights, input, ids);
    if (shape.accumulator != GGML_PREC_UNDEFINED) {
        BICORE_CHECK(ggml_prec_set_acc(output, shape.accumulator));
    }
    const std::array operands{weights, input, ids, output};
    const auto plan = ggml_webgpu_indexed_weight_windows(weights, limits);
    BICORE_CHECK(plan && plan->extra_bindings() == shape.extra_bindings);
    BICORE_CHECK(plan->first == shape.first_experts * weights->nb[2]);
    BICORE_CHECK(!plan->third || plan->first == plan->second);
    BICORE_CHECK(plan->total() == ggml_nbytes(weights));

    auto * load_input = ggml_new_tensor_3d(memory.get(), GGML_TYPE_F32, shape.k, shape.selected, 512);
    auto * load_ids = ggml_new_tensor_2d(memory.get(), GGML_TYPE_I32, shape.selected, 512);
    auto * load_op = ggml_mul_mat_id(memory.get(), weights, load_input, load_ids);
    if (shape.accumulator != GGML_PREC_UNDEFINED) {
        BICORE_CHECK(ggml_prec_set_acc(load_op, shape.accumulator));
    }
    transfer_cases::Buffer dummy(ggml_backend_buft_alloc_buffer(&gpu.type, 0), ggml_backend_buffer_free);
    for (auto * operand : {weights, load_input, load_ids})
        operand->buffer = dummy.get();
    BICORE_CHECK(!weights->data && ggml_backend_dev_supports_op(&gpu.device, load_op));
    {
        transfer_cases::Backend foreign(transfer_cases::BackendKind::webgpu);
        foreign.type.iface.get_name = ggml_backend_webgpu_buffer_type_get_name;
        transfer_cases::Buffer foreign_dummy(ggml_backend_buft_alloc_buffer(&foreign.type, 0),
                                             ggml_backend_buffer_free);
        weights->buffer = foreign_dummy.get();
        BICORE_CHECK(!ggml_backend_dev_supports_op(&gpu.device, load_op));
    }
    weights->buffer = nullptr;
    weights->data = webgpu_ptr_base;
    BICORE_CHECK(!ggml_backend_dev_supports_op(&gpu.device, load_op));
    weights->data = nullptr;
    const uint32_t old_storage_count = limits.maxStorageBuffersPerShaderStage;
    limits.maxStorageBuffersPerShaderStage = 4 + shape.extra_bindings;
    BICORE_CHECK(!ggml_backend_dev_supports_op(&gpu.device, load_op));
    limits.maxStorageBuffersPerShaderStage = old_storage_count;
    const uint64_t old_max = limits.maxBufferSize;
    limits.maxBufferSize = plan->total() - 1;
    BICORE_CHECK(!ggml_backend_dev_supports_op(&gpu.device, load_op));
    limits.maxBufferSize = old_max;

    std::vector<transfer_cases::Buffer> storage;
    for (auto * tensor : operands) {
        const size_t size = ggml_backend_buft_get_alloc_size(&gpu.type, tensor) + alignment;
        auto * handle = new WGPUBufferImpl;
        handle->logical_size = size;
        handle->usage = WGPUBufferUsage_Storage;
        auto * context = new ggml_backend_webgpu_buffer_context(wgpu::Buffer::Acquire(handle), "expert storage",
                                                                gpu.context->global_ctx);
        storage.emplace_back(ggml_backend_buffer_init(&gpu.type, ggml_backend_webgpu_buffer_interface, context, size),
                             ggml_backend_buffer_free);
        auto * base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(storage.back().get()));
        BICORE_CHECK(ggml_backend_tensor_alloc(storage.back().get(), tensor, base + alignment) == GGML_STATUS_SUCCESS);
    }
    BICORE_CHECK(ggml_backend_dev_supports_op(&gpu.device, output));
    auto * graph = ggml_new_graph_custom(memory.get(), 16, false);
    ggml_build_forward_expand(graph, output);
    commands = {};
    // The recorder substitutes for the device writes to the grid argument.
    const uint32_t row_tile = workgroup_storage == 16384 ? 80 : 128;
    const std::array<uint32_t, 3> expected_grid{uint32_t(CEIL_DIV(shape.m, row_tile)), 1, 1};
    if (indirect) {
        commands.on_direct_dispatch = [&](const WGPUBindGroupImpl & group) {
            BICORE_CHECK(group.bindings.size() == 6);
            const auto & args = group.bindings[4];
            const auto & uniform = group.bindings[5];
            uint32_t rows = 0;
            std::memcpy(&rows, uniform.buffer->bytes.data() + uniform.offset + 7 * sizeof(uint32_t), sizeof(rows));
            BICORE_CHECK(rows == expected_grid[0] && args.buffer == global.indexed_live_grid_args.Get());
            std::memcpy(args.buffer->bytes.data(), expected_grid.data(), sizeof(expected_grid));
        };
    }
    BICORE_CHECK(ggml_backend_graph_compute_async(&gpu.backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(&gpu.backend);
    commands.on_direct_dispatch = {};
    const std::size_t count = tokens == 1 ? 1 : 2;
    BICORE_CHECK(commands.dispatches.size() == count);
    BICORE_CHECK((commands.dispatches.back().arguments != nullptr) == indirect);
    if (indirect) {
        BICORE_CHECK(commands.indirect_grids.size() == 1 && commands.indirect_grids.front() == expected_grid);
    }
    const auto & main = gpu.context->bind_group_cache.slots[count - 1];
    const unsigned original_storage = tokens == 1 ? 4 : 5;
    BICORE_CHECK(main.bindings.size() == original_storage + shape.extra_bindings + 1);
    const auto & first = main.bindings[0];
    BICORE_CHECK(first.offset == alignment && first.size == plan->first);
    const auto & second = main.bindings[original_storage];
    BICORE_CHECK(second.buffer.Get() == first.buffer.Get() && second.offset == alignment + plan->first &&
                 second.size == plan->second);
    if (plan->third) {
        const auto & third = main.bindings[original_storage + 1];
        BICORE_CHECK(third.buffer.Get() == first.buffer.Get() &&
                     third.offset == alignment + plan->first + plan->second && third.size == plan->third);
    }
    const auto & params = main.bindings.back();
    const auto * bytes = params.buffer.Get()->bytes.data() + params.offset;
    std::array<uint32_t, 2> boundaries{};
    std::memcpy(boundaries.data(), bytes + 13 * sizeof(uint32_t), shape.extra_bindings * sizeof(uint32_t));
    BICORE_CHECK(boundaries[0] == plan->first / 4);
    if (plan->third) {
        BICORE_CHECK(boundaries[1] == (plan->first + plan->second) / 4);
    }
    const auto & shader = *main.pipeline.Get();
    BICORE_CHECK(shader.name.find(shape.extra_bindings == 2 ? "_split_weights3" : "_split_weights") !=
                 std::string::npos);
    if (tokens == 1) {
        BICORE_CHECK((shader.name.find("_independent_sg32") != std::string::npos) == subgroups);
        const uint32_t rows_per_group = subgroups ? 16 : 4;
        BICORE_CHECK(commands.dispatches.back().x == CEIL_DIV(shape.m, rows_per_group) * shape.selected);
    } else if (shape.type == GGML_TYPE_Q4_0 && shape.k == 2816 && tokens == 128) {
        BICORE_CHECK(shader.name.find("_tk64_tm8") != std::string::npos);
        BICORE_CHECK(commands.dispatches.back().x == 3432);
    } else if (shape.type == GGML_TYPE_IQ4_NL || shape.type == GGML_TYPE_NVFP4) {
        BICORE_CHECK(shader.name.find(workgroup_storage == 16384 ? "_tm10" : "_tm16") != std::string::npos);
        if (shape.type == GGML_TYPE_NVFP4) {
            BICORE_CHECK(shader.name.find("_acc_f32") != std::string::npos);
        }
    }
}

static void all() {
    for (const auto & shape : {q4_gate, q4_down, mxfp4, iq4_gate, iq4_down, nvfp4_gate, nvfp4_down}) {
        check_encoded(shape, 1);
        check_encoded(shape, 128);
    }
    check_encoded(q4_gate, 1, false);
    check_encoded(mxfp4, 1, false);
    check_encoded(iq4_down, 1, false);
    check_encoded(nvfp4_down, 1, false);
    check_encoded(iq4_gate, 19, true, true);
    check_encoded(nvfp4_gate, 19, true, true);
    check_encoded(nvfp4_gate, 128, true, true);
    check_encoded(iq4_gate, 19, true, true, 16384);
    check_encoded(nvfp4_gate, 128, true, true, 16384);
}

} // namespace indexed_weight_window_cases
