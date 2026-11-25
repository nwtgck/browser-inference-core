#pragma once

namespace indexed_live_grid_cases {

enum class Grid { direct, capped, indirect };
using MainPipelines = std::array<unsigned, 2>;

struct Graph {
    transfer_cases::Backend gpu{transfer_cases::BackendKind::webgpu};
    transfer_cases::Memory memory{ggml_init({128 * 1024, nullptr, true}), ggml_free};
    ggml_cgraph * graph = ggml_new_graph_custom(memory.get(), 16, false);
    std::array<ggml_backend_buffer, 6> allocations{};
    std::vector<ggml_backend_webgpu_buffer_context> storage;
    const bool split;
    const uint32_t row_groups;
    const uint32_t full_grid;

    explicit Graph(bool split_weights)
        : split(split_weights), row_groups(split ? 80 : 22), full_grid(split ? 42880 : 1716) {
        auto & global = *gpu.context->global_ctx;
        global.command_submit_batch_size = 32;
        auto & limits = global.capabilities.limits;
        limits.maxBufferSize = 256 * 1024 * 1024;
        limits.maxStorageBufferBindingSize = 128 * 1024 * 1024;
        const int64_t k = split ? 640 : 2816;
        const int64_t m = split ? 2560 : 1408;
        const int64_t used = split ? 10 : 8;
        auto * weights =
            ggml_new_tensor_3d(memory.get(), split ? GGML_TYPE_Q2_0 : GGML_TYPE_Q4_0, k, m, split ? 512 : 48);
        auto * inputs = ggml_new_tensor_3d(memory.get(), GGML_TYPE_F32, k, split ? used : 1, 128);
        auto * ids_a = ggml_new_tensor_2d(memory.get(), GGML_TYPE_I32, used, 128);
        auto * ids_b = ggml_new_tensor_2d(memory.get(), GGML_TYPE_I32, used, 128);
        auto * out_a = ggml_mul_mat_id(memory.get(), weights, inputs, ids_a);
        auto * out_b = ggml_mul_mat_id(memory.get(), weights, inputs, ids_b);
        ggml_build_forward_expand(graph, out_a);
        ggml_build_forward_expand(graph, out_b);
        BICORE_CHECK(ggml_graph_n_nodes(graph) == 2);

        // The recorder needs binding bounds for the resident operands. Actual
        // bytes are used only for parameters and the device-written argument.
        const std::array tensors{weights, inputs, ids_a, ids_b, out_a, out_b};
        storage.reserve(tensors.size());
        for (size_t i = 0; i < tensors.size(); ++i) {
            auto * tensor = tensors[i];
            const size_t size = ggml_backend_buft_get_alloc_size(&gpu.type, tensor);
            auto * handle = new WGPUBufferImpl;
            handle->logical_size = size;
            handle->usage = WGPUBufferUsage_Storage;
            storage.emplace_back(wgpu::Buffer::Acquire(handle), "indexed operand", gpu.context->global_ctx);
            auto & allocation = allocations[i];
            allocation.buft = &gpu.type;
            allocation.iface.get_base = ggml_backend_webgpu_buffer_get_base;
            allocation.context = &storage.back();
            allocation.size = size;
            tensor->data = webgpu_ptr_base;
            tensor->buffer = &allocation;
        }
        BICORE_CHECK(ggml_backend_dev_supports_op(&gpu.device, out_a));
        BICORE_CHECK(ggml_backend_dev_supports_op(&gpu.device, out_b));
    }

    MainPipelines run(Grid expected) {
        commands = {};
        const bool indirect = expected == Grid::indirect;
        if (indirect) {
            // Model only the two producer writes. The indirect API recorder
            // reads each buffer at execution time, so reordering or reusing a
            // stale grid breaks the nonzero-to-zero transition.
            commands.on_direct_dispatch = [rows = row_groups, writes = 0](const WGPUBindGroupImpl & group) mutable {
                BICORE_CHECK(group.bindings.size() == 6 && writes < 2);
                const auto & args = group.bindings[4];
                const auto & params = group.bindings[5];
                BICORE_CHECK(args.binding == 4 && params.binding == 5 && args.offset == 0 && args.size == 12);
                BICORE_CHECK(args.buffer->usage == (WGPUBufferUsage_Storage | WGPUBufferUsage_Indirect));
                uint32_t recorded_rows = 0;
                std::memcpy(&recorded_rows, params.buffer->bytes.data() + params.offset + 7 * sizeof(uint32_t),
                            sizeof(recorded_rows));
                BICORE_CHECK(recorded_rows == rows);
                const std::array<uint32_t, 3> grid{writes++ == 0 ? 3 * rows : 0, 1, 1};
                std::memcpy(args.buffer->bytes.data(), grid.data(), sizeof(grid));
            };
        }

        const auto prior_waits = waits;
        BICORE_CHECK(ggml_backend_graph_compute_async(&gpu.backend, graph) == GGML_STATUS_SUCCESS);
        BICORE_CHECK(commands.queue_waits == 0);
#ifdef GGML_WEBGPU_GPU_PROFILE
        BICORE_CHECK(waits == prior_waits + 1 && commands.passes == 4);
        BICORE_CHECK(commands.timestamp_pairs.size() == 4 && commands.resolved_queries == std::vector<uint32_t>{8});
#else
        BICORE_CHECK(waits == prior_waits && commands.passes == 1 && commands.executed == 0);
#endif
        drain_queue(gpu.context->global_ctx->queue.Get());
        commands.on_direct_dispatch = {};
        BICORE_CHECK(commands.executed == 4 && commands.dispatches.size() == 4);

        MainPipelines main_pipelines{};
        WGPUBuffer arguments = nullptr;
        const auto & slots = gpu.context->bind_group_cache.slots;
        BICORE_CHECK(slots.size() == commands.dispatches.size());
        for (size_t op = 0; op < main_pipelines.size(); ++op) {
            const auto & gather = commands.dispatches[2 * op];
            const auto & main = commands.dispatches[2 * op + 1];
            BICORE_CHECK(!gather.arguments && gather.x == 1 && gather.y == 1 && gather.z == 1);
            BICORE_CHECK(bool(main.arguments) == indirect);
            if (indirect) {
                const auto & bindings = slots[2 * op].bindings;
                BICORE_CHECK(main.argument_offset == 0 && bindings[4].buffer.Get() == main.arguments);
                BICORE_CHECK(op == 0 || arguments == main.arguments);
                arguments = main.arguments;
            } else if (expected == Grid::capped) {
                BICORE_CHECK(main.x < full_grid && main.y > 1 && uint64_t(main.x) * main.y >= full_grid);
            } else {
                BICORE_CHECK(main.x == full_grid && main.y == 1 && main.z == 1);
            }
            const auto & bindings = slots[2 * op + 1].bindings;
            BICORE_CHECK(bindings.size() == (split ? 7 : 6));
            if (indirect) {
                BICORE_CHECK(std::none_of(bindings.begin(), bindings.end(),
                                          [&](const auto & binding) { return binding.buffer.Get() == arguments; }));
            }
            if (split) {
                constexpr uint64_t window = 117964800;
                BICORE_CHECK(bindings[0].size == window && bindings[5].size == window &&
                             bindings[0].buffer.Get() == bindings[5].buffer.Get() &&
                             bindings[0].offset + window == bindings[5].offset);
            }
            main_pipelines[op] = main.pipeline;
        }
        const std::vector<std::array<uint32_t, 3>> grids =
            indirect ? std::vector<std::array<uint32_t, 3>>{{3 * row_groups, 1, 1}, {0, 1, 1}}
                     : std::vector<std::array<uint32_t, 3>>{};
        BICORE_CHECK(commands.indirect_grids == grids);
        return main_pipelines;
    }
};

static void deliver_scope() {
    BICORE_CHECK(!spontaneous_scopes.empty());
    auto callback = spontaneous_scopes.extract(std::prev(spontaneous_scopes.end()));
    callback.mapped().complete();
}

static void check_activation(bool split) {
    Graph fixture(split);
    auto & limits = fixture.gpu.context->global_ctx->capabilities.limits;
    const unsigned prior_pushes = pushes;
    limits.maxComputeWorkgroupsPerDimension = fixture.full_grid - 1;
    const auto capped = fixture.run(Grid::capped);
    BICORE_CHECK(pushes == prior_pushes && spontaneous_scopes.empty());
    limits.maxComputeWorkgroupsPerDimension = fixture.full_grid;
    const auto direct = fixture.run(Grid::direct);
    BICORE_CHECK(capped == direct && spontaneous_scopes.size() == 3 && pushes == prior_pushes + 3);
    deliver_scope();
    BICORE_CHECK(fixture.run(Grid::direct) == direct && spontaneous_scopes.size() == 2);
    deliver_scope();
    deliver_scope();
    BICORE_CHECK(fixture.run(Grid::indirect) == direct);
    limits.maxComputeWorkgroupsPerDimension = fixture.full_grid - 1;
    BICORE_CHECK(fixture.run(Grid::capped) == direct && pushes == prior_pushes + 3 && spontaneous_scopes.empty());
}

static void check_rejection() {
    Graph fixture(false);
    resources.reject_indirect_buffer = true;
    const auto direct = fixture.run(Grid::direct);
    BICORE_CHECK(!resources.reject_indirect_buffer && commands.buffers_created == 1 && spontaneous_scopes.size() == 3);
    while (!spontaneous_scopes.empty()) {
        deliver_scope();
    }
    const unsigned prior_pushes = pushes;
    BICORE_CHECK(fixture.run(Grid::direct) == direct);
    BICORE_CHECK(pushes == prior_pushes && commands.buffers_created == 0 && spontaneous_scopes.empty());
}

static void check_teardown() {
    std::weak_ptr<webgpu_global_context_struct> device;
    std::array<std::weak_ptr<ggml_webgpu_optional::ScopeData>, 3> scopes;
    {
        Graph fixture(false);
        device = fixture.gpu.context->global_ctx;
        fixture.run(Grid::direct);
        BICORE_CHECK(spontaneous_scopes.size() == 3);
        const auto & pending = fixture.gpu.context->global_ctx->indexed_live_grid_pending;
        std::copy(pending.scopes.begin(), pending.scopes.end(), scopes.begin());
    }
    // Callback userdata must outlive both the backend and its device.
    BICORE_CHECK(device.expired());
    for (auto scope = scopes.rbegin(); scope != scopes.rend(); ++scope) {
        BICORE_CHECK(!scope->expired());
        deliver_scope();
        BICORE_CHECK(scope->expired());
    }
    BICORE_CHECK(spontaneous_scopes.empty());
}

static void all() {
    check_activation(true);
    check_activation(false);
    check_rejection();
    check_teardown();
}

} // namespace indexed_live_grid_cases
