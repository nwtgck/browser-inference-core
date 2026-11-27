namespace indexed_q2_cases {

struct Shape {
    int64_t width;
    int64_t rows;
    int64_t experts;
    int64_t selected;
};
constexpr Shape small{640, 2560, 16, 10};
constexpr Shape down{1152, 2304, 128, 8};
constexpr uint64_t alignment = 256;

static void check_registry() {
    ggml_backend_reg registry{};
    registry.iface.get_proc_address = ggml_backend_webgpu_reg_get_proc_address;
    const auto query = reinterpret_cast<int (*)()>(
        ggml_backend_reg_get_proc_address(&registry, "ggml_backend_webgpu_indexed_q2_placement_v1"));
    BICORE_CHECK(query && query() == 1);
    BICORE_CHECK(!ggml_backend_reg_get_proc_address(&registry, "ggml_backend_webgpu_indexed_q2_placement_v2"));
}

static void check_weight_windows() {
    transfer_cases::Memory memory(ggml_init({128 * 1024, nullptr, true}), ggml_free);
    BICORE_CHECK(memory);
    wgpu::Limits limits{};
    const auto * large = ggml_new_tensor_3d(memory.get(), GGML_TYPE_Q2_0, small.width, small.rows, 512);
    limits.minStorageBufferOffsetAlignment = alignment;
    limits.maxBufferSize = 256 * 1024 * 1024;
    limits.maxStorageBufferBindingSize = 128 * 1024 * 1024;
    const auto large_windows = ggml_webgpu_q2_weight_windows(large, limits);
    BICORE_CHECK(large_windows && large_windows->first == 117964800 && large_windows->second == large_windows->first);
    limits.maxStorageBufferBindingSize = large_windows->first - 4;
    BICORE_CHECK(!ggml_webgpu_q2_weight_windows(large, limits));
}

static void check_encoded(const Shape & shape, int64_t tokens, bool split, uint32_t subgroup_min, uint32_t subgroup_max,
                          uint32_t vector_width, uint32_t vector_rows) {
    transfer_cases::Backend gpu(transfer_cases::BackendKind::webgpu);
    transfer_cases::Memory memory(ggml_init({128 * 1024, nullptr, true}), ggml_free);
    BICORE_CHECK(memory);
    auto & caps = gpu.context->global_ctx->capabilities;
    auto & limits = caps.limits;
    const auto [width, rows, experts, selected] = shape;
    auto * weights = ggml_new_tensor_3d(memory.get(), GGML_TYPE_Q2_0, width, rows, experts);
    auto * input = ggml_new_tensor_3d(memory.get(), GGML_TYPE_F32, width, selected, tokens);
    auto * ids = ggml_new_tensor_2d(memory.get(), GGML_TYPE_I32, selected, tokens);
    auto * result = ggml_mul_mat_id(memory.get(), weights, input, ids);
    const uint64_t packed = ggml_nbytes(weights);
    const uint64_t low_bytes = split ? packed / 2 : packed;
    limits.maxBufferSize = 256 * 1024 * 1024;
    limits.maxStorageBufferBindingSize = low_bytes;
    caps.supports_subgroups = subgroup_min != 0;
    caps.min_subgroup_size = subgroup_min;
    caps.max_subgroup_size = subgroup_max;
    auto dummy = transfer_cases::Buffer(ggml_backend_buft_alloc_buffer(&gpu.type, 0), ggml_backend_buffer_free);
    weights->buffer = dummy.get();
    BICORE_CHECK(!weights->data && !ggml_backend_dev_supports_op(&gpu.device, result));
    weights->buffer = nullptr;

    const auto place = [&](ggml_tensor * tensor) -> transfer_cases::Buffer {
        const size_t size = ggml_backend_buft_get_alloc_size(&gpu.type, tensor) + alignment;
        auto * handle = new WGPUBufferImpl;
        handle->logical_size = size;
        handle->usage = WGPUBufferUsage_Storage;
        auto * backing = new ggml_backend_webgpu_buffer_context(wgpu::Buffer::Acquire(handle), "indexed Q2 operand",
                                                                gpu.context->global_ctx);
        transfer_cases::Buffer buffer(
            ggml_backend_buffer_init(&gpu.type, ggml_backend_webgpu_buffer_interface, backing, size),
            ggml_backend_buffer_free);
        BICORE_CHECK(buffer);
        auto * addr = static_cast<uint8_t *>(ggml_backend_buffer_get_base(buffer.get())) + alignment;
        BICORE_CHECK(ggml_backend_tensor_alloc(buffer.get(), tensor, addr) == GGML_STATUS_SUCCESS);
        return buffer;
    };
    const auto weight_buffer = place(weights);
    const auto input_buffer = place(input);
    const auto id_buffer = place(ids);
    const auto result_buffer = place(result);
    BICORE_CHECK(ggml_backend_dev_supports_op(&gpu.device, result));

    if (split && tokens == 5) {
        limits.maxStorageBuffersPerShaderStage = 5;
        BICORE_CHECK(!ggml_backend_dev_supports_op(&gpu.device, result));
        limits.maxStorageBuffersPerShaderStage = 8;
        transfer_cases::Backend foreign(transfer_cases::BackendKind::webgpu);
        for (const auto * allocation :
             {weight_buffer.get(), input_buffer.get(), id_buffer.get(), result_buffer.get()}) {
            auto * backing = static_cast<ggml_backend_webgpu_buffer_context *>(allocation->context);
            backing->global_ctx = foreign.context->global_ctx;
            BICORE_CHECK(!ggml_backend_dev_supports_op(&gpu.device, result));
            backing->global_ctx = gpu.context->global_ctx;
        }
        auto * view = ggml_view_tensor(memory.get(), input);
        view->data = nullptr;
        BICORE_CHECK(!view->buffer && view->view_src == input);
        result->src[1] = view;
        BICORE_CHECK(ggml_backend_dev_supports_op(&gpu.device, result));
        auto * backing = static_cast<ggml_backend_webgpu_buffer_context *>(input_buffer->context);
        backing->global_ctx = foreign.context->global_ctx;
        BICORE_CHECK(!ggml_backend_dev_supports_op(&gpu.device, result));
        backing->global_ctx = gpu.context->global_ctx;
        result->src[1] = input;
    }

    auto * graph = ggml_new_graph_custom(memory.get(), 16, false);
    ggml_build_forward_expand(graph, result);
    commands = {};
    BICORE_CHECK(ggml_backend_graph_compute_async(&gpu.backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(&gpu.backend);
    const size_t count = tokens == 1 ? 1 : 2;
    BICORE_CHECK(commands.dispatches.size() == count);
    const auto & slot = gpu.context->bind_group_cache.slots[count - 1];
    const auto & low = slot.bindings.front();
    const size_t ordinary_storage = tokens == 1 ? 4 : 5;
    BICORE_CHECK(slot.bindings.size() == ordinary_storage + 1 + size_t(split));
    BICORE_CHECK(low.buffer.Get() == ggml_webgpu_tensor_buf(weights).Get());
    BICORE_CHECK(low.offset == alignment && low.size == low_bytes);
    if (split) {
        const auto & high = slot.bindings[ordinary_storage];
        BICORE_CHECK(high.binding_index == ordinary_storage && high.buffer.Get() == low.buffer.Get());
        BICORE_CHECK(high.offset == low.offset + low_bytes && high.size == packed - low_bytes);
    }
    uint32_t split_words = 0;
    const auto & arena = gpu.context->param_arena;
    const size_t offset = (count - 1) * arena.slot_stride + 13 * sizeof(uint32_t);
    std::memcpy(&split_words, arena.upload_data.data() + offset, sizeof(split_words));
    BICORE_CHECK(split_words == (split ? low_bytes / sizeof(uint32_t) : 0));
    const auto & pipeline = *slot.pipeline.Get();
    BICORE_CHECK((pipeline.name.find("_split_weights") != std::string::npos) == split);
    if (tokens == 1) {
        const auto & dispatch = commands.dispatches.front();
        BICORE_CHECK(dispatch.x == rows / vector_rows * selected && dispatch.y == 1);
        BICORE_CHECK(pipeline.code.find("@workgroup_size(" + std::to_string(vector_width) + ")") != std::string::npos);
    }
}

static void all() {
    check_registry();
    check_weight_windows();
    check_encoded(small, 1, false, 16, 16, 64, 4);
    check_encoded(small, 1, true, 16, 16, 64, 4);
    check_encoded(small, 5, false, 16, 16, 0, 0);
    check_encoded(small, 5, true, 16, 16, 0, 0);
    check_encoded(down, 1, false, 16, 16, 128, 32);
    check_encoded(down, 1, false, 32, 32, 128, 16);
    check_encoded(down, 1, false, 16, 32, 128, 4);
    check_encoded(down, 1, true, 16, 16, 128, 4);
    check_encoded(down, 1, false, 0, 0, 256, 4);
    check_encoded(down, 128, false, 16, 16, 0, 0);
}

} // namespace indexed_q2_cases
