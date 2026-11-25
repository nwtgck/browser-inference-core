namespace indexed_q2_cases {

constexpr int64_t width = 640;
constexpr int64_t rows = 2560;
constexpr int64_t selected = 10;
constexpr uint64_t alignment = 256;

static void check_encoded(int64_t tokens, bool split) {
    transfer_cases::Backend gpu(transfer_cases::BackendKind::webgpu);
    transfer_cases::Memory memory(ggml_init({128 * 1024, nullptr, true}), ggml_free);
    BICORE_CHECK(memory);
    auto & caps = gpu.context->global_ctx->capabilities;
    auto & limits = caps.limits;
    const auto * large = ggml_new_tensor_3d(memory.get(), GGML_TYPE_Q2_0, width, rows, 512);
    limits.maxBufferSize = 256 * 1024 * 1024;
    limits.maxStorageBufferBindingSize = 128 * 1024 * 1024;
    const auto large_windows = ggml_webgpu_q2_weight_windows(large, limits);
    BICORE_CHECK(large_windows && large_windows->first == 117964800 && large_windows->second == large_windows->first);
    limits.maxStorageBufferBindingSize = large_windows->first - 4;
    BICORE_CHECK(!ggml_webgpu_q2_weight_windows(large, limits));

    auto * weights = ggml_new_tensor_3d(memory.get(), GGML_TYPE_Q2_0, width, rows, 16);
    auto * input = ggml_new_tensor_3d(memory.get(), GGML_TYPE_F32, width, selected, tokens);
    auto * ids = ggml_new_tensor_2d(memory.get(), GGML_TYPE_I32, selected, tokens);
    auto * result = ggml_mul_mat_id(memory.get(), weights, input, ids);
    const uint64_t packed = ggml_nbytes(weights);
    const uint64_t low_bytes = split ? packed / 2 : packed;
    limits.maxStorageBufferBindingSize = low_bytes;
    caps.supports_subgroups = true;
    caps.min_subgroup_size = caps.max_subgroup_size = 16;
    auto dummy = transfer_cases::Buffer(ggml_backend_buft_alloc_buffer(&gpu.type, 0), ggml_backend_buffer_free);
    weights->buffer = dummy.get();
    BICORE_CHECK(!weights->data && !ggml_backend_dev_supports_op(&gpu.device, result));
    weights->buffer = nullptr;

    const auto place = [&](ggml_tensor * tensor) {
        const size_t bytes = ggml_backend_buft_get_alloc_size(&gpu.type, tensor);
        transfer_cases::Buffer buffer(ggml_backend_buft_alloc_buffer(&gpu.type, bytes + alignment),
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
    const auto & name = slot.pipeline.Get()->name;
    BICORE_CHECK((name.find("_split_weights") != std::string::npos) == split);
    BICORE_CHECK((name.find("_wg64") != std::string::npos) == (tokens == 1));
}

static void all() {
    check_encoded(1, false);
    check_encoded(1, true);
    check_encoded(5, false);
    check_encoded(5, true);
}

} // namespace indexed_q2_cases
