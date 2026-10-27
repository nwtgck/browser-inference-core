static int checks = 0;
static void check(bool condition) { ++checks; if (!condition) throw std::runtime_error("dispatch check " + std::to_string(checks)); }
struct fixture {
    ggml_context * context = ggml_init({1 << 20, nullptr, true});
    ggml_cgraph * graph;
    ggml_tensor * source, * first, * unary, * output;
    webgpu_context ctx = std::make_shared<Context>();
    ggml_backend_webgpu_buffer_context source_ctx{{1}, ctx->global_ctx}, output_ctx{{2}, ctx->global_ctx};
    ggml_backend_buffer source_buffer{}, output_buffer{};
    int count;
    fixture(int width, int tokens, bool sigmoid, bool final, uint32_t prefix, bool inplace) : count(final ? 3 : 2) {
        if (!context) throw std::bad_alloc();
        graph = ggml_new_graph_custom(context, 32, false);
        source = ggml_new_tensor_2d(context, GGML_TYPE_F32, width, tokens);
        first = ggml_scale_bias(context, source, 0.25f, 0.125f);
        unary = sigmoid ? ggml_sigmoid(context, first) : ggml_silu(context, first);
        output = final ? ggml_scale_bias(context, unary, 2.0f, -0.125f) : unary;
        ggml_set_output(output); ggml_build_forward_expand(graph, output);
        source_buffer.iface.get_base = output_buffer.iface.get_base = ggml_backend_webgpu_buffer_get_base;
        source_buffer.context = &source_ctx; output_buffer.context = &output_ctx;
        source_buffer.size = output_buffer.size = 1ull << 30;
        source->buffer = &source_buffer; first->buffer = unary->buffer = output->buffer = &output_buffer;
        source->data = reinterpret_cast<void *>(uintptr_t(0x1000 + prefix));
        first->data = unary->data = output->data = reinterpret_cast<void *>(uintptr_t(0x1000 + prefix + 256));
        if (inplace) { output_ctx.buffer.id = source_ctx.buffer.id; first->data = unary->data = output->data = source->data; }
    }
    ~fixture() { ggml_free(context); }
    auto run(int & consumed) { return ggml_webgpu_scale_unary(ctx, graph, 0, consumed); }
};
int main() {
    for (int width : {1, 7, 127, 128, 129, 257}) for (int tokens : {1, 3})
    for (bool sigmoid : {false, true}) for (bool final : {false, true})
    for (bool inplace : {false, true}) for (uint32_t prefix : {0u, 4u, 16u, 252u, 256u, 260u}) {
        fixture f(width, tokens, sigmoid, final, prefix, inplace);
        int consumed = 1;
        auto accepted = f.run(consumed);
        check(bool(accepted) && consumed == f.count);
        check(accepted->pipeline == (unsigned(sigmoid) | (unsigned(final) << 1) | (unsigned(inplace) << 2)));
        check(accepted->params == std::vector<uint32_t>({uint32_t(width * tokens), (prefix % 256) / 4,
            (prefix % 256) / 4, 0, ggml_webgpu_u32_from_f32(0.25f), ggml_webgpu_u32_from_f32(0.125f),
            ggml_webgpu_u32_from_f32(final ? 2.0f : 1.0f), ggml_webgpu_u32_from_f32(final ? -0.125f : 0.0f)}));
        check(accepted->entries.size() == (inplace ? 1u : 2u));
        check(accepted->entries[0].offset == prefix - prefix % 256);
        check(accepted->entries[0].size == prefix % 256 + uint64_t(width * tokens) * 4);
        // Changed coefficient values are uploaded at every encode, independent of factory/cache keys.
        float changed = -0.75f; std::memcpy(&f.first->op_params[0], &changed, 4);
        consumed = 1; accepted = f.run(consumed);
        check(accepted && accepted->params[4] == ggml_webgpu_u32_from_f32(changed));
        auto reject = [&] {
            consumed = 1; const auto before = encoded, pipelines = f.ctx->shader_lib->calls;
            check(!f.run(consumed)); check(consumed == 1 && before == encoded && pipelines == f.ctx->shader_lib->calls);
        };
        auto & limits = f.ctx->global_ctx->capabilities.limits;
        limits.maxComputeInvocationsPerWorkgroup = 127; reject(); limits.maxComputeInvocationsPerWorkgroup = 256;
        limits.maxComputeWorkgroupSizeX = 127; reject(); limits.maxComputeWorkgroupSizeX = 256;
        limits.maxStorageBuffersPerShaderStage = 0; reject(); limits.maxStorageBuffersPerShaderStage = 8;
        limits.maxUniformBuffersPerShaderStage = 0; reject(); limits.maxUniformBuffersPerShaderStage = 12;
        limits.maxBindingsPerBindGroup = 1; reject(); limits.maxBindingsPerBindGroup = 1000;
        limits.maxComputeWorkgroupsPerDimension = 0; reject(); limits.maxComputeWorkgroupsPerDimension = 65535;
        limits.maxStorageBufferBindingSize = 1; reject(); limits.maxStorageBufferBindingSize = 1ull << 30;
        limits.minStorageBufferOffsetAlignment = 3; reject(); limits.minStorageBufferOffsetAlignment = 256;
        auto data = f.source->data; f.source->data = reinterpret_cast<void *>(uintptr_t(data) + 1); reject(); f.source->data = data;
        f.source_buffer.size = prefix + ggml_nbytes(f.source) - 1; reject(); f.source_buffer.size = 1ull << 30;
        f.source_ctx.buffer.bytes = prefix + ggml_nbytes(f.source) - 1; reject(); f.source_ctx.buffer.bytes = 1ull << 30;
        f.source_ctx.buffer.usage = wgpu::BufferUsage::None; reject(); f.source_ctx.buffer.usage = wgpu::BufferUsage::Storage;
        f.output_ctx.global_ctx = std::make_shared<Global>(); reject(); f.output_ctx.global_ctx = f.ctx->global_ctx;
        f.source_buffer.iface.get_base = nullptr; f.source_buffer.context = reinterpret_cast<void *>(uintptr_t(1)); reject();
        f.source_buffer.iface.get_base = ggml_backend_webgpu_buffer_get_base; f.source_buffer.context = &f.source_ctx;
        f.source_buffer.context = nullptr; reject(); f.source_buffer.context = &f.source_ctx;
        // Partial logical overlaps and overlapping aligned bindings remain on the staged path.
        if (width * tokens > 1) {
            f.output_ctx.buffer.id = f.source_ctx.buffer.id;
            f.unary->data = f.output->data = reinterpret_cast<void *>(uintptr_t(f.source->data) + 4);
            reject();
        }
    }
    // A third operation failing device/range checks still allows the valid two-op prefix.
    {
        fixture f(129, 1, true, true, 4, false);
        f.output->buffer = nullptr;
        int consumed = 1; auto result = f.run(consumed);
        check(result && consumed == 2 && result->pipeline == 1);
    }
    // Same native buffer, disjoint aligned bindings: accepted. Logical disjoint
    // ranges with overlapping aligned storage bindings: rejected.
    {
        fixture f(7, 1, false, false, 4, false);
        f.output_ctx.buffer.id = 1;
        int consumed = 1; check(bool(f.run(consumed)));
        f.unary->data = reinterpret_cast<void *>(uintptr_t(f.source->data) + 32);
        consumed = 1; check(!f.run(consumed) && consumed == 1);
    }
    // Grid flattening is truly two-dimensional for small device limits.
    {
        fixture f(257, 1, false, false, 0, false);
        f.ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension = 2;
        int consumed = 1; auto result = f.run(consumed);
        check(result && result->x == 2 && result->y == 2);
    }
    std::cout << checks << " actual production scale/unary dispatch checks passed\n";
}
