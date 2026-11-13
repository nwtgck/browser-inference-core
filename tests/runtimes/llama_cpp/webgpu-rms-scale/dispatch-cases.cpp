struct device_fixture : fixture {
    webgpu_context ctx = std::make_shared<Context>();
    ggml_backend_webgpu_buffer_context source_ctx{{1}, ctx->global_ctx}, output_ctx{{2}, ctx->global_ctx};
    ggml_backend_buffer source_buffer{}, output_buffer{};
    device_fixture(int width, int heads, int mode, uint32_t prefix, bool inplace) : fixture(width, heads, 2, 2, mode) {
        source_buffer.iface.get_base = output_buffer.iface.get_base = ggml_backend_webgpu_buffer_get_base;
        source_buffer.context = &source_ctx;
        output_buffer.context = &output_ctx;
        source_buffer.size = output_buffer.size = 1ull << 30;
        source_root->buffer = &source_buffer;
        source_root->data = reinterpret_cast<void *>(uintptr_t(0x1000 + prefix));
        source->buffer = &source_buffer;
        source->data = reinterpret_cast<void *>(uintptr_t(source_root->data) + source->view_offs);
        rms->buffer = output->buffer = &output_buffer;
        rms->data = output->data = reinterpret_cast<void *>(uintptr_t(0x1000 + prefix + 256));
        if (inplace) {
            output_ctx.buffer.id = 1;
            output->data = source->data;
        }
    }
    auto run(int & consumed) {
        return ggml_webgpu_rms_scale(ctx, graph, index, consumed);
    }
};
int main() {
    try {
        for (int width : {1, 7, 127, 128, 129, 257})
            for (int heads : {1, 3, 16})
                for (int mode : {0, 1, 2})
                    for (uint32_t prefix : {0u, 4u, 252u, 256u, 260u}) {
                        device_fixture f(width, heads, mode, prefix, false);
                        int consumed = 1;
                        auto result = f.run(consumed);
                        check(result && consumed == 2 && result->pipeline == 0);
                        const uint64_t offset = prefix + f.source->view_offs;
                        const uint64_t span = ggml_nbytes(f.source);
                        check(result->params ==
                              std::vector<uint32_t>(
                                  {uint32_t(offset % 256 / 4), uint32_t(prefix % 256 / 4),
                                   uint32_t(f.source->nb[1] / 4), uint32_t(f.source->nb[2] / 4),
                                   uint32_t(f.source->nb[3] / 4), uint32_t(f.output->nb[1] / 4),
                                   uint32_t(f.output->nb[2] / 4), uint32_t(f.output->nb[3] / 4), uint32_t(width),
                                   uint32_t(heads), 2, 2, ggml_webgpu_u32_from_f32(1e-5f / float(width)),
                                   ggml_webgpu_u32_from_f32(-0.3f), ggml_webgpu_u32_from_f32(0.07f)}));
                        check(result->entries.size() == 2 && result->entries[0].offset == offset - offset % 256 &&
                              result->entries[0].size == offset % 256 + span);
                        check(result->x * result->y >= uint32_t(heads * 4));
                        auto reject = [&] {
                            const auto before = encoded, calls = f.ctx->shader_lib->calls;
                            consumed = 1;
                            check(!f.run(consumed));
                            check(consumed == 1 && before == encoded && calls == f.ctx->shader_lib->calls);
                        };
                        float eps = 0.003f, scale = -1.7f, bias = 0.13f;
                        std::memcpy(f.rms->op_params, &eps, 4);
                        std::memcpy(f.output->op_params, &scale, 4);
                        std::memcpy(f.output->op_params + 1, &bias, 4);
                        consumed = 1;
                        result = f.run(consumed);
                        check(result && result->params[12] == ggml_webgpu_u32_from_f32(eps) &&
                              result->params[13] == ggml_webgpu_u32_from_f32(scale) &&
                              result->params[14] == ggml_webgpu_u32_from_f32(bias));
                        auto & limits = f.ctx->global_ctx->capabilities.limits;
                        limits.maxComputeInvocationsPerWorkgroup = 127;
                        reject();
                        limits.maxComputeInvocationsPerWorkgroup = 256;
                        limits.maxComputeWorkgroupSizeX = 127;
                        reject();
                        limits.maxComputeWorkgroupSizeX = 256;
                        limits.maxComputeWorkgroupStorageSize = 1023;
                        reject();
                        limits.maxComputeWorkgroupStorageSize = 32768;
                        limits.maxUniformBufferBindingSize = 127;
                        reject();
                        limits.maxUniformBufferBindingSize = 65536;
                        limits.maxUniformBuffersPerShaderStage = 0;
                        reject();
                        limits.maxUniformBuffersPerShaderStage = 12;
                        limits.maxStorageBuffersPerShaderStage = 1;
                        reject();
                        limits.maxStorageBuffersPerShaderStage = 8;
                        limits.maxBindingsPerBindGroup = 2;
                        reject();
                        limits.maxBindingsPerBindGroup = 1000;
                        limits.maxStorageBufferBindingSize = 1;
                        reject();
                        limits.maxStorageBufferBindingSize = 1ull << 30;
                        limits.minStorageBufferOffsetAlignment = 3;
                        reject();
                        limits.minStorageBufferOffsetAlignment = 256;
                        limits.maxComputeWorkgroupsPerDimension = 0;
                        reject();
                        limits.maxComputeWorkgroupsPerDimension = 65535;
                        const auto data = f.source->data;
                        f.source->data = reinterpret_cast<void *>(uintptr_t(data) + 1);
                        reject();
                        f.source->data = data;
                        const auto rootdata = f.source_root->data;
                        f.source_root->data = reinterpret_cast<void *>(uintptr_t(0x800));
                        reject();
                        f.source_root->data = rootdata;
                        f.source_buffer.size = offset + span - 1;
                        reject();
                        f.source_buffer.size = 1ull << 30;
                        f.source_ctx.buffer.bytes = offset + span - 1;
                        reject();
                        f.source_ctx.buffer.bytes = 1ull << 30;
                        f.source_ctx.buffer.usage = wgpu::BufferUsage::None;
                        reject();
                        f.source_ctx.buffer.usage = wgpu::BufferUsage::Storage;
                        f.output_ctx.global_ctx = std::make_shared<Global>();
                        reject();
                        f.output_ctx.global_ctx = f.ctx->global_ctx;
                        if (mode) {
                            auto * saved = f.source_root->buffer;
                            f.source_root->buffer = nullptr;
                            reject();
                            f.source_root->buffer = saved;
                        }
                        f.source_buffer.iface.get_base = nullptr;
                        f.source_buffer.context = reinterpret_cast<void *>(uintptr_t(1));
                        reject();
                        f.source_buffer.iface.get_base = ggml_backend_webgpu_buffer_get_base;
                        f.source_buffer.context = &f.source_ctx;
                        f.source_buffer.context = nullptr;
                        reject();
                        f.source_buffer.context = &f.source_ctx;
                        f.output_ctx.buffer.id = 1;
                        f.output->data = reinterpret_cast<void *>(uintptr_t(f.source->data) + 4);
                        reject();
                        // Same base with a different row mapping is never called inplace.
                        f.output->data = f.source->data;
                        if (rms_scale_same_mapping(f.source, f.output)) {
                            consumed = 1;
                            result = f.run(consumed);
                            check(result && result->pipeline == 1 && result->entries.size() == 1);
                        } else
                            reject();
                    }
        {
            device_fixture f(7, 1, 0, 4, false);
            f.output_ctx.buffer.id = 1;
            f.output->data = reinterpret_cast<void *>(uintptr_t(f.source->data) + 64);
            int consumed = 1;
            check(!f.run(consumed) && consumed == 1);
            f.output->data = reinterpret_cast<void *>(uintptr_t(f.source->data) + 512);
            consumed = 1;
            check(bool(f.run(consumed)));
        }
        {
            device_fixture f(128, 3, 2, 4, false);
            f.ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension = 4;
            int consumed = 1;
            const auto result = f.run(consumed);
            check(result && result->x == 4 && result->y == 3);
            f.source->view_offs += 4;
            consumed = 1;
            check(!f.run(consumed) && consumed == 1);
        }
        for (bool sigmoid : {false, true})
            for (int mode : {0, 2}) {
                device_fixture f(128, 3, mode, 4, false);
                f.output->flags &= ~GGML_TENSOR_FLAG_OUTPUT;
                auto * unary = sigmoid ? ggml_sigmoid(f.context, f.output) : ggml_silu(f.context, f.output);
                ggml_set_output(unary);
                ggml_build_forward_expand(f.graph, unary);
                // With only SCALE/UNARY downstream, both choices save one dispatch.
                int consumed = 1;
                check(f.run(consumed) && consumed == 2);
                unary->flags &= ~GGML_TENSOR_FLAG_OUTPUT;
                auto * final = ggml_scale_bias(f.context, unary, 0.7f, -0.2f);
                ggml_set_output(final);
                ggml_build_forward_expand(f.graph, final);
                scale_unary_plan longer;
                check(plan_scale_unary(f.graph, f.index + 1, 3, longer));
                const auto before = encoded, calls = f.ctx->shader_lib->calls;
                consumed = 1;
                check(!f.run(consumed) && consumed == 1 && encoded == before && f.ctx->shader_lib->calls == calls);
                // Intermediate visibility makes the longer fusion ineligible again.
                ggml_set_output(unary);
                consumed = 1;
                check(f.run(consumed) && consumed == 2);
            }
        std::cout << checks << " actual RMS/scale production binding/dispatch checks passed\n";
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
