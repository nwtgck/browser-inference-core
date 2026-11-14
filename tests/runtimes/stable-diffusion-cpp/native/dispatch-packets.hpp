#pragma once

#include <array>
#include <limits>

namespace {
// The large-domain cases encode real builders with logical external buffers.
// Their payload cannot be read: this checks host packets, not GPU allocation,
// operator admission, shader arithmetic or shader padding execution.
struct LogicalBinding {
    std::unique_ptr<ggml_backend_webgpu_buffer_context> context;
    ggml_backend_buffer buffer{};
    explicit LogicalBinding(const webgpu_global_context & global) {
        auto handle = new WGPUBufferImpl;
        handle->logical_size = uint64_t(64) << 30;
        context =
            std::make_unique<ggml_backend_webgpu_buffer_context>(wgpu::Buffer::Acquire(handle), "logical", global);
        buffer.context = context.get();
        buffer.size = handle->logical_size;
    }
};
struct Packet {
    std::vector<uint32_t> words;
    std::pair<uint32_t, uint32_t> grid;
    size_t bindings;
    size_t written;
};
webgpu_context packet_context(uint32_t workgroup) {
    auto runtime = std::make_shared<webgpu_context_struct>();
    runtime->global_ctx = global_context();
    runtime->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup = workgroup;
    runtime->shader_lib = std::make_unique<ggml_webgpu_shader_lib>(runtime->global_ctx->device);
    runtime->param_arena.init(runtime->global_ctx->device, 128, 8, 256);
    return runtime;
}
ggml_tensor * packet_tensor(ggml_context * metadata, ggml_type type, uint64_t columns, uint32_t rows,
                            LogicalBinding & binding, size_t offset) {
    auto tensor = ggml_new_tensor_2d(metadata, type, columns, rows);
    tensor->buffer = &binding.buffer;
    tensor->data = reinterpret_cast<void *>(uintptr_t(webgpu_ptr_base) + offset);
    const float epsilon = 1e-5f;
    std::memcpy(tensor->op_params, &epsilon, sizeof(epsilon));
    return tensor;
}
template <class Builder> Packet capture_packet(webgpu_context & runtime, Builder builder) {
    auto & api = browser_api::state;
    api.uniforms.clear();
    api.dispatches.clear();
    api.binding_counts.clear();
    api.write_sizes.clear();
    runtime->active_command_encoder = runtime->global_ctx->device.CreateCommandEncoder();
    runtime->active_compute_pass = runtime->active_command_encoder.BeginComputePass();
    const auto encoded = builder();
    (void)encoded;
    runtime->active_compute_pass.End();
    runtime->active_compute_pass = nullptr;
#ifdef GGML_WEBGPU_BATCH_PARAM_UPLOADS
    runtime->param_arena.flush(runtime->global_ctx->queue);
#endif
    const auto commands = runtime->active_command_encoder.Finish();
    runtime->global_ctx->queue.Submit(1, &commands);
    runtime->active_command_encoder = nullptr;
    runtime->param_arena.reset();
    browser_api::drain();
    SDC_CHECK(api.uniforms.size() == 1 && api.dispatches.size() == 1 && api.binding_counts.size() == 1);
    SDC_CHECK(api.write_sizes.size() == 1);
    return {api.uniforms.front(), api.dispatches.front(), api.binding_counts.front(), api.write_sizes.front()};
}
void packet_contract(const Packet & packet, uint64_t logical, uint32_t count, size_t words, size_t count_index,
                     size_t bindings) {
    const auto [x, y] = packet.grid;
    SDC_CHECK(x > 0 && y > 0 && x <= 65535 && y <= 65535);
    SDC_CHECK(uint64_t(x) * y >= logical && uint64_t(x) * y <= UINT32_MAX);
    SDC_CHECK(uint64_t(x - 1) * y < logical);
    if (logical <= 65535)
        SDC_CHECK(x == logical && y == 1);
    SDC_CHECK(packet.words.size() == 32 && packet.words[count_index] == count);
    SDC_CHECK(packet.bindings == bindings);
#ifdef GGML_WEBGPU_BATCH_PARAM_UPLOADS
    SDC_CHECK(packet.written == 256);
    for (size_t index = words; index < packet.words.size(); ++index)
        SDC_CHECK(packet.words[index] == 0);
#else
    SDC_CHECK(packet.written == words * sizeof(uint32_t));
#endif
}
void dispatch_packets() {
    size_t cases = 0;
    for (uint32_t wg : {64U, 256U, 1024U}) {
        auto runtime = packet_context(wg);
        LogicalBinding a(runtime->global_ctx), b(runtime->global_ctx), c(runtime->global_ctx), d(runtime->global_ctx);
        for (uint32_t rows : {1U, 9U, 257U, 65535U, 65536U, 65537U, 131360U})
            for (bool inplace : {false, true})
                for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
                    auto metadata = ggml_init({1024 * 1024, nullptr, true});
                    auto input = packet_tensor(metadata, type, 4, rows, a, 16);
                    auto output = inplace ? input : packet_tensor(metadata, type, 4, rows, b, 32);
                    output->op = GGML_OP_NORM;
                    const auto packet =
                        capture_packet(runtime, [&] { return ggml_webgpu_row_norm(runtime, input, output); });
                    packet_contract(packet, rows, rows, 13, 12, inplace ? 2 : 3);
                    SDC_CHECK(packet.words[0] == 16 / ggml_type_size(type));
                    SDC_CHECK(packet.words[1] == (inplace ? 16 : 32) / ggml_type_size(type));
                    ++cases;
                    if (type == GGML_TYPE_F32)
                        for (bool masked : {false, true})
                            for (bool sink : {false, true}) {
                                auto mask = packet_tensor(metadata, GGML_TYPE_F32, 4, rows, c, 48);
                                auto sinks = packet_tensor(metadata, GGML_TYPE_F32, 1, 1, d, 64);
                                const auto soft = capture_packet(runtime, [&] {
                                    return ggml_webgpu_soft_max(runtime, input, masked ? mask : nullptr,
                                                                sink ? sinks : nullptr, output);
                                });
                                packet_contract(soft, rows, rows, 24, 23,
                                                (inplace ? 2 : 3) + size_t(masked) + size_t(sink));
                                ++cases;
                            }
                    ggml_free(metadata);
                }
        for (int mode = 0; mode < 4; ++mode)
            for (uint32_t rows : {7U, 65537U})
                for (bool swapped : {false, true}) {
                    auto metadata = ggml_init({1024 * 1024, nullptr, true});
                    auto input = packet_tensor(metadata, GGML_TYPE_F32, 8, rows, a, 16);
                    auto norm = packet_tensor(metadata, GGML_TYPE_F32, 8, rows, b, 48);
                    auto factor = packet_tensor(metadata, GGML_TYPE_F32, 8, 1, mode == 2 ? a : c, 32);
                    auto output = packet_tensor(metadata, GGML_TYPE_F32, 8, rows, d, 64);
                    if (mode == 0) {
                        output->buffer = input->buffer;
                        output->data = input->data;
                    }
                    if (mode == 1) {
                        output->buffer = factor->buffer;
                        output->data = factor->data;
                    }
                    const auto packet = capture_packet(runtime, [&] {
                        const auto encoded = ggml_webgpu_rms_norm_mul(runtime, input, norm, swapped ? factor : norm,
                                                                      swapped ? norm : factor, output);
                        SDC_CHECK(encoded.has_value());
                        return *encoded;
                    });
                    packet_contract(packet, rows, rows, 21, 20, mode == 3 ? 4 : 3);
                    ++cases;
                    ggml_free(metadata);
                }
    }
    for (uint32_t wg : {1U, 32U, 256U, 1024U}) {
        auto runtime = packet_context(wg);
        LogicalBinding a(runtime->global_ctx), b(runtime->global_ctx), c(runtime->global_ctx);
        for (uint32_t ne : {1U, 17U, 65536U, 16777472U, 268435455U, UINT32_MAX})
            for (int mode = 0; mode < 3; ++mode) {
                const uint64_t groups = 1 + (uint64_t(ne) - 1) / wg;
                auto metadata = ggml_init({1024 * 1024, nullptr, true});
                auto input = packet_tensor(metadata, GGML_TYPE_F32, uint64_t(ne) * (mode == 0 ? 2 : 1), 1, a, 16);
                auto second = packet_tensor(metadata, GGML_TYPE_F32, ne, 1, mode == 1 ? a : b, 32);
                auto output = packet_tensor(metadata, GGML_TYPE_F32, ne, 1, c, 64);
                output->op = GGML_OP_GLU;
                reinterpret_cast<int32_t *>(output->op_params)[0] = GGML_GLU_OP_SWIGLU;
                const auto build = [&] {
                    return ggml_webgpu_glu(runtime, input, mode == 0 ? nullptr : second, output);
                };
                if (groups > uint64_t(65535) * 65535) {
                    rejects([&] { build(); });
                } else {
                    const auto packet = capture_packet(runtime, build);
                    packet_contract(packet, groups, ne, 19, 12, mode == 2 ? 4 : 3);
                }
                ++cases;
                ggml_free(metadata);
            }
    }
    // BF16 gathers dispatch one lane per selected row, regardless of width.
    // Treating the packed-u32 source as the F16/F32 parallel path must change
    // these real-builder packets and fail without executing a shader double.
    for (uint32_t wg : {32U, 64U, 256U, 1024U}) {
        auto runtime = packet_context(wg);
        LogicalBinding a(runtime->global_ctx), b(runtime->global_ctx), c(runtime->global_ctx);
        for (uint32_t width : {1U, 3U, 16U, 1025U})
            for (uint32_t rows : {1U, wg, wg + 1}) {
                auto metadata = ggml_init({1024 * 1024, nullptr, true});
                SDC_CHECK(metadata != nullptr);
                auto weight = packet_tensor(metadata, GGML_TYPE_BF16, width, 5, a, 2);
                auto indices = packet_tensor(metadata, GGML_TYPE_I32, rows, 1, b, 4);
                auto output = packet_tensor(metadata, GGML_TYPE_F32, width, rows, c, 12);
                const auto packet =
                    capture_packet(runtime, [&] { return ggml_webgpu_get_rows(runtime, weight, indices, output); });
                packet_contract(packet, 1 + (rows - 1) / wg, rows, 18, 13, 4);
                SDC_CHECK(packet.words[0] == 1 && packet.words[1] == 1 && packet.words[2] == 3);
                SDC_CHECK(packet.words[12] == width);
                ++cases;
                ggml_free(metadata);
            }
    }
    std::cout << "real-builder uniform/binding/grid packets=" << cases << '\n';
}
} // namespace
