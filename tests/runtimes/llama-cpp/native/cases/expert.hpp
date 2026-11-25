#pragma once

#include "indexed-q4-0-paired.hpp"
#include "indexed-wide-m.hpp"

namespace expert_cases {
static void nvfp4_accumulator() {
    auto ctx = recording_context();
    ggml_tensor weights{}, input{}, indices{}, output{};
    weights.type = GGML_TYPE_NVFP4;
    input.type = output.type = GGML_TYPE_F32;
    indices.type = GGML_TYPE_I32;
    output.op = GGML_OP_MUL_MAT_ID;
    weights.ne[0] = input.ne[0] = 1152;
    weights.ne[1] = output.ne[0] = 36;
    weights.ne[2] = 17;
    ggml_webgpu_shader_lib_context context{};
    context.src0 = &weights;
    context.src1 = &input;
    context.src2 = &indices;
    context.dst = &output;
    const auto select = [&](ggml_prec precision) {
        BICORE_CHECK(ggml_prec_set_acc(&output, precision));
        return ctx->shader_lib->get_mul_mat_id_pipeline(context, false).pipeline;
    };
    const auto narrow = select(GGML_PREC_UNDEFINED);
    const auto wide = select(GGML_PREC_BF16);
    BICORE_CHECK(narrow.Get() != wide.Get());
    BICORE_CHECK(narrow.Get()->code.find("src0_tile[tm] * src1_val") != std::string::npos);
    BICORE_CHECK(wide.Get()->code.find("array<array<f32,") != std::string::npos);
    BICORE_CHECK(wide.Get()->code.find("f32(src0_tile[tm]) * f32(src1_val)") != std::string::npos);
    BICORE_CHECK(select(GGML_PREC_F32).Get() == wide.Get());
    BICORE_CHECK(select(GGML_PREC_UNDEFINED).Get() == narrow.Get());
}

static void sparse_matrix(const std::filesystem::path & output) {
    auto ctx = recording_context();
    for (const auto type : {GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_NL, GGML_TYPE_Q4_K}) {
        ggml_tensor a{}, x{};
        a.type = type;
        x.type = GGML_TYPE_F32;
        a.ne[0] = x.ne[0] = type == GGML_TYPE_IQ4_NL ? 640 : 2560;
        a.ne[1] = 640;
        a.ne[2] = 512;
        x.ne[1] = x.ne[3] = 1;
        x.ne[2] = 5;
        ggml_webgpu_shader_lib_context context{};
        context.src0 = &a;
        context.src1 = &x;
        const auto pipeline = ctx->shader_lib->get_mul_mat_id_pipeline(context, true);
        const auto & d = *static_cast<const ggml_webgpu_mul_mat_shader_decisions *>(pipeline.context.get());
        BICORE_CHECK(d.wg_size == 64 && d.tile_k == 32 && d.tile_m * d.wg_size_m == 32 && d.tile_n * d.wg_size_n == 32);
        x.ne[2] = 128;
        BICORE_CHECK(ctx->shader_lib->get_mul_mat_id_pipeline(context, true).pipeline.Get() == pipeline.pipeline.Get());
        std::ofstream shader(output / (pipeline.name + ".wgsl"));
        shader << pipeline.pipeline.Get()->code;
        BICORE_CHECK(bool(shader));
    }
}

static void indexed_wide_k(const std::filesystem::path & output) {
    auto ctx = recording_context();
    for (const auto type : {GGML_TYPE_IQ4_NL, GGML_TYPE_NVFP4, GGML_TYPE_Q4_K}) {
        ggml_tensor a{}, x{}, dst{};
        a.type = type;
        x.type = dst.type = GGML_TYPE_F32;
        dst.op = GGML_OP_MUL_MAT_ID;
        a.ne[2] = 128;
        auto c = ggml_webgpu_shader_lib_context{};
        c.src0 = &a;
        c.src1 = &x;
        c.dst = &dst;
        const auto precision = type == GGML_TYPE_NVFP4 ? GGML_PREC_F32 : GGML_PREC_UNDEFINED;
        BICORE_CHECK(ggml_prec_set_acc(&dst, precision));
        const bool paired = type == GGML_TYPE_Q4_K;
        const int64_t base_k = paired ? 512 : 2304;
        const int64_t base_m = paired ? 2048 : 2304;
        const auto select = [&](int64_t k, int64_t m, int64_t tokens, uint32_t shared, uint32_t expected) {
            a.ne[0] = x.ne[0] = k;
            a.ne[1] = dst.ne[0] = m;
            x.ne[2] = dst.ne[2] = tokens;
            c.wg_mem_limit_bytes = shared;
            const auto pipeline = ctx->shader_lib->get_mul_mat_id_pipeline(c, false);
            const auto & d = *static_cast<const ggml_webgpu_mul_mat_shader_decisions *>(pipeline.context.get());
            BICORE_CHECK(d.tile_k == expected && d.wg_size == 64);
            BICORE_CHECK((pipeline.name.find("_tk64") != std::string::npos) == (expected == 64));
            const bool expect_paired = paired && expected == 64;
            BICORE_CHECK((pipeline.name.find("_paired") != std::string::npos) == expect_paired);
            BICORE_CHECK((pipeline.pipeline.Get()->code.find("store_q4_k_nibbles") != std::string::npos) ==
                         expect_paired);
            return pipeline;
        };
        const auto narrow = select(base_k, base_m, 12, 8447, 32);
        const auto wide = select(base_k, base_m, 12, 8448, 64);
        BICORE_CHECK(narrow.pipeline.Get() != wide.pipeline.Get());
        c.prefer_narrow_indexed_k = true;
        const auto apple = select(base_k, base_m, 128, 32768, paired ? 64 : 32);
        BICORE_CHECK(apple.pipeline.Get() == (paired ? wide : narrow).pipeline.Get());
        c.prefer_narrow_indexed_k = false;
        BICORE_CHECK(select(base_k, base_m, 11, 32768, 32).pipeline.Get() == narrow.pipeline.Get());
        BICORE_CHECK(select(base_k, base_m - 1, 128, 32768, 32).pipeline.Get() == narrow.pipeline.Get());
        if (paired) {
            BICORE_CHECK(select(2048, 512, 128, 32768, 64).pipeline.Get() == wide.pipeline.Get());
            BICORE_CHECK(select(2048, 512, 128, 8447, 32).pipeline.Get() == narrow.pipeline.Get());
            BICORE_CHECK(select(2048, 512, 11, 32768, 32).pipeline.Get() == narrow.pipeline.Get());
            BICORE_CHECK(select(768, 2048, 128, 32768, 32).pipeline.Get() == narrow.pipeline.Get());
        } else {
            BICORE_CHECK(select(2304, 512, 128, 32768, 64).pipeline.Get() == wide.pipeline.Get());
            BICORE_CHECK(select(1152, 2304, 19, 32768, 64).pipeline.Get() == wide.pipeline.Get());
        }
        x.type = GGML_TYPE_F16;
        select(base_k, base_m, 128, 32768, 32);
        x.type = GGML_TYPE_F32;
        if (type == GGML_TYPE_NVFP4) {
            BICORE_CHECK(ggml_prec_set_acc(&dst, GGML_PREC_UNDEFINED));
            BICORE_CHECK(select(2304, 2304, 128, 32768, 32).pipeline.Get() != narrow.pipeline.Get());
            BICORE_CHECK(ggml_prec_set_acc(&dst, GGML_PREC_BF16));
            BICORE_CHECK(select(2304, 2304, 128, 32768, 64).pipeline.Get() == wide.pipeline.Get());
        }
        std::ofstream shader(output / (wide.name + ".wgsl"));
        shader << wide.pipeline.Get()->code;
        BICORE_CHECK(bool(shader));
    }
}

static void indexed_k_adapter_selection() {
    for (const auto type : {GGML_TYPE_IQ4_NL, GGML_TYPE_NVFP4}) {
        auto ctx = recording_context();
        auto & limits = ctx->global_ctx->capabilities.limits;
        limits.maxBufferSize = 1024 * 1024 * 1024;
        limits.maxStorageBufferBindingSize = limits.maxBufferSize;
        limits.minStorageBufferOffsetAlignment = 256;
        limits.maxStorageBuffersPerShaderStage = 8;
        limits.maxComputeWorkgroupSizeX = limits.maxComputeInvocationsPerWorkgroup = 1024;
        limits.maxComputeWorkgroupsPerDimension = 65535;
        limits.maxComputeWorkgroupStorageSize = 32768;

        const auto graph = std::unique_ptr<ggml_context, decltype(&ggml_free)>(
            ggml_init({ggml_tensor_overhead() * 6, nullptr, true}), &ggml_free);
        auto * weights = ggml_new_tensor_3d(graph.get(), type, 2304, 2304, 128);
        auto * input = ggml_new_tensor_3d(graph.get(), GGML_TYPE_F32, 2304, 1, 128);
        auto * ids = ggml_new_tensor_2d(graph.get(), GGML_TYPE_I32, 8, 128);
        auto * dst = ggml_mul_mat_id(graph.get(), weights, input, ids);
        const auto precision = type == GGML_TYPE_NVFP4 ? GGML_PREC_F32 : GGML_PREC_UNDEFINED;
        BICORE_CHECK(ggml_prec_set_acc(dst, precision));

        ggml_backend_webgpu_device_context device_context{};
        device_context.webgpu_global_ctx = ctx->global_ctx;
        ggml_backend_device device{};
        device.context = &device_context;
        ggml_backend_buffer_type buffer_type{};
        buffer_type.device = &device;
        const std::array operands{weights, input, ids, dst};
        std::array<ggml_backend_buffer, operands.size()> buffers{};
        std::vector<ggml_backend_webgpu_buffer_context> backing;
        backing.reserve(operands.size());
        for (size_t i = 0; i < operands.size(); ++i) {
            backing.emplace_back(wgpu::Buffer::Acquire(new WGPUBufferImpl), "metadata-only binding", ctx->global_ctx);
            buffers[i].context = &backing.back();
            buffers[i].size = ggml_backend_webgpu_buffer_type_get_alloc_size(&buffer_type, operands[i]);
            operands[i]->data = webgpu_ptr_base;
            operands[i]->buffer = &buffers[i];
        }

        const auto record = [&](const char * vendor, uint32_t graph_index) {
            ctx->global_ctx->vendor = vendor;
            commands = {};
            ctx->active_command_encoder = ctx->global_ctx->device.CreateCommandEncoder();
            ctx->bind_group_cache.begin_graph(graph_index);
#ifndef GGML_WEBGPU_GPU_PROFILE
            ctx->batch_compute_passes = true;
            ctx->active_compute_pass = ctx->active_command_encoder.BeginComputePass();
#endif
            const auto encoded = ggml_webgpu_mul_mat_id(ctx, weights, input, ids, dst);
#ifndef GGML_WEBGPU_GPU_PROFILE
            ctx->active_compute_pass.End();
            ctx->active_compute_pass = nullptr;
#endif
            ctx->bind_group_cache.end_graph();
            ctx->param_arena.flush(ctx->global_ctx->queue);
            BICORE_CHECK(encoded.num_kernels == 2 && commands.dispatches.size() == 2);
            const auto dispatch = commands.dispatches.back();
            BICORE_CHECK(dispatch.x == 11232 && dispatch.y == 1 && dispatch.z == 1);
            const auto encoded_commands = ctx->active_command_encoder.Finish();
            ctx->global_ctx->queue.Submit(1, &encoded_commands);
            drain_queue(ctx->global_ctx->queue.Get());
            ctx->param_arena.reset();
            return dispatch.pipeline;
        };

        ggml_webgpu_shader_lib_context selection{};
        selection.src0 = weights;
        selection.src1 = input;
        selection.src2 = ids;
        selection.dst = dst;
        selection.wg_mem_limit_bytes = limits.maxComputeWorkgroupStorageSize;
        const auto wide = ctx->shader_lib->get_mul_mat_id_pipeline(selection, false);
        selection.prefer_narrow_indexed_k = true;
        const auto narrow = ctx->shader_lib->get_mul_mat_id_pipeline(selection, false);
        BICORE_CHECK(wide.pipeline.Get() != narrow.pipeline.Get());
        BICORE_CHECK(record("intel", 0) == wide.pipeline.Get()->id);
        BICORE_CHECK(record("apple", 1) == narrow.pipeline.Get()->id);
        BICORE_CHECK(record("", 2) == wide.pipeline.Get()->id);
    }
}

static void all(const std::filesystem::path & output) {
    nvfp4_accumulator();
    sparse_matrix(output);
    indexed_wide_k(output);
    indexed_k_adapter_selection();
    indexed_wide_m(output);
    indexed_q4_0_paired(output);
    auto ctx = recording_context();
    std::ofstream cases(output / "cases.json");
    cases << "[\n";
    bool first = true;
    for (const auto type : {GGML_TYPE_Q5_1, GGML_TYPE_IQ4_NL}) {
        ggml_tensor a{}, x{}, ids{}, dst{};
        a.type = type;
        x.type = dst.type = GGML_TYPE_F32;
        ids.type = GGML_TYPE_I32;
        for (auto * tensor : {&a, &x, &ids, &dst})
            for (auto & extent : tensor->ne)
                extent = 1;
        a.ne[2] = 512;
        x.ne[1] = ids.ne[0] = dst.ne[1] = 10;
        ggml_webgpu_shader_lib_context context{};
        context.src0 = &a;
        context.src1 = &x;
        context.src2 = &ids;
        context.dst = &dst;
        context.min_subgroup_size = context.max_subgroup_size = 32;
        context.supports_subgroups = true;
        context.max_wg_size = 256;
        const auto select = [&](int64_t k, int64_t m) {
            a.ne[0] = x.ne[0] = k;
            a.ne[1] = dst.ne[0] = m;
            return ctx->shader_lib->get_mul_mat_id_vec_pipeline(context, true);
        };
        const auto short_pipeline = select(640, 2560);
        const auto fallback = select(1056, 256);
        BICORE_CHECK(short_pipeline.pipeline.Get() != fallback.pipeline.Get());
        const auto created_before_warm = created;
        const char * name = type == GGML_TYPE_Q5_1 ? "Q5_1" : "IQ4_NL";
        for (const auto [k, m, expected] : {std::tuple{640, 2560, 128u}, {1056, 256, 256u}}) {
            const auto pipeline = select(k, m);
            const auto & decision =
                *static_cast<const ggml_webgpu_mul_mat_vec_shader_decisions *>(pipeline.context.get());
            BICORE_CHECK(ggml_webgpu_expert_short_workgroup(context) == expected && decision.wg_size == expected);
            BICORE_CHECK(pipeline.pipeline.Get() == (expected == 128 ? short_pipeline : fallback).pipeline.Get());
            BICORE_CHECK(created == created_before_warm);
            BICORE_CHECK(pipeline.pipeline.Get()->code.find("@workgroup_size(" + std::to_string(expected) + ")") !=
                         std::string::npos);
            std::ofstream shader(output / (std::string(name) + "-" + std::to_string(expected) + ".wgsl"));
            shader << pipeline.pipeline.Get()->code;
            BICORE_CHECK(bool(shader));
            if (!first)
                cases << ",\n";
            first = false;
            cases << "{\"type\":\"" << name << "\",\"k\":" << k << ",\"m\":" << m
                  << ",\"mode\":0,\"sharedInput\":false,\"storedExperts\":11,\"chosen\":" << expected << "}";
        }
        BICORE_CHECK(select(640, 2560).pipeline.Get() == short_pipeline.pipeline.Get() &&
                     created == created_before_warm);
    }
    cases << "\n]\n";
    BICORE_CHECK(bool(cases));
}
} // namespace expert_cases
