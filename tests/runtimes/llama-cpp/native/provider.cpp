#include "support/webgpu-recorder.hpp"
#include "ggml-webgpu.cpp"
#include "support/provider-context.hpp"
#include "cases/q8-short.hpp"

static std::array<uint32_t, 2> production_dispatch(const ggml_webgpu_shader_lib_context & input, uint32_t dimension,
                                                   bool overlap) {
    auto ctx = recording_context();
    auto & capabilities = ctx->global_ctx->capabilities;
    capabilities.limits.maxComputeInvocationsPerWorkgroup = input.max_wg_size;
    capabilities.limits.maxComputeWorkgroupsPerDimension = dimension;
    capabilities.limits.minStorageBufferOffsetAlignment = 256;
    capabilities.supports_subgroups = input.supports_subgroups;
    capabilities.subgroup_size_control_enabled = input.subgroup_size_control_enabled;
    capabilities.supports_subgroup_matrix = input.supports_subgroup_matrix;
    capabilities.sg_mat_m = input.sg_mat_m;
    capabilities.sg_mat_n = input.sg_mat_n;
    capabilities.sg_mat_k = input.sg_mat_k;
    capabilities.supports_dot_product = input.supports_dot_product;
    capabilities.min_subgroup_size = input.min_subgroup_size;
    capabilities.max_subgroup_size = input.max_subgroup_size;
    ctx->global_ctx->vendor = input.vendor;
    ggml_tensor tensors[]{*input.src0, *input.src1, *input.dst};
    ggml_backend_buffer buffers[3]{};
    std::vector<ggml_backend_webgpu_buffer_context> backing;
    backing.reserve(3);
    for (size_t i = 0; i < 3; ++i) {
        auto & tensor = tensors[i];
        tensor.nb[0] = ggml_type_size(tensor.type);
        tensor.nb[1] = ggml_row_size(tensor.type, tensor.ne[0]);
        tensor.nb[2] = tensor.nb[1] * tensor.ne[1];
        tensor.nb[3] = tensor.nb[2] * tensor.ne[2];
        tensor.data = webgpu_ptr_base;
        tensor.view_src = nullptr;
        tensor.view_offs = 0;
        backing.emplace_back(wgpu::Buffer::Acquire(new WGPUBufferImpl), "metadata-only binding", ctx->global_ctx);
        buffers[i].context = &backing.back();
        buffers[i].size = ggml_nbytes(&tensor);
        tensor.buffer = &buffers[i];
    }
    if (overlap)
        tensors[1].buffer = tensors[0].buffer;
    commands = {};
#ifndef GGML_WEBGPU_GPU_PROFILE
    ctx->active_compute_pass = ctx->active_command_encoder.BeginComputePass();
#endif
    const auto encoded = ggml_webgpu_mul_mat(ctx, &tensors[0], &tensors[1], &tensors[2]);
    BICORE_CHECK(encoded.num_kernels == commands.dispatches.size() && !commands.dispatches.empty());
    const auto dispatch = commands.dispatches.back();
#ifndef GGML_WEBGPU_GPU_PROFILE
    ctx->active_compute_pass.End();
#endif
    return {dispatch.x, dispatch.y};
}

#include "cases/quant-matvec.hpp"

static void geometry_and_offsets() {
    for (uint32_t limit : {1u, 2u, 7u, 32u, 65535u}) {
        for (uint32_t count = 0; count < std::min(uint64_t(limit) * limit, UINT64_C(5000)); ++count) {
            uint32_t x = 0, y = 0;
            compute_2d_workgroups(count, limit, x, y);
            BICORE_CHECK(y >= 1 && y <= limit && x <= limit);
            BICORE_CHECK(uint64_t(x) * y >= count);
            BICORE_CHECK(x == (count + y - 1) / y);
        }
    }
    ggml_tensor base{}, view{};
    base.data = reinterpret_cast<void *>(uintptr_t(webgpu_ptr_base) + 96);
    BICORE_CHECK(ggml_webgpu_tensor_offset(&base) == 96);
    view.view_src = &base;
    view.view_offs = 20;
    BICORE_CHECK(ggml_webgpu_tensor_offset(&view) == 116);
}

static void cache_identity_and_lifetime() {
    resources = {};
    auto device = wgpu::Device::Acquire(new WGPUDeviceImpl);
    auto pipeline = wgpu::ComputePipeline::Acquire(new WGPUComputePipelineImpl{1, 1, "", "controlled"});
    auto other_pipeline = wgpu::ComputePipeline::Acquire(new WGPUComputePipelineImpl{1, 2, "", "other"});
    auto buffer = wgpu::Buffer::Acquire(new WGPUBufferImpl);
    std::vector<wgpu::BindGroupEntry> bindings(1);
    bindings[0].buffer = buffer;
    bindings[0].binding = 0;
    bindings[0].offset = 8;
    bindings[0].size = 4;
    {
        webgpu_bind_group_cache actual;
        auto get = [&](uint64_t generation, uint32_t group_index = 0) {
            actual.begin_graph(generation);
            auto result = actual.get(device, pipeline, group_index, bindings, "case", generation);
            actual.end_graph();
            return result;
        };
        auto first = get(0);
        BICORE_CHECK(resources.groups_created == 1 && resources.groups_alive == 1);
        auto warm = get(0);
        BICORE_CHECK(warm.Get() == first.Get() && resources.groups_created == 1);
        auto fresh = get(1);
        BICORE_CHECK(fresh.Get() != first.Get() && resources.groups_created == 2);
        first = nullptr;
        warm = nullptr;
        BICORE_CHECK(resources.groups_alive == 1);
        // Every field of the cache key must invalidate the dispatch-position slot.
        for (int field = 0; field < 7; ++field) {
            auto before = get(1);
            const auto count = resources.groups_created;
            auto changed = bindings;
            switch (field) {
            case 0:
                changed[0].binding++;
                break;
            case 1:
                changed[0].offset++;
                break;
            case 2:
                changed[0].size++;
                break;
            case 3:
                changed[0].buffer = wgpu::Buffer::Acquire(new WGPUBufferImpl);
                break;
            case 4:
                changed.push_back(changed[0]);
                break;
            default:
                break;
            }
            actual.begin_graph(1);
            auto after =
                actual.get(device, field == 5 ? other_pipeline : pipeline, field == 6 ? 1 : 0, changed, "case", 1);
            actual.end_graph();
            BICORE_CHECK(after.Get() != before.Get() && resources.groups_created == count + 1);
        }
        // Uncacheable descriptors keep the actual API path, rather than a stale slot.
        for (int condition = 0; condition < 5; ++condition) {
            auto changed = bindings;
            WGPUChainedStruct chain{};
            switch (condition) {
            case 0:
                changed[0].nextInChain = reinterpret_cast<const wgpu::ChainedStruct *>(&chain);
                break;
            case 1:
                changed[0].sampler = wgpu::Sampler::Acquire(new WGPUSamplerImpl);
                break;
            case 2:
                changed[0].textureView = wgpu::TextureView::Acquire(new WGPUTextureViewImpl);
                break;
            case 3:
                changed[0].buffer = nullptr;
                break;
            case 4:
                while (changed.size() <= actual.max_buffer_entries)
                    changed.push_back(changed[0]);
                break;
            }
            const auto count = resources.groups_created;
            for (int repeat = 0; repeat < 2; ++repeat) {
                actual.begin_graph(1);
                actual.get(device, pipeline, 0, changed, "uncacheable", 1);
                actual.end_graph();
            }
            BICORE_CHECK(resources.groups_created == count + 2);
        }
        // A null or throwing driver allocation may not publish a cache hit.
        uint64_t generation = 2;
        for (auto failure : {GroupFailure::null_result, GroupFailure::exception}) {
            resources.group_failure = failure;
            actual.begin_graph(generation);
            const auto attempts = resources.groups_created;
            bool threw = false;
            try {
                BICORE_CHECK(!actual.get(device, pipeline, 0, bindings, "failure", generation));
            } catch (const std::runtime_error & error) {
                BICORE_CHECK(std::string(error.what()) == "Recorded bind-group allocation failure");
                threw = true;
            }
            BICORE_CHECK(threw == (failure == GroupFailure::exception));
            BICORE_CHECK(resources.groups_created == attempts + 1);
            resources.group_failure = GroupFailure::none;
            const auto count = resources.groups_created;
            auto restored = get(generation);
            BICORE_CHECK(restored && resources.groups_created == count + 1);
            ++generation;
        }
        actual.begin_graph(generation);
        actual.end_graph();
        BICORE_CHECK(actual.slots.empty());
    }
    BICORE_CHECK(resources.groups_alive == 0 && buffer.Get()->refs == 2);
}

static void encoding_and_upload() {
    auto ctx = recording_context();
    webgpu_pipeline pipeline{wgpu::ComputePipeline::Acquire(new WGPUComputePipelineImpl{1, 41, "", "encoded"}),
                             "encoded", nullptr};
    const std::vector<webgpu_dispatch_desc> input{{pipeline, {3, 5, 7}, {}, {11, 2}}, {pipeline, {13}, {}, {17, 3}}};
    for (const bool batched : {false, true}) {
        commands = {};
        ctx->active_command_encoder = ctx->global_ctx->device.CreateCommandEncoder();
        ctx->batch_compute_passes = batched;
        ctx->param_arena.reset();
        ctx->bind_group_cache.begin_graph(0);
        if (batched)
            ctx->active_compute_pass = ctx->active_command_encoder.BeginComputePass();
        const auto result = ggml_backend_webgpu_build_multi(ctx, input);
        ctx->bind_group_cache.end_graph();
        BICORE_CHECK(result.num_kernels == 2 && commands.dispatches.size() == 2);
        BICORE_CHECK(commands.dispatches[0].x == 11 && commands.dispatches[0].y == 2);
        BICORE_CHECK(commands.dispatches[1].x == 17 && commands.dispatches[1].y == 3);
        for (const auto & dispatch : commands.dispatches) {
            BICORE_CHECK(dispatch.pipeline == 41 && dispatch.z == 1);
        }
        BICORE_CHECK(commands.writes.empty());
        ctx->param_arena.flush(ctx->global_ctx->queue);
        BICORE_CHECK(commands.writes.size() == 1);
        const auto & upload = commands.writes[0];
        BICORE_CHECK(upload.offset == 0 && upload.bytes.size() == 512);
        BICORE_CHECK(std::memcmp(upload.bytes.data(), input[0].params.data(), 12) == 0);
        BICORE_CHECK(std::memcmp(upload.bytes.data() + 256, input[1].params.data(), 4) == 0);
        for (size_t i = 12; i < 256; ++i)
            BICORE_CHECK(upload.bytes[i] == 0);
        for (size_t i = 260; i < 512; ++i)
            BICORE_CHECK(upload.bytes[i] == 0);
#ifdef GGML_WEBGPU_GPU_PROFILE
        BICORE_CHECK(commands.timestamp_pairs.size() == 2);
        BICORE_CHECK(commands.ends == 2);
        BICORE_CHECK(result.pipeline_names == std::vector<std::string>({"encoded", "encoded"}));
        for (size_t i = 0; i < commands.timestamp_pairs.size(); ++i) {
            const uint32_t offset = batched ? 4 : 0;
            BICORE_CHECK(commands.timestamp_pairs[i].first == offset + 2 * i);
            BICORE_CHECK(commands.timestamp_pairs[i].second == offset + 2 * i + 1);
        }
#else
        BICORE_CHECK(commands.timestamp_pairs.empty());
        BICORE_CHECK(commands.passes == (batched ? 1 : 2));
        BICORE_CHECK(commands.ends == (batched ? 0 : 2));
#endif
        if (batched)
            ctx->active_compute_pass.End();
        ctx->active_compute_pass = nullptr;
        const auto recorded = ctx->active_command_encoder.Finish();
        ctx->global_ctx->queue.Submit(1, &recorded);
        BICORE_CHECK(commands.executed == 0);
        drain_queue(ctx->global_ctx->queue.Get());
        BICORE_CHECK(commands.executed == 2);
    }
    commands = {};
    ctx->param_arena.reset();
    const auto empty = ggml_backend_webgpu_build_multi(ctx, {});
    ctx->param_arena.flush(ctx->global_ctx->queue);
    BICORE_CHECK(empty.num_kernels == 0 && commands.writes.empty() && commands.dispatches.empty());
}

#include "cases/host-encode.hpp"
#include "cases/readback.hpp"
#include "cases/copy.hpp"
#include "cases/profile-query.hpp"
#include "cases/subgroup.hpp"
#include "cases/matrix.hpp"
#include "cases/cache.hpp"
#include "cases/expert.hpp"

int main(int argc, char ** argv) {
    BICORE_CHECK(argc == 2);
    BICORE_CHECK(run_q8_short_cases(argc, argv) == 0);
    BICORE_CHECK(run_quant_matvec_cases(argc, argv) == 0);
    geometry_and_offsets();
    cache_identity_and_lifetime();
    encoding_and_upload();
    host_encoding_contracts();
    readback_cases::scalar();
    copy_cases::all();
    subgroup_cases::all(argv[1]);
    matrix_cases::all(argv[1]);
    cache_cases::all();
    expert_cases::all(argv[1]);
#ifdef GGML_WEBGPU_GPU_PROFILE
    profile_query_cases::all();
#endif
    std::cout << "complete-provider geometry/cache/lifetime contracts passed\n";
}
