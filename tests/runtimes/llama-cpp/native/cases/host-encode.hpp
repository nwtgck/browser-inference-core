namespace host_encoding {
enum class Mode { lvalue, rvalue, multi1, multi2, multi0, multi74 };

static uint64_t key(const webgpu_bind_group_cache::slot & slot) {
    uint64_t value = uintptr_t(slot.pipeline.Get());
    auto mix = [&](uint64_t item) { value = (value ^ item) * UINT64_C(1099511628211); };
    mix(slot.group_index);
    for (const auto & entry : slot.bindings) {
        mix(entry.binding_index);
        mix(uintptr_t(entry.buffer.Get()));
        mix(entry.offset);
        mix(entry.size);
    }
    return value;
}

static void run(Mode mode, size_t parameter_count, size_t storage_count, bool long_name, bool batch_pass = true) {
    auto ctx = recording_context();
    ctx->param_arena.cleanup();
    ctx->param_arena.init(ctx->global_ctx->device, 128, 74, 256);
    ctx->batch_compute_passes = batch_pass;
    wgpu::BufferDescriptor descriptor{};
    descriptor.size = 1024;
    descriptor.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    auto data = ctx->global_ctx->device.CreateBuffer(&descriptor);
    const auto count = mode == Mode::multi0 ? 0u : mode == Mode::multi74 ? 74u : mode == Mode::multi2 ? 2u : 1u;
    const std::string original_name = long_name ? "mul_mat_vec_q8_0_f32_independent_sg16_wg128_r4" : "test";
    webgpu_pipeline pipeline{wgpu::ComputePipeline::Acquire(new WGPUComputePipelineImpl{1, 42, "", "host"}),
                             original_name, nullptr};
    std::array<uint64_t, 74> keys{};
    size_t warm_calls = 0, warm_bytes = 0;
    for (unsigned repetition = 0; repetition < 9; ++repetition) {
        BICORE_CHECK(ctx->global_ctx->queue.Get()->pending.empty());
        commands = {};
        commands.order.reserve(5 * 74);
        commands.dispatches.reserve(74);
        commands.timestamp_pairs.reserve(74);
        ctx->active_command_encoder = ctx->global_ctx->device.CreateCommandEncoder();
#ifndef GGML_WEBGPU_GPU_PROFILE
        if (batch_pass)
            ctx->active_compute_pass = ctx->active_command_encoder.BeginComputePass();
#endif
        commands.order.clear();
        pipeline.name = original_name;
        const uint32_t seed = 1000 + 13 * repetition;
        std::vector<uint32_t> params(parameter_count);
        for (size_t word = 0; word < parameter_count; ++word)
            params[word] = seed + uint32_t(word);
        std::vector<wgpu::BindGroupEntry> entries;
        for (size_t binding = 0; binding < storage_count; ++binding) {
            entries.push_back(ggml_webgpu_make_bind_group_entry(uint32_t(binding), data, 4 * binding, 4));
        }
        std::vector<webgpu_dispatch_desc> dispatches;
        if (mode != Mode::lvalue && mode != Mode::rvalue) {
            dispatches.reserve(count);
            for (size_t index = 0; index < count; ++index) {
                auto values = params;
                for (auto & value : values)
                    value += uint32_t(100 * index);
                dispatches.push_back({pipeline, std::move(values), entries, {3 + uint32_t(index), 2}});
            }
        }
        ctx->param_arena.reset();
        ctx->bind_group_cache.begin_graph(0);
        const auto groups = resources.groups_created, layouts = resources.layouts_created;
        allocation_fault::state = {repetition > 0, -1, 0};
        webgpu_encoded_op result;
        if (mode == Mode::lvalue)
            result = ggml_backend_webgpu_build(ctx, pipeline, params, entries, 3, 2);
        else if (mode == Mode::rvalue)
            result = ggml_backend_webgpu_build(ctx, pipeline, std::move(params), std::move(entries), 3, 2);
        else
            result = ggml_backend_webgpu_build_multi(ctx, dispatches);
        allocation_fault::state.observing = false;
        const auto calls = allocation_fault::state.allocations, bytes = allocation_fault::state.bytes;
        BICORE_CHECK(result.num_kernels == count && ctx->bind_group_cache.cursor == count);
        bool separate_pass = !batch_pass;
#ifdef GGML_WEBGPU_GPU_PROFILE
        separate_pass = true;
        BICORE_CHECK(result.pipeline_names.size() == count);
#endif
        std::string order;
        for (size_t index = 0; index < count; ++index)
            order += separate_pass ? "CPBDE" : "PBD";
        BICORE_CHECK(std::string(commands.order.begin(), commands.order.end()) == order);
        BICORE_CHECK(ctx->bind_group_cache.slots.size() == count);
        for (size_t index = 0; index < count; ++index) {
            const auto & slot = ctx->bind_group_cache.slots[index];
            BICORE_CHECK(slot.pipeline.Get() == pipeline.pipeline.Get() && slot.bindings.size() == storage_count + 1);
            for (size_t binding = 0; binding < storage_count; ++binding) {
                const auto & entry = slot.bindings[binding];
                BICORE_CHECK(entry.binding_index == binding && entry.buffer.Get() == data.Get());
                BICORE_CHECK(entry.offset == 4 * binding && entry.size == 4);
            }
            const auto & uniform = slot.bindings.back();
            BICORE_CHECK(uniform.binding_index == storage_count &&
                         uniform.buffer.Get() == ctx->param_arena.buffer.Get());
            BICORE_CHECK(uniform.offset == 256 * index && uniform.size == 128);
            for (size_t word = 0; word < 64; ++word) {
                uint32_t value = 0;
                std::memcpy(&value, ctx->param_arena.upload_data.data() + 256 * index + 4 * word, 4);
                BICORE_CHECK(value == (word < parameter_count ? seed + uint32_t(word + 100 * index) : 0));
            }
            if (!repetition)
                keys[index] = key(slot);
            BICORE_CHECK(keys[index] == key(slot));
            const auto & dispatch = commands.dispatches[index];
            BICORE_CHECK(dispatch.x == 3 + index && dispatch.y == 2 && dispatch.z == 1);
        }
        if (repetition) {
            BICORE_CHECK(resources.groups_created == groups && resources.layouts_created == layouts);
            // At most two batch arrays and one binding list (including the
            // uniform slice) per dispatch. Caller params/names need no copies.
            size_t call_budget = count ? 2 + count : 0;
            size_t byte_budget =
                count * (sizeof(wgpu::BindGroup) + sizeof(size_t) + (storage_count + 1) * sizeof(wgpu::BindGroupEntry));
#ifdef GGML_WEBGPU_GPU_PROFILE
            // Profile labels are owned by the result. Geometric vector growth
            // stays below four string objects per output; a long label adds
            // one owned copy per output. These are absent in ordinary builds.
            size_t growth = 0;
            for (size_t capacity = 1; capacity < 2 * count; capacity *= 2)
                ++growth;
            call_budget += growth + (long_name ? count : 0);
            byte_budget += 4 * count * sizeof(std::string) + (long_name ? count * (original_name.size() + 1) : 0);
#endif
            BICORE_CHECK(calls <= call_budget && bytes <= byte_budget);
            if (repetition == 1) {
                warm_calls = calls;
                warm_bytes = bytes;
            }
            BICORE_CHECK(calls == warm_calls && bytes == warm_bytes);
        }
        // No deferred work may retain caller-owned parameters or names.
        std::fill(params.begin(), params.end(), 0xdeadbeefu);
        params.clear();
        entries.clear();
        dispatches.clear();
        pipeline.name = "x";
        for (size_t index = 0; index < count; ++index)
            BICORE_CHECK(keys[index] == key(ctx->bind_group_cache.slots[index]));
#ifdef GGML_WEBGPU_GPU_PROFILE
        for (const auto & name : result.pipeline_names)
            BICORE_CHECK(name == original_name);
        BICORE_CHECK(ctx->profile_timestamp_query_count == 2 * count * (repetition + 1));
#else
        BICORE_CHECK(commands.timestamp_pairs.empty());
#endif
        BICORE_CHECK(commands.writes.empty() && ctx->global_ctx->queue.Get()->pending.empty());
        ctx->bind_group_cache.end_graph();
        ctx->param_arena.flush(ctx->global_ctx->queue);
        BICORE_CHECK(commands.writes.size() == (count ? 1u : 0u));
#ifndef GGML_WEBGPU_GPU_PROFILE
        if (batch_pass) {
            ctx->active_compute_pass.End();
            ctx->active_compute_pass = nullptr;
        }
#endif
        const auto encoded = ctx->active_command_encoder.Finish();
        ctx->global_ctx->queue.Submit(1, &encoded);
        BICORE_CHECK(commands.executed == 0 && commands.submissions == 1);
        drain_queue(ctx->global_ctx->queue.Get());
        BICORE_CHECK(commands.executed == count && commands.observed_params.size() == count);
        for (size_t index = 0; index < count; ++index) {
            BICORE_CHECK(commands.observed_params[index] == (parameter_count ? seed + 100 * index : 0));
        }
        BICORE_CHECK(std::equal(ctx->param_arena.upload_data.begin(),
                                ctx->param_arena.upload_data.begin() + 256 * count,
                                ctx->param_arena.buffer.Get()->bytes.begin()));
    }
}
static void guards() {
    for (unsigned kind : {0u, 1u, 2u}) {
#ifndef GGML_WEBGPU_GPU_PROFILE
        if (kind == 2)
            continue;
#endif
        auto ctx = recording_context();
        ctx->param_arena.cleanup();
        ctx->param_arena.init(ctx->global_ctx->device, 128, 74, 256);
        webgpu_pipeline pipeline{wgpu::ComputePipeline::Acquire(new WGPUComputePipelineImpl{1, 43, "", "guard"}),
                                 "guard", nullptr};
        std::vector<webgpu_dispatch_desc> dispatches;
        const size_t count = kind == 1 ? 75 : 1;
        for (size_t index = 0; index < count; ++index) {
            dispatches.push_back({pipeline, std::vector<uint32_t>(kind == 0 ? 33 : 1, 7), {}, {1, 1}});
        }
#ifdef GGML_WEBGPU_GPU_PROFILE
        if (kind == 2)
            ctx->profile_timestamp_query_count = WEBGPU_MAX_PROFILE_QUERY_COUNT - 1;
#endif
        commands = {};
        bool rejected = false;
        try {
            (void)ggml_backend_webgpu_build_multi(ctx, dispatches);
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        BICORE_CHECK(rejected && commands.order.empty() && commands.dispatches.empty());
        BICORE_CHECK(ctx->param_arena.next_slot == (kind == 0 ? 0u : kind == 1 ? 74u : 1u));
    }
}
} // namespace host_encoding

static void host_encoding_contracts() {
    using host_encoding::Mode;
    size_t cases = 0;
    for (const size_t parameters : {0u, 1u, 8u, 32u}) {
        for (const size_t storage : {0u, 1u, 3u, 8u, 15u}) {
            for (const bool long_name : {false, true}) {
                for (const auto mode : {Mode::lvalue, Mode::rvalue, Mode::multi1, Mode::multi2}) {
                    host_encoding::run(mode, parameters, storage, long_name);
                    ++cases;
                }
            }
        }
    }
    host_encoding::run(Mode::multi0, 0, 0, false);
    ++cases;
    host_encoding::run(Mode::multi74, 32, 15, true);
    ++cases;
    host_encoding::run(Mode::multi74, 0, 0, false);
    ++cases;
    host_encoding::run(Mode::multi2, 8, 3, true, false);
    ++cases;
    BICORE_CHECK(cases == 164);
    host_encoding::guards();
    std::cout << "164 current host-encoding allocation/order/upload/lifetime cases passed\n";
}
