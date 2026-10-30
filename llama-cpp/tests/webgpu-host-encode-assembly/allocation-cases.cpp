static bool count_allocations = false;
static size_t allocation_calls, allocation_bytes, deletion_calls, allocation_sizes[512];
[[gnu::noinline]] void * operator new(std::size_t bytes) {
    if (void * pointer = std::malloc(bytes ? bytes : 1)) {
        if (count_allocations) {
            assert(allocation_calls < 512);
            allocation_sizes[allocation_calls++] = bytes;
            allocation_bytes += bytes;
        }
        return pointer;
    }
    throw std::bad_alloc();
}
[[gnu::noinline]] void * operator new[](std::size_t bytes) {
    return ::operator new(bytes);
}
[[gnu::noinline]] void operator delete(void * pointer) noexcept {
    if (pointer && count_allocations)
        ++deletion_calls;
    std::free(pointer);
}
[[gnu::noinline]] void operator delete(void * pointer, std::size_t) noexcept {
    ::operator delete(pointer);
}
[[gnu::noinline]] void operator delete[](void * pointer) noexcept {
    ::operator delete(pointer);
}
[[gnu::noinline]] void operator delete[](void * pointer, std::size_t) noexcept {
    ::operator delete(pointer);
}

struct api_event {
    char kind;
    uintptr_t identity;
    uint32_t x, y, z;
};
static api_event api_events[1024];
static size_t api_count;
static void event(char kind, uintptr_t identity = 0, uint32_t x = 0, uint32_t y = 0, uint32_t z = 0) {
    assert(api_count < 1024);
    api_events[api_count++] = {kind, identity, x, y, z};
}

// @MODEL_AND_ACTUAL@

static uint64_t mix(uint64_t hash, uint64_t value) {
    return (hash ^ value) * UINT64_C(1099511628211);
}
static uint64_t key_hash(const webgpu_bind_group_cache::slot & slot) {
    uint64_t hash = mix(UINT64_C(1469598103934665603), uintptr_t(slot.pipeline.Get()));
    hash = mix(hash, slot.group_index);
    for (const auto & entry : slot.bindings) {
        hash = mix(hash, entry.binding_index);
        hash = mix(hash, uintptr_t(entry.buffer.Get()));
        hash = mix(hash, entry.offset);
        hash = mix(hash, entry.size);
    }
    return hash;
}

enum class mode { lvalue_single, rvalue_single, owning_multi1, owning_multi2, owning_multi0, owning_multi74 };
static const char * mode_name(mode value) {
    switch (value) {
    case mode::lvalue_single:
        return "lvalue-single";
    case mode::rvalue_single:
        return "rvalue-single";
    case mode::owning_multi1:
        return "owning-multi1";
    case mode::owning_multi2:
        return "owning-multi2";
    case mode::owning_multi0:
        return "owning-multi0";
    case mode::owning_multi74:
        return "owning-multi74";
    }
    std::abort();
}
static void run_case(mode selected, size_t parameters, size_t storage, bool long_name, bool & first,
                     bool batch_pass = true) {
    fixture f;
    webgpu_pipeline pipeline{f.pipeline, long_name ? "mul_mat_vec_q8_0_f32_independent_sg16_wg128_r4" : "test"};
    f.ctx->batch_compute_passes = batch_pass;
    const std::string original_name = pipeline.name;
    const size_t count = selected == mode::owning_multi0    ? 0
                         : selected == mode::owning_multi74 ? 74
                         : selected == mode::owning_multi2  ? 2
                                                            : 1;
    size_t expected_calls = 0, expected_bytes = 0;
    uint64_t expected_keys[74]{};
    size_t sizes[512]{}, size_count = 0;
    // First encode, cache population and all input construction are unmeasured.
    for (size_t repetition = 0; repetition < 9; ++repetition) {
        f.ctx->active_command_encoder = f.global->device.CreateCommandEncoder();
        f.ctx->active_command_encoder.encoded->reserve(74);
#ifndef GGML_WEBGPU_GPU_PROFILE
        if (batch_pass)
            f.ctx->active_compute_pass = f.ctx->active_command_encoder.BeginComputePass();
#endif
        pipeline.name = original_name;
        const uint32_t seed = 1000 + uint32_t(repetition) * 13;
        std::vector<uint32_t> params(parameters);
        for (size_t word = 0; word < parameters; ++word)
            params[word] = seed + uint32_t(word);
        std::vector<wgpu::BindGroupEntry> entries;
        for (size_t binding = 0; binding < storage; ++binding)
            entries.push_back(ggml_webgpu_make_bind_group_entry(uint32_t(binding), f.data, binding * 4, 4));
        std::vector<webgpu_dispatch_desc> dispatches;
        if (selected == mode::owning_multi1 || selected == mode::owning_multi2 || selected == mode::owning_multi0 ||
            selected == mode::owning_multi74) {
            dispatches.reserve(count);
            for (size_t i = 0; i < count; ++i) {
                auto values = params;
                for (auto & value : values)
                    value += uint32_t(i * 100);
                dispatches.push_back({pipeline, std::move(values), entries, {3 + uint32_t(i), 2}});
            }
        }
        f.ctx->param_arena.reset();
        f.ctx->bind_group_cache.begin_graph(0);
        api_count = 0;
        const auto layouts = f.global->stats->layouts, groups = f.global->stats->groups;
        allocation_calls = allocation_bytes = deletion_calls = 0;
        count_allocations = repetition > 0;
        webgpu_encoded_op result;
        switch (selected) {
        case mode::lvalue_single:
            result = ggml_backend_webgpu_build(f.ctx, pipeline, params, entries, 3, 2);
            break;
        case mode::rvalue_single:
            result = ggml_backend_webgpu_build(f.ctx, pipeline, std::move(params), std::move(entries), 3, 2);
            break;
        case mode::owning_multi1:
        case mode::owning_multi2:
        case mode::owning_multi0:
        case mode::owning_multi74:
            result = ggml_backend_webgpu_build_multi(f.ctx, dispatches);
            break;
        }
        count_allocations = false;
        assert(result.num_kernels == count);

#ifdef GGML_WEBGPU_GPU_PROFILE
        const bool separate_pass = true;
        assert(result.pipeline_names.size() == count);
        for (const auto & name : result.pipeline_names)
            assert(name == pipeline.name);
#else
        const bool separate_pass = !batch_pass;
#endif
        const size_t events_per_dispatch = separate_pass ? 5 : 3;
        assert(api_count == count * events_per_dispatch);
        assert(f.ctx->bind_group_cache.cursor == count && f.ctx->bind_group_cache.slots.size() == count);
        for (size_t i = 0; i < count; ++i) {
            const auto & slot = f.ctx->bind_group_cache.slots[i];
            assert(slot.pipeline.Get() == f.pipeline.Get() && slot.group_index == 0 &&
                   slot.bindings.size() == storage + 1);
            for (size_t binding = 0; binding < storage; ++binding) {
                const auto & key = slot.bindings[binding];
                assert(key.binding_index == binding && key.buffer.Get() == f.data.Get() && key.offset == binding * 4 &&
                       key.size == 4);
            }
            const auto & uniform = slot.bindings.back();
            assert(uniform.binding_index == storage && uniform.buffer.Get() == f.ctx->param_arena.buffer.Get() &&
                   uniform.offset == i * 256 && uniform.size == 128);
            const auto offset = i * f.ctx->param_arena.slot_stride;
            for (size_t word = 0; word < 64; ++word) {
                uint32_t value;
                std::memcpy(&value, f.ctx->param_arena.upload_data.data() + offset + word * 4, 4);
                assert(value == (word < parameters ? seed + uint32_t(word) + uint32_t(i * 100) : 0));
            }
            const size_t event_index = i * events_per_dispatch + (separate_pass ? 1 : 0);
            if (separate_pass) {
                assert(api_events[event_index - 1].kind == 'C');
                assert(api_events[event_index + 3].kind == 'E');
            }
            assert(api_events[event_index].kind == 'P' &&
                   api_events[event_index].identity == uintptr_t(f.pipeline.Get()));
            assert(api_events[event_index + 1].kind == 'B' &&
                   api_events[event_index + 1].identity == uintptr_t(slot.group.Get()));
            const auto & dispatch = api_events[event_index + 2];
            assert(dispatch.kind == 'D' && dispatch.x == 3 + i && dispatch.y == 2 && dispatch.z == 1);
            const uint64_t key = key_hash(slot);
            if (!repetition)
                expected_keys[i] = key;
            else
                assert(key == expected_keys[i]);
        }
        if (repetition) {
            assert(f.global->stats->layouts == layouts && f.global->stats->groups == groups);
            if (repetition == 1) {
                expected_calls = allocation_calls;
                expected_bytes = allocation_bytes;
                size_count = allocation_calls;
                std::copy(allocation_sizes, allocation_sizes + size_count, sizes);
            } else {
                assert(allocation_calls == expected_calls && allocation_bytes == expected_bytes);
            }
        }
        // Destroy or poison caller-owned input storage before deferred queue use.
        std::fill(params.begin(), params.end(), 0xdeadbeefu);
        params.clear();
        entries.clear();
        dispatches.clear();
        pipeline.name = "x";
        for (size_t i = 0; i < count; ++i)
            assert(key_hash(f.ctx->bind_group_cache.slots[i]) == expected_keys[i]);
#ifdef GGML_WEBGPU_GPU_PROFILE
        for (const auto & name : result.pipeline_names)
            assert(name == original_name);
        assert(f.ctx->profile_timestamp_query_count == 2 * count * (repetition + 1));
#endif
        assert(f.global->stats->pending.empty());
        f.ctx->bind_group_cache.end_graph();
        f.ctx->param_arena.flush(f.global->queue); // Outside helper allocation window, as at actual batch submit.
        assert(f.global->stats->writes == (count ? repetition + 1 : 0));
        assert(f.global->stats->pending.size() == (count ? 1u : 0u));
#ifndef GGML_WEBGPU_GPU_PROFILE
        if (batch_pass) {
            f.ctx->active_compute_pass.End();
            f.ctx->active_compute_pass = nullptr;
        }
#endif
        auto commands = f.ctx->active_command_encoder.Finish();
        f.global->queue.Submit(1, &commands);
        f.global->stats->drain();
        assert(f.global->stats->submissions == repetition + 1 &&
               f.global->stats->dispatches == count * (repetition + 1));
        assert(f.global->stats->pending.empty());
        for (size_t i = 0; i < count; ++i)
            assert(f.global->stats->observed_params[count * repetition + i] == (parameters ? seed + i * 100 : 0));
        assert(std::equal(f.ctx->param_arena.upload_data.begin(), f.ctx->param_arena.upload_data.begin() + count * 256,
                          f.ctx->param_arena.buffer.p->bytes.begin()));
    }
    std::sort(sizes, sizes + size_count);
    if (!first) {
        std::cout << ',';
    }
    first = false;
    std::cout << "{\"mode\":\"" << mode_name(selected) << "\",\"params\":" << parameters
              << ",\"storageEntries\":" << storage << ",\"pipelineNameBytes\":" << original_name.size()
              << ",\"dispatches\":" << count << ",\"warmRepetitions\":8,\"allocationsPerOp\":" << expected_calls
              << ",\"allocationBytesPerOp\":" << expected_bytes << ",\"allocationSizesSorted\":[";
    for (size_t i = 0; i < size_count; ++i) {
        if (i)
            std::cout << ',';
        std::cout << sizes[i];
    }
    std::cout << "],\"warmCreates\":0,\"apiOrder\":\"" << ([&] {
        std::string value;
        for (size_t i = 0; i < count; ++i)
            value +=
#ifdef GGML_WEBGPU_GPU_PROFILE
                "CPBDE";
#else
                batch_pass ? "PBD" : "CPBDE";
#endif
        return value;
    })() << "\",\"paramBytesAndKeysVerified\":true,\"callerStorageReleasedBeforeFlush\":true,\"batchComputePasses\":"
              << (batch_pass ? "true" : "false") << "}";
}
static void run_guard_cases() {
    for (int kind : {0, 1, 2}) {
#ifndef GGML_WEBGPU_GPU_PROFILE
        if (kind == 2)
            continue;
#endif
        fixture f;
        f.ctx->active_command_encoder = f.global->device.CreateCommandEncoder();
        f.ctx->active_command_encoder.encoded->reserve(75);
#ifndef GGML_WEBGPU_GPU_PROFILE
        f.ctx->active_compute_pass = f.ctx->active_command_encoder.BeginComputePass();
#endif
        webgpu_pipeline pipeline{f.pipeline, "guard"};
        std::vector<webgpu_dispatch_desc> dispatches;
        const size_t count = kind == 1 ? 75 : 1;
        for (size_t i = 0; i < count; ++i)
            dispatches.push_back({pipeline, std::vector<uint32_t>(kind == 0 ? 33 : 1, 7), f.entries(), {1, 1}});
        if (kind == 2)
            f.ctx->profile_timestamp_query_count = WEBGPU_MAX_PROFILE_QUERY_COUNT - 1;
        api_count = 0;
        bool rejected = false;
        try {
            ggml_backend_webgpu_build_multi(f.ctx, dispatches);
        } catch (const std::logic_error &) {
            rejected = true;
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        assert(rejected && api_count == 0);
        assert(f.ctx->param_arena.next_slot == (kind == 0 ? 0u : kind == 1 ? 74u : 1u));
    }
}
int main() {
    std::cout
        << "{\"status\":\"passed\",\"scope\":\"O3 actual extracted helpers with CPU mock API, not GPU/browser timing\",\"cases\":[";
    bool first = true;
    for (size_t parameters : {size_t(0), size_t(1), size_t(8), size_t(32)})
        for (size_t storage : {size_t(0), size_t(1), size_t(3), size_t(8), size_t(15)})
            for (bool long_name : {false, true})
                for (auto selected :
                     {mode::lvalue_single, mode::rvalue_single, mode::owning_multi1, mode::owning_multi2})
                    run_case(selected, parameters, storage, long_name, first);
    run_case(mode::owning_multi0, 0, 0, false, first);
    run_case(mode::owning_multi74, 32, 15, true, first);
    run_case(mode::owning_multi74, 0, 0, false, first);
    run_case(mode::owning_multi2, 8, 3, true, first, false);
    run_guard_cases();
    assert(live_resources == 0);
    std::cout << "],\"mockBindGroupEntryBytes\":" << sizeof(wgpu::BindGroupEntry)
              << ",\"actualPipelineStructBytesInThisMock\":" << sizeof(webgpu_pipeline) << "}\n";
}
