#pragma once

namespace transfer_cases {
static Scheduler scheduler(Backend & gpu, Backend & cpu, bool parallel = false) {
    ggml_backend_t backends[]{&gpu.backend, &cpu.backend};
    ggml_backend_buffer_type_t types[]{&gpu.type, &cpu.type};
    return Scheduler(ggml_backend_sched_new(backends, types, 2, 64, parallel, false), ggml_backend_sched_free);
}

static void discovery() {
    BICORE_CHECK(ggml_backend_webgpu_reg_get_proc_address(nullptr, GGML_BACKEND_HOST_INPUT_SNAPSHOT_V1) ==
                 reinterpret_cast<void *>(ggml_backend_webgpu_copy_host_input_snapshot_v1));
    BICORE_CHECK(ggml_backend_webgpu_reg_get_proc_address(nullptr, nullptr) == nullptr);
    BICORE_CHECK(ggml_backend_webgpu_reg_get_proc_address(nullptr, "ggml_backend_copy_host_input_snapshot_v2") ==
                 nullptr);
    for (const bool offer : {false, true}) {
        Backend gpu(BackendKind::webgpu), cpu(BackendKind::host);
        gpu.offer_snapshot = offer;
        auto sched = scheduler(gpu, cpu);
        BICORE_CHECK(sched && gpu.lookups == 1 && cpu.lookups == 1);
        BICORE_CHECK(gpu.lookup_names == std::vector<std::string>{GGML_BACKEND_HOST_INPUT_SNAPSHOT_V1});
        BICORE_CHECK((sched->host_input_snapshot[0] != nullptr) == offer && sched->host_input_snapshot[1] == nullptr);
        const auto cached = sched->host_input_snapshot[0];
        gpu.offer_snapshot = !offer;
        BICORE_CHECK(sched->host_input_snapshot[0] == cached);
    }
    {
        Backend gpu(BackendKind::webgpu), cpu(BackendKind::host);
        auto sched = scheduler(gpu, cpu, true);
        BICORE_CHECK(sched && ggml_backend_sched_get_n_copies(sched.get()) == GGML_SCHED_MAX_COPIES);
        BICORE_CHECK(gpu.lookups == 0 && cpu.lookups == 0);
        BICORE_CHECK(!sched->host_input_snapshot[0] && !sched->host_input_snapshot[1]);
        for (int copy = 0; copy < GGML_SCHED_MAX_COPIES; ++copy)
            BICORE_CHECK(sched->events[0][copy] != nullptr && sched->events[1][copy] == nullptr);
    }
    for (const bool no_registry : {false, true}) {
        Backend gpu(BackendKind::webgpu), cpu(BackendKind::host);
        if (no_registry)
            gpu.device.reg = nullptr;
        else
            gpu.registry.iface.get_proc_address = nullptr;
        auto sched = scheduler(gpu, cpu);
        BICORE_CHECK(sched && gpu.lookups == 0 && sched->host_input_snapshot[0] == nullptr);
    }
}

static void public_graph(bool offered, bool user_input, bool first_required = true, bool second_required = true) {
    Backend gpu(BackendKind::webgpu), cpu(BackendKind::host);
    gpu.offer_snapshot = offered;
    const auto memory = Memory(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
    BICORE_CHECK(memory);
    auto * left = ggml_new_tensor_1d(memory.get(), GGML_TYPE_F32, 4);
    auto * right = ggml_new_tensor_1d(memory.get(), GGML_TYPE_F32, 4);
    auto * output = ggml_add(memory.get(), left, right);
    if (user_input) {
        ggml_set_input(left);
        ggml_set_input(right);
    }
    auto * graph = ggml_new_graph_custom(memory.get(), 64, false);
    ggml_build_forward_expand(graph, output);
    auto input = Buffer(ggml_backend_buft_alloc_buffer(&cpu.type, 512), ggml_backend_buffer_free);
    BICORE_CHECK(input);
    auto * base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(input.get()));
    BICORE_CHECK(ggml_backend_tensor_alloc(input.get(), left, base) == GGML_STATUS_SUCCESS);
    BICORE_CHECK(ggml_backend_tensor_alloc(input.get(), right, base + 256) == GGML_STATUS_SUCCESS);
    auto sched = scheduler(gpu, cpu);
    BICORE_CHECK(sched);
    ggml_backend_sched_set_tensor_backend(sched.get(), left, &cpu.backend);
    ggml_backend_sched_set_tensor_backend(sched.get(), right, &cpu.backend);
    ggml_backend_sched_set_tensor_backend(sched.get(), output, &gpu.backend);
    BICORE_CHECK(ggml_backend_sched_alloc_graph(sched.get(), graph));
    BICORE_CHECK(ggml_backend_sched_get_n_splits(sched.get()) == 1);
    // The allocated real scheduler also permits a focused guard test for an
    // already-resident input. No constructor/hash/copy body is substituted.
    sched->hv_tensor_requires_copy[hash_id(left)] = first_required;
    sched->hv_tensor_requires_copy[hash_id(right)] = second_required;
    std::array<uint8_t, 16> resident;
    resident.fill(9);
    ggml_backend_tensor_set(tensor_copy(left, 0, 0), resident.data(), 0, resident.size());
    resident.fill(10);
    ggml_backend_tensor_set(tensor_copy(right, 0, 0), resident.data(), 0, resident.size());
    ggml_backend_synchronize(&gpu.backend);
    std::array<std::array<uint8_t, 2>, 3> observed{};
    size_t observed_count = 0;
    gpu.before_graph = [&](ggml_cgraph * current) {
        BICORE_CHECK(current->n_nodes == 1 && current->nodes[0]->op == GGML_OP_ADD);
        const auto * node = current->nodes[0];
        BICORE_CHECK(node->src[0]->buffer->iface.get_base == ggml_backend_webgpu_buffer_get_base &&
                     node->src[1]->buffer->iface.get_base == ggml_backend_webgpu_buffer_get_base);
        const auto first = ggml_webgpu_tensor_buf(node->src[0]), second = ggml_webgpu_tensor_buf(node->src[1]);
        const auto first_offset = ggml_webgpu_tensor_offset(node->src[0]),
                   second_offset = ggml_webgpu_tensor_offset(node->src[1]);
        gpu.context->global_ctx->queue.Get()->pending.push_back([&, first, second, first_offset, second_offset] {
            BICORE_CHECK(observed_count < observed.size());
            observed[observed_count++] = {first.Get()->bytes.at(first_offset), second.Get()->bytes.at(second_offset)};
        });
    };
    commands = {};
    readback = {};
    const auto cpu_syncs = cpu.synchronizations;
    size_t first_pending = 0;
    const std::array<uint8_t, 3> seeds{17, 42, 79};
    // The constructor caches both presence and absence for backend lifetime.
    gpu.offer_snapshot = !offered;
    for (const auto seed : seeds) {
        const auto produce = [&, seed] {
            std::memset(left->data, seed, ggml_nbytes(left));
            std::memset(right->data, seed + 1, ggml_nbytes(right));
        };
        if (user_input)
            produce();
        else {
            std::memset(left->data, 0xcc, ggml_nbytes(left));
            std::memset(right->data, 0xcc, ggml_nbytes(right));
            cpu.complete = produce;
        }
        BICORE_CHECK(ggml_backend_sched_graph_compute_async(sched.get(), graph) == GGML_STATUS_SUCCESS);
        std::memset(left->data, 0xee, ggml_nbytes(left));
        std::memset(right->data, 0xee, ggml_nbytes(right));
        const auto pending = gpu.context->global_ctx->queue.Get()->pending.size();
        if (!first_pending)
            first_pending = pending;
        if (first_required || second_required)
            BICORE_CHECK(pending <= first_pending);
    }
    const bool snapshot = offered && first_required && second_required;
    BICORE_CHECK(gpu.snapshots == (snapshot ? seeds.size() : 0));
    BICORE_CHECK(gpu.lookups == 1 && cpu.lookups == 1);
    const auto transfers = unsigned(first_required) + unsigned(second_required);
    BICORE_CHECK(commands.queue_waits == seeds.size() * (transfers - unsigned(snapshot)));
    BICORE_CHECK(cpu.synchronizations - cpu_syncs == seeds.size() * (user_input ? unsigned(snapshot) : transfers));
    size_t input_writes = 0;
    for (const auto & write : commands.writes)
        input_writes += write.buffer != gpu.context->param_arena.buffer.Get();
    BICORE_CHECK(input_writes == seeds.size() * transfers);
    // External caller storage is no longer needed once each copy returned.
    cpu.complete = {};
    input.reset();
    ggml_backend_sched_synchronize(sched.get());
    BICORE_CHECK(observed_count == seeds.size());
    for (size_t index = 0; index < seeds.size(); ++index)
        BICORE_CHECK(
            (observed[index] == std::array<uint8_t, 2>{first_required ? seeds[index] : uint8_t(9),
                                                       second_required ? uint8_t(seeds[index] + 1) : uint8_t(10)}));
    BICORE_CHECK(gpu.context->global_ctx->queue.Get()->pending.empty());
}

static void event_prefix() {
    Backend gpu(BackendKind::webgpu);
    auto * queue = gpu.context->global_ctx->queue.Get();
    unsigned progress = 0;
    queue->pending.push_back([&] { progress = 1; });
    const auto event = std::unique_ptr<ggml_backend_event, decltype(&ggml_backend_event_free)>(
        ggml_backend_event_new(&gpu.device), ggml_backend_event_free);
    BICORE_CHECK(event);
    ggml_backend_event_record(event.get(), &gpu.backend);
    queue->pending.push_back([&] { progress = 2; });
    ggml_backend_event_synchronize(event.get());
    BICORE_CHECK(progress == 1 && queue->pending.size() == 1);
    const auto completed_waits = waits;
    ggml_backend_event_synchronize(event.get());
    BICORE_CHECK(waits == completed_waits && progress == 1);
    drain_queue(queue);
    BICORE_CHECK(progress == 2);
    // An already-completed prefix must not execute later work either.
    ggml_backend_event_record(event.get(), &gpu.backend);
    queue->pending.push_back([&] { progress = 3; });
    ggml_backend_event_synchronize(event.get());
    BICORE_CHECK(progress == 2 && queue->pending.size() == 1);
    drain_queue(queue);
    BICORE_CHECK(progress == 3);
}

static void prefetch_order() {
    Backend gpu(BackendKind::webgpu), cpu(BackendKind::host);
    const auto memory = Memory(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
    BICORE_CHECK(memory);
    auto * weight = ggml_new_tensor_1d(memory.get(), GGML_TYPE_F32, 4);
    auto * normal = ggml_new_tensor_1d(memory.get(), GGML_TYPE_F32, 4);
    auto * earlier = ggml_new_tensor_1d(memory.get(), GGML_TYPE_F32, 4);
    auto * device_output = ggml_add(memory.get(), weight, normal);
    auto * host_output = ggml_scale(memory.get(), earlier, 2);
    auto * graph = ggml_new_graph_custom(memory.get(), 64, false);
    ggml_build_forward_expand(graph, device_output);
    ggml_build_forward_expand(graph, host_output);
    auto weights = Buffer(ggml_backend_buft_alloc_buffer(&cpu.type, 256), ggml_backend_buffer_free);
    auto previous = Buffer(ggml_backend_buft_alloc_buffer(&gpu.type, 256), ggml_backend_buffer_free);
    BICORE_CHECK(weights && previous);
    ggml_backend_buffer_set_usage(weights.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_set_usage(previous.get(), GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    BICORE_CHECK(ggml_backend_tensor_alloc(weights.get(), weight, ggml_backend_buffer_get_base(weights.get())) ==
                 GGML_STATUS_SUCCESS);
    BICORE_CHECK(ggml_backend_tensor_alloc(previous.get(), earlier, ggml_backend_buffer_get_base(previous.get())) ==
                 GGML_STATUS_SUCCESS);
    std::array<uint8_t, 16> previous_bytes;
    previous_bytes.fill(42);
    ggml_backend_tensor_set(earlier, previous_bytes.data(), 0, previous_bytes.size());
    ggml_backend_synchronize(&gpu.backend);
    auto sched = scheduler(gpu, cpu);
    for (auto * tensor : {weight, normal, host_output})
        ggml_backend_sched_set_tensor_backend(sched.get(), tensor, &cpu.backend);
    for (auto * tensor : {earlier, device_output})
        ggml_backend_sched_set_tensor_backend(sched.get(), tensor, &gpu.backend);
    BICORE_CHECK(ggml_backend_sched_alloc_graph(sched.get(), graph));
    BICORE_CHECK(sched->n_splits == 2 && sched->splits[0].inputs[0] == weight);
    auto * prefetched = tensor_copy(earlier, 1, 0);
    // Force the legal host-memory reuse hazard independently of allocator
    // placement choices, using the real allocated tensor and buffer types.
    prefetched->data = normal->data;
    prefetched->buffer = normal->buffer;
    std::memset(normal->data, 17, ggml_nbytes(normal));
    std::memset(weight->data, 24, ggml_nbytes(weight));
    std::vector<std::string> trace;
    std::pair callback_state{&trace, normal};
    unsigned attempt = 0;
    ggml_backend_sched_set_copy_callback(
        sched.get(),
        [](ggml_backend_t, const ggml_tensor * source, ggml_tensor * destination, ggml_cgraph *, void * data) {
            auto & state = *static_cast<std::pair<std::vector<std::string> *, ggml_tensor *> *>(data);
            BICORE_CHECK(*static_cast<uint8_t *>(state.second->data) == 42);
            state.first->push_back("weight-callback");
            ggml_backend_tensor_set(destination, source->data, 0, ggml_nbytes(source));
            return true;
        },
        &callback_state);
    gpu.before_graph = [&](ggml_cgraph *) {
        BICORE_CHECK(trace == std::vector<std::string>{"weight-callback"});
        BICORE_CHECK(sched->n_prefetched == 1 && gpu.snapshots == attempt + 1);
        const auto copied_normal = tensor_copy(normal, 0, 0);
        BICORE_CHECK(ggml_webgpu_tensor_buf(copied_normal).Get()->bytes.at(ggml_webgpu_tensor_offset(copied_normal)) ==
                     17);
        trace.push_back("gpu-encode");
        gpu.context->global_ctx->queue.Get()->pending.push_back([&] { trace.push_back("later-gpu-work"); });
    };
    cpu.host_compute = [&](ggml_cgraph * current) {
        BICORE_CHECK(current->n_nodes == 1 && current->nodes[0]->src[0]->data == normal->data);
        BICORE_CHECK(*static_cast<uint8_t *>(normal->data) == 42);
#ifdef GGML_WEBGPU_GPU_PROFILE
        BICORE_CHECK((trace == std::vector<std::string>{"weight-callback", "gpu-encode", "later-gpu-work"}));
#else
        BICORE_CHECK((trace == std::vector<std::string>{"weight-callback", "gpu-encode"}));
#endif
        trace.push_back("cpu-compute");
        return attempt == 0 ? GGML_STATUS_ABORTED : GGML_STATUS_SUCCESS;
    };
    for (; attempt < 2; ++attempt) {
        commands = {};
        readback = {};
        trace.clear();
        std::memset(normal->data, 17, ggml_nbytes(normal));
        BICORE_CHECK(ggml_backend_sched_graph_compute_async(sched.get(), graph) ==
                     (attempt == 0 ? GGML_STATUS_ABORTED : GGML_STATUS_SUCCESS));
        BICORE_CHECK(gpu.snapshots == attempt + 1 && sched->n_prefetched == 0);
        // The caller completes already-submitted work after an external
        // backend cancellation, then reuses the actual allocated scheduler.
        ggml_backend_sched_synchronize(sched.get());
        BICORE_CHECK(std::count(trace.begin(), trace.end(), "later-gpu-work") == 1 &&
                     gpu.context->global_ctx->queue.Get()->pending.empty());
    }
}

static void all() {
    discovery();
    for (const bool offered : {false, true})
        for (const bool input : {false, true})
            public_graph(offered, input);
    public_graph(true, false, false, true);
    public_graph(true, true, false, true);
    public_graph(true, false, false, false);
    event_prefix();
    prefetch_order();
    BICORE_CHECK(futures.empty());
}
} // namespace transfer_cases
