struct LookupTrace { int calls = 0; std::vector<std::string> names; bool offer = true; };
static void * tracked_webgpu_lookup(ggml_backend_reg_t reg, const char * name) {
    auto & trace = *static_cast<LookupTrace *>(reg->context);
    ++trace.calls; trace.names.push_back(name);
    return trace.offer ? ggml_backend_webgpu_reg_get_proc_address(reg, name) : nullptr;
}
static int snapshot_calls = 0;
static bool counted_snapshot(ggml_backend_t src_backend, ggml_backend_t dst_backend,
                             const ggml_tensor * src, ggml_tensor * dst) {
    ++snapshot_calls;
    return ggml_backend_webgpu_copy_host_input_snapshot_v1(src_backend, dst_backend, src, dst);
}
static bool unsupported_snapshot(ggml_backend_t, ggml_backend_t, const ggml_tensor *, ggml_tensor *) {
    ++snapshot_calls; return false;
}
static ggml_backend_sched single_scheduler(Fixture & f) {
    ggml_backend_sched sched;
    sched.n_copies = 1; sched.backends[0] = &f.gpu; sched.source = &f.cpu; sched.copy = &f.dst;
    sched.host_input_snapshot[0] = counted_snapshot;
    return sched;
}
// These local contract cases exercise a later input; first-input barriers and
// complete repeated splits are exercised separately below.
static ggml_backend_sched_split subsequent_input_split(Fixture & f) {
    ggml_backend_sched_split split;
    split.n_inputs = 2; split.inputs[0] = &f.gpu_src; split.inputs[1] = &f.src;
    return split;
}
static void discovery_and_cache() {
    Fixture a, b;
    CHECK(ggml_backend_webgpu_reg_get_proc_address(nullptr, nullptr) == nullptr);
    CHECK(ggml_backend_webgpu_reg_get_proc_address(nullptr, "ggml_backend_copy_host_input_snapshot_v2") == nullptr);
    auto * address = ggml_backend_webgpu_reg_get_proc_address(nullptr, GGML_BACKEND_HOST_INPUT_SNAPSHOT_V1);
    CHECK(address == reinterpret_cast<void *>(ggml_backend_webgpu_copy_host_input_snapshot_v1));
    ggml_backend_reg cpu_registry{{ggml_backend_cpu_get_proc_address}, nullptr};
    CHECK(ggml_backend_reg_get_proc_address(&cpu_registry, GGML_BACKEND_HOST_INPUT_SNAPSHOT_V1) == nullptr);
    ggml_backend_reg no_proc{};
    CHECK(ggml_backend_reg_get_proc_address(&no_proc, GGML_BACKEND_HOST_INPUT_SNAPSHOT_V1) == nullptr);
    LookupTrace trace;
    ggml_backend_reg webgpu_registry{{tracked_webgpu_lookup}, &trace};
    a.gpu_device.reg = b.gpu_device.reg = &webgpu_registry;
    a.cpu_device.reg = &cpu_registry;
    ggml_backend missing_device = a.gpu; missing_device.device = nullptr;
    ggml_backend_device no_registry{{device_type}, GGML_BACKEND_DEVICE_TYPE_GPU, nullptr};
    ggml_backend absent_registry = a.gpu; absent_registry.device = &no_registry;
    ggml_backend_device no_proc_device{{device_type}, GGML_BACKEND_DEVICE_TYPE_GPU, &no_proc};
    ggml_backend absent_proc = a.gpu; absent_proc.device = &no_proc_device;
    ggml_backend_t backends[] = {&a.gpu, &b.gpu, &a.cpu, &missing_device, &absent_registry, &absent_proc};
    ggml_backend_sched sched;
    actual_constructor_discovery(&sched, backends, nullptr, 6);
    CHECK(trace.calls == 2 && trace.names == std::vector<std::string>(2, GGML_BACKEND_HOST_INPUT_SNAPSHOT_V1));
    CHECK(sched.host_input_snapshot[0] == sched.host_input_snapshot[1]);
    CHECK(reinterpret_cast<void *>(sched.host_input_snapshot[0]) == address);
    for (int index = 2; index < 6; ++index) CHECK(sched.host_input_snapshot[index] == nullptr);
    sched.source = &a.cpu; sched.copy = &a.dst;
    auto split = subsequent_input_split(a);
    for (bool input : {false, true}) {
        a.src.flags = input ? GGML_TENSOR_FLAG_INPUT : 0;
        ggml_backend_sched_copy_input(&sched, &split, &a.src);
    }
    CHECK(trace.calls == 2 && a.cpu_state.waits == 2 && a.g->queue.waits == 0);
    a.g->queue.drain(); a.copied(5);
    // Cache the registry decision per backend, including absence. A registry
    // changing later is outside the scheduler's immutable backend lifetime.
    trace.offer = false;
    ggml_backend_sched absent;
    actual_constructor_discovery(&absent, backends, nullptr, 2);
    CHECK(trace.calls == 4 && absent.host_input_snapshot[0] == nullptr && absent.host_input_snapshot[1] == nullptr);
    trace.offer = true;
    absent.source = &a.cpu; absent.copy = &a.dst; a.src.flags = 0;
    ggml_backend_sched_copy_input(&absent, &split, &a.src);
    CHECK(trace.calls == 4 && a.g->queue.waits == 1);
    ggml_backend_sched parallel; parallel.n_copies = 4;
    const auto old_events = allocated_events.size();
    actual_constructor_discovery(&parallel, backends, nullptr, 2);
    CHECK(trace.calls == 4 && allocated_events.size() == old_events + 8);
    CHECK(parallel.host_input_snapshot[0] == nullptr && parallel.host_input_snapshot[1] == nullptr);
}
static void subsequent_snapshot_order_and_lifetime() {
    for (bool input : {false, true}) {
        Fixture f; std::vector<int> observed;
        f.src.flags = input ? GGML_TENSOR_FLAG_INPUT : 0;
        auto sched = single_scheduler(f); auto split = subsequent_input_split(f);
        snapshot_calls = 0;
        // This later-input fixture starts after the first input barrier. Check
        // snapshot lifetime and reader ordering; split backpressure is separate.
        f.g->queue.pending.push_back([&] { observed.push_back(f.b.buffer.p->bytes[8]); });
        for (int value : {17, 42, 79}) {
            f.cpu_state.complete = [&, value] { std::fill_n(f.host.begin() + 4, 16, value); };
            ggml_backend_sched_copy_input(&sched, &split, &f.src);
            std::fill(f.host.begin(), f.host.end(), 99);
            f.g->queue.pending.push_back([&] { observed.push_back(f.b.buffer.p->bytes[8]); });
        }
        CHECK(snapshot_calls == 3 && f.cpu_state.waits == 3 && f.g->queue.waits == 0 && f.g->queue.writes == 3);
        CHECK(observed.empty() && f.b.buffer.p->bytes[8] == 0xcc);
        f.host.clear(); f.host.shrink_to_fit();
        ggml_backend_synchronize(&f.gpu);
        CHECK(observed == std::vector<int>({0xcc, 17, 42, 79}));
        CHECK(f.g->queue.waits == 1); f.guard();
    }
    // Revalidation happens on each cached call, even for the same registry.
    { Fixture f; auto sched = single_scheduler(f); auto split = subsequent_input_split(f);
      ggml_backend_sched_copy_input(&sched, &split, &f.src);
      f.b.global_ctx = std::make_shared<global_context>(); snapshot_calls = 0;
      CHECK(!counted_snapshot(&f.cpu, &f.gpu, &f.src, &f.dst));
      CHECK(snapshot_calls == 1 && f.cpu_state.waits == 1 && f.g->queue.writes == 1); }
}
static void first_input_keeps_original_wait() {
    for (bool input : {false, true}) {
        Fixture f; f.src.flags = input ? GGML_TENSOR_FLAG_INPUT : 0;
        std::fill_n(f.host.begin() + 4, 16, 77);
        std::vector<int> readers;
        f.g->queue.pending.push_back([&] { readers.push_back(f.b.buffer.p->bytes[8]); });
        auto sched = single_scheduler(f); ggml_backend_sched_split split;
        split.n_inputs = 1; split.inputs[0] = &f.src; snapshot_calls = 0;
        ggml_backend_sched_copy_input(&sched, &split, &f.src);
        CHECK(snapshot_calls == 0 && f.g->queue.waits == 1);
        CHECK(f.cpu_state.waits == (input ? 0 : 1) && f.g->queue.writes == 1);
        CHECK(readers == std::vector<int>({0xcc}));
        CHECK(f.g->queue.pending.size() == 1 && f.g->queue.pending_snapshot_bytes == 16);
        std::fill(f.host.begin(), f.host.end(), 99);
        f.g->queue.pending.push_back([&] { readers.push_back(f.b.buffer.p->bytes[8]); });
        f.g->queue.drain();
        CHECK(readers == std::vector<int>({0xcc, 77}) && f.g->queue.pending_snapshot_bytes == 0);
        f.guard();
    }
}
static void repeated_splits_without_readback_keep_work_bounded() {
    constexpr int iterations = 256;
    for (bool first_is_input : {false, true}) for (bool later_is_input : {false, true}) {
        Fixture f; auto sched = single_scheduler(f);
        ggml_tensor first = f.src;
        first.flags = first_is_input ? GGML_TENSOR_FLAG_INPUT : 0;
        f.src.flags = later_is_input ? GGML_TENSOR_FLAG_INPUT : 0;
        ggml_backend_sched_split split;
        split.n_inputs = 2; split.inputs[0] = &first; split.inputs[1] = &f.src;
        const int previous_gets = gets;
        int completed_graphs = 0; snapshot_calls = 0;
        size_t peak_pending_commands = 0;
        for (int iteration = 0; iteration < iterations; ++iteration) {
            // A caller-owned completed first input preserves the old INPUT
            // contract. Ordinary first inputs still synchronize their producer.
            const uint8_t first_value = uint8_t(iteration % 100 + 1);
            const uint8_t later_value = uint8_t(iteration % 100 + 121);
            std::fill_n(f.host.begin() + 4, 16, first_value);
            sched.copy = &f.gpu_src;
            ggml_backend_sched_copy_input(&sched, &split, &first);
            // Its existing destination wait drains the previous submitted graph
            // and all prior WriteBuffer snapshots, even with n_outputs == 0.
            CHECK(completed_graphs == iteration && f.g->queue.waits == iteration + 1);
            CHECK(snapshot_calls == iteration && gets == previous_gets);
            CHECK(f.g->queue.pending.size() == 1 && f.g->queue.pending_snapshot_bytes == 16);
            f.cpu_state.complete = [&, later_value] { std::fill_n(f.host.begin() + 4, 16, later_value); };
            sched.copy = &f.dst;
            ggml_backend_sched_copy_input(&sched, &split, &f.src);
            CHECK(snapshot_calls == iteration + 1 && f.g->queue.waits == iteration + 1);
            CHECK(f.g->queue.pending.size() == 2 && f.g->queue.pending_snapshot_bytes == 32);
            std::fill(f.host.begin(), f.host.end(), 99);
            wgpu::CommandBuffer graph{[&, first_value, later_value] {
                for (int byte = 0; byte < 16; ++byte) {
                    CHECK(f.a.buffer.p->bytes[4 + byte] == first_value);
                    CHECK(f.b.buffer.p->bytes[8 + byte] == later_value);
                }
                ++completed_graphs;
            }};
            f.g->queue.Submit(1, &graph);
            peak_pending_commands = std::max(peak_pending_commands, f.g->queue.pending.size());
            CHECK(f.g->queue.pending.size() == 3 && f.g->queue.pending_snapshot_bytes == 32);
            CHECK(completed_graphs == iteration); // No readback, output wait, or implicit eager queue.
        }
        CHECK(f.g->queue.waits == iterations && snapshot_calls == iterations && gets == previous_gets);
        CHECK(f.cpu_state.waits == iterations * (first_is_input ? 1 : 2));
        CHECK(peak_pending_commands == 3 && f.g->queue.peak_snapshot_bytes == 32);
        // Fixture teardown only; the loop itself never reads back or explicitly
        // synchronizes, and always retains at most this split's 2 uploads+graph.
        f.g->queue.drain();
        CHECK(completed_graphs == iterations && f.g->queue.pending_snapshot_bytes == 0);
        f.guard();
    }
}
static void guarded_fallback_and_weights() {
    for (bool input : {false, true}) for (bool event : {false, true}) for (int copies : {1, 2}) {
        Fixture f; f.src.flags = input ? GGML_TENSOR_FLAG_INPUT : 0;
        auto sched = single_scheduler(f); sched.n_copies = copies;
        Event e{0, 0, &f.gpu}; if (event) sched.events[0][0] = &e;
        auto split = subsequent_input_split(f); snapshot_calls = 0;
        ggml_backend_sched_copy_input(&sched, &split, &f.src);
        const bool eligible = !event && copies == 1;
        CHECK(snapshot_calls == (eligible ? 1 : 0));
        CHECK(f.g->queue.waits == (eligible ? 0 : 1));
        CHECK(f.cpu_state.waits == (eligible || !input ? 1 : 0));
        CHECK(e.syncs == (event && input ? 1 : 0));
        CHECK(e.waits == (event && !input ? 1 : 0));
        f.g->queue.drain(); f.copied(5);
    }
    for (bool input : {false, true}) for (bool handled : {false, true}) for (bool view : {false, true}) {
        Fixture f; f.src.flags = input ? GGML_TENSOR_FLAG_INPUT : 0;
        f.hb.usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
        ggml_tensor parent = f.src;
        ggml_backend_buffer child_buffer = f.hb;
        child_buffer.usage = GGML_BACKEND_BUFFER_USAGE_ANY;
        if (view) { f.src.view_src = &parent; f.src.buffer = &child_buffer; }
        auto sched = single_scheduler(f); auto split = subsequent_input_split(f);
        struct Callback { int calls = 0; bool handled; } callback{0, handled};
        sched.callback_copy_user_data = &callback;
        sched.callback_copy = [](ggml_backend_t, ggml_tensor *, ggml_tensor *, ggml_cgraph *, void * user) {
            auto & callback = *static_cast<Callback *>(user); ++callback.calls; return callback.handled;
        };
        snapshot_calls = 0;
        // The provider excludes resolved WEIGHTS even when the scheduler's
        // unchanged upstream helper cannot see a view's parent buffer.
        CHECK(!counted_snapshot(&f.cpu, &f.gpu, &f.src, &f.dst));
        CHECK(f.cpu_state.waits == 0 && f.g->queue.waits == 0 && f.g->queue.writes == 0);
        snapshot_calls = 0;
        ggml_backend_sched_copy_input(&sched, &split, &f.src);
        CHECK(snapshot_calls == (view ? 1 : 0));
        CHECK(callback.calls == (!input && !view ? 1 : 0));
        const bool callback_handled = !input && !view && handled;
        CHECK(f.g->queue.waits == 1);
        CHECK(f.cpu_state.waits == (!input && !callback_handled ? 1 : 0));
        CHECK(f.g->queue.writes == (callback_handled ? 0 : 1));
        f.g->queue.drain(); if (!callback_handled) f.copied(5);
    }
    // Unsupported optional function has no side effect; the original INPUT
    // and ordinary paths retain their respective waits and copying semantics.
    for (bool input : {false, true}) {
        Fixture f; f.src.flags = input ? GGML_TENSOR_FLAG_INPUT : 0;
        auto sched = single_scheduler(f); sched.host_input_snapshot[0] = unsupported_snapshot;
        auto split = subsequent_input_split(f); snapshot_calls = 0;
        ggml_backend_sched_copy_input(&sched, &split, &f.src);
        CHECK(snapshot_calls == 1 && f.g->queue.waits == 1 && f.cpu_state.waits == (input ? 0 : 1));
        f.g->queue.drain(); f.copied(5);
    }
}
static void scheduler_valid_unsupported_and_cpu_destination() {
    for (bool input : {false, true}) for (int mode : {0, 1, 2}) {
        Fixture f; f.src.flags = input ? GGML_TENSOR_FLAG_INPUT : 0;
        if (mode == 0) f.dst.data = (void *)0x1009; // byte head
        if (mode == 1) f.src.nbytes = f.dst.nbytes = 15; // byte tail
        if (mode == 2) f.src.nbytes = f.dst.nbytes = 0;
        auto sched = single_scheduler(f); auto split = subsequent_input_split(f); snapshot_calls = 0;
        ggml_backend_sched_copy_input(&sched, &split, &f.src);
        CHECK(snapshot_calls == 1 && f.g->queue.waits == (input ? 1 : 2));
        CHECK(f.cpu_state.waits == (input ? 0 : 1));
        f.g->queue.drain();
        const size_t begin = mode == 0 ? 9 : 8, end = begin + f.src.nbytes;
        for (size_t i = begin; i < end; ++i) CHECK(f.b.buffer.p->bytes[i] == 5 + i - begin);
        CHECK(f.b.buffer.p->bytes[begin - 1] == 0xcc && f.b.buffer.p->bytes[end] == 0xcc);
        if (mode == 2) CHECK(f.g->queue.writes == 0);
    }
    { Fixture f; f.cpu_device.type = GGML_BACKEND_DEVICE_TYPE_GPU;
      f.cpu_state.complete = [&] { std::fill_n(f.host.begin() + 4, 16, 83); };
      auto sched = single_scheduler(f); auto split = subsequent_input_split(f); snapshot_calls = 0;
      ggml_backend_sched_copy_input(&sched, &split, &f.src);
      CHECK(snapshot_calls == 1 && f.cpu_state.waits == 1 && f.g->queue.waits == 2);
      f.g->queue.drain(); CHECK(f.b.buffer.p->bytes[8] == 83); }
    for (bool input : {false, true}) {
        Fixture f; f.src.flags = input ? GGML_TENSOR_FLAG_INPUT : 0;
        CpuState destination_state; ggml_backend cpu_destination = f.cpu; cpu_destination.context = &destination_state;
        ggml_backend_reg cpu_registry{{ggml_backend_cpu_get_proc_address}, nullptr}; f.cpu_device.reg = &cpu_registry;
        std::vector<uint8_t> host_destination(64, 0xcc);
        ggml_backend_buffer host_buffer{{host_base, host_set, host_get, nullptr}, &f.host_type, host_destination.data(), 64};
        ggml_tensor output{nullptr, &host_buffer, host_destination.data() + 8, 0, 16, 0, 0};
        ggml_backend_sched sched; ggml_backend_t backends[] = {&cpu_destination};
        actual_constructor_discovery(&sched, backends, nullptr, 1);
        CHECK(sched.host_input_snapshot[0] == nullptr);
        sched.source = &f.cpu; sched.copy = &output; auto split = subsequent_input_split(f);
        ggml_backend_sched_copy_input(&sched, &split, &f.src);
        CHECK(destination_state.waits == (input ? 1 : 2) && f.cpu_state.waits == (input ? 0 : 1));
        CHECK(f.g->queue.waits == 0 && f.g->queue.writes == 0);
        for (int i = 0; i < 16; ++i) CHECK(host_destination[8 + i] == i + 5);
    }
}
static void provider_boundary_and_failures() {
    // A view with no child buffer can be rejected safely by the provider;
    // do not send it through upstream generic copy, which requires buffer.
    { Fixture f; ggml_tensor parent = f.src; f.hb.usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
      f.src.view_src = &parent; f.src.buffer = nullptr;
      CHECK(!ggml_backend_webgpu_copy_host_input_snapshot_v1(&f.cpu, &f.gpu, &f.src, &f.dst));
      CHECK(f.cpu_state.waits == 0 && f.g->queue.waits == 0 && f.g->queue.writes == 0); }
    { Fixture f; f.cpu_device.type = GGML_BACKEND_DEVICE_TYPE_ACCEL;
      CHECK(!ggml_backend_webgpu_copy_host_input_snapshot_v1(&f.cpu, &f.gpu, &f.src, &f.dst));
      CHECK(f.cpu_state.waits == 0 && f.g->queue.waits == 0 && f.g->queue.writes == 0); }
    // The stronger registry contract never exports the GPU-to-GPU success path.
    { Fixture f; CHECK(!ggml_backend_webgpu_copy_host_input_snapshot_v1(&f.gpu, &f.gpu, &f.gpu_src, &f.dst));
      CHECK(f.g->queue.submissions == 0 && f.g->queue.writes == 0 && f.g->queue.waits == 0);
      CHECK(f.gpu_copy()); }
    { Fixture f;
      CHECK(!ggml_backend_webgpu_copy_host_input_snapshot_v1(nullptr, &f.gpu, &f.src, &f.dst));
      CHECK(!ggml_backend_webgpu_copy_host_input_snapshot_v1(&f.cpu, nullptr, &f.src, &f.dst));
      CHECK(!ggml_backend_webgpu_copy_host_input_snapshot_v1(&f.cpu, &f.gpu, nullptr, &f.dst));
      CHECK(!ggml_backend_webgpu_copy_host_input_snapshot_v1(&f.cpu, &f.gpu, &f.src, nullptr));
      CHECK(f.cpu_state.waits == 0 && f.g->queue.waits == 0 && f.g->queue.writes == 0); }
    for (bool input : {false, true}) for (bool producer : {false, true}) {
        Fixture f; f.src.flags = input ? GGML_TENSOR_FLAG_INPUT : 0;
        auto sched = single_scheduler(f); auto split = subsequent_input_split(f);
        f.cpu_state.fail = producer; f.g->queue.fail_write = !producer;
        throws([&] { ggml_backend_sched_copy_input(&sched, &split, &f.src); }, producer ? "cpu" : "write");
        CHECK(f.cpu_state.waits == 1 && f.g->queue.waits == 0 && f.g->queue.writes == 0);
        CHECK(f.g->queue.pending.empty());
    }
    // true does not claim completion: explicit readback and synchronization
    // still observe queued work and errors using production accessor bodies.
    { Fixture f; auto sched = single_scheduler(f); auto split = subsequent_input_split(f);
      ggml_backend_sched_copy_input(&sched, &split, &f.src); CHECK(f.b.buffer.p->bytes[8] == 0xcc);
      uint8_t result[16]{}; const int old_gets = gets;
      ggml_backend_webgpu_get_tensor_async(&f.gpu, &f.dst, result, 0, 16);
      CHECK(gets == old_gets + 1 && f.g->queue.waits == 0);
      for (int i = 0; i < 16; ++i) CHECK(result[i] == i + 5); }
    for (bool readback : {false, true}) {
        Fixture f; auto sched = single_scheduler(f); auto split = subsequent_input_split(f);
        ggml_backend_sched_copy_input(&sched, &split, &f.src); f.g->queue.fail_wait = true;
        throws([&] { if (readback) { uint8_t result[16]; ggml_backend_webgpu_get_tensor_async(&f.gpu, &f.dst, result, 0, 16); }
                      else ggml_backend_synchronize(&f.gpu); }, "queue");
    }
}
int main() {
    // Reuse the existing guard/range/view/CPU-lifetime cases, now executing
    // host_copy via the new stronger provider, without copying that harness.
    host_success_and_lifetime(); host_guards(); gpu_order_guards_and_fallback(); scheduler_and_cpu_mode(); errors();
    discovery_and_cache(); subsequent_snapshot_order_and_lifetime(); first_input_keeps_original_wait(); repeated_splits_without_readback_keep_work_bounded(); guarded_fallback_and_weights(); scheduler_valid_unsupported_and_cpu_destination(); provider_boundary_and_failures();
    std::cout << "PASS: " << checks << " source-bound snapshot/registry/cache/scheduler/queue/guard checks (CPU model only)\n";
}
