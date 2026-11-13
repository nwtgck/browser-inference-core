// New flags are controlled scheduler bookkeeping. Both loops and copy_input
// are extracted from production; this does not execute split_graph allocation.
static void requires_copy_bound_case(const char * name, bool first_required, bool first_weight, bool first_input,
                                     bool later_required, bool empty_prefetch) {
    constexpr int iterations = 128;
    Fixture f;
    auto sched = single_scheduler(f);
    auto & split = sched.splits[0];
    sched.n_splits = empty_prefetch ? 2 : 1;
    ggml_backend_buffer weight_buffer = f.hb;
    weight_buffer.usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
    ggml_tensor first = f.src;
    first.buffer = first_weight ? &weight_buffer : &f.hb;
    first.data = f.host.data() + 24;
    first.nbytes = 8;
    first.flags = first_input ? GGML_TENSOR_FLAG_INPUT : 0;
    ggml_tensor first_dst = f.gpu_src;
    first_dst.nbytes = 8;
    f.src.flags = GGML_TENSOR_FLAG_INPUT;
    split.n_inputs = 2;
    split.inputs[0] = &first;
    split.inputs[1] = &f.src;
    sched.copies = {{&first, &first_dst}, {&f.src, &f.dst}};
    sched.hv_tensor_requires_copy[&first] = first_required;
    sched.hv_tensor_requires_copy[&f.src] = later_required;
    if (empty_prefetch) {
        // The actual prefetch helper returns because this backend has no async
        // getter, even though another split exists. No extra read/wait helps.
        sched.splits[1].backend_id = 1;
        sched.backends[1] = &f.cpu;
        f.gpu.iface.get_tensor_async = nullptr;
    }
    int weight_callbacks = 0;
    sched.callback_copy_user_data = &weight_callbacks;
    sched.callback_copy = [](ggml_backend_t, ggml_tensor *, ggml_tensor *, ggml_cgraph *, void * data) {
        ++*static_cast<int *>(data);
        return false;
    };
    int completed = 0;
    size_t peak_commands = 0;
    snapshot_calls = 0;
    const int before_gets = gets;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const auto first_value = uint8_t(iteration % 80 + 1);
        const auto later_value = uint8_t(iteration % 80 + 121);
        std::fill_n(f.host.begin() + 24, 8, first_value);
        std::fill_n(f.host.begin() + 4, 16, later_value);
        actual_input_phase(&sched, 0);
        wgpu::CommandBuffer graph{[&, first_value, later_value] {
            if (first_required) {
                CHECK(std::all_of(f.a.buffer.p->bytes.begin() + 4, f.a.buffer.p->bytes.begin() + 12,
                                  [=](uint8_t v) { return v == first_value; }));
            }
            if (later_required) {
                CHECK(std::all_of(f.b.buffer.p->bytes.begin() + 8, f.b.buffer.p->bytes.begin() + 24,
                                  [=](uint8_t v) { return v == later_value; }));
            }
            ++completed;
        }};
        f.g->queue.Submit(1, &graph);
        peak_commands = std::max(peak_commands, f.g->queue.pending.size());
        std::fill(f.host.begin(), f.host.end(), 0xfe);
    }
    const auto outstanding_bytes = f.g->queue.pending_snapshot_bytes;
    const int completed_before_drain = completed;
    const bool transfers_first = first_required;
    const bool transfers_later = later_required;
    CHECK(f.g->queue.writes == iterations * (int(transfers_first) + int(transfers_later)));
    CHECK(weight_callbacks == (first_weight && transfers_first ? iterations : 0));
    CHECK(gets == before_gets && sched.n_prefetched == 0);
    const bool all_skipped = !transfers_first && !transfers_later;
    if (all_skipped) {
        // No retained upload bytes exist to bound. This is upstream's graph
        // submission behavior, not evidence that the snapshot fix adds a wait.
        CHECK(f.g->queue.waits == 0 && f.g->queue.peak_snapshot_bytes == 0);
        CHECK(peak_commands == size_t(iterations) && snapshot_calls == 0);
    } else {
        CHECK(f.g->queue.waits == iterations && f.g->queue.peak_snapshot_bytes <= 24);
        CHECK(outstanding_bytes <= 24 && peak_commands <= 3 && completed == iterations - 1);
        CHECK(snapshot_calls == (transfers_later && (first_required) ? iterations : 0));
    }
    std::cout << "{\"case\":\"" << name << "\",\"iterations\":" << iterations << ",\"gpuWaits\":" << f.g->queue.waits
              << ",\"writes\":" << f.g->queue.writes << ",\"snapshots\":" << snapshot_calls
              << ",\"peakSnapshotBytes\":" << f.g->queue.peak_snapshot_bytes
              << ",\"peakPendingCommands\":" << peak_commands << ",\"completedBeforeDrain\":" << completed_before_drain
              << ",\"weightCallbacks\":" << weight_callbacks << "}\n";
    f.g->queue.drain();
    CHECK(completed == iterations && f.g->queue.pending_snapshot_bytes == 0);
    f.guard();
}

static void prefetched_input_zero(bool required) {
    Fixture f;
    f.ab.usage = GGML_BACKEND_BUFFER_USAGE_COMPUTE;
    auto sched = single_scheduler(f);
    sched.backends[1] = &f.cpu;
    sched.n_splits = 2;
    auto & next = sched.splits[1];
    next.backend_id = 1;
    next.n_inputs = 1;
    next.inputs[0] = &f.gpu_src;
    ggml_tensor host_result{nullptr, &f.hb, f.host.data() + 24, 0, 16, 0, 0};
    sched.backend_ids[&f.gpu_src] = 0;
    sched.copies[&f.gpu_src] = &host_result;
    sched.hv_tensor_requires_copy[&f.gpu_src] = required;
    f.gpu.iface.get_tensor_async = ggml_backend_webgpu_get_tensor_async;
    const auto old_gets = gets;
    actual_input_phase(&sched, 0);
    CHECK(gets == old_gets + 1 && sched.n_prefetched == 1);
    const auto old_syncs = sched.prefetch_events[0]->syncs;
    actual_input_phase(&sched, 1);
    const int expected_waits = required ? 1 : 0;
    CHECK(sched.prefetch_events[0]->syncs - old_syncs == expected_waits && gets == old_gets + 1);
    std::cout << "{\"case\":\"prefetched0-" << (required ? "required" : "unused")
              << "\",\"prefetchGets\":1,\"copyPhaseEventSyncs\":" << expected_waits << "}\n";
    // The inherited CPU event mock synchronizes the full backend. This case
    // checks control flow only, never real event-prefix backpressure.
}

static void requires_copy_case_matrix() {
    requires_copy_bound_case("active0-host", true, false, false, true, false);
    requires_copy_bound_case("active0-input", true, false, true, true, false);
    requires_copy_bound_case("inactive0-host", false, false, false, true, false);
    requires_copy_bound_case("inactive0-input", false, false, true, true, false);
    requires_copy_bound_case("active0-hostweight", true, true, false, true, false);
    requires_copy_bound_case("inactive0-hostweight", false, true, false, true, false);
    requires_copy_bound_case("inactive0-no-async-prefetch", false, false, false, true, true);
    requires_copy_bound_case("all-inactive", false, false, false, false, false);
    prefetched_input_zero(true);
    prefetched_input_zero(false);
}
