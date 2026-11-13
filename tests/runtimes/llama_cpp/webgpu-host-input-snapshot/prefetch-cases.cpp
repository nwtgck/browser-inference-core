// Actual prefetch/copy/input-loop bodies use modeled event and queue bookkeeping.
static void prefetch_and_snapshot_order() {
    {
        // Input zero is a host weight, but the actual input loop snapshots a
        // later normal input before prefetch reuses its host storage.
        Fixture f;
        f.ab.usage = GGML_BACKEND_BUFFER_USAGE_COMPUTE;
        auto sched = single_scheduler(f);
        sched.backends[1] = &f.cpu;
        sched.n_splits = 2;
        auto & current = sched.splits[0];
        auto & next = sched.splits[1];
        next.backend_id = 1;
        auto weight_buffer = f.hb;
        weight_buffer.usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
        auto weight = f.src;
        weight.buffer = &weight_buffer;
        auto weight_result = f.dst;
        ggml_tensor host_result{nullptr, &f.hb, f.src.data, 0, 16, 0, 0};
        current.n_inputs = 2;
        current.inputs[0] = &weight;
        current.inputs[1] = &f.src;
        next.n_inputs = 1;
        next.inputs[0] = &f.gpu_src;
        sched.backend_ids = {{&weight, 1}, {&f.src, 1}, {&f.gpu_src, 0}};
        sched.copies = {{&weight, &weight_result}, {&f.src, &f.dst}, {&f.gpu_src, &host_result}};
        f.gpu.iface.get_tensor_async = [](ggml_backend_t backend, const ggml_tensor * tensor, void * data,
                                          size_t offset, size_t size) {
            ggml_backend_webgpu_get_tensor_async(backend, tensor, data, offset, size);
            prefetch_trace.push_back("read-complete");
        };
        const auto expected = std::vector<uint8_t>(f.host.begin() + 4, f.host.begin() + 20);
        sched.callback_copy_user_data = &f;
        sched.callback_copy = [](ggml_backend_t, ggml_tensor *, ggml_tensor *, ggml_cgraph *, void * data) {
            auto & fixture = *static_cast<Fixture *>(data);
            CHECK(fixture.g->queue.waits == 1);
            CHECK(prefetch_trace == std::vector<std::string>({"read-complete", "record"}));
            CHECK(fixture.b.buffer.p->bytes[8] == 5);
            CHECK(fixture.host[4] == fixture.a.buffer.p->bytes[4]);
            prefetch_trace.push_back("weight-callback");
            return true;
        };
        prefetch_trace.clear();
        snapshot_calls = 0;
        actual_input_phase(&sched, 0);
        CHECK(snapshot_calls == 1 && sched.n_prefetched == 1);
        CHECK(prefetch_trace.back() == "weight-callback");
        CHECK(std::equal(expected.begin(), expected.end(), f.b.buffer.p->bytes.begin() + 8));
    }
    {
        Fixture f;
        f.ab.usage = GGML_BACKEND_BUFFER_USAGE_COMPUTE;
        auto sched = single_scheduler(f);
        sched.backends[1] = &f.cpu;
        sched.n_splits = 2;
        auto & current = sched.splits[0];
        auto & next = sched.splits[1];
        next.backend_id = 1;
        ggml_tensor later = f.src;
        later.data = f.host.data() + 24;
        later.nbytes = 8;
        ggml_tensor later_dst = f.dst;
        later_dst.data = (void *)0x1020;
        later_dst.nbytes = 8;
        ggml_tensor host_result{nullptr, &f.hb, later.data, 0, 16, 0, 0};
        ggml_tensor second = f.gpu_src;
        second.view_src = &f.gpu_src;
        second.data = (void *)0x1008;
        second.view_offs = 4;
        second.nbytes = 8;
        CHECK(second.buffer == second.view_src->buffer);
        CHECK((uintptr_t)second.data == (uintptr_t)second.view_src->data + second.view_offs);
        std::vector<uint8_t> second_bytes(16, 0xcc);
        ggml_tensor second_result{nullptr, &f.hb, second_bytes.data(), 0, 8, 0, 0};
        current.n_inputs = 2;
        current.inputs[0] = &f.src;
        current.inputs[1] = &later;
        next.n_inputs = 2;
        next.inputs[0] = &f.gpu_src;
        next.inputs[1] = &second;
        sched.backend_ids = {{&f.src, 1}, {&later, 1}, {&f.gpu_src, 0}, {&second, 0}};
        sched.copies = {{&f.src, &f.dst}, {&later, &later_dst}, {&f.gpu_src, &host_result}, {&second, &second_result}};
        f.gpu.iface.get_tensor_async = [](ggml_backend_t backend, const ggml_tensor * tensor, void * data,
                                          size_t offset, size_t size) {
            ggml_backend_webgpu_get_tensor_async(backend, tensor, data, offset, size);
            prefetch_trace.push_back("read-complete");
        };
        const auto expected_later = std::vector<uint8_t>(f.host.begin() + 24, f.host.begin() + 32);
        snapshot_calls = 0;
        prefetch_trace.clear();
        actual_input_phase(&sched, 0);
        CHECK(snapshot_calls == 1 && f.g->queue.waits == 1);
        CHECK(sched.n_prefetched == 2 && sched.prefetch_split == 1);
        CHECK(prefetch_trace == std::vector<std::string>({"read-complete", "read-complete", "record"}));
        CHECK(std::equal(expected_later.begin(), expected_later.end(), f.b.buffer.p->bytes.begin() + 32));
        CHECK(std::equal(f.a.buffer.p->bytes.begin() + 4, f.a.buffer.p->bytes.begin() + 20, f.host.begin() + 24));
        // The modeled current computation runs after the blocking prefetch.
        prefetch_trace.push_back("current-compute");
        const int previous_gets = gets;
        ggml_backend_sched_copy_input(&sched, &next, &f.gpu_src);
        ggml_backend_sched_copy_input(&sched, &next, &second);
        // Count scheduler API calls only; real WebGPU event recorded-state and
        // WaitAny elision after the first wait are outside this CPU event model.
        CHECK(gets == previous_gets && sched.prefetch_events[0]->syncs == 2);
        CHECK(prefetch_trace.back() == "current-compute");
        CHECK(std::equal(second_bytes.begin(), second_bytes.begin() + 8, f.a.buffer.p->bytes.begin() + 8));
    }
    // Each upstream exclusion is reached through the real prefetch body.
    for (int exclusion = 0; exclusion < 13; ++exclusion) {
        Fixture f;
        f.ab.usage = GGML_BACKEND_BUFFER_USAGE_COMPUTE;
        auto sched = single_scheduler(f);
        sched.backends[1] = &f.cpu;
        sched.n_splits = 2;
        auto & current = sched.splits[0];
        auto & next = sched.splits[1];
        next.backend_id = 1;
        next.n_inputs = 1;
        next.inputs[0] = &f.gpu_src;
        ggml_tensor result{nullptr, &f.hb, f.host.data() + 24, 0, 16, 0, 0};
        ggml_tensor writer = f.gpu_src;
        sched.backend_ids[&f.gpu_src] = 0;
        sched.copies[&f.gpu_src] = &result;
        f.gpu.iface.get_tensor_async = ggml_backend_webgpu_get_tensor_async;
        switch (exclusion) {
        case 0:
            f.gpu.iface.get_tensor_async = nullptr;
            break;
        case 1:
            deny_prefetch_event = true;
            break;
        case 2:
            sched.n_splits = 1;
            break;
        case 3:
            result.buffer = &f.bb;
            break;
        case 4:
            f.gpu_src.flags = GGML_TENSOR_FLAG_INPUT;
            break;
        case 5:
            sched.backend_ids[&f.gpu_src] = 1;
            break;
        case 6:
            current.graph.n_nodes = 1;
            current.graph.nodes[0] = &f.gpu_src;
            break;
        case 7:
            writer.view_src = &f.gpu_src;
            current.graph.n_nodes = 1;
            current.graph.nodes[0] = &writer;
            break;
        case 8:
            result.buffer = nullptr;
            break;
        case 9:
            f.ab.usage = GGML_BACKEND_BUFFER_USAGE_ANY;
            break;
        case 10:
            f.ab.usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
            break;
        case 11:
            f.gpu_src.buffer = nullptr;
            break;
        case 12:
            // Descriptor rejection only: this unallocated root must never
            // reach a getter or the scheduler's allocated-input copy path.
            writer.buffer = nullptr;
            f.gpu_src.view_src = &writer;
            break;
        }
        prefetch_trace.clear();
        const int previous_gets = gets;
        ggml_backend_sched_prefetch_inputs(&sched, 0);
        CHECK(sched.n_prefetched == 0 && gets == previous_gets && prefetch_trace.empty());
        deny_prefetch_event = false;
    }
    {
        Fixture f;
        f.ab.usage = GGML_BACKEND_BUFFER_USAGE_COMPUTE;
        auto sched = single_scheduler(f);
        sched.backends[1] = &f.cpu;
        sched.n_splits = 2;
        auto & next = sched.splits[1];
        next.backend_id = 1;
        next.n_inputs = 1;
        next.inputs[0] = &f.gpu_src;
        ggml_tensor result{nullptr, &f.hb, f.host.data() + 24, 0, 16, 0, 0};
        sched.backend_ids[&f.gpu_src] = 0;
        sched.copies[&f.gpu_src] = &result;
        f.gpu.iface.get_tensor_async = ggml_backend_webgpu_get_tensor_async;
        f.g->queue.pending.push_back([] { throw std::runtime_error("prefetch read failure"); });
        prefetch_trace.clear();
        throws([&] { ggml_backend_sched_prefetch_inputs(&sched, 0); }, "prefetch read failure");
        CHECK(sched.n_prefetched == 0 && prefetch_trace.empty());
    }
}

static void prefetch_excludes_callback_owned_buffers() {
    for (auto usage : {GGML_BACKEND_BUFFER_USAGE_ANY, GGML_BACKEND_BUFFER_USAGE_WEIGHTS}) {
        for (bool view : {false, true}) {
            Fixture f;
            f.ab.usage = usage;
            auto sched = single_scheduler(f);
            sched.backends[1] = &f.cpu;
            sched.n_splits = 2;
            auto weight_buffer = f.hb;
            weight_buffer.usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
            auto weight = f.src;
            weight.buffer = &weight_buffer;
            auto cached = f.gpu_src;
            if (view) {
                cached.view_src = &f.gpu_src;
                CHECK(cached.buffer == cached.view_src->buffer);
                CHECK(cached.data == cached.view_src->data);
            }
            ggml_tensor host_result{nullptr, &f.hb, f.host.data() + 24, 0, 16, 0, 0};
            sched.splits[0].n_inputs = 1;
            sched.splits[0].inputs[0] = &weight;
            sched.splits[1].backend_id = 1;
            sched.splits[1].n_inputs = 1;
            sched.splits[1].inputs[0] = &cached;
            sched.backend_ids = {{&weight, 1}, {&cached, 0}};
            sched.copies = {{&weight, &f.dst}, {&cached, &host_result}};
            f.gpu.iface.get_tensor_async = [](ggml_backend_t backend, const ggml_tensor * tensor, void * data,
                                              size_t offset, size_t size) {
                prefetch_trace.push_back("prefetch-read");
                ggml_backend_webgpu_get_tensor_async(backend, tensor, data, offset, size);
            };
            sched.callback_copy_user_data = &f;
            sched.callback_copy = [](ggml_backend_t, ggml_tensor *, ggml_tensor *, ggml_cgraph *, void * data) {
                auto & fixture = *static_cast<Fixture *>(data);
                std::fill_n(fixture.a.buffer.p->bytes.begin() + 4, 16, 0x91);
                prefetch_trace.push_back("cache-write");
                return true;
            };
            const int previous_gets = gets;
            prefetch_trace.clear();
            actual_input_phase(&sched, 0);
            CHECK(sched.n_prefetched == 0 && gets == previous_gets);
            CHECK(prefetch_trace == std::vector<std::string>({"cache-write"}));
            actual_input_phase(&sched, 1);
            CHECK(gets == previous_gets + 1);
            CHECK(std::all_of(f.host.begin() + 24, f.host.begin() + 40, [](uint8_t value) { return value == 0x91; }));
            f.guard();
        }
    }
}
