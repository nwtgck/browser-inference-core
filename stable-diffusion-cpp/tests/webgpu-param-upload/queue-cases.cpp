int main(int argc, char** argv) {
    if (argc > 1) {
        webgpu_param_arena arena; arena.init({}, 128, 74, 256); uint32_t value = 1;
        int mode = std::atoi(argv[1]);
        if (mode == 1) arena.alloc_slot(132);
        if (mode == 2) for (int i = 0; i < 75; ++i) arena.alloc_slot(4);
#ifdef GGML_WEBGPU_BATCH_PARAM_UPLOADS
        if (mode == 3) arena.stage(0, &value, 4);
        arena.alloc_slot(4);
        if (mode == 4) arena.stage(1, &value, 4);
        if (mode == 5) arena.stage(0, &value, 132);
        if (mode == 6) { arena.upload_data.resize(8); arena.stage(0, &value, 4); }
#else
        (void)value;
#endif
        return 0;
    }
    size_t cases = 0;
    for (bool passes : {false, true}) for (size_t alignment : {128u, 256u, 512u})
        for (uint32_t batch_size : {1u, 8u, 64u}) for (uint32_t inflight : {1u, 3u, 1000u}) {
        auto ctx = std::make_shared<context>();
        ctx->global_ctx = std::make_shared<global_context>();
        ctx->global_ctx->command_submit_batch_size = batch_size;
        ctx->global_ctx->max_inflight_batches = inflight;
        ctx->batch_compute_passes = passes;
        ctx->param_arena.init({}, 128, batch_size + 10, alignment);
        ggml_backend_webgpu_context wrapper{ctx}; backend backend{&wrapper};
        auto& queue = ctx->global_ctx->queue;
#ifdef GGML_WEBGPU_BATCH_PARAM_UPLOADS
        ctx->param_arena.flush(queue);
        assert(queue.writes == 0);
        assert(ctx->param_arena.upload_data.size() == (batch_size + 10) * alignment);
#endif
        size_t expected_kernels = 0, expected_submits = 0, expected_waits = 0;
        for (int count : {0, 1, 63, 64, 65, 128, 129, 300}) {
            std::vector<ggml_tensor> nodes;
            std::vector<ggml_tensor*> pointers;
            size_t batch = 0, submits = 0;
            for (int i = 0; i < count; ++i) {
                int kernels = i % 7 == 0 ? 0 : i % 4 + 1;
                nodes.push_back({i == 3 ? GGML_OP_SET_ROWS : 1, kernels, uint32_t(count * 23 + i * 17)});
                expected_kernels += kernels; batch += kernels;
                if (batch >= batch_size) { ++submits; batch = 0; }
            }
            if (batch) ++submits;
            expected_submits += submits;
            expected_waits += submits ? (submits - 1) / inflight : 0;
            for (auto& node : nodes) pointers.push_back(&node);
            ggml_cgraph graph{count, pointers.data()};
            assert(ggml_backend_webgpu_graph_compute(&backend, &graph) == GGML_STATUS_SUCCESS);
            assert(ctx->param_arena.next_slot == 0);
            assert(queue.submits == expected_submits && queue.waits == expected_waits);
            ++cases;
        }
        assert(queue.kernels == expected_kernels);
#ifdef GGML_WEBGPU_BATCH_PARAM_UPLOADS
        assert(queue.writes == queue.submits);
#else
        assert(queue.writes == expected_kernels);
#endif
        queue.drain();
        std::printf("passes=%d alignment=%zu batch=%u inflight=%u kernels=%zu submits=%zu writes=%zu bytes=%zu waits=%zu passed\n",
                    passes, alignment, batch_size, inflight, expected_kernels,
                    queue.submits, queue.writes, queue.uploaded_bytes, queue.waits);
    }
    assert(checks == 324);
    std::printf("actual-graph cases=%zu passed\n", cases);
}
