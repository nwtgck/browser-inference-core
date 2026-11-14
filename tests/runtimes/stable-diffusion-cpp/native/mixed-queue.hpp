#pragma once

namespace {
struct QueueKernel {
    size_t parameter_words;
    std::vector<std::pair<size_t, uint32_t>> fields;
};

// Public ggml graph construction selects the production encoder and builders:
// VIEW emits zero kernels; ARGSORT widths 256/512/1024/2048 emit 1/2/3/4.
// The SDK observer checks their command packets and deferred uniform ownership,
// without substituting an encoder or executing the numerical shader operators.
void graph_mixed_queue() {
    size_t cases = 0;
    for (bool passes : {false, true})
        for (size_t alignment : {128U, 256U, 512U})
            for (uint32_t batch : {1U, 3U, 8U})
                for (uint32_t inflight : {1U, 3U, 1000U}) {
                    auto global = global_context();
                    global->command_submit_batch_size = batch;
                    global->max_inflight_batches = inflight;
                    auto runtime = std::make_shared<webgpu_context_struct>();
                    runtime->global_ctx = global;
                    runtime->shader_lib = std::make_unique<ggml_webgpu_shader_lib>(global->device);
                    runtime->batch_compute_passes = passes;
                    runtime->param_arena.init(global->device, 128, batch + 10, alignment);
#ifdef GGML_WEBGPU_CHECK_SET_ROWS
                    // Exercise the actual optional SET_ROWS post-graph helper.
                    // Production configurations keep this diagnostic disabled.
                    ggml_webgpu_create_buffer(global->device, runtime->set_rows_dev_error_buf, sizeof(uint32_t),
                                              wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc, "set-errors");
                    ggml_webgpu_create_buffer(global->device, runtime->set_rows_host_error_buf, sizeof(uint32_t),
                                              wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst, "set-readback");
#endif
                    ggml_backend_webgpu_context wrapper{runtime, "mixed queue contract"};
                    ggml_backend backend{};
                    backend.context = &wrapper;
                    for (int workload = 0; workload < 4; ++workload) {
                        const auto metadata = std::unique_ptr<ggml_context, decltype(&ggml_free)>(
                            ggml_init({1024 * 1024, nullptr, true}), ggml_free);
                        SDC_CHECK(metadata != nullptr);
                        const auto buffer =
                            wgpu::Buffer::Acquire(new WGPUBufferImpl{1, std::vector<unsigned char>(128 * 1024), false});
                        ggml_backend_webgpu_buffer_context buffer_context(buffer, "mixed queue", global);
                        ggml_backend_buffer backing{};
                        backing.context = &buffer_context;
                        backing.size = buffer.GetSize();
                        size_t cursor = 0;
                        const auto bind = [&](ggml_tensor * tensor, bool scratch) {
                            tensor->data = reinterpret_cast<void *>(uintptr_t(webgpu_ptr_base) + cursor);
                            tensor->buffer = &backing;
                            // ARGSORT owns a second index array after its output.
                            cursor += GGML_PAD(ggml_nbytes(tensor), 256) * (scratch ? 2 : 1);
                            SDC_CHECK(cursor <= backing.size);
                        };
                        auto graph = ggml_new_graph_custom(metadata.get(), 32, false);
                        auto source = ggml_new_tensor_1d(metadata.get(), GGML_TYPE_F32, 4);
                        bind(source, false);
                        std::vector<QueueKernel> expected;
                        const auto append = [&](ggml_tensor * tensor) {
                            if (tensor->view_src)
                                SDC_CHECK(ggml_backend_view_init(tensor) == GGML_STATUS_SUCCESS);
                            ggml_build_forward_expand(graph, tensor);
                        };
                        const auto view = [&] { append(ggml_view_1d(metadata.get(), source, 4, 0)); };
                        const auto sort = [&](uint32_t kernels) {
                            const uint32_t width = 256U << (kernels - 1);
                            auto input = ggml_new_tensor_1d(metadata.get(), GGML_TYPE_F32, width);
                            bind(input, false);
                            auto output = ggml_argsort(metadata.get(), input, GGML_SORT_ORDER_ASC);
                            bind(output, true);
                            append(output);
                            expected.push_back({15, {{8, width}, {11, width}, {12, 256}, {13, width / 256}, {14, 1}}});
                            for (uint32_t step = 1; step < kernels; ++step) {
                                const uint32_t length = 256U << (step - 1);
                                expected.push_back(
                                    {19,
                                     {{12, width}, {15, width}, {16, length}, {17, width / (2 * length)}, {18, 1}}});
                            }
                        };
                        const auto set_rows = [&](bool empty) {
                            auto target = ggml_new_tensor_2d(metadata.get(), GGML_TYPE_F32, 4, 3);
                            auto input = ggml_new_tensor_2d(metadata.get(), GGML_TYPE_F32, 4, empty ? 0 : 2);
                            auto indices = ggml_new_tensor_1d(metadata.get(), GGML_TYPE_I32, empty ? 0 : 2);
                            for (auto tensor : {target, input, indices})
                                bind(tensor, false);
                            append(ggml_set_rows(metadata.get(), target, input, indices));
                            if (!empty)
                                expected.push_back({18, {{12, 4}, {13, 2}, {14, 1}, {15, 1}, {16, 1}, {17, 1}}});
                        };
                        if (workload == 1) {
                            // The destination is nonempty, but its empty source
                            // makes the real SET_ROWS encoder return no command.
                            set_rows(true);
                        } else {
                            view();
                            if (workload >= 2) {
                                sort(1);
                                sort(2);
                                if (workload == 3)
                                    set_rows(false); // SET_ROWS is node 3, as in the legacy mixed case.
                                view();
                                sort(3);
                                sort(4);
                                view();
                            }
                        }
                        SDC_CHECK(ggml_graph_n_nodes(graph) == (workload < 2 ? 1 : workload == 2 ? 7 : 8));
                        // Authored submission traces, independent of the
                        // production loop's kernel accumulation algorithm.
                        std::vector<size_t> submissions;
                        if (workload == 2)
                            submissions = batch == 1   ? std::vector<size_t>{1, 2, 3, 4}
                                          : batch == 3 ? std::vector<size_t>{3, 3, 4}
                                                       : std::vector<size_t>{10};
                        if (workload == 3)
                            submissions = batch == 1   ? std::vector<size_t>{1, 2, 1, 3, 4}
                                          : batch == 3 ? std::vector<size_t>{3, 4, 4}
                                                       : std::vector<size_t>{11};
                        const auto compute_submissions = submissions;
                        bool diagnostic = false;
#ifdef GGML_WEBGPU_CHECK_SET_ROWS
                        diagnostic = workload == 1 || workload == 3;
                        if (diagnostic)
                            submissions.push_back(0); // Actual error-buffer copy, not a compute dispatch.
#endif
                        auto & api = browser_api::state;
                        browser_api::reset_wait();
                        api.uniforms.clear();
                        api.dispatches.clear();
                        api.submission_dispatches.clear();
                        api.write_sizes.clear();
                        const auto before_submits = api.submissions;
                        const auto before_reads = api.reads;
                        const auto before_copies = api.copies;
                        SDC_CHECK(ggml_backend_webgpu_graph_compute(&backend, graph) == GGML_STATUS_SUCCESS);
                        browser_api::drain();
                        SDC_CHECK(api.submissions - before_submits == submissions.size());
                        SDC_CHECK(api.submission_dispatches == submissions);
                        const auto throttle_waits = submissions.empty() ? 0 : (submissions.size() - 1) / inflight;
                        SDC_CHECK(api.waits == throttle_waits + size_t(diagnostic));
                        SDC_CHECK(api.reads - before_reads == size_t(diagnostic));
                        SDC_CHECK(api.copies - before_copies == size_t(diagnostic));
                        SDC_CHECK(api.dispatches.size() == expected.size() && api.uniforms.size() == expected.size());
                        SDC_CHECK(runtime->param_arena.next_slot == 0);
                        std::vector<size_t> uploads;
#ifdef GGML_WEBGPU_BATCH_PARAM_UPLOADS
                        for (const auto kernels : compute_submissions)
                            uploads.push_back(kernels * alignment);
#else
                        for (const auto & kernel : expected)
                            uploads.push_back(kernel.parameter_words * sizeof(uint32_t));
#endif
                        SDC_CHECK(api.write_sizes == uploads);
                        for (size_t index = 0; index < expected.size(); ++index) {
                            const auto & words = api.uniforms[index];
                            SDC_CHECK(words.size() == 32);
                            for (const auto & [field, value] : expected[index].fields)
                                SDC_CHECK(words[field] == value);
#ifdef GGML_WEBGPU_BATCH_PARAM_UPLOADS
                            for (size_t field = expected[index].parameter_words; field < words.size(); ++field)
                                SDC_CHECK(words[field] == 0);
#endif
                        }
                        ++cases;
                    }
                }
    std::cout << "real graph zero/multi-kernel/SET_ROWS queue cases=" << cases << '\n';
}
} // namespace
