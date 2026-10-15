// Workloads are synthetic, but dispatch parameter allocation/staging, binding,
// the graph loop and queue throttling below are the actual prepared functions.
static std::optional<webgpu_encoded_op> ggml_webgpu_encode(webgpu_context& ctx,
                                                          ggml_cgraph* graph, int index, int&) {
    auto node = graph->nodes[index];
    if (!node->kernels) return {};
    std::vector<webgpu_dispatch_desc> dispatches;
    for (int j = 0; j < node->kernels; ++j) {
        webgpu_dispatch_desc dispatch;
        dispatch.params.resize((node->seed + j) % 33, node->seed * 17 + j + 1);
        dispatch.pipeline.name = "synthetic-dispatch";
        dispatch.pipeline.pipeline.expected.resize(dispatch.params.size() * sizeof(uint32_t));
        if (!dispatch.params.empty()) {
            std::memcpy(dispatch.pipeline.pipeline.expected.data(), dispatch.params.data(),
                        dispatch.pipeline.pipeline.expected.size());
        }
        dispatches.push_back(std::move(dispatch));
    }
    auto result = ggml_backend_webgpu_build_multi(ctx, dispatches);
    // Catch deferred reads of a parameter vector after it was overwritten/freed.
    for (auto& dispatch : dispatches) std::fill(dispatch.params.begin(), dispatch.params.end(), 0xdeadbeefu);
    return result;
}
