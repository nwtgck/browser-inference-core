#pragma once

namespace matrix_cases {
struct Case {
    ggml_type left;
    ggml_type right = GGML_TYPE_F32, output = GGML_TYPE_F32;
    int64_t k = 256, m = 64, n = 32, batch2 = 1, batch3 = 1;
    bool overlap = false, subgroup = false;
    std::optional<unsigned> expected_mode = std::nullopt;
    uint32_t expected_grid_x = 0;
    std::optional<unsigned> expected_small_m = std::nullopt;
    uint32_t shared_bytes = 0;
    bool packed_half = false;
};
static unsigned mode(const Case & c) {
    if (c.expected_mode)
        return *c.expected_mode;
    if (c.right != GGML_TYPE_F32 || c.output != GGML_TYPE_F32 || c.k <= 0 || c.n < 5 || c.overlap || c.subgroup)
        return 0;
    const bool q4 = (c.left == GGML_TYPE_Q4_0 && c.k % 32 == 0) || (c.left == GGML_TYPE_Q4_K && c.k % 256 == 0);
    if (c.left == GGML_TYPE_Q6_K && c.k % 256 == 0 && c.m >= 64 && c.m % 64 == 0 && c.n >= 32 && c.batch2 > 0 &&
        c.batch3 > 0) {
        // Independent division-based parallelism oracle: avoid reproducing
        // the production saturated-product implementation for large extents.
        uint64_t remaining = 8;
        for (const int64_t factor : {c.m / 64, c.n / 32 + (c.n % 32 != 0), c.batch2, c.batch3})
            remaining = remaining / uint64_t(factor) + (remaining % uint64_t(factor) != 0);
        return remaining == 1 ? 2 : 0;
    }
    return !q4 ? 0 : c.m >= 64 && c.m % 64 == 0 && c.n >= 32 ? 2 : 1;
}

static void exercise(const std::vector<Case> & inputs, const std::filesystem::path & output) {
    auto ctx = recording_context();
    std::map<std::tuple<ggml_type, ggml_type, bool, bool, bool, unsigned, uint32_t, uint32_t, uint32_t, uint32_t>,
             unsigned>
        cache;
    for (const auto & input : inputs) {
        ggml_tensor a{}, b{}, dst{};
        a.type = input.left;
        b.type = input.right;
        dst.type = input.output;
        a.ne[0] = b.ne[0] = input.k;
        a.ne[1] = dst.ne[0] = input.m;
        b.ne[1] = dst.ne[1] = input.n;
        a.ne[2] = a.ne[3] = 1;
        b.ne[2] = dst.ne[2] = input.batch2;
        b.ne[3] = dst.ne[3] = input.batch3;
        ggml_webgpu_shader_lib_context c{};
        c.src0 = &a;
        c.src1 = &b;
        c.dst = &dst;
        c.max_wg_size = 256;
        c.wg_mem_limit_bytes = input.shared_bytes;
        c.supports_subgroup_matrix = input.subgroup;
        c.min_subgroup_size = c.max_subgroup_size = 16;
        c.sg_mat_m = c.sg_mat_n = c.sg_mat_k = 8;
        const auto pipeline = ctx->shader_lib->get_mul_mat_fast_pipeline(c, input.overlap, true);
        BICORE_CHECK(ctx->shader_lib->get_mul_mat_fast_pipeline(c, input.overlap, true).pipeline.Get() ==
                     pipeline.pipeline.Get());
        const auto & d = *static_cast<const ggml_webgpu_mul_mat_shader_decisions *>(pipeline.context.get());
        const auto selected = mode(input);
        BICORE_CHECK((pipeline.name.find("_half_pairs") != std::string::npos) == (selected != 0));
        BICORE_CHECK((pipeline.name.find("_half_pairs_m8") != std::string::npos) == (selected == 2));
        BICORE_CHECK(d.use_subgroup_matrix == input.subgroup);
        BICORE_CHECK(pipeline.name.ends_with("_packed_k64") == input.packed_half);
        if (input.packed_half) {
            const bool one_column = input.expected_small_m == 4;
            BICORE_CHECK(d.tile_m == 4 && d.tile_n == (one_column ? 1u : 4u) && d.tile_k == 64 &&
                         d.wg_size == (one_column ? 32u : 64u));
        }
        if (selected) {
            BICORE_CHECK(d.tile_m == (selected == 2 ? 8u : 4u) && d.tile_n == 4 && d.tile_k == 32 && d.wg_size == 64);
            BICORE_CHECK(pipeline.pipeline.Get()->code.find(
                             "array<vec2<f16>, (TILE_SRC0_SHMEM + TILE_SRC1_SHMEM) / 2u>") != std::string::npos);
        }
        if (input.expected_small_m) {
            const auto expected = *input.expected_small_m;
            BICORE_CHECK((pipeline.name.find("_small_m") != std::string::npos) == (expected != 0));
            if (expected) {
                BICORE_CHECK(pipeline.name.ends_with("_small_m" + std::to_string(expected) +
                                                     (input.packed_half ? "_packed_k64" : "")));
                BICORE_CHECK(d.tile_m == expected && d.tile_n == 1 && d.tile_k == (input.packed_half ? 64u : 8u));
                BICORE_CHECK(d.wg_size_m == 1 && d.wg_size_n == 32 && d.wg_size == 32);
            } else if (input.n == 4) {
                BICORE_CHECK((production_dispatch(c, 65535, input.overlap, true) == std::array<uint32_t, 2>{1, 1}));
            }
        }
        const bool vectorized = input.packed_half || (input.k % 4 == 0 && input.m % 4 == 0 &&
                                                      (input.left == GGML_TYPE_F32 || input.left == GGML_TYPE_F16));
        const auto key = std::tuple{input.left, input.right, vectorized, input.subgroup, input.overlap,
                                    selected,   d.tile_m,    d.tile_n,   d.tile_k,       d.wg_size};
        const auto [position, inserted] = cache.emplace(key, pipeline.pipeline.Get()->id);
        BICORE_CHECK(inserted || position->second == pipeline.pipeline.Get()->id);
        for (const auto & item : cache)
            BICORE_CHECK(item.first == key || item.second != pipeline.pipeline.Get()->id);
        const bool valid_layout = input.k > 0 && input.k % ggml_blck_size(input.left) == 0 && input.m > 0 &&
                                  input.n >= 5 && input.batch2 > 0 && input.batch3 > 0;
        if (input.expected_grid_x) {
            BICORE_CHECK(valid_layout && !input.subgroup);
            const bool verify_pipeline =
                input.expected_small_m.has_value() || input.expected_mode.has_value() || input.shared_bytes != 0;
            const auto expected_matvec = verify_pipeline ? std::optional{false} : std::nullopt;
            BICORE_CHECK((production_dispatch(c, 65535, input.overlap, expected_matvec) ==
                          std::array<uint32_t, 2>{input.expected_grid_x, 1}));
        } else if (valid_layout && !input.subgroup && input.m < 1000 && input.n < 1000 && input.batch2 < 100 &&
                   input.batch3 < 100) {
            const uint64_t groups = ((input.m + d.tile_m * d.wg_size_m - 1) / (d.tile_m * d.wg_size_m)) *
                                    ((input.n + d.tile_n * d.wg_size_n - 1) / (d.tile_n * d.wg_size_n)) * input.batch2 *
                                    input.batch3;
            BICORE_CHECK(groups <= 65535);
            BICORE_CHECK(
                (production_dispatch(c, 65535, input.overlap) == std::array<uint32_t, 2>{uint32_t(groups), 1}));
        }
        if (!output.empty() && inserted) {
            std::ofstream shader(output / (pipeline.name + ".wgsl"));
            shader << pipeline.pipeline.Get()->code;
            BICORE_CHECK(bool(shader));
        }
    }
}

static void supported_type_controls(std::vector<Case> & cases) {
    auto ctx = recording_context();
    ctx->global_ctx->capabilities.limits.maxStorageBufferBindingSize = UINT64_MAX;
    ggml_backend_webgpu_device_context bridge;
    bridge.webgpu_global_ctx = ctx->global_ctx;
    ggml_backend_device device{};
    device.context = &bridge;
    // Query the real provider instead of parsing its source's switch labels.
    for (int index = 0; index < GGML_TYPE_COUNT; ++index) {
        const auto type = static_cast<ggml_type>(index);
        const auto * traits = ggml_get_type_traits(type);
        if (!traits->type_size || !traits->blck_size)
            continue;
        q4_current::Fixture f(type, traits->blck_size * 8, 64);
        f.x.ne[1] = f.dst.ne[1] = 32;
        f.dst.op = GGML_OP_MUL_MAT;
        f.dst.src[0] = &f.a;
        f.dst.src[1] = &f.x;
        for (auto * tensor : {&f.a, &f.x, &f.dst}) {
            tensor->nb[0] = ggml_type_size(tensor->type);
            tensor->nb[1] = ggml_row_size(tensor->type, tensor->ne[0]);
            tensor->nb[2] = tensor->nb[1] * tensor->ne[1];
            tensor->nb[3] = tensor->nb[2];
        }
        if (ggml_backend_webgpu_device_supports_op(&device, &f.dst))
            cases.push_back({type, GGML_TYPE_F32, GGML_TYPE_F32, f.a.ne[0]});
    }
}

static void q2_paired_prefill(const std::filesystem::path & output) {
    auto ctx = recording_context();
    auto & limits = ctx->global_ctx->capabilities.limits;
    limits.minStorageBufferOffsetAlignment = 256;
    limits.maxBufferSize = limits.maxStorageBufferBindingSize = 128 * 1024 * 1024;
    limits.maxComputeInvocationsPerWorkgroup = limits.maxComputeWorkgroupSizeX = 64;
    limits.maxComputeWorkgroupStorageSize = 6144;
    limits.maxComputeWorkgroupsPerDimension = 65535;
    ctx->batch_compute_passes = false;

    struct Shape {
        int64_t k, m;
        uint32_t scalar_grid, paired_grid;
    };
    constexpr Shape gate{5120, 17408, 2176, 1088};
    enum class Precision { automatic, accumulator, input };
    const auto run = [&](const Shape & shape, int64_t n, size_t prefix, bool overlap, bool paired, const char * capture,
                         Precision precision = Precision::automatic) {
        std::array<ggml_tensor, 3> tensors{};
        const std::array columns{shape.k, shape.k, shape.m};
        const std::array rows{shape.m, n, n};
        std::array<ggml_backend_buffer, 3> allocations{};
        std::vector<ggml_backend_webgpu_buffer_context> backing;
        backing.reserve(tensors.size());
        for (size_t i = 0; i < tensors.size(); ++i) {
            auto & tensor = tensors[i];
            tensor.type = i == 0 ? GGML_TYPE_Q2_0 : GGML_TYPE_F32;
            tensor.ne[0] = columns[i];
            tensor.ne[1] = rows[i];
            tensor.ne[2] = tensor.ne[3] = 1;
            tensor.nb[0] = ggml_type_size(tensor.type);
            tensor.nb[1] = ggml_row_size(tensor.type, tensor.ne[0]);
            tensor.nb[2] = tensor.nb[3] = tensor.nb[1] * tensor.ne[1];
            const size_t offset = i == 0 ? prefix : 0;
            auto * buffer = new WGPUBufferImpl;
            buffer->logical_size = ROUNDUP_POW2(offset + ggml_nbytes(&tensor), 4);
            buffer->usage = WGPUBufferUsage_Storage;
            backing.emplace_back(wgpu::Buffer::Acquire(buffer), "Q2 prefill operand", ctx->global_ctx);
            allocations[i].context = &backing.back();
            allocations[i].size = buffer->logical_size;
            tensor.data = reinterpret_cast<void *>(uintptr_t(webgpu_ptr_base) + offset);
            tensor.buffer = &allocations[i];
        }
        if (overlap) {
            tensors[1].buffer = tensors[0].buffer;
        }
        tensors[2].op = GGML_OP_MUL_MAT;
        if (precision == Precision::accumulator) {
            BICORE_CHECK(ggml_prec_set_acc(&tensors[2], GGML_PREC_F32));
        } else if (precision == Precision::input) {
            BICORE_CHECK(ggml_prec_set_src(&tensors[2], GGML_PREC_F32, 1));
        }
        ctx->param_arena.reset();
        ctx->active_command_encoder = ctx->global_ctx->device.CreateCommandEncoder();
        ctx->bind_group_cache.begin_graph(0);
        commands = {};
        BICORE_CHECK(ggml_webgpu_mul_mat(ctx, &tensors[0], &tensors[1], &tensors[2]).num_kernels == 1);
        ctx->bind_group_cache.end_graph();
        BICORE_CHECK(commands.dispatches.size() == 1 && ctx->bind_group_cache.slots.size() == 1);
        const auto & dispatch = commands.dispatches.front();
        const auto & slot = ctx->bind_group_cache.slots.front();
        const auto & pipeline = *slot.pipeline.Get();
        BICORE_CHECK(pipeline.id == dispatch.pipeline && pipeline.name.ends_with("_half_pairs_m8") == paired);
        const uint32_t groups = paired ? shape.paired_grid : shape.scalar_grid;
        BICORE_CHECK(dispatch.x <= limits.maxComputeWorkgroupsPerDimension && dispatch.z == 1);
        BICORE_CHECK(limits.maxComputeWorkgroupsPerDimension < groups ? dispatch.y > 1
                                                                      : dispatch.x == groups && dispatch.y == 1);
        ctx->param_arena.flush(ctx->global_ctx->queue);
        BICORE_CHECK(commands.writes.size() == 1 && commands.writes.front().bytes.size() >= 64);
        std::array<uint32_t, 16> params{};
        std::memcpy(params.data(), commands.writes.front().bytes.data(), sizeof(params));
        if (!overlap) {
            const size_t blocks_per_row = shape.k / 64;
            const size_t weight_bytes = prefix + shape.m * blocks_per_row * 18;
            BICORE_CHECK(slot.bindings.size() == 4 && slot.bindings[0].offset == 0 &&
                         slot.bindings[0].size == ROUNDUP_POW2(weight_bytes, 4));
            BICORE_CHECK(params[0] == prefix / 18 && params[6] == blocks_per_row);
        }
        const auto command = ctx->active_command_encoder.Finish();
        ctx->global_ctx->queue.Submit(1, &command);
        drain_queue(ctx->global_ctx->queue.Get());
        if (capture && !output.empty()) {
            std::ofstream(output / (std::string(capture) + ".wgsl")) << pipeline.code;
            std::ofstream bytecode(output / (std::string(capture) + ".params"), std::ios::binary);
            bytecode.write(reinterpret_cast<const char *>(params.data()), sizeof(params));
            BICORE_CHECK(bool(bytecode));
        }
        return pipeline.id;
    };

    const auto wide = run(gate, 128, 0, false, true, "q2-dense-wide");
    // The extra block puts the final four packed bytes across the last two
    // aligned words. The binding includes exactly two bytes of terminal pad.
    BICORE_CHECK(run(gate, 128, 18, false, true, nullptr) == wide);
    --limits.maxComputeWorkgroupSizeX;
    const auto scalar = run(gate, 128, 0, false, false, "q2-dense-stock");
    ++limits.maxComputeWorkgroupSizeX;
    --limits.maxComputeInvocationsPerWorkgroup;
    BICORE_CHECK(run(gate, 128, 0, false, false, nullptr) == scalar);
    ++limits.maxComputeInvocationsPerWorkgroup;
    --limits.maxComputeWorkgroupStorageSize;
    BICORE_CHECK(run(gate, 128, 0, false, false, nullptr) == scalar);
    ++limits.maxComputeWorkgroupStorageSize;
    limits.maxComputeWorkgroupsPerDimension = 1087;
    BICORE_CHECK(run(gate, 128, 0, false, false, nullptr) == scalar);
    limits.maxComputeWorkgroupsPerDimension = 65535;
    BICORE_CHECK(run(gate, 127, 0, false, false, nullptr) == scalar);
    BICORE_CHECK(run(gate, 128, 0, true, false, nullptr) != wide);
    BICORE_CHECK(run(gate, 128, 0, false, false, nullptr, Precision::accumulator) == scalar);
    BICORE_CHECK(run(gate, 128, 0, false, false, nullptr, Precision::input) == scalar);
    BICORE_CHECK(scalar != wide && run(gate, 128, 0, false, true, nullptr) == wide);

    const auto check_shape = [&](const Shape & shape, const char * wide_capture, const char * scalar_capture) {
        BICORE_CHECK(run(shape, 128, 0, false, true, wide_capture) == wide);
        --limits.maxComputeWorkgroupStorageSize;
        BICORE_CHECK(run(shape, 128, 0, false, false, scalar_capture) == scalar);
        ++limits.maxComputeWorkgroupStorageSize;
    };
    check_shape({17408, 5120, 640, 320}, "q2-dense-down-wide", "q2-dense-down-stock");
    check_shape({6144, 5120, 640, 320}, "q2-dense-out-wide", "q2-dense-out-stock");
    check_shape({5120, 10240, 1280, 640}, "q2-dense-qkv-wide", "q2-dense-qkv-stock");
    BICORE_CHECK(run({6144, 10240, 1280, 0}, 128, 0, false, false, nullptr) == scalar);
}

static void all(const std::filesystem::path & output) {
    q2_paired_prefill(output);
    std::vector<Case> cases;
    for (const auto type : {GGML_TYPE_Q4_0, GGML_TYPE_Q4_K}) {
        // Keep independent admission boundaries, then exercise interactions at
        // the cache/mode transitions below rather than a full Cartesian grid.
        for (int64_t k : {0, 31, 32, 33, 96, 255, 256, 257, 512, 2048})
            cases.push_back({type, GGML_TYPE_F32, GGML_TYPE_F32, k, 64, 32});
        for (int64_t m : {0, 1, 31, 32, 33, 63, 64, 65, 128})
            cases.push_back({type, GGML_TYPE_F32, GGML_TYPE_F32, 256, m, 32});
        for (int64_t n : {1, 4, 5, 31, 32, 33, 128})
            cases.push_back({type, GGML_TYPE_F32, GGML_TYPE_F32, 256, 64, n});
        cases.push_back({type, GGML_TYPE_F16});
        cases.push_back({type, GGML_TYPE_F32, GGML_TYPE_F16});
        cases.push_back({type, GGML_TYPE_F32, GGML_TYPE_F32, 256, 128, 33, 1, 1, true});
        cases.push_back({type, GGML_TYPE_F32, GGML_TYPE_F32, 256, 128, 33, 1, 1, false, true});
        for (int64_t batches : {1, 4})
            cases.push_back({type, GGML_TYPE_F32, GGML_TYPE_F32, 256, 33, 5, batches});
    }
    for (int64_t k : {0, 255, 256, 257, 512, 10752})
        cases.push_back({GGML_TYPE_Q6_K, GGML_TYPE_F32, GGML_TYPE_F32, k, 512, 32});
    for (int64_t m : {0, 63, 64, 65, 448, 512, 576})
        cases.push_back({GGML_TYPE_Q6_K, GGML_TYPE_F32, GGML_TYPE_F32, 256, m, 32});
    for (int64_t n : {4, 31, 32, 224, 225, 256, 257})
        cases.push_back({GGML_TYPE_Q6_K, GGML_TYPE_F32, GGML_TYPE_F32, 256, 64, n});
    for (const auto [batch2, batch3] : {std::pair{1, 1},
                                        {1, 7},
                                        {1, 8},
                                        {1, 9},
                                        {2, 3},
                                        {2, 4},
                                        {4, 2},
                                        {7, 1},
                                        {8, 1},
                                        {9, 1},
                                        {0, 8},
                                        {8, 0},
                                        {-1, 8},
                                        {8, -1}})
        cases.push_back({GGML_TYPE_Q6_K, GGML_TYPE_F32, GGML_TYPE_F32, 256, 64, 32, batch2, batch3});
    // The expected mode/grid are explicit boundaries, independent of the production predicate.
    const auto q8 = [](int64_t k, int64_t m, int64_t n, unsigned mode, uint32_t grid = 0) {
        return Case{GGML_TYPE_Q8_0, GGML_TYPE_F32, GGML_TYPE_F32, k, m, n, 1, 1, false, false, mode, grid};
    };
    cases.push_back(q8(5120, 10240, 128, 2, 640));
    cases.push_back(q8(5120, 10176, 128, 0, 1272));
    cases.push_back(q8(5120, 8192, 160, 2, 640));
    cases.push_back(q8(5120, 8192, 159, 0, 1280));
    cases.push_back(q8(5120, 10241, 128, 0, 1284));
    cases.push_back(q8(5088, 10240, 128, 0, 1280));
    cases.push_back(q8(5121, 10240, 128, 0));
    cases.push_back(q8(5120, 16384, 96, 0, 1536));
    cases.push_back(q8(5120, 10240, 4, 0));
    cases.push_back(q8(5120, 48, 128, 0, 8));
    cases.push_back(q8(2304, 1024, 128, 2, 64));
    cases.push_back(q8(2304, 3072, 128, 2, 192));
    cases.push_back(q8(3072, 2304, 128, 2, 144));
    cases.push_back(q8(3104, 2304, 128, 0, 288));
    cases.push_back(q8(3072, 2240, 128, 0, 280));
    cases.push_back(q8(3072, 2304, 160, 0, 360));
    cases.push_back(q8(2304, 384, 128, 1, 48));
    cases.push_back(q8(2304, 384, 127, 0, 48));
    cases.push_back(q8(2272, 384, 128, 0, 48));
    cases.push_back(q8(2304, 416, 128, 0, 52));
    cases.push_back(q8(2304, 1088, 128, 0, 136));
    cases.push_back(q8(2336, 3072, 128, 0, 384));
    cases.push_back(q8(2304, 3072, 129, 0, 480));
    cases.push_back(q8(2304, 3072, 64, 0, 192));
    cases.push_back(q8(10240, 320, 128, 1, 40));
    cases.push_back(q8(10208, 320, 128, 0, 40));
    cases.push_back(q8(10240, 384, 128, 0, 48));
    cases.push_back(q8(10240, 320, 127, 0, 40));
    auto long_k_batched = q8(10240, 320, 128, 0, 80);
    long_k_batched.batch2 = 2;
    cases.push_back(long_k_batched);
    auto long_k_excluded = q8(10240, 320, 128, 0);
    long_k_excluded.overlap = true;
    cases.push_back(long_k_excluded);
    long_k_excluded.overlap = false;
    long_k_excluded.subgroup = true;
    cases.push_back(long_k_excluded);
    auto short_batched = q8(2304, 3072, 128, 0, 768);
    short_batched.batch2 = 2;
    cases.push_back(short_batched);
    auto excluded = q8(5120, 10240, 128, 0);
    excluded.batch2 = 2;
    cases.push_back(excluded);
    excluded.batch2 = 1;
    excluded.batch3 = 2;
    cases.push_back(excluded);
    excluded.batch3 = 1;
    excluded.overlap = true;
    cases.push_back(excluded);
    excluded.overlap = false;
    excluded.subgroup = true;
    cases.push_back(excluded);
    excluded.subgroup = false;
    excluded.right = GGML_TYPE_F16;
    cases.push_back(excluded);
    excluded.right = GGML_TYPE_F32;
    excluded.output = GGML_TYPE_F16;
    cases.push_back(excluded);
    cases.push_back(q8(5120, 10240, 128, 2));
    supported_type_controls(cases);
    for (const auto right : {GGML_TYPE_F32, GGML_TYPE_F16})
        for (const auto result : {GGML_TYPE_F32, GGML_TYPE_F16})
            cases.push_back({GGML_TYPE_Q6_K, right, result, 256, 512, 32});
    cases.push_back({GGML_TYPE_Q6_K, GGML_TYPE_F32, GGML_TYPE_F32, 256, 512, 32, 1, 1, true});
    cases.push_back({GGML_TYPE_Q6_K, GGML_TYPE_F32, GGML_TYPE_F32, 256, 512, 32, 1, 1, false, true});
    for (int64_t batches : {INT64_MAX, int64_t(0), int64_t(-1)})
        cases.push_back(
            {GGML_TYPE_Q6_K, GGML_TYPE_F32, GGML_TYPE_F32, 256, INT64_MAX - 63, INT64_MAX, INT64_MAX, batches});
    for (int64_t m : {1, 4, 64, 65})
        cases.push_back({GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, 256, m, 32});
    const auto bf16 = [](int64_t k, int64_t m, int64_t n, unsigned tile, uint32_t grid = 0) {
        return Case{GGML_TYPE_BF16, GGML_TYPE_F32, GGML_TYPE_F32, k, m, n, 1, 1, false, false, 0, grid, tile};
    };
    cases.push_back(bf16(10240, 4, 4, 0));
    cases.push_back(bf16(10240, 4, 5, 4, 1));
    cases.push_back(bf16(10240, 4, 32, 4, 1));
    cases.push_back(bf16(10240, 4, 96, 4, 3));
    cases.push_back(bf16(10240, 4, 109, 4, 4));
    cases.push_back(bf16(10240, 4, 128, 4, 4));
    cases.push_back(bf16(10240, 4, 129, 0, 5));
    cases.push_back(bf16(10240, 3, 128, 0, 4));
    cases.push_back(bf16(10240, 5, 128, 0, 4));
    cases.push_back(bf16(10239, 4, 128, 0, 4));
    cases.push_back(bf16(10241, 4, 128, 0, 4));
    cases.push_back(bf16(2560, 1, 128, 0, 4));
    auto bf16_fallback = bf16(10240, 4, 128, 0);
    bf16_fallback.batch2 = 2;
    cases.push_back(bf16_fallback);
    bf16_fallback.batch2 = 1;
    bf16_fallback.batch3 = 2;
    cases.push_back(bf16_fallback);
    bf16_fallback.batch3 = 1;
    bf16_fallback.overlap = true;
    cases.push_back(bf16_fallback);
    bf16_fallback.overlap = false;
    bf16_fallback.subgroup = true;
    cases.push_back(bf16_fallback);
    bf16_fallback.subgroup = false;
    bf16_fallback.right = GGML_TYPE_F16;
    cases.push_back(bf16_fallback);
    bf16_fallback.right = GGML_TYPE_F32;
    bf16_fallback.output = GGML_TYPE_F16;
    cases.push_back(bf16_fallback);
    const auto f32 = [](int64_t k, int64_t m, int64_t n, uint32_t shared, bool packed) {
        Case test{GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, k, m, n};
        test.shared_bytes = shared;
        test.packed_half = packed;
        test.expected_grid_x = uint32_t((m + 31) / 32 * ((n + 31) / 32));
        return test;
    };
    cases.push_back(f32(4608, 64, 19, 8191, false));
    cases.push_back(f32(4608, 64, 19, 8192, true));
    cases.push_back(f32(4608, 64, 16, 8192, true));
    cases.push_back(f32(4608, 64, 33, 8192, true));
    cases.push_back(f32(4608, 64, 128, 8192, true));
    cases.push_back(f32(4608, 64, 15, 8192, false));
    cases.push_back(f32(4608, 64, 129, 8192, false));
    cases.push_back(f32(4604, 64, 64, 8192, false));
    cases.push_back(f32(4608, 68, 64, 8192, false));
    cases.push_back(f32(2304, 128, 128, 8192, true));
    cases.push_back(f32(2048, 256, 128, 8192, true));
    cases.push_back(f32(2560, 512, 128, 8192, true));
    cases.push_back(f32(2560, 48, 64, 8192, true));
    cases.push_back(f32(5120, 48, 128, 8192, true));
    cases.push_back(f32(2560, 52, 64, 8192, false));
    const auto four_rows = [&](int64_t k, int64_t n, uint32_t shared, bool packed) {
        auto test = f32(k, 4, n, shared, packed);
        test.expected_small_m = 4;
        return test;
    };
    cases.push_back(four_rows(10240, 19, 4607, false));
    cases.push_back(four_rows(10240, 19, 4608, true));
    cases.push_back(four_rows(10240, 128, 4608, true));
    cases.push_back(four_rows(10240, 129, 4608, false));
    cases.push_back(four_rows(10236, 128, 4608, false));
    const auto bf16_half = [&](int64_t k, int64_t m, int64_t n, uint32_t shared, bool packed) {
        auto test = m == 4 ? four_rows(k, n, shared, packed) : f32(k, m, n, shared, packed);
        test.left = GGML_TYPE_BF16;
        return test;
    };
    cases.push_back(bf16_half(10240, 4, 19, 4607, false));
    cases.push_back(bf16_half(10240, 4, 19, 4608, true));
    cases.push_back(bf16_half(10240, 4, 128, 4608, true));
    cases.push_back(bf16_half(10240, 4, 15, 4608, false));
    cases.push_back(bf16_half(10240, 320, 128, 8191, false));
    cases.push_back(bf16_half(10240, 320, 128, 8192, true));
    cases.push_back(bf16_half(10240, 316, 128, 8192, false));
    cases.push_back(bf16_half(2560, 128, 64, 8192, true));
    cases.push_back(bf16_half(2560, 48, 64, 8192, true));
    cases.push_back(bf16_half(2560, 512, 128, 8192, true));
    auto packed_fallback = f32(4608, 64, 64, 8192, false);
    packed_fallback.overlap = true;
    cases.push_back(packed_fallback);
    packed_fallback.overlap = false;
    packed_fallback.batch2 = 2;
    packed_fallback.expected_grid_x = 8;
    cases.push_back(packed_fallback);
    packed_fallback.batch2 = 1;
    packed_fallback.expected_grid_x = 0;
    packed_fallback.subgroup = true;
    cases.push_back(packed_fallback);
    for (const auto & transition : std::vector<Case>{{GGML_TYPE_Q4_K, GGML_TYPE_F32, GGML_TYPE_F32, 256, 64, 31},
                                                     {GGML_TYPE_Q4_K, GGML_TYPE_F32, GGML_TYPE_F32, 256, 64, 32},
                                                     {GGML_TYPE_Q4_K, GGML_TYPE_F32, GGML_TYPE_F32, 256, 65, 32},
                                                     {GGML_TYPE_Q4_K, GGML_TYPE_F32, GGML_TYPE_F32, 255, 64, 32},
                                                     {GGML_TYPE_Q4_K, GGML_TYPE_F32, GGML_TYPE_F32, 256, 64, 32}})
        cases.push_back(transition);
    exercise(cases, output);
    std::reverse(cases.begin(), cases.end());
    exercise(cases, {});
    using Key = ggml_webgpu_mul_mat_pipeline_key;
    Key scalar{}, paired{}, wide{};
    paired.shared_mode = Key::quant_shared_mode::half_pairs;
    wide.shared_mode = Key::quant_shared_mode::half_pairs_m8;
    BICORE_CHECK(!(scalar == paired) && !(scalar == wide) && !(paired == wide));
    std::cout << "matrix factory/dispatch cases passed: " << cases.size() << " in each order\n";
}
} // namespace matrix_cases
