static webgpu_global_context allocation_global;
static std::vector<std::shared_ptr<wgpu::Data>> allocated;
static void free_memory(ggml_backend_buffer_t b) {
    delete static_cast<ggml_backend_webgpu_buffer_context *>(b->context);
}
static ggml_backend_buffer_t make_memory(ggml_backend_buffer_type_t t, size_t n) {
    auto data = std::make_shared<wgpu::Data>(n);
    allocated.push_back(data);
    ggml_backend_buffer_i i{};
    i.free_buffer = free_memory;
    i.get_base = ggml_backend_webgpu_buffer_get_base;
    return ggml_backend_buffer_init(t, i, new ggml_backend_webgpu_buffer_context{{data}, allocation_global}, n);
}
static const char * memory_name(ggml_backend_buffer_type_t) {
    return "tile-bound-recorded-memory";
}
static size_t memory_alignment(ggml_backend_buffer_type_t) {
    return 256;
}
static size_t memory_max(ggml_backend_buffer_type_t) {
    return 1ull << 26;
}
static wgpu::Buffer tensor_buffer(ggml_tensor * t) {
    return static_cast<ggml_backend_webgpu_buffer_context *>(t->buffer->context)->buffer;
}
static uint8_t * at(wgpu::Buffer b, size_t off, size_t n) {
    check(off <= b.GetSize() && n <= b.GetSize() - off, "physical range");
    return b.data->bytes.data() + 64 + off;
}
static void fill_tensor(ggml_tensor * t, const void * data) {
    std::memcpy(at(tensor_buffer(t), ggml_webgpu_tensor_offset(t), ggml_nbytes(t)), data, ggml_nbytes(t));
}
struct Spec {
    int e, u, n, m, mode;
};
struct Result {
    std::vector<float> output;
    std::vector<std::vector<uint32_t>> params, grids;
    uint32_t capacity;
    size_t allocation, descriptor;
};
static Result run(bool changed, Spec s, int epoch) {
    allocated.clear();
    auto global = std::make_shared<Global>();
    allocation_global = global;
    ggml_backend_webgpu_device_context devctx{global};
    ggml_backend_device device{};
    device.context = &devctx;
    ggml_backend_buffer_type type{};
    type.device = &device;
    type.iface.get_name = memory_name;
    type.iface.get_alignment = memory_alignment;
    type.iface.get_max_size = memory_max;
    type.iface.alloc_buffer = make_memory;
    type.iface.get_alloc_size = changed ? candidate::ggml_backend_webgpu_buffer_type_get_alloc_size
                                        : baseline::ggml_backend_webgpu_buffer_type_get_alloc_size;
    auto * ctx = ggml_init({1024 * 1024, nullptr, true});
    check(ctx != nullptr, "real ggml context");
    auto * w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, s.m, s.e);
    auto * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 1, s.n);
    auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, s.u, s.n);
    auto * out = ggml_mul_mat_id(ctx, w, x, ids);
    ggml_set_output(out);
    auto * graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, out);
    auto * allocator = ggml_gallocr_new(&type);
    check(ggml_gallocr_alloc_graph(allocator, graph), "real gallocr allocation");
    const int slots = s.u * s.n, a = std::min(s.e, slots);
    std::vector<int32_t> routing(size_t(slots), 0);
    for (int i = slots - a + 1; i < slots; ++i)
        routing[size_t(i)] = i - (slots - a);
    if (s.mode == 1)
        routing[0] = -1;
    if (s.mode == 2)
        std::fill(routing.begin(), routing.end(), -1);
    if (epoch)
        for (auto & id : routing)
            if (id >= 0)
                id = s.e - 1 - id;
    std::vector<float> weights(size_t(s.e * s.m * 32)), input(size_t(s.n * 32));
    for (int e = 0; e < s.e; ++e)
        for (int r = 0; r < s.m; ++r)
            for (int k = 0; k < 32; ++k)
                weights[size_t((e * s.m + r) * 32 + k)] = float((e * 3 + r * 5 + k * 7 + epoch) % 17 - 8) / 16;
    for (int t = 0; t < s.n; ++t)
        for (int k = 0; k < 32; ++k)
            input[size_t(t * 32 + k)] = float((t * 7 + k * 3 + epoch) % 13 - 6) / 8;
    fill_tensor(w, weights.data());
    fill_tensor(x, input.data());
    fill_tensor(ids, routing.data());
    auto context = std::make_shared<Context>();
    context->global_ctx = global;
    auto ops = changed ? candidate::ggml_webgpu_mul_mat_id(context, w, x, ids, out)
                       : baseline::ggml_webgpu_mul_mat_id(context, w, x, ids, out);
    check(ops.size() == 2, "same two direct dispatches");
    check(ops[0].pipeline.id == 1 && ops[1].pipeline.id == 2, "gather then matrix");
    Result result;
    result.capacity =
        changed ? candidate::ggml_webgpu_mul_mat_id_max_tiles(out) : baseline::ggml_webgpu_mul_mat_id_max_tiles(out);
    result.allocation = type.iface.get_alloc_size(&type, out);
    result.descriptor = 4 + 12 * size_t(result.capacity);
    check(result.allocation == ROUNDUP_POW2(ggml_nbytes(out) + size_t(slots) * 4 + result.descriptor + 512, 4),
          "actual allocation callback uses capacity");
    check(ggml_webgpu_tensor_offset(out) + result.allocation <= tensor_buffer(out).GetSize(),
          "allocator reserves full scratch span");
    for (const auto & op : ops) {
        result.params.push_back(op.params);
        result.grids.push_back(op.workgroups);
        for (const auto & entry : op.bind_group_entries)
            check(entry.offset <= entry.buffer.GetSize() && entry.size <= entry.buffer.GetSize() - entry.offset,
                  "actual binding fits physical allocation");
    }
    check(ops[0].bind_group_entries[2].size == result.descriptor &&
              ops[1].bind_group_entries[4].size == result.descriptor,
          "gather/matrix exact descriptor size");
    check(ops[0].bind_group_entries[2].offset == ops[1].bind_group_entries[4].offset, "shared descriptor offset");
    check(ops[0].bind_group_entries[1].size == size_t(slots) * 4, "IDs cache unchanged");
    for (const auto & entry : ops[1].bind_group_entries)
        check(entry.offset % 256 == 0 && entry.size % 4 == 0, "aligned storage bindings");
    const auto & grid = ops[1].workgroups;
    const uint32_t row_tiles = uint32_t((s.m + 31) / 32), logical = row_tiles * result.capacity;
    check(grid[0] <= 32 && grid[1] <= 32 && uint64_t(grid[0]) * grid[1] >= logical, "legal covering rectangle");
    std::vector<std::vector<uint8_t>> before, mask;
    for (auto & data : allocated) {
        before.push_back(data->bytes);
        mask.emplace_back(data->bytes.size(), 0);
    }
    auto write = [&](const wgpu::BindGroupEntry & entry, size_t index, uint32_t value) {
        check(index < entry.size / 4, "descriptor/output binding boundary");
        const size_t off = entry.offset + index * 4;
        std::memcpy(at(entry.buffer, off, 4), &value, 4);
        auto it = std::find(allocated.begin(), allocated.end(), entry.buffer.data);
        check(it != allocated.end(), "owned recorded resource");
        const size_t bi = size_t(it - allocated.begin());
        for (size_t i = 0; i < 4; ++i)
            mask[bi][64 + off + i] = 1;
    };
    bool rejected = false;
    try {
        write(ops[0].bind_group_entries[2], result.descriptor / 4, 0);
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    check(rejected, "oracle rejects first word past binding");
    std::vector<std::vector<uint32_t>> positions(size_t(s.e));
    for (int i = 0; i < slots; ++i)
        if (routing[size_t(i)] >= 0)
            positions[size_t(routing[size_t(i)])].push_back(uint32_t(i));
    std::vector<std::array<uint32_t, 3>> tiles;
    uint32_t base = 0;
    const auto & positions_entry = ops[0].bind_group_entries[1];
    const auto & tiles_entry = ops[0].bind_group_entries[2];
    for (int e = 0; e < s.e; ++e) {
        const auto & p = positions[size_t(e)];
        for (size_t i = 0; i < p.size(); ++i)
            write(positions_entry, base + i, p[i]);
        for (size_t i = 0; i < p.size(); i += 32) {
            tiles.push_back({uint32_t(e), base + uint32_t(i), uint32_t(std::min(size_t(32), p.size() - i))});
        }
        base += uint32_t(p.size());
    }
    check(tiles.size() <= result.capacity, "independent histogram capacity");
    if (s.mode == 0)
        check(changed ? tiles.size() == result.capacity : tiles.size() < result.capacity, "attaining extremal routing");
    write(tiles_entry, 0, uint32_t(tiles.size()));
    for (size_t i = 0; i < tiles.size(); ++i)
        for (size_t j = 0; j < 3; ++j)
            write(tiles_entry, 1 + 3 * i + j, tiles[i][j]);
    auto store = [&](size_t slot, size_t row, float value) {
        uint32_t bits;
        std::memcpy(&bits, &value, 4);
        write(ops[1].bind_group_entries[2], ops[1].params[2] + slot * size_t(s.m) + row, bits);
    };
    for (int slot = 0; slot < slots; ++slot)
        if (routing[size_t(slot)] == -1)
            for (int r = 0; r < s.m; ++r)
                store(size_t(slot), size_t(r), 0.0f);
    // CPU store/guard oracle only. GPU math is not executed by this harness.
    for (uint64_t linear = 0; linear < uint64_t(grid[0]) * grid[1]; ++linear) {
        const size_t tile = size_t(linear / row_tiles);
        if (tile >= tiles.size())
            continue;
        const auto d = tiles[tile];
        for (uint32_t j = 0; j < d[2]; ++j) {
            const auto & p = positions[size_t(d[0])];
            uint32_t expert_base = 0;
            for (uint32_t e = 0; e < d[0]; ++e)
                expert_base += uint32_t(positions[e].size());
            const uint32_t slot = p[size_t(d[1] - expert_base + j)];
            for (uint32_t row = uint32_t(linear % row_tiles) * 32;
                 row < std::min(uint32_t(s.m), uint32_t(linear % row_tiles) * 32 + 32); ++row) {
                float sum = 0;
                for (int k = 0; k < 32; ++k)
                    sum += weights[size_t((int(d[0]) * s.m + int(row)) * 32 + k)] *
                           input[size_t(int(slot) / s.u * 32 + k)];
                store(slot, row, sum);
            }
        }
    }
    result.output.resize(size_t(slots * s.m));
    std::memcpy(result.output.data(), at(tensor_buffer(out), ggml_webgpu_tensor_offset(out), ggml_nbytes(out)),
                ggml_nbytes(out));
    for (int slot = 0; slot < slots; ++slot)
        for (int row = 0; row < s.m; ++row) {
            double sum = 0;
            const int expert = routing[size_t(slot)];
            if (expert >= 0)
                for (int k = 0; k < 32; ++k)
                    sum += double(weights[size_t((expert * s.m + row) * 32 + k)]) *
                           double(input[size_t(slot / s.u * 32 + k)]);
            const float expected = float(sum);
            check(std::memcmp(&expected, &result.output[size_t(slot * s.m + row)], 4) == 0, "exact dyadic CPU oracle");
        }
    for (size_t b = 0; b < allocated.size(); ++b)
        for (size_t i = 0; i < before[b].size(); ++i)
            if (!mask[b][i])
                check(allocated[b]->bytes[i] == before[b][i], "input/scratch/pool guard unchanged");
    ggml_gallocr_free(allocator);
    ggml_free(ctx);
    return result;
}
int main() {
    try {
        for (uint32_t e : {0u, 1u, 2u, 31u, 32u, 33u, 512u, UINT32_MAX - 1, UINT32_MAX})
            for (uint32_t slots : {0u,
                                   1u,
                                   2u,
                                   31u,
                                   32u,
                                   33u,
                                   34u,
                                   63u,
                                   64u,
                                   65u,
                                   511u,
                                   512u,
                                   513u,
                                   1090u,
                                   1280u,
                                   UINT32_MAX - 32,
                                   UINT32_MAX - 31,
                                   UINT32_MAX - 30,
                                   UINT32_MAX - 1,
                                   UINT32_MAX}) {
                ggml_tensor w{}, out{};
                w.ne[2] = e;
                out.src[0] = &w;
                out.ne[1] = slots;
                out.ne[2] = 1;
                const uint64_t a = std::min(e, slots), bound = a + (uint64_t(slots) - a) / 32;
                check(candidate::ggml_webgpu_mul_mat_id_max_tiles(&out) == bound && bound <= slots,
                      "actual uint32 helper boundary");
                if (a) {
                    const uint64_t count = slots - a + 1;
                    check((count + 31) / 32 + a - 1 == bound, "attaining compressed histogram");
                }
            }
        const std::array<Spec, 6> cases = {{{4, 2, 2, 33, 0},
                                            {4, 1, 35, 1, 0},
                                            {4, 1, 36, 33, 0},
                                            {4, 2, 18, 33, 1},
                                            {4, 2, 18, 33, 2},
                                            {512, 10, 128, 1, 0}}};
        for (const auto & s : cases)
            for (int epoch : {0, 1}) {
                const auto old = run(false, s, epoch), now = run(true, s, epoch);
                check(old.output == now.output, "baseline/candidate CPU-oracle output");
                check(old.params == now.params, "actual wire parameters unchanged");
                check(old.capacity >= now.capacity, "capacity nonincrease");
                check(old.allocation - now.allocation == 12 * (old.capacity - now.capacity), "raw allocation savings");
                for (size_t i = 0; i < old.grids.size(); ++i)
                    check(uint64_t(now.grids[i][0]) * now.grids[i][1] <= uint64_t(old.grids[i][0]) * old.grids[i][1],
                          "physical rectangle nonincrease");
            }
        std::cout << checks << " real ggml allocator / recorded dispatcher / capacity and CPU guard checks passed\n";
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
