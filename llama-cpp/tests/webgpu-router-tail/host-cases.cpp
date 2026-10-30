#include "ggml-webgpu-router-tail.hpp"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include <array>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

static size_t checks = 0;
static void check(bool ok, const char * message) {
    ++checks;
    if (!ok) {
        throw std::runtime_error(std::to_string(checks) + ": " + message);
    }
}

namespace wgpu {
enum class BufferUsage : uint32_t { None = 0, Storage = 1, CopyDst = 2 };
static BufferUsage operator&(BufferUsage a, BufferUsage b) {
    return BufferUsage(uint32_t(a) & uint32_t(b));
}
struct BufferData {
    uint64_t size;
    BufferUsage usage = BufferUsage::Storage;
};
struct Buffer {
    std::shared_ptr<BufferData> data;
    explicit operator bool() const {
        return bool(data);
    }
    BufferData * Get() const {
        return data.get();
    }
    uint64_t GetSize() const {
        return data->size;
    }
    BufferUsage GetUsage() const {
        return data->usage;
    }
};
struct BindGroupEntry {
    uint32_t binding;
    Buffer buffer;
    uint64_t offset, size;
};
} // namespace wgpu
struct Limits {
    uint32_t maxComputeInvocationsPerWorkgroup = 256, maxComputeWorkgroupSizeX = 256;
    uint32_t maxComputeWorkgroupStorageSize = 16384, maxComputeWorkgroupsPerDimension = 65535;
    uint32_t maxStorageBuffersPerShaderStage = 8, maxBindingsPerBindGroup = 8, maxUniformBuffersPerShaderStage = 8;
    uint64_t maxUniformBufferBindingSize = 65536, maxStorageBufferBindingSize = 1ull << 30;
    uint64_t minStorageBufferOffsetAlignment = 256;
};
struct Global {
    struct {
        Limits limits;
    } capabilities;
};
using webgpu_global_context = std::shared_ptr<Global>;
struct ggml_backend_webgpu_buffer_context {
    wgpu::Buffer buffer;
    webgpu_global_context global_ctx;
};
static void * ggml_backend_webgpu_buffer_get_base(ggml_backend_buffer_t) {
    return reinterpret_cast<void *>(0x1000);
}
static void * foreign_base(ggml_backend_buffer_t) {
    return reinterpret_cast<void *>(0x1000);
}
static void * const webgpu_ptr_base = reinterpret_cast<void *>(0x1000);
static size_t ggml_webgpu_tensor_offset(const ggml_tensor * tensor) {
    const ggml_tensor * base_tensor = tensor->view_src ? tensor->view_src : tensor;
    return (size_t)((uintptr_t)base_tensor->data - (uintptr_t)webgpu_ptr_base) + tensor->view_offs;
}
static uint32_t ggml_webgpu_u32_from_f32(float x) {
    uint32_t out;
    std::memcpy(&out, &x, 4);
    return out;
}
struct Library {
    std::unordered_map<uint64_t, int> keys;
    int calls = 0, created = 0;
    int get_router_tail_pipeline(uint32_t wg, bool merged) {
        ++calls;
        const auto key = uint64_t(wg) * 2 + merged;
        if (!keys.count(key)) {
            keys[key] = ++created;
        }
        return keys[key];
    }
};
struct Context {
    webgpu_global_context global_ctx = std::make_shared<Global>();
    std::shared_ptr<Library> shader_lib = std::make_shared<Library>();
};
using webgpu_context = std::shared_ptr<Context>;
struct webgpu_encoded_op {
    std::vector<uint32_t> params;
    std::vector<wgpu::BindGroupEntry> entries;
    uint32_t x, y;
};
static wgpu::BindGroupEntry ggml_webgpu_make_bind_group_entry(uint32_t index, wgpu::Buffer b, uint64_t off,
                                                              uint64_t bytes) {
    return {index, b, off, bytes};
}
static int build_calls = 0;
static webgpu_encoded_op ggml_backend_webgpu_build(webgpu_context &, int, const std::vector<uint32_t> & params,
                                                   const std::vector<wgpu::BindGroupEntry> & entries, uint32_t x,
                                                   uint32_t y) {
    ++build_calls;
    return {params, entries, x, y};
}
constexpr uint64_t WEBGPU_PARAMS_BUF_SIZE_BYTES = 128;
#include "dispatch.inc"

static webgpu_global_context allocation_global;
static void free_buffer(ggml_backend_buffer_t b) {
    delete static_cast<ggml_backend_webgpu_buffer_context *>(b->context);
}
static ggml_backend_buffer_t make_buffer(ggml_backend_buffer_type_t type, size_t bytes) {
    ggml_backend_buffer_i iface{};
    iface.free_buffer = free_buffer;
    iface.get_base = ggml_backend_webgpu_buffer_get_base;
    auto * context = new ggml_backend_webgpu_buffer_context{
        {std::make_shared<wgpu::BufferData>(wgpu::BufferData{bytes})}, allocation_global};
    return ggml_backend_buffer_init(type, iface, context, bytes);
}
static const char * memory_name(ggml_backend_buffer_type_t) {
    return "router-tail-host-stub";
}
static size_t memory_alignment(ggml_backend_buffer_type_t) {
    return 256;
}
static ggml_backend_buffer_type memory_type = [] {
    ggml_backend_buffer_type t{};
    t.iface.get_name = memory_name;
    t.iface.get_alignment = memory_alignment;
    t.iface.alloc_buffer = make_buffer;
    return t;
}();

struct Fixture {
    ggml_context * ctx;
    ggml_tensor *root, *probs, *ids_root, *ids, *gather, *flattened, *sum, *clamp, *div, *output;
    ggml_cgraph * graph;
    int index = -1;
    Fixture(int64_t experts = 512, int64_t used = 10, int64_t tokens = 3, bool padded = true, bool produced = false,
            bool preserve = false, bool preserve_ids = false, bool late_ids = false) {
        ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
        check(ctx != nullptr, "context");
        root = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, experts, tokens);
        if (produced) {
            root = ggml_scale(ctx, root, 0.5f);
        }
        if (preserve) {
            ggml_set_output(root);
        }
        probs = ggml_reshape_3d(ctx, root, 1, experts, tokens);
        ids_root = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, padded ? experts : used, tokens);
        if (preserve_ids) {
            ggml_set_output(ids_root);
        }
        ids = padded ? ggml_view_2d(ctx, ids_root, used, tokens, ids_root->nb[1], 0) : ids_root;
        gather = ggml_get_rows(ctx, probs, ids);
        flattened = ggml_reshape_2d(ctx, gather, used, tokens);
        sum = ggml_sum_rows(ctx, flattened);
        clamp = ggml_clamp(ctx, sum, 0.00006103515625f, std::numeric_limits<float>::infinity());
        div = ggml_div(ctx, flattened, clamp);
        output = ggml_reshape_3d(ctx, div, 1, used, tokens);
        auto * consumer = ggml_scale(ctx, output, 1.0f);
        ggml_set_output(consumer);
        graph = ggml_new_graph_custom(ctx, 256, false);
        ggml_build_forward_expand(graph, consumer);
        if (late_ids) {
            auto * experts_data = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 1, experts);
            auto * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 1, tokens);
            auto * expert_consumer = ggml_mul_mat_id(ctx, experts_data, x, ids);
            ggml_set_output(expert_consumer);
            ggml_build_forward_expand(graph, expert_consumer);
        }
        for (int i = 0; i < graph->n_nodes; ++i) {
            if (graph->nodes[i] == gather) {
                index = i;
            }
        }
        check(index >= 0, "find gather");
    }
    ~Fixture() {
        ggml_free(ctx);
    }
    std::array<ggml_tensor *, 6> nodes() const {
        return {gather, flattened, sum, clamp, div, output};
    }
};

static void bind_tensor(ggml_tensor * tensor, ggml_backend_buffer_t buffer, uint64_t offset) {
    auto * root = tensor->view_src ? tensor->view_src : tensor;
    root->buffer = buffer;
    tensor->buffer = buffer;
    root->data = reinterpret_cast<void *>(0x1000 + offset);
    tensor->data = reinterpret_cast<void *>(0x1000 + offset + tensor->view_offs);
}

int main() {
    try {
        for (int64_t e : {1, 5, 16, 512})
            for (int64_t t : {1, 3, 128})
                for (bool padded : {false, true}) {
                    Fixture f(e, std::min<int64_t>(e, 10), t, padded);
                    router_tail_plan plan;
                    auto admitted = [&] { return plan_router_tail(f.graph, f.index, plan); };
                    check(admitted(), "actual six-node graph");
                    check(plan.probs == f.probs && plan.ids == f.ids && plan.output == f.output, "pointers");
                    check(plan.experts == e && plan.used == std::min<int64_t>(e, 10) && plan.tokens == t, "dimensions");
                    check(plan.probs_bytes == ggml_nbytes(f.probs) && plan.ids_bytes == ggml_nbytes(f.ids) &&
                              plan.output_bytes == ggml_nbytes(f.output),
                          "spans");
                    for (int i = 0; i < 6; ++i) {
                        auto * n = f.nodes()[i];
                        const int flags = n->flags;
                        n->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
                        check(!admitted(), "COMPUTE required for every raw node");
                        n->flags = flags;
                        n->flags |= GGML_TENSOR_FLAG_OUTPUT;
                        check(i == 5 ? admitted() : !admitted(), "only final output survives");
                        n->flags = flags;
                        const auto type = n->type;
                        n->type = GGML_TYPE_F16;
                        check(!admitted(), "F32 intermediate");
                        n->type = type;
                        const auto nb = n->nb[0];
                        n->nb[0] = 8;
                        check(!admitted(), "dense intermediate");
                        n->nb[0] = nb;
                        const auto op = n->op;
                        n->op = GGML_OP_NONE;
                        check(!admitted(), "exact raw op");
                        n->op = op;
                        n->src[5] = f.probs;
                        check(!admitted(), "extra source rejected");
                        n->src[5] = nullptr;
                    }
                    auto view = ggml_graph_view(f.graph, f.index, f.index + 5);
                    check(!plan_router_tail(&view, 0, plan), "callback truncation");
                    check(!plan_router_tail(f.graph, -1, plan) && !plan_router_tail(f.graph, f.graph->n_nodes, plan),
                          "index guards");
                    const auto stride = f.ids->nb[1];
                    f.ids->nb[1] = 3;
                    check(!admitted(), "unaligned ids stride");
                    f.ids->nb[1] = stride;
                    const auto shape = f.output->ne[0];
                    f.output->ne[0] = 2;
                    check(!admitted(), "exact output shape");
                    f.output->ne[0] = shape;
                    const auto view_offs = f.flattened->view_offs;
                    f.flattened->view_offs = 4;
                    check(!admitted(), "zero reshape offset");
                    f.flattened->view_offs = view_offs;
                    auto * parent = f.flattened->view_src;
                    f.flattened->view_src = f.probs;
                    check(!admitted(), "internal reshape parent");
                    f.flattened->view_src = parent;
                    float params[2];
                    std::memcpy(params, f.clamp->op_params, 8);
                    const float bad[2] = {1, -1};
                    std::memcpy(f.clamp->op_params, bad, 8);
                    check(!admitted(), "unordered clamp");
                    std::memcpy(f.clamp->op_params, params, 8);
                    ggml_set_output(f.probs);
                    ggml_set_output(f.ids);
                    check(admitted(), "source outputs preserved");
                }
        for (int target = 0; target < 5; ++target) {
            Fixture f;
            router_tail_plan plan;
            auto * side = ggml_scale(f.ctx, f.nodes()[target], 2);
            ggml_build_forward_expand(f.graph, side);
            check(!plan_router_tail(f.graph, f.index, plan), "intermediate side-user");
        }
        {
            Fixture f;
            router_tail_plan plan;
            ggml_set_output(f.output);
            check((f.div->flags & GGML_TENSOR_FLAG_OUTPUT) != 0, "public set_output propagates to view root");
            check(!plan_router_tail(f.graph, f.index, plan), "DIV output remains a conservative fallback");
        }
        {
            Fixture f;
            router_tail_plan plan;
            auto * side =
                ggml_view_2d(f.ctx, f.flattened, f.flattened->ne[0], f.flattened->ne[1], f.flattened->nb[1], 0);
            ggml_build_forward_expand(f.graph, side);
            check(!plan_router_tail(f.graph, f.index, plan), "intermediate outside view");
        }
        {
            Fixture f;
            router_tail_plan plan;
            auto * a = ggml_scale(f.ctx, f.output, 2);
            auto * b = ggml_scale(f.ctx, f.output, 3);
            ggml_build_forward_expand(f.graph, a);
            ggml_build_forward_expand(f.graph, b);
            check(plan_router_tail(f.graph, f.index, plan), "final multiple users");
        }
        {
            ggml_tensor tensor{};
            tensor.type = GGML_TYPE_F32;
            for (int i = 0; i < 4; ++i) {
                tensor.ne[i] = 1;
                tensor.nb[i] = 4;
            }
            uint64_t bytes;
            tensor.ne[0] = UINT32_MAX;
            for (int i = 1; i < 4; ++i) {
                tensor.nb[i] = uint64_t(UINT32_MAX) * 4;
            }
            check(router_tail_span(&tensor, GGML_TYPE_F32, true, bytes) && bytes == uint64_t(UINT32_MAX) * 4,
                  "u32 exact limit");
            tensor.ne[1] = 2;
            check(!router_tail_span(&tensor, GGML_TYPE_F32, false, bytes), "u32 product overflow");
            tensor.ne[0] = 1;
            tensor.nb[1] = UINT64_MAX;
            check(!router_tail_span(&tensor, GGML_TYPE_F32, false, bytes), "span overflow");
        }
        auto ctx = std::make_shared<Context>();
        allocation_global = ctx->global_ctx;
        for (bool produced : {false, true})
            for (bool preserve : {false, true})
                for (bool preserve_ids : {false, true}) {
                    Fixture f(512, 10, 3, true, produced, preserve, preserve_ids);
                    auto * allocator = ggml_gallocr_new(&memory_type);
                    check(ggml_gallocr_alloc_graph(allocator, f.graph), "real gallocr");
                    router_tail_plan plan;
                    check(plan_router_tail(f.graph, f.index, plan), "real allocator graph admitted structurally");
                    int count = 1;
                    const auto encoded = ggml_webgpu_router_tail(ctx, f.graph, f.index, count);
                    webgpu_router_tail_binding a, b, c;
                    check(ggml_webgpu_router_tail_binding(ctx, plan.probs, plan.probs_bytes, a),
                          "allocator source binding");
                    check(ggml_webgpu_router_tail_binding(ctx, plan.ids, plan.ids_bytes, b), "allocator ids binding");
                    check(ggml_webgpu_router_tail_binding(ctx, plan.output, plan.output_bytes, c),
                          "allocator output binding");
                    const bool disjoint = router_tail_disjoint(a.offset, a.size, b.offset, b.size) &&
                                          router_tail_disjoint(a.offset, a.size, c.offset, c.size) &&
                                          router_tail_disjoint(b.offset, b.size, c.offset, c.size);
                    check(bool(encoded) == disjoint, "allocator reused input falls back; disjoint fuses");
                    if (preserve && preserve_ids) {
                        check(bool(encoded), "retained inputs have disjoint live output");
                    }
                    check(count == (encoded ? 6 : 1), "raw node consumption only on success");
                    std::cout << "allocator produced=" << produced << " preserve=" << preserve
                              << " preserve_ids=" << preserve_ids << " fused=" << bool(encoded)
                              << " source=" << a.tensor_offset << " ids=" << b.tensor_offset
                              << " output=" << c.tensor_offset << '\n';
                    ggml_gallocr_free(allocator);
                }
        for (bool produced : {false, true})
            for (int64_t tokens : {1, 2, 3, 128}) {
                Fixture f(512, 10, tokens, true, produced, false, false, true);
                check(!(f.probs->flags & GGML_TENSOR_FLAG_OUTPUT) && !(f.ids->flags & GGML_TENSOR_FLAG_OUTPUT),
                      "natural inputs are not artificial outputs");
                auto * allocator = ggml_gallocr_new(&memory_type);
                check(ggml_gallocr_alloc_graph(allocator, f.graph), "natural live ids graph allocator");
                int count = 1;
                const auto encoded = ggml_webgpu_router_tail(ctx, f.graph, f.index, count);
                check(bool(encoded) == (tokens == 1), "natural alias accepted only for single workgroup");
                check(count == (tokens == 1 ? 6 : 1), "natural raw consumption");
                if (encoded) {
                    check(encoded->entries.size() == 2, "merged source/output uses one storage binding");
                }
                std::cout << "natural allocator produced=" << produced << " tokens=" << tokens
                          << " fused=" << bool(encoded) << " source=" << ggml_webgpu_tensor_offset(f.probs)
                          << " ids=" << ggml_webgpu_tensor_offset(f.ids)
                          << " output=" << ggml_webgpu_tensor_offset(f.output) << '\n';
                ggml_gallocr_free(allocator);
            }
        {
            Fixture f;
            std::vector<ggml_backend_buffer_t> buffers;
            for (int i = 0; i < 3; ++i) {
                buffers.push_back(make_buffer(&memory_type, 1 << 20));
            }
            bind_tensor(f.probs, buffers[0], 12);
            bind_tensor(f.ids, buffers[1], 20);
            bind_tensor(f.output, buffers[2], 28);
            int count = 1;
            auto encode = [&] { return ggml_webgpu_router_tail(ctx, f.graph, f.index, count); };
            auto good = encode();
            check(bool(good) && count == 6 && good->x == 3 && good->y == 1, "checked dispatch");
            check(good->params == std::vector<uint32_t>({3, 5, 7, 512, 512, 10, 10, 3, 0x38800000, 0x7f800000}),
                  "all uniform words");
            check(good->entries.size() == 3 && good->entries[0].offset == 0 &&
                      good->entries[0].size == 12 + 512 * 3 * 4,
                  "binding prefix and span");
            const float bounds[2] = {0.25f, 0.75f};
            std::memcpy(f.clamp->op_params, bounds, 8);
            auto changed = encode();
            check(changed && changed->params[8] == 0x3e800000 && changed->params[9] == 0x3f400000,
                  "changed clamp on reused pipeline");
            auto reject = [&] {
                count = 1;
                const int old_calls = build_calls, old_pipeline_calls = ctx->shader_lib->calls;
                check(!encode(), "unsupported dispatch fallback");
                check(count == 1 && build_calls == old_calls && ctx->shader_lib->calls == old_pipeline_calls,
                      "fallback has no build or pipeline side effect");
            };
            auto * bc = static_cast<ggml_backend_webgpu_buffer_context *>(buffers[0]->context);
            const auto old_global = bc->global_ctx;
            bc->global_ctx = std::make_shared<Global>();
            reject();
            bc->global_ctx = old_global;
            auto callback = buffers[0]->iface.get_base;
            buffers[0]->iface.get_base = foreign_base;
            reject();
            buffers[0]->iface.get_base = callback;
            bc->buffer.data->usage = wgpu::BufferUsage::CopyDst;
            reject();
            bc->buffer.data->usage = wgpu::BufferUsage::Storage;
            const auto old_size = buffers[0]->size;
            buffers[0]->size = 16;
            reject();
            buffers[0]->size = old_size;
            bc->buffer.data->size = 16;
            reject();
            bc->buffer.data->size = old_size;
            auto * old_data = f.probs->data;
            f.probs->data = reinterpret_cast<void *>(0x1001);
            reject();
            f.probs->data = old_data;
            const auto old_root_type = f.root->type;
            f.root->type = GGML_TYPE_I32;
            reject();
            f.root->type = old_root_type;
            const auto saved_limits = ctx->global_ctx->capabilities.limits;
            auto & limits = ctx->global_ctx->capabilities.limits;
            limits.maxComputeInvocationsPerWorkgroup = 3;
            reject();
            limits = saved_limits;
            limits.maxComputeInvocationsPerWorkgroup = 8;
            reject();
            limits = saved_limits;
            limits.maxComputeWorkgroupSizeX = 128;
            reject();
            limits = saved_limits;
            limits.maxComputeWorkgroupStorageSize = 512;
            reject();
            limits = saved_limits;
            limits.maxComputeWorkgroupsPerDimension = 2;
            reject();
            limits = saved_limits;
            limits.maxStorageBuffersPerShaderStage = 2;
            reject();
            limits = saved_limits;
            limits.maxBindingsPerBindGroup = 3;
            reject();
            limits = saved_limits;
            limits.maxUniformBuffersPerShaderStage = 0;
            reject();
            limits = saved_limits;
            limits.maxUniformBufferBindingSize = 64;
            reject();
            limits = saved_limits;
            limits.maxStorageBufferBindingSize = 64;
            reject();
            limits = saved_limits;
            limits.minStorageBufferOffsetAlignment = 6;
            reject();
            limits = saved_limits;
            auto * output_root = f.output->view_src;
            bind_tensor(f.output, buffers[0], 12);
            reject();
            bind_tensor(f.output, buffers[0], 12 + 512 * 3 * 4 + 4);
            reject(); // logical disjoint, aligned prefix overlaps.
            bind_tensor(f.output, buffers[0], 8192);
            check(bool(encode()), "same allocation disjoint aligned regions");
            bind_tensor(f.output, buffers[2], 28);
            check(output_root == f.div, "output root retained");
            f.output->data = reinterpret_cast<void *>(0x1000 + 32);
            reject();
            bind_tensor(f.output, buffers[2], 28);
            for (auto * buffer : buffers) {
                ggml_backend_buffer_free(buffer);
            }
        }
        {
            Fixture f(512, 10, 1, true, false, false, false, true);
            auto * buffer = make_buffer(&memory_type, 1 << 20);
            auto * ids_buffer = make_buffer(&memory_type, 1 << 20);
            bind_tensor(f.probs, buffer, 12);
            bind_tensor(f.ids, ids_buffer, 20);
            bind_tensor(f.output, buffer, 12);
            int count = 1;
            auto encode = [&] { return ggml_webgpu_router_tail(ctx, f.graph, f.index, count); };
            auto same = encode();
            check(bool(same) && same->entries.size() == 2 && same->params[0] == 3 && same->params[2] == 3,
                  "exact same logical range T1");
            bind_tensor(f.output, buffer, 76);
            auto partial = encode();
            check(bool(partial) && partial->params[2] == 19, "partial payload overlap T1");
            bind_tensor(f.output, buffer, 2064);
            auto prefix = encode();
            check(bool(prefix) && prefix->entries[0].size == 2104, "logical disjoint but aligned overlap merged");
            bind_tensor(f.probs, buffer, 1036);
            bind_tensor(f.output, buffer, 1028);
            auto reverse = encode();
            check(reverse && reverse->entries[0].offset == 1024 && reverse->params[0] == 3 && reverse->params[2] == 1,
                  "output before source rebased");
            bind_tensor(f.probs, buffer, 12);
            bind_tensor(f.output, buffer, 76);
            bind_tensor(f.ids, buffer, 2064);
            check(!encode(), "ids logical disjoint but aligned overlap with union rejected");
            bind_tensor(f.ids, buffer, 4096);
            check(bool(encode()), "ids disjoint region in same allocation");
            bind_tensor(f.ids, ids_buffer, 20);
            bind_tensor(f.output, buffer, 2064);
            ctx->global_ctx->capabilities.limits.maxStorageBufferBindingSize = 2060;
            check(!encode(), "union binding cap beyond each valid individual binding");
            ctx->global_ctx->capabilities.limits.maxStorageBufferBindingSize = 1ull << 30;
            bind_tensor(f.output, buffer, 4096);
            auto disjoint = encode();
            check(disjoint && disjoint->entries.size() == 3, "same allocation disjoint preserves original pipeline");
            const auto * original = static_cast<ggml_backend_webgpu_buffer_context *>(buffer->context);
            auto * wrapper_context = new ggml_backend_webgpu_buffer_context{original->buffer, original->global_ctx};
            auto * wrapper = ggml_backend_buffer_init(&memory_type, buffer->iface, wrapper_context, buffer->size);
            bind_tensor(f.output, wrapper, 76);
            auto different_wrapper = encode();
            check(different_wrapper && different_wrapper->entries.size() == 2,
                  "physical identity recognized across distinct buffer wrappers");
            bind_tensor(f.ids, wrapper, 76);
            check(!encode(), "IDS overlap found across wrappers by physical identity");
            bind_tensor(f.ids, ids_buffer, 20);
            ggml_backend_buffer_free(wrapper);
            bind_tensor(f.output, buffer, 76);
            auto * original_mutable = static_cast<ggml_backend_webgpu_buffer_context *>(buffer->context);
            const auto saved_size = buffer->size, saved_physical_size = original_mutable->buffer.data->size;
            buffer->size = 1ull << 37;
            original_mutable->buffer.data->size = buffer->size;
            const auto saved_limits = ctx->global_ctx->capabilities.limits;
            ctx->global_ctx->capabilities.limits.minStorageBufferOffsetAlignment = 1ull << 36;
            ctx->global_ctx->capabilities.limits.maxStorageBufferBindingSize = 1ull << 37;
            bind_tensor(f.probs, buffer, 1ull << 35);
            bind_tensor(f.output, buffer, (1ull << 35) + 64);
            check(!encode(), "element-prefix overflow rejected before narrowing/subtraction");
            ctx->global_ctx->capabilities.limits = saved_limits;
            buffer->size = saved_size;
            original_mutable->buffer.data->size = saved_physical_size;
            ggml_backend_buffer_free(buffer);
            ggml_backend_buffer_free(ids_buffer);
        }
        {
            // T>1 cannot use a single-WG snapshot guarantee. This legal schedule
            // overwrites the first token's not-yet-read probabilities.
            float probs[] = {1, 1, 1, 3, 3, 1, 0, 0};
            const float correct0 = probs[2] / (probs[2] + probs[3]);
            const float correct1 = probs[3] / (probs[2] + probs[3]);
            const float second_sum = probs[4] + probs[5];
            probs[2] = probs[4] / second_sum;
            probs[3] = probs[5] / second_sum;
            const float corrupted0 = probs[2] / (probs[2] + probs[3]);
            const float corrupted1 = probs[3] / (probs[2] + probs[3]);
            check(correct0 == 0.25f && correct1 == 0.75f && corrupted0 == 0.75f && corrupted1 == 0.25f,
                  "two-WG alias schedule is unsafe");
        }
        std::cout << checks << " source-bound router graph/binding/caps/fallback checks passed\n";
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
