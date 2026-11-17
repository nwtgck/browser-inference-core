#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"

#include <cstdio>
#include <limits>
#include <memory>
#include <vector>

namespace {
struct BufferPlan {
    size_t first;
    size_t second;
    size_t maximum;
    std::vector<size_t> allocations;
};

BufferPlan & plan(ggml_backend_buffer_type_t type) {
    return *static_cast<BufferPlan *>(type->context);
}

const char * name(ggml_backend_buffer_type_t) {
    return "size-only-buffer-plan";
}

void * base(ggml_backend_buffer_t) {
    return reinterpret_cast<void *>(0x1000);
}

size_t alignment(ggml_backend_buffer_type_t) {
    return 1;
}

size_t maximum(ggml_backend_buffer_type_t type) {
    return plan(type).maximum;
}

size_t tensor_size(ggml_backend_buffer_type_t type, const ggml_tensor * tensor) {
    return tensor->name[0] == 'a' ? plan(type).first : plan(type).second;
}

ggml_backend_buffer_t allocate(ggml_backend_buffer_type_t type, size_t size) {
    plan(type).allocations.push_back(size);
    ggml_backend_buffer_i iface{};
    iface.get_base = base;
    // Record logical GPU-sized allocations without allocating their payloads.
    return ggml_backend_buffer_init(type, iface, nullptr, size);
}

bool check_plan(size_t first, size_t second, size_t limit, const std::vector<size_t> & expected) {
    BufferPlan state{first, second, limit, {}};
    ggml_backend_buffer_type type{};
    type.context = &state;
    type.iface.get_name = name;
    type.iface.alloc_buffer = allocate;
    type.iface.get_alignment = alignment;
    type.iface.get_max_size = maximum;
    type.iface.get_alloc_size = tensor_size;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(
        ggml_init({2 * ggml_tensor_overhead() + 1024, nullptr, true}), ggml_free);
    if (!context) {
        return false;
    }
    ggml_set_name(ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, 1), "a");
    ggml_set_name(ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, 1), "b");
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors_from_buft(context.get(), &type), ggml_backend_buffer_free);
    return buffer && state.allocations == expected;
}
} // namespace

int main() {
    const size_t maximum_size = std::numeric_limits<size_t>::max();
    if (!check_plan(80, 48, 256, {128}) || !check_plan(80, 48, 96, {80, 48}) ||
        !check_plan(maximum_size - 65535, 131072, maximum_size - 4096, {maximum_size - 65535, 131072})) {
        std::fprintf(stderr, "GPU-sized buffers must split before exceeding their limit, including size_t overflow.\n");
        return 1;
    }
    return 0;
}
