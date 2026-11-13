// Host-only contracts for explicit BF16 retention. The fake backend allocates
// ordinary CPU memory and plans graphs; it never creates or emulates a GPU.
#include "core/browser_bf16_retention.h"
#include "core/compute_workspace.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "model_manager.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
void require(bool value, const char * message) {
    if (!value)
        throw std::runtime_error(message);
}
template <class F> void rejected(F operation, const char * message) {
    bool failed = false;
    try {
        operation();
    } catch (const std::runtime_error &) {
        failed = true;
    }
    require(failed, message);
}
Context context(bool allocate = false) {
    return Context(ggml_init({2 * 1024 * 1024, nullptr, !allocate}), ggml_free);
}

struct FakeWebgpu {
    struct Storage {
        std::vector<uint8_t> bytes;
        void * base;
        explicit Storage(size_t size) : bytes(size + 64) {
            base = reinterpret_cast<void *>((reinterpret_cast<uintptr_t>(bytes.data()) + 63) & ~uintptr_t(63));
        }
    };
    ggml_guid guid{};
    ggml_backend_reg registry{};
    ggml_backend_device device{};
    ggml_backend_buffer_type buft{};
    ggml_backend backend{};
    bool admit = true;
    size_t matmul_probes = 0;
    static const char * name(ggml_backend_t) {
        return "WebGPU";
    }
    FakeWebgpu() {
        registry.api_version = GGML_BACKEND_API_VERSION;
        registry.iface.get_name = [](ggml_backend_reg_t) { return "WebGPU"; };
        registry.iface.get_device_count = [](ggml_backend_reg_t) { return size_t(1); };
        registry.iface.get_device = [](ggml_backend_reg_t reg, size_t) {
            return &static_cast<FakeWebgpu *>(reg->context)->device;
        };
        registry.context = this;
        device.reg = &registry;
        device.context = this;
        device.iface.get_name = [](ggml_backend_dev_t) { return "WebGPU"; };
        device.iface.get_description = [](ggml_backend_dev_t) { return "host-only retention contract"; };
        device.iface.get_type = [](ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_GPU; };
        device.iface.get_memory = [](ggml_backend_dev_t, size_t * free, size_t * total) { *free = *total = 0; };
        device.iface.get_props = [](ggml_backend_dev_t, ggml_backend_dev_props * props) {
            *props = {};
            props->name = "WebGPU";
            props->description = "host-only retention contract";
            props->type = GGML_BACKEND_DEVICE_TYPE_GPU;
        };
        device.iface.get_buffer_type = [](ggml_backend_dev_t dev) {
            return &static_cast<FakeWebgpu *>(dev->context)->buft;
        };
        device.iface.supports_op = [](ggml_backend_dev_t dev, const ggml_tensor * op) {
            auto & self = *static_cast<FakeWebgpu *>(dev->context);
            // The pinned backend always owns preallocated leaves and metadata
            // views, even when it cannot compute the consumer operation.
            if (op->op == GGML_OP_NONE || op->op == GGML_OP_VIEW || op->op == GGML_OP_RESHAPE ||
                op->op == GGML_OP_PERMUTE || op->op == GGML_OP_TRANSPOSE)
                return true;
            if (op->op == GGML_OP_MUL_MAT)
                ++self.matmul_probes;
            return self.admit;
        };
        device.iface.supports_buft = [](ggml_backend_dev_t dev, ggml_backend_buffer_type_t type) {
            return type == &static_cast<FakeWebgpu *>(dev->context)->buft;
        };
        buft.device = &device;
        buft.iface.get_name = [](ggml_backend_buffer_type_t) { return "host-only-WebGPU-contract"; };
        buft.iface.get_alignment = [](ggml_backend_buffer_type_t) { return size_t(64); };
        buft.iface.get_max_size = [](ggml_backend_buffer_type_t) { return size_t(UINT32_MAX - 3); };
        buft.iface.alloc_buffer = [](ggml_backend_buffer_type_t type, size_t size) {
            ggml_backend_buffer_i functions{};
            functions.free_buffer = [](ggml_backend_buffer_t buffer) {
                delete static_cast<Storage *>(buffer->context);
            };
            functions.get_base = [](ggml_backend_buffer_t buffer) {
                return static_cast<Storage *>(buffer->context)->base;
            };
            functions.set_tensor = [](ggml_backend_buffer_t, ggml_tensor * tensor, const void * data, size_t offset,
                                      size_t size) {
                std::memcpy(static_cast<uint8_t *>(tensor->data) + offset, data, size);
            };
            functions.get_tensor = [](ggml_backend_buffer_t, const ggml_tensor * tensor, void * data, size_t offset,
                                      size_t size) {
                std::memcpy(data, static_cast<const uint8_t *>(tensor->data) + offset, size);
            };
            functions.clear = [](ggml_backend_buffer_t buffer, uint8_t value) {
                std::memset(static_cast<Storage *>(buffer->context)->base, value, buffer->size);
            };
            return ggml_backend_buffer_init(type, functions, new Storage(size), size);
        };
        backend.guid = &guid;
        backend.device = &device;
        backend.context = this;
        backend.iface.get_name = name;
        backend.iface.synchronize = [](ggml_backend_t) {};
        // Graph computation is deliberately unavailable: these tests cover
        // metadata/storage/scheduler contracts, not GPU arithmetic or speed.
        backend.iface.graph_compute = [](ggml_backend_t, ggml_cgraph *) { return GGML_STATUS_FAILED; };
    }
};

const char * fixture_path = "bf16-retention-contract.gguf";
std::string file_bytes() {
    std::ifstream file(fixture_path, std::ios::binary);
    return {std::istreambuf_iterator<char>(file), {}};
}
void write_fixture(const char * path, ggml_type source_type) {
    auto source = context(true);
    std::unique_ptr<gguf_context, decltype(&gguf_free)> gguf(gguf_init_empty(), gguf_free);
    for (const auto & item : std::vector<std::pair<const char *, std::vector<int64_t>>>{{"linear", {32, 32}},
                                                                                        {"lookup", {32, 64}},
                                                                                        {"norm", {32}},
                                                                                        {"unreviewed", {32, 32}},
                                                                                        {"control", {32, 32}}}) {
        auto * tensor =
            ggml_new_tensor(source.get(), std::strcmp(item.first, "control") == 0 ? GGML_TYPE_F16 : source_type,
                            int(item.second.size()), item.second.data());
        ggml_set_name(tensor, item.first);
        if (tensor->type == GGML_TYPE_BF16) {
            for (int64_t i = 0; i < ggml_nelements(tensor); ++i)
                static_cast<ggml_bf16_t *>(tensor->data)[i] = ggml_fp32_to_bf16(float(i % 17 - 8) / 4.f);
        } else {
            std::memset(tensor->data, 0, ggml_nbytes(tensor));
        }
        gguf_add_tensor(gguf.get(), tensor);
    }
    require(gguf_write_to_file(gguf.get(), path, false), "fixture write failed");
}

void check_registration(ggml_type selected, bool cpu, bool supported) {
    FakeWebgpu fake;
    fake.admit = supported;
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> host(cpu ? ggml_backend_cpu_init() : nullptr,
                                                                     ggml_backend_free);
    auto backend = cpu ? host.get() : &fake.backend;
    auto params = context();
    ModelManager manager;
    manager.set_webgpu_bf16_type(selected);
    manager.set_n_threads(1);
    manager.set_enable_mmap(false);
    manager.set_prefetch_disabled(true);
    require(manager.add_file(fixture_path), "fixture inspection failed");
    std::map<std::string, ggml_tensor *> tensors;
    for (const auto & entry : manager.loader().get_tensor_storage_map())
        tensors[entry.first] = ggml_new_tensor(params.get(), entry.second.type, entry.second.n_dims, entry.second.ne);
    std::map<ggml_tensor *, ggml_op> roles{{tensors.at("linear"), GGML_OP_MUL_MAT},
                                           {tensors.at("lookup"), GGML_OP_GET_ROWS}};
    size_t registered = 0;
    const auto original = file_bytes();
    const bool accepted =
        manager.register_param_tensors(ModelComponent::Diffusion, tensors, ModelManager::ResidencyMode::ParamBackend,
                                       backend, backend, &registered, false, false, &roles);
    if (!cpu && selected == GGML_TYPE_BF16 && !supported) {
        require(!accepted && registered == 0, "unsupported retention must fail atomically");
        for (const auto & entry : tensors)
            if (entry.first != "control")
                require(entry.second->type == GGML_TYPE_BF16, "failed registration changed a weight");
        return;
    }
    require(accepted, "registration failed");
    if (!cpu && selected != GGML_TYPE_BF16)
        require(fake.matmul_probes == 0, "new Linear role changed a legacy placement probe");
    size_t expected_bytes = 0;
    for (const auto & entry : tensors) {
        const bool retained = entry.first == "linear" || entry.first == "lookup";
        const auto expected = entry.first == "control"     ? GGML_TYPE_F16
                              : cpu                        ? GGML_TYPE_BF16
                              : selected == GGML_TYPE_BF16 ? retained ? GGML_TYPE_BF16 : GGML_TYPE_F32
                                                           : selected;
        require(entry.second->type == expected, "wrong retained or fallback type");
        require(manager.is_browser_retained_bf16_param(entry.second) ==
                    (!cpu && selected == GGML_TYPE_BF16 && retained),
                "registration identity gate disagrees with retention");
        require(ggml_is_contiguous(entry.second), "changed storage lost contiguous strides");
        expected_bytes += ggml_nbytes(entry.second);
        require(manager.loader().get_tensor_storage_map().at(entry.first).type ==
                    (entry.first == "control" ? GGML_TYPE_F16 : GGML_TYPE_BF16),
                "source file dtype changed");
    }
    require(registered == expected_bytes &&
                manager.registered_params_size({ModelComponent::Diffusion}) == expected_bytes,
            "resident payload accounting disagrees with selected types");
    require(manager.load_all_params_eagerly(), "mixed retained/fallback file load failed");
    const auto & source = manager.loader().get_tensor_storage_map();
    require(source.at("linear").type == GGML_TYPE_BF16 && file_bytes() == original,
            "loading changed the original file");
    for (const auto & entry : tensors) {
        if (entry.first == "control")
            continue;
        std::vector<uint8_t> actual(ggml_nbytes(entry.second));
        ggml_backend_tensor_get(entry.second, actual.data(), 0, actual.size());
        for (int64_t i = 0; i < ggml_nelements(entry.second); ++i) {
            const auto bf16 = ggml_fp32_to_bf16(float(i % 17 - 8) / 4.f);
            const auto fp32 = ggml_bf16_to_fp32(bf16);
            const auto fp16 = ggml_fp32_to_fp16(fp32);
            const auto width = ggml_type_size(entry.second->type);
            const void * expected = entry.second->type == GGML_TYPE_BF16  ? static_cast<const void *>(&bf16)
                                    : entry.second->type == GGML_TYPE_F16 ? static_cast<const void *>(&fp16)
                                                                          : &fp32;
            require(std::memcmp(actual.data() + size_t(i) * width, expected, width) == 0,
                    "loaded scalar differs from its source");
        }
    }
}

void changed_source_and_lora_contracts() {
    FakeWebgpu fake;
    auto params = context();
    ModelManager manager;
    manager.set_webgpu_bf16_type(GGML_TYPE_BF16);
    manager.set_n_threads(1);
    manager.set_enable_mmap(false);
    manager.set_prefetch_disabled(true);
    require(manager.add_file(fixture_path), "source-change fixture inspection failed");
    std::map<std::string, ggml_tensor *> tensors;
    for (const auto & entry : manager.loader().get_tensor_storage_map())
        tensors[entry.first] = ggml_new_tensor(params.get(), entry.second.type, entry.second.n_dims, entry.second.ne);
    std::map<ggml_tensor *, ggml_op> roles{{tensors.at("linear"), GGML_OP_MUL_MAT},
                                           {tensors.at("lookup"), GGML_OP_GET_ROWS}};
    size_t registered = 0;
    require(manager.register_param_tensors(ModelComponent::Diffusion, tensors,
                                           ModelManager::ResidencyMode::ParamBackend, &fake.backend, &fake.backend,
                                           &registered, false, false, &roles),
            "source-change registration failed");
    require(manager.load_all_params_eagerly(), "source-change initial load failed");
    ModelManager::LoraSpec lora;
    lora.path = fixture_path;
    require(manager.set_loras({lora}, VERSION_MINIMAX_H3), "LoRA source registration failed");
    std::vector<ggml_tensor *> required;
    for (const auto & entry : tensors)
        required.push_back(entry.second);
    require(!manager.prepare_params(required), "destructive LoRA changed a retained weight");
    require(manager.set_loras({}, VERSION_MINIMAX_H3), "LoRA source removal failed");
    const char * replacement_path = "bf16-retention-replacement.gguf";
    struct RemoveReplacement {
        const char * path;
        ~RemoveReplacement() {
            std::remove(path);
        }
    } remove{replacement_path};
    write_fixture(replacement_path, GGML_TYPE_F16);
    ModelLoader replacement;
    require(replacement.add_file(replacement_path), "replacement fixture inspection failed");
    require(manager.set_loader(std::move(replacement)), "idle source replacement failed");
    require(!manager.load_all_params_eagerly(), "retained source silently changed from BF16 to F16");
    require(manager.unregister_param_tensors(ModelComponent::Diffusion, &registered) && registered == 0,
            "retained registration cleanup failed");
    for (const auto & entry : tensors)
        require(!manager.is_browser_retained_bf16_param(entry.second), "unregistered weight retained stale identity");
}

void graph_contracts() {
    auto params = context();
    auto graphs = context();
    auto * weight = ggml_new_tensor_2d(params.get(), GGML_TYPE_BF16, 32, 32);
    auto * norm = ggml_new_tensor_1d(params.get(), GGML_TYPE_BF16, 32);
    require(sd::browser_bf16_matrix_role(weight, GGML_OP_MUL_MAT), "Linear matrix role rejected");
    require(sd::browser_bf16_matrix_role(weight, GGML_OP_GET_ROWS), "Embedding role rejected");
    require(!sd::browser_bf16_matrix_role(weight, GGML_OP_NONE), "shape alone admitted an unknown role");
    require(!sd::browser_bf16_matrix_role(norm, GGML_OP_MUL_MAT), "one-dimensional norm admitted as a matrix");
    auto * input = ggml_new_tensor_2d(graphs.get(), GGML_TYPE_F32, 32, 2);
    auto make_graph = [&](ggml_tensor * output) {
        auto * graph = ggml_new_graph_custom(graphs.get(), 64, false);
        ggml_build_forward_expand(graph, output);
        return graph;
    };
    auto * output = ggml_mul_mat(graphs.get(), weight, input);
    auto * graph = make_graph(output);
    sd::BrowserBf16RetainedGraph::validate_unallocated(graph, {weight});
    auto * indices = ggml_new_tensor_1d(graphs.get(), GGML_TYPE_I32, 3);
    sd::BrowserBf16RetainedGraph::validate_unallocated(make_graph(ggml_get_rows(graphs.get(), weight, indices)),
                                                       {weight});
    // Logical H3 table sizes only: no payload or GPU allocation.
    auto * h3_embedding = ggml_new_tensor_2d(params.get(), GGML_TYPE_BF16, 5120, 151936);
    require(ggml_nbytes(h3_embedding) == size_t(1555824640), "H3 retained embedding unexpectedly widened");
    sd::BrowserBf16RetainedGraph::validate_unallocated(make_graph(ggml_get_rows(graphs.get(), h3_embedding, indices)),
                                                       {h3_embedding});
    auto * too_wide = ggml_new_tensor_2d(params.get(), GGML_TYPE_BF16, 5120, 500000);
    rejected(
        [&] {
            sd::BrowserBf16RetainedGraph::validate_unallocated(
                make_graph(ggml_get_rows(graphs.get(), too_wide, indices)), {too_wide});
        },
        "unreviewed over-4GiB source span accepted");
    rejected(
        [&] {
            sd::BrowserBf16RetainedGraph::validate_unallocated(
                make_graph(ggml_cast(graphs.get(), weight, GGML_TYPE_F32)), {weight});
        },
        "hidden F32 cast accepted");
    rejected(
        [&] {
            sd::BrowserBf16RetainedGraph::validate_unallocated(make_graph(ggml_dup(graphs.get(), weight)), {weight});
        },
        "full weight copy accepted");
    auto * transposed = ggml_transpose(graphs.get(), weight);
    auto * transposed_use = ggml_mul_mat(graphs.get(), weight, input);
    // Inject invalid metadata after the valid builder call so the guard, rather
    // than a ggml constructor assertion, owns this rejection test.
    transposed_use->src[0] = transposed;
    rejected([&] { sd::BrowserBf16RetainedGraph::validate_unallocated(make_graph(transposed_use), {weight}); },
             "non-contiguous weight view accepted");
    auto * excessive = ggml_new_tensor_1d(graphs.get(), GGML_TYPE_I32, 65536);
    rejected(
        [&] {
            sd::BrowserBf16RetainedGraph::validate_unallocated(
                make_graph(ggml_get_rows(graphs.get(), weight, excessive)), {weight});
        },
        "oversized lookup dispatch accepted");
    auto * saved_input = output->src[1];
    output->src[1] = weight;
    rejected([&] { sd::BrowserBf16RetainedGraph::validate_unallocated(graph, {weight}); },
             "BF16 weight used as a right operand");
    output->src[1] = saved_input;
    auto * view = ggml_view_2d(graphs.get(), weight, 32, 16, weight->nb[1], 0);
    view->view_offs = ggml_nbytes(weight);
    rejected(
        [&] {
            sd::BrowserBf16RetainedGraph::validate_unallocated(make_graph(ggml_mul_mat(graphs.get(), view, input)),
                                                               {weight});
        },
        "out-of-bounds weight view accepted");
}

void allocation_contracts() {
    FakeWebgpu fake;
    auto weights = context();
    auto graph_ctx = context();
    auto * weight = ggml_new_tensor_2d(weights.get(), GGML_TYPE_BF16, 32, 32);
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(weights.get(), &fake.backend), ggml_backend_buffer_free);
    require(bool(buffer), "fake weight allocation failed");
    ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    auto * input = ggml_new_tensor_2d(graph_ctx.get(), GGML_TYPE_F32, 32, 2);
    ggml_set_input(input);
    auto * output = ggml_mul_mat(graph_ctx.get(), weight, input);
    ggml_set_output(output);
    auto * graph = ggml_new_graph_custom(graph_ctx.get(), 64, false);
    ggml_build_forward_expand(graph, output);
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu(ggml_backend_cpu_init(), ggml_backend_free);
    ggml_backend_t backends[] = {&fake.backend, cpu.get()};
    std::unique_ptr<ggml_backend_sched, decltype(&ggml_backend_sched_free)> scheduler(
        ggml_backend_sched_new(backends, nullptr, 2, 128, false, false), ggml_backend_sched_free);
    sd::BrowserBf16RetainedGraph retained;
    fake.admit = false;
    rejected([&] { retained.prepare(graph, scheduler.get(), &fake.backend, {weight}); },
             "unsupported actual weight use accepted");
    require(ggml_backend_sched_get_tensor_backend(scheduler.get(), output) == nullptr,
            "failed admission assigned half a graph");
    fake.admit = true;
    retained.prepare(graph, scheduler.get(), &fake.backend, {weight});
    require(ggml_backend_sched_alloc_graph(scheduler.get(), graph), "fake scheduler allocation failed");
    retained.verify_allocated(scheduler.get());
    auto * replacement = ggml_dup_tensor(graph_ctx.get(), weight);
    output->src[0] = replacement;
    rejected([&] { retained.verify_allocated(scheduler.get()); }, "scheduler weight copy accepted");
    output->src[0] = weight;
    ggml_backend_sched_set_tensor_backend(scheduler.get(), output, cpu.get());
    rejected([&] { retained.verify_allocated(scheduler.get()); }, "CPU placement accepted");
    ggml_backend_sched_set_tensor_backend(scheduler.get(), output, &fake.backend);
    auto * saved = output->buffer;
    output->buffer = weight->buffer;
    rejected([&] { retained.verify_allocated(scheduler.get()); }, "weight/activation buffer overlap accepted");
    output->buffer = saved;
    retained.verify_allocated(scheduler.get());
}
void legacy_allocation_contract() {
    FakeWebgpu fake;
    auto weights = context();
    auto graph_ctx = context();
    auto * weight = ggml_new_tensor_2d(weights.get(), GGML_TYPE_BF16, 32, 32);
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(weights.get(), &fake.backend), ggml_backend_buffer_free);
    require(bool(buffer), "legacy fake weight allocation failed");
    ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    auto * input = ggml_new_tensor_2d(graph_ctx.get(), GGML_TYPE_F32, 32, 2);
    ggml_set_input(input);
    auto * output = ggml_mul_mat(graph_ctx.get(), weight, input);
    ggml_set_output(output);
    auto * graph = ggml_new_graph_custom(graph_ctx.get(), 64, false);
    ggml_build_forward_expand(graph, output);
    fake.admit = false;
    sd::ComputeWorkspace workspace(&fake.backend);
    require(workspace.allocate(graph, [](ggml_backend_sched_t, ggml_cgraph *) {}),
            "non-retained BF16 graph acquired a new rejection");
    require(ggml_backend_sched_get_tensor_backend(workspace.scheduler(), output) == workspace.cpu_backend(),
            "non-retained BF16 weight was unexpectedly forced onto WebGPU");
}
} // namespace

int main() {
    struct Cleanup {
        ~Cleanup() {
            std::remove(fixture_path);
        }
    } cleanup;
    try {
        write_fixture(fixture_path, GGML_TYPE_BF16);
        for (auto selected : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16}) {
            check_registration(selected, false, true);
            check_registration(selected, true, true);
        }
        check_registration(GGML_TYPE_BF16, false, false);
        changed_source_and_lora_contracts();
        graph_contracts();
        allocation_contracts();
        legacy_allocation_contract();
        std::cout << "BF16 retention storage/role/graph/scheduler contracts passed (host-only fake backend)\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
