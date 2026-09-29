// Tiny synthetic files exercise the production loader and parameter allocation.
// This is not trained-model image inference and is linked into test variants only.
#include "core/compute_workspace.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "model_manager.h"
#include "pipeline/diffusion_engine.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <vector>

using ProbeContext = std::unique_ptr<ggml_context, decltype(&ggml_free)>;

static std::string file_bytes(const char* path) {
    std::ifstream input(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(input), {});
}

static bool write_fixture(const char* path, bool safetensors, ggml_context* ctx) {
    auto weight = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, 32, 32);
    auto range = ggml_new_tensor_1d(ctx, GGML_TYPE_BF16, 8);
    auto control = ggml_new_tensor_1d(ctx, safetensors ? GGML_TYPE_F16 : GGML_TYPE_Q4_K, 256);
    ggml_set_name(weight, "weight"); ggml_set_name(range, "range"); ggml_set_name(control, "control");
    std::memset(weight->data, 0, ggml_nbytes(weight));
    std::memset(control->data, 0, ggml_nbytes(control));
    for (int i = 0; i < 32; ++i) static_cast<ggml_bf16_t*>(weight->data)[i * 32 + i] = ggml_fp32_to_bf16(0.75f);
    // F32 must preserve the BF16 exponent range; opt-in F16 may overflow/underflow.
    const float values[] = {0.f, -0.f, 1.f, -2.f, 65536.f, -65536.f, 0x1p-30f, -0x1p-30f};
    ggml_fp32_to_bf16_row(values, static_cast<ggml_bf16_t*>(range->data), 8);
    if (!safetensors) {
        std::unique_ptr<gguf_context, decltype(&gguf_free)> file(gguf_init_empty(), gguf_free);
        for (auto tensor : {weight, range, control}) gguf_add_tensor(file.get(), tensor);
        return gguf_write_to_file(file.get(), path, false);
    }
    const std::string header =
        "{\"weight\":{\"dtype\":\"BF16\",\"shape\":[32,32],\"data_offsets\":[0,2048]},"
        "\"range\":{\"dtype\":\"BF16\",\"shape\":[8],\"data_offsets\":[2048,2064]},"
        "\"control\":{\"dtype\":\"F16\",\"shape\":[256],\"data_offsets\":[2064,2576]}}";
    std::ofstream file(path, std::ios::binary);
    for (int i = 0; i < 8; ++i) file.put(char(uint64_t(header.size()) >> (i * 8)));
    file << header;
    for (auto tensor : {weight, range, control}) file.write(static_cast<const char*>(tensor->data), ggml_nbytes(tensor));
    return bool(file);
}

static int check_file(ggml_backend_t backend, bool safetensors, ggml_type target_type) {
    const char* path = safetensors ? "bf16-weights.safetensors" : "bf16-weights.gguf";
    struct RemoveFile { const char* path; ~RemoveFile() { std::remove(path); } } cleanup{path};
    ProbeContext source(ggml_init({1024 * 1024, nullptr, false}), ggml_free);
    ProbeContext params(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
    ProbeContext graph_ctx(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
    if (!source || !params || !graph_ctx || !write_fixture(path, safetensors, source.get())) return -20;
    const auto original = file_bytes(path);
    ModelManager manager;
    manager.set_n_threads(1);
    manager.set_enable_mmap(false);
    manager.set_prefetch_disabled(true);
    manager.set_webgpu_bf16_type(target_type);
    if (!manager.add_file(path)) return -21;
    std::map<std::string, ggml_tensor*> weights;
    for (const auto& entry : manager.loader().get_tensor_storage_map()) {
        weights[entry.first] = ggml_new_tensor(params.get(), entry.second.type, entry.second.n_dims, entry.second.ne);
    }
    size_t registered_bytes = 0;
    if (!ggml_backend_is_cpu(backend)) {
        const auto stride = weights.at("weight")->nb[1];
        weights.at("weight")->nb[1] += 4;
        if (manager.register_param_tensors(ModelComponent::Diffusion, weights, ModelManager::ResidencyMode::ParamBackend,
                                          backend, backend, &registered_bytes) || registered_bytes != 0 ||
            weights.at("range")->type != GGML_TYPE_BF16 || weights.at("weight")->type != GGML_TYPE_BF16) return -34;
        weights.at("weight")->nb[1] = stride;
    }
    if (!manager.register_param_tensors(ModelComponent::Diffusion, weights, ModelManager::ResidencyMode::ParamBackend,
                                       backend, backend, &registered_bytes)) return -22;
    const auto expected_type = ggml_backend_is_cpu(backend) ? GGML_TYPE_BF16 : target_type;
    const auto control_type = safetensors ? GGML_TYPE_F16 : GGML_TYPE_Q4_K;
    const size_t expected_bytes = 1032 * ggml_type_size(expected_type) + ggml_row_size(control_type, 256);
    if (registered_bytes != expected_bytes || manager.registered_params_size({ModelComponent::Diffusion}) != expected_bytes ||
        weights.at("weight")->type != expected_type || weights.at("range")->type != expected_type ||
        weights.at("control")->type != control_type || !ggml_is_contiguous(weights.at("weight"))) return -23;
    const auto before_load = manager.memory_info();
    if (before_load.registered_tensor_count != 3 || before_load.registered_tensor_bytes != expected_bytes ||
        before_load.manager_host_buffer_bytes != 0 || before_load.manager_device_buffer_bytes != 0 || before_load.saturated) return -35;
    if (manager.loader().get_tensor_storage_map().at("weight").type != GGML_TYPE_BF16 || !manager.load_all_params_eagerly()) return -24;
    const auto after_load = manager.memory_info();
    std::set<ggml_backend_buffer_t> buffers;
    uint64_t host_bytes = 0, device_bytes = 0, host_count = 0, device_count = 0;
    for (const auto& weight : weights) {
        const auto buffer = weight.second->buffer;
        if (!buffer || !buffers.insert(buffer).second) continue;
        const bool host = ggml_backend_buffer_is_host(buffer);
        (host ? host_bytes : device_bytes) += ggml_backend_buffer_get_size(buffer);
        ++(host ? host_count : device_count);
    }
    if (buffers.empty() || after_load.manager_host_buffer_bytes != host_bytes ||
        after_load.manager_device_buffer_bytes != device_bytes || after_load.manager_host_buffer_count != host_count ||
        after_load.manager_device_buffer_count != device_count || after_load.registered_tensor_bytes != expected_bytes ||
        after_load.saturated) return -36;
    // Synthetic last-published reports, not allocations: sum beyond 4 GiB on wasm32 too.
    manager.update_runtime_residency(1, backend, size_t(3) << 30);
    manager.update_runtime_residency(2, backend, size_t(2) << 30);
    const auto runtime = manager.memory_info();
    const bool cpu = ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_CPU;
    if ((cpu ? runtime.tracked_runtime_cpu_bytes : runtime.tracked_runtime_non_cpu_bytes) != (uint64_t(5) << 30) ||
        runtime.tracked_runtime_unknown_bytes != 0 || runtime.manager_host_buffer_bytes != host_bytes ||
        runtime.manager_device_buffer_bytes != device_bytes) return -37;
    manager.update_runtime_residency(1, backend, 1024);
    manager.update_runtime_residency(2, backend, 0);
    const auto replaced = manager.memory_info();
    if ((cpu ? replaced.tracked_runtime_cpu_bytes : replaced.tracked_runtime_non_cpu_bytes) != 1024) return -38;
    manager.update_runtime_residency(1, backend, 0);
    if (SIZE_MAX == UINT64_MAX) {
        manager.update_runtime_residency(1, backend, SIZE_MAX);
        manager.update_runtime_residency(2, backend, SIZE_MAX);
        const auto overflow = manager.memory_info();
        if (!overflow.saturated || (cpu ? overflow.tracked_runtime_cpu_bytes : overflow.tracked_runtime_non_cpu_bytes) != UINT64_MAX) return -42;
        manager.update_runtime_residency(1, backend, 0);
        manager.update_runtime_residency(2, backend, 0);
    }

    std::vector<uint8_t> loaded(ggml_nbytes(weights.at("range")));
    ggml_backend_tensor_get(weights.at("range"), loaded.data(), 0, loaded.size());
    const auto original_range = static_cast<const ggml_bf16_t*>(ggml_get_tensor(source.get(), "range")->data);
    for (int i = 0; i < 8; ++i) {
        float value = ggml_bf16_to_fp32(original_range[i]);
        if (expected_type == GGML_TYPE_F32) {
            if (std::memcmp(loaded.data() + i * sizeof(float), &value, sizeof(float)) != 0) return -25;
        } else if (expected_type == GGML_TYPE_F16) {
            auto narrowed = ggml_fp32_to_fp16(value);
            if (std::memcmp(loaded.data() + i * sizeof(narrowed), &narrowed, sizeof(narrowed)) != 0) return -26;
        } else if (std::memcmp(loaded.data() + i * sizeof(ggml_bf16_t), &original_range[i], sizeof(ggml_bf16_t)) != 0) return -27;
    }
    auto input = ggml_new_tensor_2d(graph_ctx.get(), GGML_TYPE_F32, 32, 2);
    ggml_set_input(input);
    auto output = ggml_mul_mat(graph_ctx.get(), weights.at("weight"), input);
    ggml_set_output(output);
    auto graph = ggml_new_graph_custom(graph_ctx.get(), 32, false);
    ggml_build_forward_expand(graph, output);
    sd::ComputeWorkspace workspace(backend);
    if (!workspace.allocate(graph, {})) return -28;
    auto scheduler = workspace.scheduler();
    if (workspace.cpu_backend() != nullptr) ggml_backend_cpu_set_n_threads(workspace.cpu_backend(), 1);
    auto assigned = scheduler != nullptr ? ggml_backend_sched_get_tensor_backend(scheduler, output) : backend;
    if (assigned != backend || !ggml_backend_supports_op(backend, output)) return -29;
    std::vector<float> input_values(64), actual(64);
    for (size_t i = 0; i < input_values.size(); ++i) input_values[i] = (int(i) - 32) / 16.f;
    ggml_backend_tensor_set(input, input_values.data(), 0, input_values.size() * sizeof(float));
    auto status = scheduler != nullptr ? ggml_backend_sched_graph_compute(scheduler, graph) : ggml_backend_graph_compute(backend, graph);
    workspace.synchronize();
    if (status != GGML_STATUS_SUCCESS) return -30;
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || std::abs(actual[i] - input_values[i] * 0.75f) > 0.002f) return -31;
    }
    if (file_bytes(path) != original) return -32;
    if (!manager.unregister_param_tensors(ModelComponent::Diffusion, &registered_bytes) || registered_bytes != 0) return -33;
    const auto released = manager.memory_info();
    if (released.registered_tensor_count != 0 || released.registered_tensor_bytes != 0 ||
        released.manager_host_buffer_bytes != 0 || released.manager_device_buffer_bytes != 0 ||
        released.tracked_runtime_cpu_bytes != 0 || released.tracked_runtime_non_cpu_bytes != 0) return -39;

    return 1;
}

extern "C" int sdc_test_bf16_weights(const char* backend_name) {
    // Configuration snapshot pointers must belong to the context, not its caller.
    sd_ctx_params_t params{};
    sd_ctx_params_init(&params);
    char audio_path[] = "/models/audio.safetensors";
    params.audio_encoder_path = audio_path;
    StableDiffusionGGML::ModelConfig config(params);
    if (config.params.audio_encoder_path == audio_path || std::strcmp(config.params.audio_encoder_path, audio_path) != 0) return -40;
    audio_path[0] = 'X';
    if (config.params.audio_encoder_path[0] != '/') return -41;

    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
        ggml_backend_init_by_name(backend_name, nullptr), ggml_backend_free);
    if (!backend) return -1;
    if (ggml_backend_is_cpu(backend.get())) ggml_backend_cpu_set_n_threads(backend.get(), 1);
    for (bool safetensors : {false, true}) {
        for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
            const int result = check_file(backend.get(), safetensors, type);
            if (result != 1) return result;
        }
    }
    return 1;
}
