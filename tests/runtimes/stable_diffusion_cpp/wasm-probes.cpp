// Test builds only: exercise the actual patched GGUF reader and C++ stream.
// These probes neither allocate a model nor perform inference.
#include "stable-diffusion.h"
#include "model_io/gguf_io.h"
#include "core/util.h"
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <vector>
extern "C" {
uint64_t sdc_test_gguf_offset(const char * path) {
    std::vector<TensorStorage> tensors;
    std::string error;
    if (!read_gguf_file(path, tensors, &error) || tensors.empty())
        return UINT64_MAX;
    return tensors.back().offset;
}
uint32_t sdc_test_gguf_value(const char * path) {
    const uint64_t offset = sdc_test_gguf_offset(path);
    if (offset == UINT64_MAX)
        return 0;
    std::ifstream file(path, std::ios::binary);
    file.seekg(static_cast<std::streamoff>(offset));
    uint8_t data[4] = {};
    file.read(reinterpret_cast<char *>(data), sizeof(data));
    if (!file)
        return 0;
    return uint32_t(data[0]) | uint32_t(data[1]) << 8 | uint32_t(data[2]) << 16 | uint32_t(data[3]) << 24;
}
void sdc_test_callbacks() {
    log_printf(SD_LOG_INFO, __FILE__, __LINE__, "native callback probe");
    pretty_progress(1, 4, 0.125f);
    static uint64_t previous_window = 0;
    const auto epoch = sd_graph_diagnostics_epoch();
    if (epoch == 0) {
        uint64_t observed = 0;
        if (sd_begin_graph_diagnostics(observed) || observed != 0)
            throw std::runtime_error("disabled graph diagnostics changed state");
    } else {
        if (epoch == previous_window)
            throw std::runtime_error("enable did not start a new diagnostic window");
        previous_window = epoch;
        for (const char * outcome : {"end", "failed"}) {
            uint64_t observed = 0;
            if (!sd_begin_graph_diagnostics(observed))
                throw std::runtime_error("missing diagnostic window");
            sd_log_graph_stage("callback-probe", "attempt", "begin");
            sd_log_graph_stage("callback-probe", "attempt", outcome);
            if (sd_begin_graph_diagnostics(observed))
                throw std::runtime_error("diagnostic attempt repeated after completion/failure");
        }
    }
}
}

#include "model_io/safetensors_io.h"
#include "model_loader.h"
#include "ggml-cpu.h"
#include <cmath>
#include <memory>
extern "C" uint64_t sdc_test_safetensors_offset(const char * path) {
    std::vector<TensorStorage> tensors;
    std::string error;
    if (!read_safetensors_file(path, tensors, &error) || tensors.empty())
        return UINT64_MAX;
    return tensors.back().offset;
}
extern "C" uint32_t sdc_test_safetensors_value(const char * path) {
    const uint64_t offset = sdc_test_safetensors_offset(path);
    if (offset == UINT64_MAX)
        return 0;
    std::ifstream file(path, std::ios::binary);
    file.seekg(static_cast<std::streamoff>(offset));
    uint8_t b[4] = {};
    file.read(reinterpret_cast<char *>(b), 4);
    return file ? uint32_t(b[0]) | uint32_t(b[1]) << 8 | uint32_t(b[2]) << 16 | uint32_t(b[3]) << 24 : 0;
}
extern "C" uint32_t sdc_test_model_tensor_count(const char * path) {
    try {
        ModelLoader loader;
        if (!loader.init_from_file(path))
            return 0;
        return static_cast<uint32_t>(loader.get_tensor_storage_map().size());
    } catch (...) {
        return 0;
    }
}

extern "C" uint64_t sdc_test_gguf_shape(const char * path, const char * tensor_name, int dimension) {
    try {
        if (!path || !tensor_name || dimension < 0 || dimension >= GGML_MAX_DIMS)
            return UINT64_MAX;
        ModelLoader loader;
        if (!loader.init_from_file(path))
            return UINT64_MAX;
        const auto & tensors = loader.get_tensor_storage_map();
        const auto found = tensors.find(tensor_name);
        return found == tensors.end() ? UINT64_MAX : static_cast<uint64_t>(found->second.ne[dimension]);
    } catch (...) {
        return UINT64_MAX;
    }
}

// The self-authored fixture is F32 [32,2], with values 1..64. Exercise the
// production loader and CPU matrix arithmetic, not only a metadata-only probe.
extern "C" int sdc_test_gguf_original_shape_values(const char * path) {
    try {
        ModelLoader loader;
        loader.set_n_threads(1);
        if (!loader.init_from_file(path))
            return -1;
        const auto & tensors = loader.get_tensor_storage_map();
        const auto found = tensors.find("weight");
        if (tensors.size() != 1 || found == tensors.end() || found->second.type != GGML_TYPE_F32 ||
            found->second.ne[0] != 32 || found->second.ne[1] != 2 || found->second.ne[2] != 1 ||
            found->second.ne[3] != 1)
            return -2;
        std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({1024 * 1024, nullptr, false}), ggml_free);
        if (!ctx)
            return -3;
        auto weight = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 32, 2);
        std::map<std::string, ggml_tensor *> destinations{{"weight", weight}};
        if (!loader.load_tensors(destinations, {}, false))
            return -4;
        const auto values = static_cast<const float *>(weight->data);
        for (int i = 0; i < 64; ++i)
            if (values[i] != float(i + 1))
                return -5;
        auto input = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 32);
        ggml_set_f32(input, 1.f);
        auto output = ggml_mul_mat(ctx.get(), weight, input);
        auto graph = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(graph, output);
        if (ggml_graph_compute_with_ctx(ctx.get(), graph, 1) != GGML_STATUS_SUCCESS)
            return -6;
        const auto actual = static_cast<const float *>(output->data);
        return std::isfinite(actual[0]) && std::isfinite(actual[1]) && actual[0] == 528.f && actual[1] == 1552.f ? 1
                                                                                                                 : -7;
    } catch (...) {
        return -8;
    }
}
