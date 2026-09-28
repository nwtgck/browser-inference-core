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
uint64_t sdc_test_gguf_offset(const char* path) {
    std::vector<TensorStorage> tensors;
    std::string error;
    if (!read_gguf_file(path, tensors, &error) || tensors.empty()) return UINT64_MAX;
    return tensors.back().offset;
}
uint32_t sdc_test_gguf_value(const char* path) {
    const uint64_t offset = sdc_test_gguf_offset(path);
    if (offset == UINT64_MAX) return 0;
    std::ifstream file(path, std::ios::binary);
    file.seekg(static_cast<std::streamoff>(offset));
    uint8_t data[4] = {};
    file.read(reinterpret_cast<char*>(data), sizeof(data));
    if (!file) return 0;
    return uint32_t(data[0]) | uint32_t(data[1]) << 8 | uint32_t(data[2]) << 16 | uint32_t(data[3]) << 24;
}
void sdc_test_callbacks() {
    log_printf(SD_LOG_INFO, __FILE__, __LINE__, "native callback probe");
    pretty_progress(1, 4, 0.125f);
    static uint64_t previous_window = 0;
    const auto epoch = sd_graph_diagnostics_epoch();
    if (epoch == 0) {
        uint64_t observed = 0;
        if (sd_begin_graph_diagnostics(observed) || observed != 0) throw std::runtime_error("disabled graph diagnostics changed state");
    } else {
        if (epoch == previous_window) throw std::runtime_error("enable did not start a new diagnostic window");
        previous_window = epoch;
        for (const char* outcome : {"end", "failed"}) {
            uint64_t observed = 0;
            if (!sd_begin_graph_diagnostics(observed)) throw std::runtime_error("missing diagnostic window");
            sd_log_graph_stage("callback-probe", "attempt", "begin");
            sd_log_graph_stage("callback-probe", "attempt", outcome);
            if (sd_begin_graph_diagnostics(observed)) throw std::runtime_error("diagnostic attempt repeated after completion/failure");
        }
    }
}
}

#include "model_io/safetensors_io.h"
#include "model_loader.h"
extern "C" uint64_t sdc_test_safetensors_offset(const char* path) {
    std::vector<TensorStorage> tensors; std::string error;
    if (!read_safetensors_file(path, tensors, &error) || tensors.empty()) return UINT64_MAX;
    return tensors.back().offset;
}
extern "C" uint32_t sdc_test_safetensors_value(const char* path) {
    const uint64_t offset = sdc_test_safetensors_offset(path);
    if (offset == UINT64_MAX) return 0;
    std::ifstream file(path, std::ios::binary); file.seekg(static_cast<std::streamoff>(offset));
    uint8_t b[4] = {}; file.read(reinterpret_cast<char*>(b), 4);
    return file ? uint32_t(b[0]) | uint32_t(b[1]) << 8 | uint32_t(b[2]) << 16 | uint32_t(b[3]) << 24 : 0;
}
extern "C" uint32_t sdc_test_model_tensor_count(const char* path) {
    try {
        ModelLoader loader;
        if (!loader.init_from_file(path)) return 0;
        return static_cast<uint32_t>(loader.get_tensor_storage_map().size());
    } catch (...) { return 0; }
}
