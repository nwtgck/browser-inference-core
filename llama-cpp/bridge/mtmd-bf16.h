#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <istream>
#include <limits>
#include <stdexcept>
#include <vector>

// Backend display names include an adapter suffix ("WebGPU: ..."). Match the
// registry identity instead, without requiring the WebGPU library in host tests.
inline bool lcb_mtmd_is_webgpu(ggml_backend_t backend) {
    if (!backend) return false;
    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    ggml_backend_reg_t registry = device ? ggml_backend_dev_backend_reg(device) : nullptr;
    const char * name = registry ? ggml_backend_reg_name(registry) : nullptr;
    return name && std::strcmp(name, "WebGPU") == 0;
}

// Legacy expansion of Brain Floating Point 16 (BF16) vision weights. The current
// upstream has BF16 kernels, but target-model speed/quality acceptance remains
// separate from storage support. Expand weights only, not the model file.
// This conversion preserves finite BF16 values; it does not change the existing
// WebGPU kernels' intermediate precision. Resident converted weights double in
// size; bounded scratch does not remove that cost. Keep scratch space bounded
// and reuse it across tensors rather than staging a second complete weight.
// See docs/webgpu-bf16-projector.md before retaining this on an upstream update.
class lcb_bf16_upload {
public:
    // 256K elements need 0.5 MiB of BF16 input + 1 MiB of F32 output scratch.
    static constexpr size_t chunk_elements = 256 * 1024;

    // source_bytes counts on-disk BF16 bytes; Write receives F32 byte offsets
    // and lengths. Write must consume/copy the chunk before returning because
    // the next iteration reuses it; it must not retain destination.data().
    template <typename Write>
    void read(std::istream & input, size_t source_bytes, Write write) {
        if (source_bytes % sizeof(ggml_bf16_t) != 0) {
            throw std::runtime_error("unaligned BF16 tensor data");
        }
        if (source_bytes > std::numeric_limits<size_t>::max() / 2) {
            throw std::runtime_error("expanded BF16 tensor exceeds the address space");
        }
        const size_t elements = source_bytes / sizeof(ggml_bf16_t);
        if (elements != 0 && source.empty()) {
            // Allocate once at the bound. Repeated resize(count) could grow a
            // vector's capacity geometrically beyond the stated scratch limit.
            source.resize(chunk_elements);
            destination.resize(chunk_elements);
        }
        for (size_t offset = 0; offset < elements;) {
            const size_t count = std::min(chunk_elements, elements - offset);
            input.read(reinterpret_cast<char *>(source.data()), count * sizeof(ggml_bf16_t));
            if (!input) {
                throw std::runtime_error("truncated BF16 tensor data");
            }
            ggml_bf16_to_fp32_row(source.data(), destination.data(), count);
            write(destination.data(), offset * sizeof(float), count * sizeof(float));
            offset += count;
        }
    }

private:
    std::vector<ggml_bf16_t> source;
    std::vector<float> destination;
};
