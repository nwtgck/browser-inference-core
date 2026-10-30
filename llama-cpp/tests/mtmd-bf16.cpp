#include "mtmd-bf16.h"
#include "ggml-backend-impl.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>

static void require(bool condition, const char * message) {
    if (!condition)
        throw std::runtime_error(message);
}

static std::string bf16_data(size_t count) {
    std::string data(count * 2, '\0');
    for (size_t i = 0; i < count; ++i) {
        // Sweep all bit patterns, including zeros, extremes and NaNs.
        const uint16_t bits = static_cast<uint16_t>(i);
        std::memcpy(data.data() + i * 2, &bits, 2);
    }
    return data;
}

static void test_backend_identity() {
    ggml_backend_reg registry{};
    registry.api_version = GGML_BACKEND_API_VERSION;
    registry.iface.get_name = [](ggml_backend_reg_t) { return "WebGPU"; };
    ggml_backend_device device{};
    device.reg = &registry;
    ggml_backend backend{};
    backend.device = &device;
    backend.iface.get_name = [](ggml_backend_t) { return "WebGPU: adapter"; };
    require(std::strcmp(ggml_backend_name(&backend), "WebGPU") != 0,
            "fixture must exercise an adapter-suffixed display name");
    require(lcb_mtmd_is_webgpu(&backend), "WebGPU registry with a display suffix not recognized");
    registry.iface.get_name = [](ggml_backend_reg_t) { return "CPU"; };
    backend.iface.get_name = [](ggml_backend_t) { return "WebGPU"; };
    require(!lcb_mtmd_is_webgpu(&backend), "backend display name must not override registry identity");
    registry.iface.get_name = [](ggml_backend_reg_t) -> const char * { return nullptr; };
    require(!lcb_mtmd_is_webgpu(&backend), "missing registry name accepted");
    device.reg = nullptr;
    require(!lcb_mtmd_is_webgpu(&backend), "missing registry accepted");
    backend.device = nullptr;
    require(!lcb_mtmd_is_webgpu(&backend), "missing device accepted");
    require(!lcb_mtmd_is_webgpu(nullptr), "missing backend accepted");
}

int main() {
    try {
        test_backend_identity();
        lcb_bf16_upload upload;
        for (size_t count : {size_t(0), size_t(1), size_t(65536), lcb_bf16_upload::chunk_elements,
                             lcb_bf16_upload::chunk_elements + 3, lcb_bf16_upload::chunk_elements * 2 + 7}) {
            const std::string raw = bf16_data(count);
            std::istringstream input(raw + "next tensor");
            size_t written = 0;
            upload.read(input, raw.size(), [&](const float * values, size_t offset, size_t bytes) {
                require(offset == written, "non-contiguous destination offsets");
                require(bytes <= lcb_bf16_upload::chunk_elements * sizeof(float), "unbounded scratch chunk");
                for (size_t i = 0; i < bytes / 4; ++i) {
                    uint32_t actual;
                    std::memcpy(&actual, values + i, sizeof(actual));
                    const uint32_t expected = uint32_t(uint16_t(offset / 4 + i)) << 16;
                    // Native vector implementations may quiet signalling NaNs.
                    if ((expected & 0x7fffffffU) > 0x7f800000U) {
                        require((actual & 0x7fffffffU) > 0x7f800000U, "NaN class lost");
                    } else {
                        require(actual == expected, "BF16 value was rounded or changed");
                    }
                }
                written += bytes;
            });
            require(written == count * 4, "wrong destination size");
            require(input.peek() == 'n', "read into the following tensor");
        }
        for (size_t size : {size_t(2), lcb_bf16_upload::chunk_elements * 2 + 4}) {
            std::istringstream truncated(bf16_data(size / 2 - 1));
            bool failed = false;
            size_t bytes_written = 0;
            try {
                upload.read(truncated, size, [&](const float *, size_t, size_t bytes) { bytes_written += bytes; });
            } catch (const std::runtime_error &) {
                failed = true;
            }
            require(failed, "truncated input accepted");
            require(bytes_written < size * 2, "truncated chunk uploaded");
        }
        bool failed = false;
        std::istringstream odd("123");
        try {
            upload.read(odd, 3, [](const float *, size_t, size_t) {});
        } catch (const std::runtime_error &) {
            failed = true;
        }
        require(failed && odd.tellg() == 0, "misaligned size consumed input");
        failed = false;
        std::istringstream enormous;
        try {
            upload.read(enormous, std::numeric_limits<size_t>::max() - 1, [](const float *, size_t, size_t) {});
        } catch (const std::runtime_error &) {
            failed = true;
        }
        require(failed && enormous.tellg() == 0, "overflow was not rejected before reading");
        failed = false;
        std::istringstream input(bf16_data(lcb_bf16_upload::chunk_elements + 1));
        try {
            upload.read(input, (lcb_bf16_upload::chunk_elements + 1) * 2,
                        [](const float *, size_t, size_t) { throw std::runtime_error("upload failed"); });
        } catch (const std::runtime_error &) {
            failed = true;
        }
        require(failed && input.tellg() == lcb_bf16_upload::chunk_elements * 2, "upload failure swallowed");
        std::cout
            << "PASS: registry identity, bounded BF16 upload, all bit patterns, source boundaries, truncation, upload failures\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
