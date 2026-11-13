// The production host writer is inserted after these controlled backend APIs.
// This exercises host offsets and failure lifetime, without a GPU or serializer.
#include "ggml-backend-tensor-get-batch.h"
#include "llama-io.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

static ggml_backend_dev_t device = reinterpret_cast<ggml_backend_dev_t>(1);
static ggml_backend_buffer_type_t buffer_type = reinterpret_cast<ggml_backend_buffer_type_t>(2);
static ggml_backend_reg_t registry = reinterpret_cast<ggml_backend_reg_t>(3);
static int batches = 0, scalars = 0;
static bool accept = true, fail = false;
static std::vector<void *> destinations;

ggml_backend_buffer_type_t ggml_backend_buffer_get_type(ggml_backend_buffer_t) {
    return buffer_type;
}
ggml_backend_dev_t ggml_backend_buft_get_device(ggml_backend_buffer_type_t) {
    return device;
}
ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t) {
    return registry;
}
void ggml_backend_tensor_get(const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ++scalars;
    destinations.push_back(data);
    std::memcpy(data, static_cast<const uint8_t *>(tensor->data) + offset, size);
}
static bool get_batch(const ggml_backend_tensor_get_request_v1 * requests, size_t count) {
    ++batches;
    if (!accept)
        return false;
    for (size_t i = 0; i < count; ++i) {
        const auto & request = requests[i];
        destinations.push_back(request.data);
        std::memcpy(request.data, static_cast<const uint8_t *>(request.tensor->data) + request.offset, request.size);
        if (fail)
            throw std::runtime_error("injected post-write failure");
    }
    return true;
}
void * ggml_backend_reg_get_proc_address(ggml_backend_reg_t, const char * name) {
    assert(std::strcmp(name, GGML_BACKEND_TENSOR_GET_BATCH_V1) == 0);
    return reinterpret_cast<void *>(get_batch);
}

// PRODUCTION_WRITER

int main() {
    std::array<uint8_t, 64> bytes{}, output{};
    for (size_t i = 0; i < bytes.size(); ++i)
        bytes[i] = static_cast<uint8_t>(17 + i);
    ggml_tensor tensor{};
    tensor.buffer = reinterpret_cast<ggml_backend_buffer_t>(4);
    tensor.data = bytes.data();
    const std::array<uint8_t, 3> header{1, 2, 3};
    for (bool supported : {false, true}) {
        accept = supported;
        output.fill(0xcd);
        batches = scalars = 0;
        destinations.clear();
        llama_io_write_host writer(output.data() + 2, output.size() - 4);
        writer.write(header.data(), header.size());
        writer.write_tensor(&tensor, 7, 5);
        writer.write(header.data(), header.size());
        writer.write_tensor(&tensor, 3, 9);
        assert(batches == 0 && scalars == 0);
        writer.flush();
        assert(batches == 1 && scalars == (supported ? 0 : 2));
        assert((destinations == std::vector<void *>{output.data() + 5, output.data() + 13}));
        assert(std::equal(bytes.begin() + 7, bytes.begin() + 12, output.begin() + 5));
        assert(std::equal(bytes.begin() + 3, bytes.begin() + 12, output.begin() + 13));
        assert(output[0] == 0xcd && output[1] == 0xcd && output[22] == 0xcd && output.back() == 0xcd);
        try {
            writer.flush();
            assert(false);
        } catch (const std::runtime_error &) {
        }
    }
    accept = true;
    fail = true;
    batches = scalars = 0;
    {
        llama_io_write_host writer(output.data(), output.size());
        writer.write_tensor(&tensor, 0, 5);
        writer.write_tensor(&tensor, 5, 5);
        try {
            writer.flush();
            assert(false);
        } catch (const std::runtime_error &) {
        }
        assert(batches == 1 && scalars == 0);
    }
    assert(batches == 1 && scalars == 0); // Destructor never retries caller-memory writes.
    fail = false;
    batches = scalars = 0;
    {
        llama_io_write_host writer(output.data(), output.size());
        writer.write_tensor(&tensor, 0, 5);
        writer.flush();
    }
    assert(batches == 0 && scalars == 1);
}
