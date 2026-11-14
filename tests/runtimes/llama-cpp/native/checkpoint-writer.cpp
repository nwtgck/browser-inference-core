// Compile the entire prepared context. Its ordinary inline host writer uses
// real ggml declarations; only external backend callbacks are controlled.
#include "llama-context.cpp"
#include "ggml-backend-impl.h"
#include "support/check.hpp"
#include <array>
#include <cstdio>
#include <memory>

static unsigned batches, scalars;
static bool accept = true, fail_write;
static std::vector<void *> destinations;

static bool get_batch(const ggml_backend_tensor_get_request_v1 * requests, size_t count) {
    ++batches;
    if (!accept)
        return false;
    for (size_t index = 0; index < count; ++index) {
        const auto & request = requests[index];
        destinations.push_back(request.data);
        std::memcpy(request.data, static_cast<const uint8_t *>(request.tensor->data) + request.offset, request.size);
        if (fail_write)
            throw std::runtime_error("controlled post-write failure");
    }
    return true;
}

int main() {
    ggml_backend_reg registry{};
    registry.iface.get_proc_address = [](ggml_backend_reg_t, const char * name) -> void * {
        BICORE_CHECK(std::strcmp(name, GGML_BACKEND_TENSOR_GET_BATCH_V1) == 0);
        return reinterpret_cast<void *>(get_batch);
    };
    ggml_backend_device device{};
    device.reg = &registry;
    ggml_backend_buffer_type type{};
    type.device = &device;
    ggml_backend_buffer buffer{};
    buffer.buft = &type;
    buffer.iface.get_tensor = [](ggml_backend_buffer_t, const ggml_tensor * tensor, void * output, size_t offset,
                                 size_t size) {
        ++scalars;
        destinations.push_back(output);
        std::memcpy(output, static_cast<const uint8_t *>(tensor->data) + offset, size);
    };
    std::array<uint8_t, 64> bytes{}, output{};
    for (size_t index = 0; index < bytes.size(); ++index)
        bytes[index] = uint8_t(17 + index);
    ggml_tensor tensor{};
    tensor.buffer = &buffer;
    tensor.type = GGML_TYPE_I8;
    tensor.ne[0] = bytes.size();
    tensor.ne[1] = tensor.ne[2] = tensor.ne[3] = 1;
    tensor.nb[0] = 1;
    tensor.nb[1] = tensor.nb[2] = tensor.nb[3] = bytes.size();
    tensor.data = bytes.data();
    const std::array<uint8_t, 3> header{1, 2, 3};
    for (const bool supported : {false, true}) {
        accept = supported;
        output.fill(0xcd);
        batches = scalars = 0;
        destinations.clear();
        llama_io_write_host writer(output.data() + 2, output.size() - 4);
        writer.write(header.data(), header.size());
        writer.write_tensor(&tensor, 7, 5);
        writer.write(header.data(), header.size());
        writer.write_tensor(&tensor, 3, 9);
        BICORE_CHECK(batches == 0 && scalars == 0);
        writer.flush();
        BICORE_CHECK(batches == 1 && scalars == (supported ? 0 : 2));
        BICORE_CHECK(destinations == std::vector<void *>({output.data() + 5, output.data() + 13}));
        BICORE_CHECK(std::equal(bytes.begin() + 7, bytes.begin() + 12, output.begin() + 5));
        BICORE_CHECK(std::equal(bytes.begin() + 3, bytes.begin() + 12, output.begin() + 13));
        BICORE_CHECK(output[0] == 0xcd && output[1] == 0xcd && output[22] == 0xcd && output.back() == 0xcd);
        bool caught = false;
        try {
            writer.flush();
        } catch (const std::runtime_error &) {
            caught = true;
        }
        BICORE_CHECK(caught);
    }
    accept = true;
    fail_write = true;
    batches = scalars = 0;
    {
        llama_io_write_host writer(output.data(), output.size());
        writer.write_tensor(&tensor, 0, 5);
        writer.write_tensor(&tensor, 5, 5);
        bool caught = false;
        try {
            writer.flush();
        } catch (const std::runtime_error &) {
            caught = true;
        }
        BICORE_CHECK(caught && batches == 1 && scalars == 0);
    }
    BICORE_CHECK(batches == 1 && scalars == 0);
    fail_write = false;
    batches = scalars = 0;
    {
        llama_io_write_host writer(output.data(), output.size());
        writer.write_tensor(&tensor, 0, 5);
        writer.flush();
    }
    BICORE_CHECK(batches == 0 && scalars == 1);
    {
        // Exercise the context's file writer too: its out-of-line dependency
        // must link even when a compiler discards other serialization paths.
        const auto temporary = std::unique_ptr<FILE, int (*)(FILE *)>(std::tmpfile(), std::fclose);
        BICORE_CHECK(temporary);
        {
            llama_file file(temporary.get());
            llama_io_write_file writer(&file);
            batches = scalars = 0;
            writer.write(header.data(), header.size());
            writer.write_tensor(&tensor, 7, 5);
            BICORE_CHECK(writer.n_bytes() == header.size() + 5 && batches == 0 && scalars == 1);
        }
        llama_file reader(temporary.get());
        std::array<uint8_t, 8> restored{};
        BICORE_CHECK(reader.size() == restored.size());
        reader.read_raw(restored.data(), restored.size());
        BICORE_CHECK(std::equal(header.begin(), header.end(), restored.begin()));
        BICORE_CHECK(std::equal(bytes.begin() + 7, bytes.begin() + 12, restored.begin() + header.size()));
    }
}
