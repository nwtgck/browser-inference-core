#include "ggml.h"
#include "gguf.h"
#include "wasm-probes.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct Tensor {
    uint64_t elements;
    ggml_type type;
    uint64_t offset;
};

struct Reader {
    std::vector<uint8_t> header;
    size_t reads = 0;
    size_t payload_reads = 0;

    void integer(uint64_t value, size_t bytes) {
        for (size_t i = 0; i < bytes; ++i) {
            header.push_back(static_cast<uint8_t>(value >> (8 * i)));
        }
    }
};

Reader metadata(const std::vector<Tensor> & tensors) {
    Reader result;
    result.header = {'G', 'G', 'U', 'F'};
    result.integer(GGUF_VERSION, 4);
    result.integer(tensors.size(), 8);
    result.integer(0, 8);
    for (size_t i = 0; i < tensors.size(); ++i) {
        const std::string name = "tensor." + std::to_string(i);
        result.integer(name.size(), 8);
        result.header.insert(result.header.end(), name.begin(), name.end());
        result.integer(1, 4);
        result.integer(tensors[i].elements, 8);
        result.integer(tensors[i].type, 4);
        result.integer(tensors[i].offset, 8);
    }
    result.header.resize((result.header.size() + GGUF_DEFAULT_ALIGNMENT - 1) / GGUF_DEFAULT_ALIGNMENT * GGUF_DEFAULT_ALIGNMENT);
    return result;
}

size_t read_header(void * userdata, void * output, uint64_t offset, size_t size) {
    auto & reader = *static_cast<Reader *>(userdata);
    ++reader.reads;
    check(size <= 1024, "GGUF reader exceeded the requested read bound");
    if (offset >= reader.header.size()) {
        ++reader.payload_reads;
        return 0;
    }
    const size_t count = std::min(size, reader.header.size() - static_cast<size_t>(offset));
    std::memcpy(output, reader.header.data() + static_cast<size_t>(offset), count);
    return count;
}

using Gguf = std::unique_ptr<gguf_context, decltype(&gguf_free)>;

void large_metadata() {
    constexpr uint64_t tensor_bytes = 32 * 1024 * 1024;
    std::vector<Tensor> tensors;
    for (uint64_t i = 0; i < 130; ++i) {
        tensors.push_back({tensor_bytes / sizeof(float), GGML_TYPE_F32, i * tensor_bytes});
    }
    Reader reader = metadata(tensors);
    check(reader.header.size() < 16 * 1024, "large GGUF regression allocated a large header");
    ggml_context * context = nullptr;
    Gguf parsed(gguf_init_from_callback(read_header, &reader, 1024, reader.header.size() + tensors.size() * tensor_bytes,
                                      {true, &context}), gguf_free);
    const std::unique_ptr<ggml_context, decltype(&ggml_free)> descriptors(context, ggml_free);
    check(parsed != nullptr && descriptors != nullptr, "metadata-only GGUF rejected aggregate tensor data above 4 GiB");
    check(gguf_get_n_tensors(parsed.get()) == static_cast<int64_t>(tensors.size()), "large GGUF lost tensor descriptors");
    check(gguf_get_tensor_offset(parsed.get(), 129) == 129 * tensor_bytes, "tensor file offset above 4 GiB was narrowed");
    check(gguf_get_data_offset(parsed.get()) == reader.header.size(), "GGUF data section offset changed");
    check(ggml_get_tensor(descriptors.get(), "tensor.129") != nullptr, "no-alloc GGUF omitted its last descriptor");
    check(reader.payload_reads == 0, "metadata-only GGUF read tensor payload");

    if (sizeof(size_t) == 4) {
        reader.reads = 0;
        context = nullptr;
        Gguf blob(gguf_init_from_callback(read_header, &reader, 1024, reader.header.size() + tensors.size() * tensor_bytes,
                                        {false, &context}), gguf_free);
        if (context != nullptr) {
            ggml_free(context);
        }
        check(blob == nullptr && context == nullptr, "contiguous GGUF blob exceeded the native allocation range");
        check(reader.payload_reads == 0, "oversized contiguous GGUF read payload before rejecting allocation");
    }

    tensors[1].offset += GGUF_DEFAULT_ALIGNMENT;
    Reader invalid = metadata(tensors);
    Gguf rejected(gguf_init_from_callback(read_header, &invalid, 1024, UINT64_MAX, {true, nullptr}), gguf_free);
    check(rejected == nullptr, "GGUF accepted noncontiguous tensor offsets");
}

void malformed_ranges() {
    const std::vector<std::vector<Tensor>> invalid = {
        // Each descriptor is representable on native64; its padded aggregate is not.
        {{INT64_MAX, GGML_TYPE_I8, 0}, {INT64_MAX, GGML_TYPE_I8, uint64_t{1} << 63}},
        // The aggregate fits uint64_t, but adding the actual data-section offset does not.
        {{uint64_t{INT64_MAX} - 31, GGML_TYPE_I8, 0}, {uint64_t{INT64_MAX} - 31, GGML_TYPE_I8, uint64_t{INT64_MAX} - 31}},
        // Padding an otherwise native64-representable tensor must still reject overflow.
        {{UINT64_MAX / sizeof(double), GGML_TYPE_F64, 0}},
    };
    for (const auto & tensors : invalid) {
        Reader reader = metadata(tensors);
        Gguf parsed(gguf_init_from_callback(read_header, &reader, 1024, UINT64_MAX, {true, nullptr}), gguf_free);
        check(parsed == nullptr, "GGUF accepted an overflowing tensor data range");
        check(reader.payload_reads == 0, "invalid GGUF metadata read tensor payload");
    }
}
} // namespace

extern "C" int spc_test_gguf_large_file(void) {
    try {
        large_metadata();
        malformed_ranges();
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}

#ifdef SPCB_GGUF_TEST_MAIN
int main() {
    return spc_test_gguf_large_file();
}
#endif
