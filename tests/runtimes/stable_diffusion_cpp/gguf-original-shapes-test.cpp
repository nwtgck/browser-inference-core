// Self-authored GGUFs exercise the production loader, shape validation and CPU
// arithmetic. They do not contain trained weights or perform model inference.
#include "browser-probes.h"
#include "ggml.h"
#include "model_io/gguf_io.h"
#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

static void check(bool value, const char * message) {
    if (!value)
        throw std::runtime_error(message);
}

static void integer(std::ostream & out, uint64_t value, int width) {
    for (int i = 0; i < width; ++i)
        out.put(char(value >> (8 * i)));
}

static void text(std::ostream & out, const std::string & value) {
    integer(out, value.size(), 8);
    out.write(value.data(), value.size());
}

struct Metadata {
    std::string key;
    std::vector<uint64_t> values;
    uint32_t element = 5; // Comfy original-shape metadata uses INT32.
    uint32_t type = 9;
    std::optional<uint64_t> count;
};

static Metadata shape(std::vector<uint64_t> dimensions, const char * tensor = "weight", uint32_t element = 5) {
    return {std::string("comfy.gguf.orig_shape.") + tensor, std::move(dimensions), element, 9, std::nullopt};
}

static void fixture(const char * path, const std::vector<uint64_t> & descriptor, const std::vector<Metadata> & metadata,
                    ggml_type type = GGML_TYPE_F32, const char * tensor = "weight") {
    std::ofstream out(path, std::ios::binary);
    integer(out, 0x46554747, 4);
    integer(out, 3, 4);
    integer(out, 1, 8);
    integer(out, metadata.size(), 8);
    for (const auto & item : metadata) {
        text(out, item.key);
        integer(out, item.type, 4);
        if (item.type != 9) {
            integer(out, item.values.front(), 4);
            continue;
        }
        integer(out, item.element, 4);
        integer(out, item.count.value_or(item.values.size()), 8);
        const int width = item.element <= 1 || item.element == 7 ? 1
                          : item.element <= 3                    ? 2
                          : item.element <= 6                    ? 4
                                                                 : 8;
        for (const auto value : item.values)
            integer(out, value, width);
    }
    text(out, tensor);
    integer(out, descriptor.size(), 4);
    uint64_t elements = 1;
    for (const auto dimension : descriptor) {
        integer(out, dimension, 8);
        elements *= dimension;
    }
    integer(out, type, 4);
    integer(out, 0, 8);
    const uint64_t start = (uint64_t(out.tellp()) + 31) / 32 * 32;
    out.seekp(static_cast<std::streamoff>(start));
    const uint64_t bytes = elements / ggml_blck_size(type) * ggml_type_size(type);
    if (type == GGML_TYPE_F32 && elements == 64) {
        for (int i = 0; i < 64; ++i) {
            const float value = float(i + 1);
            uint32_t bits;
            std::memcpy(&bits, &value, sizeof(bits));
            integer(out, bits, 4);
        }
    } else {
        // Large shape checks stay sparse; only the small F32 fixture is loaded.
        out.seekp(static_cast<std::streamoff>(start + bytes - 1));
        out.put(0);
    }
    check(bool(out), "could not write original-shape fixture");
}

static void dimensions(const char * path, const std::array<uint64_t, 4> & expected, const char * tensor = "weight") {
    for (int dimension = 0; dimension < 4; ++dimension)
        check(sdc_test_gguf_shape(path, tensor, dimension) == expected[dimension], "wrong restored GGUF dimensions");
}

int main() {
#ifdef SDCB_TEST_GGUF_GENERIC_FALLBACK
    const char * path = "gguf-original-shapes-fallback.gguf";
    const char * first = "gguf-original-shapes-fallback-00001-of-00002.gguf";
    const char * second = "gguf-original-shapes-fallback-00002-of-00002.gguf";
#else
    const char * path = "gguf-original-shapes.gguf";
    const char * first = "gguf-original-shapes-00001-of-00002.gguf";
    const char * second = "gguf-original-shapes-00002-of-00002.gguf";
#endif
    struct Cleanup {
        const char * path;
        const char * first;
        const char * second;
        ~Cleanup() {
            std::remove(path);
            std::remove(first);
            std::remove(second);
        }
    } cleanup{path, first, second};
    try {
        fixture(path, {32, 2}, {});
        dimensions(path, {32, 2, 1, 1});
        check(sdc_test_gguf_original_shape_values(path) == 1, "ordinary GGUF numeric load regressed");
        for (uint32_t element : {0, 1, 2, 3, 4, 5, 10, 11}) {
            fixture(path, {64}, {shape({2, 32}, "weight", element)});
            dimensions(path, {32, 2, 1, 1});
            check(sdc_test_gguf_original_shape_values(path) == 1, "restored GGUF numeric load failed");
        }
        fixture(path, {32, 2}, {shape({2, 32})});
        check(sdc_test_gguf_original_shape_values(path) == 1, "consistent descriptor metadata rejected");
        fixture(path, {64}, {shape({1, 64})});
        dimensions(path, {64, 1, 1, 1});
        fixture(path, {64}, {shape({2, 2, 4, 4})});
        dimensions(path, {4, 4, 2, 2});
        fixture(path, {512}, {shape({2, 256})}, GGML_TYPE_Q4_K);
        dimensions(path, {256, 2, 1, 1});
        fixture(path, {3096576}, {shape({576, 5376})});
        dimensions(path, {5376, 576, 1, 1});
        fixture(path, {64}, {shape({2, 32}, "another-shard-weight")});
        dimensions(path, {64, 1, 1, 1});

        std::vector<Metadata> shared{shape({2, 32}, "weight0"), shape({2, 32}, "weight1")};
        for (int index = 0; index < 2; ++index) {
            auto metadata = shared;
            metadata.push_back({"split.no", {uint64_t(index)}, 0, 4, std::nullopt});
            metadata.push_back({"split.count", {2}, 0, 4, std::nullopt});
            metadata.push_back({"split.tensors.count", {2}, 0, 4, std::nullopt});
            fixture(index == 0 ? first : second, {64}, metadata, GGML_TYPE_F32, index == 0 ? "weight0" : "weight1");
        }
        check(sdc_test_model_tensor_count(first) == 2, "original shapes broke standard shard groups");
        dimensions(first, {32, 2, 1, 1}, "weight0");
        dimensions(first, {32, 2, 1, 1}, "weight1");

        auto excessive_rank = shape({2, 32});
        excessive_rank.count = UINT64_MAX;
        auto scalar = shape({2});
        scalar.type = 4;
        const std::vector<std::vector<Metadata>> invalid{
            {shape({})},
            {shape({1, 1, 1, 1, 64})},
            {shape({0, 64})},
            {shape({UINT64_MAX, 64}, "weight", 1)},
            {shape({UINT64_MAX, 64}, "weight", 11)},
            {shape({uint64_t(INT64_MAX), 2}, "weight", 10)},
            {shape({2, 31})},
            {shape({2, 32}), shape({4, 16})},
            {shape({2, 32}, "")},
            {shape({2, 32}, "weight", 6)},
            {shape({2, 32}, "weight", 7)},
            {shape({0, 64}, "another-shard-weight")},
            {excessive_rank},
            {scalar},
        };
        for (const auto & metadata : invalid) {
            fixture(path, {64}, metadata);
            check(sdc_test_gguf_shape(path, "weight", 0) == UINT64_MAX,
                  "invalid original shape escaped through the production loader");
            check(sdc_test_model_tensor_count(path) == 0, "invalid original shape was accepted as a model");
        }
        fixture(path, {16, 4}, {shape({2, 32})});
        check(sdc_test_model_tensor_count(path) == 0, "conflicting multidimensional descriptor accepted");
        fixture(path, {512}, {shape({256, 2})}, GGML_TYPE_Q4_K);
        check(sdc_test_model_tensor_count(path) == 0, "quantization row block mismatch accepted");
#ifdef SDCB_TEST_GGUF_GENERIC_FALLBACK
        const Metadata long_key{std::string(4097, 'k'), {1}, 0, 4, std::nullopt};
        std::vector<TensorStorage> tensors;
        std::string error;
        fixture(path, {64}, {long_key});
        check(read_gguf_file(path, tensors, &error) && tensors.size() == 1 && tensors[0].ne[0] == 64,
              "ordinary native GGUF fallback regressed");
        fixture(path, {64}, {long_key, shape({2, 32})});
        check(!read_gguf_file(path, tensors, &error),
              "native fallback ignored original shapes after an earlier private-parser failure");
#endif
        std::puts(
            "GGUF original shapes: native shape restoration, CPU arithmetic, metadata rejection and shared shard metadata passed");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
