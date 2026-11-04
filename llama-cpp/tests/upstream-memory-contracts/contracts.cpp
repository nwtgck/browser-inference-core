#include "ggml-backend-impl.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

// Pinned private C++ helper: exercise the actual full-buffer writer as well as
// the public file writer. This is test-only and adds no runtime binding.
void gguf_write_to_buf(const gguf_context * ctx, std::vector<int8_t> & buf, bool only_meta);

static int checks = 0;
static void require(bool condition, const char * message) {
    ++checks;
    if (!condition)
        throw std::runtime_error(message);
}

static void rejected_view(const std::function<void()> & operation) {
    const pid_t child = fork();
    require(child >= 0, "fork failed");
    if (child == 0) {
        ggml_set_abort_callback([](const char *) { std::_Exit(73); });
        operation();
        std::_Exit(0);
    }
    int status = 0;
    require(waitpid(child, &status, 0) == child, "child was not reaped");
    require(WIFEXITED(status) && WEXITSTATUS(status) == 73, "view guard did not reject the invalid span");
}

static void test_views() {
    ggml_context * ctx = ggml_init({65536, nullptr, true});
    require(ctx != nullptr, "view context initialization failed");
    auto * root = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16);
    // Overlapping windows have more logical elements than their physical span.
    auto * overlap = ggml_view_2d(ctx, root, 8, 4, 2 * sizeof(float), 0);
    require(ggml_nelements(overlap) == 32 && ggml_nbytes(overlap) == 56, "legal overlapping windows rejected");
    auto * offset = ggml_view_2d(ctx, root, 8, 4, 2 * sizeof(float), 8);
    require(ggml_nbytes(offset) + offset->view_offs == ggml_nbytes(root), "boundary view span changed");
    rejected_view([&] { ggml_view_2d(ctx, root, 4, 4, 32, 0); });
    rejected_view([&] { ggml_view_2d(ctx, root, 8, 4, 8, 12); });
    auto * matrix = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4);
    auto * transposed = ggml_transpose(ctx, matrix);
    auto * preserved = ggml_view_tensor(ctx, transposed);
    require(preserved->nb[0] == transposed->nb[0] && preserved->nb[1] == transposed->nb[1],
            "view_tensor lost the source strides");
    rejected_view([&] { ggml_view_2d(ctx, transposed, 4, 1, 4, 0); });
    ggml_free(ctx);
}

struct access_state {
    bool host;
    int reads = 0;
};

static void test_gguf(bool attach_buffer, bool host) {
    ggml_context * ctx = ggml_init({65536, nullptr, false});
    require(ctx != nullptr, "GGUF context initialization failed");
    auto * floats = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3);
    auto * bytes = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, 7);
    ggml_set_name(floats, "floats");
    ggml_set_name(bytes, "bytes");
    const std::array<float, 3> values{-1.25f, 0.5f, 7.75f};
    const std::array<uint8_t, 7> packed{0, 127, 255, 18, 52, 86, 120};
    std::memcpy(floats->data, values.data(), sizeof(values));
    std::memcpy(bytes->data, packed.data(), packed.size());
    access_state access{host};
    ggml_backend_buffer_type type{};
    type.context = &access;
    type.iface.is_host = [](ggml_backend_buffer_type_t bt) { return static_cast<access_state *>(bt->context)->host; };
    ggml_backend_buffer buffer{};
    buffer.buft = &type;
    buffer.context = &access;
    buffer.size = 128;
    buffer.iface.get_tensor = [](ggml_backend_buffer_t b, const ggml_tensor * tensor, void * out, size_t offset,
                                 size_t size) {
        ++static_cast<access_state *>(b->context)->reads;
        std::memcpy(out, static_cast<const uint8_t *>(tensor->data) + offset, size);
    };
    if (attach_buffer) {
        floats->buffer = &buffer;
        bytes->buffer = &buffer;
    }
    gguf_context * gguf = gguf_init_empty();
    gguf_add_tensor(gguf, floats);
    gguf_add_tensor(gguf, bytes);
    const size_t meta = gguf_get_meta_size(gguf);
    const size_t alignment = gguf_get_alignment(gguf);
    std::vector<int8_t> serialized;
    gguf_write_to_buf(gguf, serialized, false);
    require(access.reads == (attach_buffer && !host ? 2 : 0), "buffer writer used the wrong data path");
    require(serialized.size() == meta + 2 * alignment, "GGUF payload padding size changed");
    for (int i = 0; i < 2; ++i) {
        const size_t start = meta + gguf_get_tensor_offset(gguf, i);
        const size_t size = i == 0 ? sizeof(values) : packed.size();
        const void * expected = i == 0 ? static_cast<const void *>(values.data()) : packed.data();
        require(std::memcmp(serialized.data() + start, expected, size) == 0, "GGUF payload bytes changed");
        require(std::all_of(serialized.begin() + start + size, serialized.begin() + start + alignment,
                            [](int8_t value) { return value == 0; }),
                "GGUF payload padding was not zero");
    }
    FILE * file = std::tmpfile();
    require(file != nullptr, "tmpfile failed");
    const std::array<uint8_t, 7> prefix{3, 1, 4, 1, 5, 9, 2};
    require(std::fwrite(prefix.data(), 1, prefix.size(), file) == prefix.size(), "prefix write failed");
    require(gguf_write_to_file_ptr(gguf, file, false), "GGUF file write failed");
    require(access.reads == (attach_buffer && !host ? 4 : 0), "file writer used the wrong data path");
    std::rewind(file);
    std::vector<uint8_t> actual(prefix.size() + serialized.size());
    require(std::fread(actual.data(), 1, actual.size(), file) == actual.size(), "GGUF file read was short");
    require(std::fgetc(file) == EOF, "GGUF file had unexpected trailing bytes");
    require(std::equal(prefix.begin(), prefix.end(), actual.begin()), "GGUF writer changed prefix bytes");
    require(std::memcmp(actual.data() + prefix.size(), serialized.data(), serialized.size()) == 0,
            "GGUF file and buffer writers differ");
    std::fclose(file);
    gguf_free(gguf);
    ggml_free(ctx);
}

int main() {
    try {
        test_views();
        test_gguf(false, false);
        test_gguf(true, true);
        test_gguf(true, false);
        std::cout << checks << " actual upstream view/GGUF span, payload, padding and access-path checks passed\n";
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
