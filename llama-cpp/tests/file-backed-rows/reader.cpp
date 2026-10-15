// Native bounded-reader tests. Does not validate a browser or a trained model.
#include "file-backed-rows.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <string>

static void check(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}
template<class F> static void rejects(F && f) {
    bool failed = false;
    try { f(); } catch (const std::runtime_error &) { failed = true; }
    check(failed, "expected reader rejection");
}
struct temporary_file {
    FILE * value = std::tmpfile();
    temporary_file() { check(value != nullptr, "tmpfile"); }
    ~temporary_file() { if (value) std::fclose(value); }
    int fd() const { return fileno(value); }
};
static void write_at(int fd, uint64_t offset, const void * bytes, size_t count) {
    check(pwrite(fd, bytes, count, off_t(offset)) == ssize_t(count), "pwrite");
}
static void compare_rows(ggml_type type, int64_t columns, uint64_t offset) {
    constexpr int64_t rows = 5;
    const size_t row_bytes = ggml_row_size(type, columns);
    const auto * traits = ggml_get_type_traits(type);
    std::vector<float> input(columns * rows);
    for (size_t i = 0; i < input.size(); ++i) input[i] = std::sin(float(i) * 0.37f) * 2;
    std::vector<uint64_t> packed((row_bytes * rows + 7) / 8);
    if (type == GGML_TYPE_F32) std::memcpy(packed.data(), input.data(), input.size() * sizeof(float));
    else {
        check(traits->from_float_ref != nullptr, "reference quantizer");
        traits->from_float_ref(input.data(), packed.data(), columns * rows);
    }
    temporary_file file;
    const uint64_t size = offset + row_bytes * rows;
    check(ftruncate(file.fd(), off_t(size)) == 0, "sparse truncate");
    write_at(file.fd(), offset, packed.data(), row_bytes * rows);
    lcb_file_backed_rows reader(file.fd(), 1, size, offset, type, columns, rows);
    check(reader.shard() == 1, "shard identity");
    std::array<int32_t, 7> ids{4, 0, 4, 1, 3, 0, 2};
    std::vector<float> actual;
    // The original loader FILE and its cursor can be closed without invalidating
    // the retained descriptor. Reads must not be implemented using seek+read.
    check(lseek(file.fd(), 13, SEEK_SET) == 13, "set shared cursor");
    reader.gather(ids.data(), ids.size(), actual);
    check(lseek(file.fd(), 0, SEEK_CUR) == 13, "pread changed cursor");
    std::fclose(file.value); file.value = nullptr;
    std::vector<float> again;
    reader.gather(ids.data(), ids.size(), again);
    check(actual == again, "descriptor lifetime");

    // Compare against a real CPU ggml_get_rows graph, not a mocked gather.
    auto ctx = std::unique_ptr<ggml_context, decltype(&ggml_free)>(
        ggml_init({2 * 1024 * 1024, nullptr, false}), ggml_free);
    check(bool(ctx), "ggml context");
    auto * tensor = ggml_new_tensor_2d(ctx.get(), type, columns, rows);
    std::memcpy(tensor->data, packed.data(), row_bytes * rows);
    auto * indices = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, ids.size());
    std::memcpy(indices->data, ids.data(), sizeof(ids));
    auto * output = ggml_get_rows(ctx.get(), tensor, indices);
    auto * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, output);
    check(ggml_graph_compute_with_ctx(ctx.get(), graph, 1) == GGML_STATUS_SUCCESS, "get_rows compute");
    check(ggml_nbytes(output) == actual.size() * sizeof(float), "output layout");
    check(std::memcmp(output->data, actual.data(), ggml_nbytes(output)) == 0, "get_rows differs");
    const int32_t negative = -1, beyond = rows;
    rejects([&] { reader.gather(&negative, 1, actual); });
    rejects([&] { reader.gather(&beyond, 1, actual); });
    rejects([&] { reader.gather(nullptr, 1, actual); });
    rejects([&] { reader.elements(lcb_file_backed_rows::max_gather_rows + 1); });
    rejects([&] { reader.elements(lcb_file_backed_rows::max_staging_bytes / 4 / size_t(columns) + 1); });
    reader.gather(nullptr, 0, actual);
    check(actual.empty(), "empty gather");
}
int main() {
    try {
        // IQ4_NL / 160 columns is the actual Unsloth PLE row format.
        for (const auto type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_IQ4_NL}) {
            compare_rows(type, 160, 0);
            compare_rows(type, 160, (uint64_t(1) << 32) + 71);
        }
        temporary_file f;
        check(ftruncate(f.fd(), 4096) == 0, "truncate fixture");
        rejects([&] { lcb_file_backed_rows r(-1, 0, 4096, 0, GGML_TYPE_F32, 32, 1); });
        rejects([&] { lcb_file_backed_rows r(f.fd(), 0, 4096, 0, GGML_TYPE_F32, 0, 1); });
        rejects([&] { lcb_file_backed_rows r(f.fd(), 0, 4096, 0, GGML_TYPE_F32, 1, -1); });
        rejects([&] { lcb_file_backed_rows r(f.fd(), 0, 4096, 0, GGML_TYPE_COUNT, 32, 1); });
        rejects([&] { lcb_file_backed_rows r(f.fd(), 0, 4096, 0, GGML_TYPE_IQ4_NL, 33, 1); });
        rejects([&] { lcb_file_backed_rows r(f.fd(), 0, 4096, 0, GGML_TYPE_I32, 32, 1); });
        rejects([&] { lcb_file_backed_rows r(f.fd(), 0, 4096, 4097, GGML_TYPE_F32, 32, 1); });
        rejects([&] { lcb_file_backed_rows r(f.fd(), 0, 4096, 4096, GGML_TYPE_F32, 32, 1); });
        rejects([&] { lcb_file_backed_rows r(f.fd(), 0, UINT64_MAX, 0, GGML_TYPE_F32, 32, 1); });
        rejects([&] { lcb_file_backed_rows r(f.fd(), UINT32_MAX, 4096, 0, GGML_TYPE_F32, 32, 1); });
        rejects([&] { lcb_file_backed_rows r(f.fd(), 0, UINT64_MAX / 2, 0, GGML_TYPE_F32, INT64_MAX, 1); });
        // A backing file truncated after successful load must fail during gather.
        lcb_file_backed_rows reader(f.fd(), 0, 4096, 0, GGML_TYPE_F32, 32, 32);
        check(ftruncate(f.fd(), 7) == 0, "truncate after load");
        int32_t index = 0; std::vector<float> output;
        rejects([&] { reader.gather(&index, 1, output); });
        std::cout << "PASS bounded positioned reads, lifetime, quantization/get_rows equivalence, >4GiB, invalid inputs and truncation\n";
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
