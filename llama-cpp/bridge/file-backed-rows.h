#pragma once

// Bounded, positioned row reads for the browser's synchronous read-only mounts.
// No mmap, worker pool, process-global cache, or file cursor is used.
#include "ggml.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>
#include <unistd.h>

class lcb_file_backed_rows {
public:
    static constexpr size_t max_staging_bytes = 64u * 1024 * 1024;
    static constexpr size_t max_gather_rows = 1024u * 1024;

    lcb_file_backed_rows(int fd, uint32_t shard, uint64_t file_bytes,
                        uint64_t offset, ggml_type type, int64_t columns, int64_t rows)
        : shard_(shard), offset_(offset), columns_(columns), rows_(rows), type_(type) {
        static_assert(sizeof(void *) == 8, "file-backed rows require 64-bit pointers");
        static_assert(sizeof(off_t) >= 8, "file-backed rows require 64-bit file offsets");
        if (columns <= 0 || rows <= 0 || shard > INT32_MAX ||
            type < 0 || type >= GGML_TYPE_COUNT) {
            throw std::runtime_error("file-backed rows: invalid geometry");
        }
        const auto * traits = ggml_get_type_traits(type);
        if (traits->blck_size <= 0 || columns % traits->blck_size != 0) {
            throw std::runtime_error("file-backed rows: invalid block alignment");
        }
        if (type != GGML_TYPE_F32 && !traits->to_float) {
            throw std::runtime_error("file-backed rows: unsupported row type");
        }
        if (uint64_t(columns) > max_staging_bytes / sizeof(float)) {
            throw std::runtime_error("file-backed rows: row exceeds staging limit");
        }
        row_bytes_ = ggml_row_size(type, columns);
        if (!row_bytes_ || row_bytes_ > max_staging_bytes || offset > file_bytes ||
            uint64_t(rows) > (file_bytes - offset) / row_bytes_ ||
            file_bytes > uint64_t(std::numeric_limits<off_t>::max())) {
            throw std::runtime_error("file-backed rows: invalid file range");
        }
        // Own a descriptor, not the loader's transient llama_file. pread never
        // changes the shared open-file cursor, even across multiple contexts.
        fd_ = dup(fd);
        if (fd_ < 0) throw std::runtime_error("file-backed rows: cannot retain file");
    }

    ~lcb_file_backed_rows() { if (fd_ >= 0) close(fd_); }
    lcb_file_backed_rows(const lcb_file_backed_rows &) = delete;
    lcb_file_backed_rows & operator=(const lcb_file_backed_rows &) = delete;

    int32_t shard() const { return int32_t(shard_); }
    int64_t columns() const { return columns_; }

    size_t elements(size_t count) const {
        if (count > max_gather_rows || count > max_staging_bytes / sizeof(float) / size_t(columns_)) {
            throw std::runtime_error("file-backed rows: batch exceeds staging limit");
        }
        return count * size_t(columns_);
    }

    // Output is in the caller's original head/token order. Sorting only changes
    // I/O order; duplicates are read and dequantized once per microbatch.
    // Do not publish any of this staging tensor to a backend until this returns.
    void gather(const int32_t * indices, size_t count, std::vector<float> & output) const {
        const size_t n = elements(count);
        if (count && !indices) throw std::runtime_error("file-backed rows: missing indices");
        std::vector<std::pair<int32_t, size_t>> order;
        order.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            if (indices[i] < 0 || int64_t(indices[i]) >= rows_) {
                throw std::runtime_error("file-backed rows: invalid row index");
            }
            order.emplace_back(indices[i], i);
        }
        std::sort(order.begin(), order.end());
        output.resize(n);
        // uint64_t storage preserves alignment for every ggml dequantizer.
        std::vector<uint64_t> packed((row_bytes_ + sizeof(uint64_t) - 1) / sizeof(uint64_t));
        const size_t float_bytes = size_t(columns_) * sizeof(float);
        for (size_t i = 0; i < order.size();) {
            const auto row = order[i].first;
            float * destination = output.data() + order[i].second * size_t(columns_);
            read_exact(offset_ + uint64_t(row) * row_bytes_, packed.data(), row_bytes_);
            if (type_ == GGML_TYPE_F32) {
                std::memcpy(destination, packed.data(), float_bytes);
            } else {
                ggml_get_type_traits(type_)->to_float(packed.data(), destination, columns_);
            }
            ++i;
            while (i < order.size() && order[i].first == row) {
                std::memcpy(output.data() + order[i].second * size_t(columns_), destination, float_bytes);
                ++i;
            }
        }
    }

private:
    void read_exact(uint64_t offset, void * buffer, size_t length) const {
        auto * bytes = static_cast<unsigned char *>(buffer);
        size_t done = 0;
        while (done < length) {
            const ssize_t count = pread(fd_, bytes + done, length - done, off_t(offset + done));
            if (count < 0 && errno == EINTR) continue;
            if (count <= 0) throw std::runtime_error("file-backed rows: read failed");
            done += size_t(count);
        }
    }

    int fd_ = -1;
    uint32_t shard_;
    uint64_t offset_;
    int64_t columns_;
    int64_t rows_;
    ggml_type type_;
    size_t row_bytes_ = 0;
};
