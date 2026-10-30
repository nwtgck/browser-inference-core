#pragma once

// Bounded, positioned row reads for the browser's synchronous read-only mounts.
// The immutable model owns a bounded packed-row cache; contexts never share
// mutable gather buffers. No mmap, worker pool, or file cursor is used.
#include "ggml.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>
#include <unistd.h>
#include <sys/stat.h>

class lcb_file_backed_rows {
  public:
    static constexpr size_t max_staging_bytes = 64u * 1024 * 1024;
    static constexpr size_t max_gather_rows = 1024u * 1024;
    static constexpr size_t max_cache_bytes = 8u * 1024 * 1024;

    lcb_file_backed_rows(int fd, uint32_t shard, uint64_t file_bytes, uint64_t offset, ggml_type type, int64_t columns,
                         int64_t rows, size_t cache_bytes = max_cache_bytes)
        : shard_(shard), file_bytes_(file_bytes), offset_(offset), columns_(columns), rows_(rows), type_(type),
          cache_budget_(std::min(cache_bytes, max_cache_bytes)) {
        static_assert(sizeof(void *) == 8, "file-backed rows require 64-bit pointers");
        static_assert(sizeof(off_t) >= 8, "file-backed rows require 64-bit file offsets");
        if (columns <= 0 || rows <= 0 || shard > INT32_MAX || type < 0 || type >= GGML_TYPE_COUNT) {
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
        if (fd_ < 0)
            throw std::runtime_error("file-backed rows: cannot retain file");
    }

    ~lcb_file_backed_rows() {
        if (fd_ >= 0)
            close(fd_);
    }
    lcb_file_backed_rows(const lcb_file_backed_rows &) = delete;
    lcb_file_backed_rows & operator=(const lcb_file_backed_rows &) = delete;

    int32_t shard() const {
        return int32_t(shard_);
    }
    int64_t columns() const {
        return columns_;
    }

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
        if (count && !indices)
            throw std::runtime_error("file-backed rows: missing indices");
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
        // One model can serve multiple native contexts. Serialize access to its
        // cache, not the caller's output lifetime; pread still leaves the cursor
        // untouched. Browser profiles execute this on their existing worker.
        std::lock_guard<std::mutex> lock(cache_mutex_);
        if (count) {
            // Cached rows must not hide a truncated/invalid retained descriptor.
            // This is a metadata-only lookup on the browser's immutable mounts,
            // not another source read. As with resident weights, same-size file
            // mutation is outside the model's immutable-source contract.
            struct stat status;
            if (fstat(fd_, &status) != 0 || status.st_size < 0 || uint64_t(status.st_size) != file_bytes_) {
                for (auto & entry : cache_entries_)
                    entry = {};
                throw std::runtime_error("file-backed rows: backing file size changed");
            }
            initialize_cache();
        }
        // Allocate scratch on the first miss only. Fully warm and empty gathers
        // need no packed staging; uint64_t preserves dequantizer alignment.
        std::vector<uint64_t> packed;
        const size_t float_bytes = size_t(columns_) * sizeof(float);
        for (size_t i = 0; i < order.size();) {
            const auto row = order[i].first;
            float * destination = output.data() + order[i].second * size_t(columns_);
            const void * bytes = read_row(row, packed);
            if (type_ == GGML_TYPE_F32) {
                std::memcpy(destination, bytes, float_bytes);
            } else {
                ggml_get_type_traits(type_)->to_float(bytes, destination, columns_);
            }
            ++i;
            while (i < order.size() && order[i].first == row) {
                std::memcpy(output.data() + order[i].second * size_t(columns_), destination, float_bytes);
                ++i;
            }
        }
    }

    // Account for both packed payload and cache metadata. This is independent
    // of the existing gather staging bound, and never grows with context length.
    size_t cache_bytes() const {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        return cache_entries_.size() * sizeof(cache_entry) + cache_data_.size() * sizeof(uint64_t);
    }

  private:
    static constexpr size_t cache_ways = 4;
    struct cache_entry {
        int32_t row = -1;
        uint32_t stamp = 0;
    };

    void initialize_cache() const {
        if (cache_initialized_)
            return;
        cache_initialized_ = true;
        cache_row_words_ = (row_bytes_ + sizeof(uint64_t) - 1) / sizeof(uint64_t);
        const size_t set_bytes = cache_ways * (cache_row_words_ * sizeof(uint64_t) + sizeof(cache_entry));
        size_t sets = 1;
        if (set_bytes > cache_budget_)
            return;
        const size_t useful_sets = (uint64_t(rows_) + cache_ways - 1) / cache_ways;
        while (sets <= cache_budget_ / set_bytes / 2 && sets < useful_sets)
            sets *= 2;
        try {
            // Construct both allocations locally. A failure cannot leave a
            // partially initialized cache, strand a descriptor, or fail a load.
            std::vector<cache_entry> entries(sets * cache_ways);
            std::vector<uint64_t> data(sets * cache_ways * cache_row_words_);
            cache_entries_.swap(entries);
            cache_data_.swap(data);
            cache_set_mask_ = sets - 1;
        } catch (const std::bad_alloc &) {
            // This is an optional optimization. The bounded uncached reader is
            // still valid; do not retry a failed cache allocation every token.
        }
    }

    const void * read_into_scratch(int32_t row, std::vector<uint64_t> & scratch) const {
        scratch.resize((row_bytes_ + sizeof(uint64_t) - 1) / sizeof(uint64_t));
        read_exact(offset_ + uint64_t(row) * row_bytes_, scratch.data(), row_bytes_);
        return scratch.data();
    }

    const void * read_row(int32_t row, std::vector<uint64_t> & scratch) const {
        if (cache_entries_.empty()) {
            return read_into_scratch(row, scratch);
        }
        // Hash n-gram row IDs rather than mapping their low bits to cache sets.
        uint32_t hash = uint32_t(row);
        hash ^= hash >> 16;
        hash *= 0x7feb352du;
        hash ^= hash >> 15;
        hash *= 0x846ca68bu;
        hash ^= hash >> 16;
        const size_t first = (size_t(hash) & cache_set_mask_) * cache_ways;
        if (++cache_clock_ == 0) {
            // A wrap can change recency, never row identity or output bytes.
            for (auto & entry : cache_entries_)
                entry = {};
            cache_clock_ = 1;
        }
        size_t selected = first;
        for (size_t slot = first; slot < first + cache_ways; ++slot) {
            auto & entry = cache_entries_[slot];
            if (entry.row == row) {
                entry.stamp = cache_clock_;
                return cache_data_.data() + slot * cache_row_words_;
            }
            if (entry.stamp < cache_entries_[selected].stamp)
                selected = slot;
        }
        // Read into scratch first. EINTR/short reads are handled by read_exact;
        // an I/O failure must neither overwrite nor publish a cache entry.
        const void * bytes = read_into_scratch(row, scratch);
        auto * cached = cache_data_.data() + selected * cache_row_words_;
        std::memcpy(cached, bytes, row_bytes_);
        cache_entries_[selected] = {row, cache_clock_};
        return cached;
    }

    void read_exact(uint64_t offset, void * buffer, size_t length) const {
        auto * bytes = static_cast<unsigned char *>(buffer);
        size_t done = 0;
        while (done < length) {
            const ssize_t count = pread(fd_, bytes + done, length - done, off_t(offset + done));
            if (count < 0 && errno == EINTR)
                continue;
            if (count <= 0)
                throw std::runtime_error("file-backed rows: read failed");
            done += size_t(count);
        }
    }

    int fd_ = -1;
    uint32_t shard_;
    uint64_t file_bytes_;
    uint64_t offset_;
    int64_t columns_;
    int64_t rows_;
    ggml_type type_;
    size_t row_bytes_ = 0;
    size_t cache_budget_;
    mutable std::mutex cache_mutex_;
    mutable bool cache_initialized_ = false;
    mutable size_t cache_row_words_ = 0;
    mutable size_t cache_set_mask_ = 0;
    mutable uint32_t cache_clock_ = 0;
    mutable std::vector<cache_entry> cache_entries_;
    mutable std::vector<uint64_t> cache_data_;
};
