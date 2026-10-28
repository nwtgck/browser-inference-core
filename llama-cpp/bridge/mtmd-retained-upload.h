#pragma once

#include "ggml-backend.h"
#include "mtmd-vision-bf16-policy.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <istream>
#include <limits>
#include <memory>
#include <stdexcept>

inline size_t lcb_mtmd_add_bytes(size_t left, size_t right) {
    if (right > std::numeric_limits<size_t>::max() - left) {
        throw std::runtime_error("vision tensor byte count exceeds the address space");
    }
    return left + right;
}

// Raw file bytes, not floats: keep the original mixed-precision representation.
// Write must consume/copy before returning. Check receives SOURCE bytes consumed,
// including zero and the final chunk; it may throw to cancel. No GPU API is
// reimplemented here: the caller uses the ordinary ggml tensor-set entry point,
// which owns partial-word writes for odd BF16 element counts.
class lcb_mtmd_retained_reader {
public:
    static constexpr size_t chunk_bytes = 512 * 1024;

    template <typename Write, typename Check>
    void read(std::istream & input, size_t source_bytes, Write write, Check check) {
        check(0);
        if (source_bytes != 0 && !scratch) {
            scratch.reset(new uint8_t[chunk_bytes]);
        }
        for (size_t offset = 0; offset < source_bytes;) {
            const size_t count = std::min(chunk_bytes, source_bytes - offset);
            input.read(reinterpret_cast<char *>(scratch.get()), static_cast<std::streamsize>(count));
            if (!input) {
                throw std::runtime_error("truncated retained vision tensor data");
            }
            check(offset);
            write(scratch.get(), offset, count);
            offset += count;
            check(offset);
        }
    }

    size_t scratch_bytes() const { return scratch ? chunk_bytes : 0; }

private:
    std::unique_ptr<uint8_t[]> scratch;
};

// This is a loader-local QUEUED-BYTE budget, not a physical memory guarantee.
// Keep the session inside the owner's weight-buffer lifetime. On every exit,
// retire writes before that buffer can be freed. A ggml synchronize failure is
// handled by the existing backend (including its fatal WebGPU error handling),
// not converted into successful model publication or a recoverable error here.
class lcb_mtmd_upload_queue {
public:
    static constexpr size_t budget_bytes = 32 * 1024 * 1024;
    static constexpr size_t write_overhead = 24; // up to two head/tail parameter writes

    explicit lcb_mtmd_upload_queue(ggml_backend_t backend) : backend(backend) {}
    lcb_mtmd_upload_queue(const lcb_mtmd_upload_queue &) = delete;
    lcb_mtmd_upload_queue & operator=(const lcb_mtmd_upload_queue &) = delete;

    ~lcb_mtmd_upload_queue() { drain(); }

    // Charge BEFORE handing data to the backend: a partly submitted failed write
    // must still be drained. A false return means no synchronization occurred.
    bool reserve(size_t bytes) {
        if (!backend) {
            throw std::runtime_error("missing retained vision upload backend");
        }
        if (bytes == 0) return false;
        if (bytes > budget_bytes - write_overhead) {
            throw std::runtime_error("retained vision upload exceeds queue budget");
        }
        const size_t charge = bytes + write_overhead;
        bool waited = false;
        if (charge > budget_bytes - pending) {
            drain();
            waited = true;
        }
        pending += charge;
        return waited;
    }

    void drain() {
        if (pending != 0) {
            ggml_backend_synchronize(backend);
            pending = 0;
        }
    }

    size_t pending_bytes() const { return pending; }

private:
    ggml_backend_t backend;
    size_t pending = 0;
};
