#include "model_io/browser_weight_reader.h"
#include "ggml-webgpu-bounded-upload.h"

#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

static void require(bool value, const char * reason) {
    if (!value)
        throw std::runtime_error(reason);
}

template <typename Call> static void rejects(Call call, const char * reason) {
    bool rejected = false;
    try {
        call();
    } catch (const std::exception &) {
        rejected = true;
    }
    require(rejected, reason);
}

struct Destination {
    bool alive = true;
    uint64_t size;
    std::vector<uint8_t> bytes;
    explicit Destination(uint64_t size, bool allocate = false) : size(size) {
        if (allocate)
            bytes.resize(static_cast<size_t>(size), 0xa5);
    }
};

// CPU fake of the backend's QUEUE-global state, shared by different buffers and
// sync/async callers. A queued write copies its CPU source before returning, but
// accesses its destination only on completion. This exposes premature frees.
struct Queue {
    ggml_webgpu_upload_budget budget;
    struct Write {
        std::weak_ptr<Destination> destination;
        uint64_t offset;
        size_t bytes;
        std::vector<uint8_t> snapshot;
    };
    std::vector<Write> pending;
    std::function<void()> during_wait;
    size_t waits = 0, largest_write = 0, peak_pending = 0;
    bool fail_write = false, fail_wait = false;

    void drain() {
        budget.begin_wait();
        try {
            if (during_wait)
                during_wait();
            if (fail_wait) {
                fail_wait = false;
                throw std::runtime_error("completion failed");
            }
            for (const auto & write : pending) {
                auto destination = write.destination.lock();
                require(destination && destination->alive, "destination freed before completion");
                if (!write.snapshot.empty()) {
                    require(write.offset <= destination->bytes.size() &&
                                write.bytes <= destination->bytes.size() - write.offset,
                            "fake destination range");
                    std::memcpy(destination->bytes.data() + write.offset, write.snapshot.data(), write.bytes);
                }
            }
            pending.clear();
            ++waits;
            budget.completed();
        } catch (...) {
            budget.failed_wait();
            throw;
        }
    }

    void write(const std::shared_ptr<Destination> & destination, uint64_t offset, const uint8_t * source, size_t bytes,
               bool partial) {
        budget.reserve(partial ? 12 : bytes, [&]() { drain(); });
        peak_pending = std::max(peak_pending, budget.pending_bytes());
        require(destination->alive && offset <= destination->size && bytes <= destination->size - offset,
                "write to invalid destination");
        Write write{destination, offset, bytes, {}};
        if (source) {
            write.snapshot.resize(bytes);
            if (partial) {
                const auto word = ggml_webgpu_upload_word(offset, source, bytes);
                for (size_t i = 0; i < bytes; ++i)
                    write.snapshot[i] = static_cast<uint8_t>(word >> (((offset + i) % 4) * 8));
            } else
                std::memcpy(write.snapshot.data(), source, bytes);
        }
        largest_write = std::max(largest_write, partial ? size_t(12) : bytes);
        pending.push_back(std::move(write));
        if (fail_write) {
            fail_write = false;
            throw std::runtime_error("partly enqueued write failed");
        }
    }

    void set(const std::shared_ptr<Destination> & destination, uint64_t offset, const void * data, uint64_t bytes) {
        require(offset <= destination->size && bytes <= destination->size - offset, "set range");
        auto source = static_cast<const uint8_t *>(data);
        ggml_webgpu_upload_ranges(
            offset, bytes,
            [&](uint64_t at, uint64_t consumed, size_t count) {
                write(destination, at, source ? source + consumed : nullptr, count, false);
            },
            [&](uint64_t at, uint64_t consumed, size_t count) {
                write(destination, at, source ? source + consumed : nullptr, count, true);
            });
    }
};

struct BufferOwner {
    Queue & queue;
    std::shared_ptr<Destination> destination;
    ~BufferOwner() {
        // Same order as buffer_free_buffer: retire the queue before Destroy.
        if (queue.budget.pending_bytes() != 0)
            queue.drain();
        destination->alive = false;
    }
};

static void ranges_and_partial_words() {
    for (size_t offset = 0; offset < 8; ++offset) {
        for (size_t bytes = 0; bytes < 20; ++bytes) {
            Queue queue;
            auto destination = std::make_shared<Destination>(32, true);
            std::vector<uint8_t> source(bytes);
            for (size_t i = 0; i < bytes; ++i)
                source[i] = static_cast<uint8_t>(i + 17);
            queue.set(destination, offset, source.data(), bytes);
            std::fill(source.begin(), source.end(), 0); // CPU source can be reused now.
            queue.drain();
            for (size_t i = 0; i < 32; ++i)
                require(destination->bytes[i] == (i >= offset && i < offset + bytes ? i - offset + 17 : 0xa5),
                        "unaligned upload changed neighboring bytes or retained CPU source");
        }
    }
    const uint64_t size = 3111649280ULL, base = (1ULL << 32) + 1;
    uint64_t consumed = 0;
    ggml_webgpu_upload_ranges(
        base, size,
        [&](uint64_t offset, uint64_t source, size_t bytes) {
            require(offset == base + consumed && source == consumed && offset % 4 == 0 && bytes % 4 == 0,
                    "64-bit aligned range continuity");
            require(bytes <= ggml_webgpu_upload_budget::chunk_bytes, "single oversized WriteBuffer");
            consumed += bytes;
        },
        [&](uint64_t offset, uint64_t source, size_t bytes) {
            require(offset == base + consumed && source == consumed && bytes <= 3, "partial range continuity");
            consumed += bytes;
        });
    require(consumed == size, "logical 3.11GB upload coverage");
    rejects([&]() { ggml_webgpu_upload_offset(UINT64_MAX - 1, 2); }, "base+offset overflow accepted");
    rejects(
        [&]() {
            ggml_webgpu_upload_ranges(
                UINT64_MAX - 1, 3, [](auto...) {}, [](auto...) {});
        },
        "offset+size overflow accepted");
}

static void queue_window_and_ownership() {
    Queue queue;
    auto first = std::make_shared<Destination>(64ULL * 1024 * 1024);
    auto second = std::make_shared<Destination>(64ULL * 1024 * 1024);
    {
        BufferOwner owner{queue, first};
        for (size_t i = 0; i < 64; ++i)
            queue.set(i % 2 ? first : second, i * 512 * 1024, nullptr, 512 * 1024);
        require(queue.waits == 0 && queue.budget.pending_bytes() == 32 * 1024 * 1024, "per-slice wait regression");
        queue.set(first, 0, nullptr, 4);
        require(queue.waits == 1 && queue.budget.pending_bytes() == 4, "queue-global window not drained");
    }
    require(!first->alive && queue.budget.pending_bytes() == 0 && queue.pending.empty(),
            "final batch survived buffer free");
    require(queue.peak_pending <= 32 * 1024 * 1024, "queued upload budget exceeded");

    Queue failure;
    auto destination = std::make_shared<Destination>(128);
    rejects(
        [&]() {
            BufferOwner owner{failure, destination};
            failure.fail_write = true;
            failure.set(destination, 0, nullptr, 128);
        },
        "write failure disappeared");
    require(!destination->alive && failure.waits == 1 && failure.budget.pending_bytes() == 0,
            "partly failed write was not drained before destruction");

    Queue completion;
    auto storage = std::make_shared<Destination>(128);
    completion.set(storage, 0, nullptr, 128);
    completion.fail_wait = true;
    rejects([&]() { completion.drain(); }, "completion failure disappeared");
    require(completion.budget.pending_bytes() == 128 && storage->alive, "failed wait cleared pending ownership");
    completion.during_wait = [&]() {
        rejects([&]() { completion.set(storage, 0, nullptr, 4); }, "JSPI reentrant upload accepted");
        rejects([&]() { completion.drain(); }, "reentrant completion wait accepted");
        require(completion.budget.pending_bytes() == 128, "reentry mutated wait snapshot");
    };
    completion.drain();
    require(completion.budget.pending_bytes() == 0, "completed wait retained pending bytes");
}

static void reader_numeric_and_raw() {
    using namespace sd::browser_weights;
    const uint16_t patterns[] = {0, 0x8000, 0x3f80, 0xc020, 1, 0x7f80, 0xff80, 0x7fc1, 0x7f7f};
    const size_t elements = Reader::source_chunk_bytes + 3; // multiple slices plus odd-element tail
    std::vector<ggml_bf16_t> source(elements);
    for (size_t i = 0; i < elements; ++i)
        source[i].bits = patterns[i % 9];
    const auto original = source;
    const uint64_t file_offset = (5ULL << 30) + 18;
    for (const auto destination_type : {GGML_TYPE_BF16, GGML_TYPE_F32, GGML_TYPE_F16}) {
        Reader reader;
        Queue queue;
        auto destination = std::make_shared<Destination>(elements * ggml_type_size(destination_type), true);
        auto conversion = mode(GGML_TYPE_BF16, destination_type, false);
        size_t largest_read = 0;
        bool ok = reader.read(
            file_offset, elements * 2, 2, ggml_type_size(destination_type), conversion != Mode::Raw,
            [&](uint64_t at, void * output, size_t bytes) {
                require(at >= file_offset && at - file_offset + bytes <= elements * 2, "64-bit file slice");
                largest_read = std::max(largest_read, bytes);
                std::memcpy(output, reinterpret_cast<const uint8_t *>(source.data()) + at - file_offset, bytes);
                return true;
            },
            [&](const void * input, void * output, size_t count) { convert_bf16(conversion, input, output, count); },
            [&](const void * data, uint64_t at, size_t bytes) {
                queue.set(destination, at, data, bytes);
                return true;
            },
            []() { return true; });
        require(ok && largest_read <= Reader::source_chunk_bytes && reader.scratch_bytes() <= 1536 * 1024,
                "bounded BF16 reader scratch");
        queue.drain();
        for (size_t i = 0; i < elements; ++i) {
            if (destination_type == GGML_TYPE_F32) {
                uint32_t observed;
                std::memcpy(&observed, destination->bytes.data() + i * 4, 4);
                require(observed == uint32_t(source[i].bits) << 16, "BF16->F32 bits changed");
            } else {
                uint16_t observed;
                std::memcpy(&observed, destination->bytes.data() + i * 2, 2);
                const auto expected = destination_type == GGML_TYPE_BF16
                                          ? source[i].bits
                                          : ggml_fp32_to_fp16(ggml_bf16_to_fp32(source[i]));
                require(observed == expected, "BF16 raw/F16 numeric mismatch");
            }
        }
        require(std::memcmp(source.data(), original.data(), elements * 2) == 0, "source BF16 representation mutated");
    }
    for (auto type : {GGML_TYPE_Q4_0, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K}) {
        Reader reader;
        const size_t block = ggml_type_size(type), bytes = block * 40001;
        std::vector<uint8_t> source_bytes(bytes), output(bytes);
        for (size_t i = 0; i < bytes; ++i)
            source_bytes[i] = static_cast<uint8_t>(i * 29 + 7);
        require(reader.read(
                    file_offset, bytes, block, block, false,
                    [&](uint64_t at, void * data, size_t count) {
                        std::memcpy(data, source_bytes.data() + at - file_offset, count);
                        return true;
                    },
                    [](auto...) { throw std::runtime_error("raw quant payload converted"); },
                    [&](const void * data, uint64_t at, size_t count) {
                        std::memcpy(output.data() + at, data, count);
                        return true;
                    },
                    []() { return true; }),
                "raw quant read failed");
        require(source_bytes == output && reader.scratch_bytes() == Reader::source_chunk_bytes,
                "raw quant bytes/scratch changed");
        require(mode(type, type, false) == Mode::Raw, "raw quant mode rejected");
    }
    require(mode(GGML_TYPE_BF16, GGML_TYPE_F32, true) == Mode::Unsupported, "decoded/FP8 source accepted");
    require(mode(GGML_TYPE_Q4_K, GGML_TYPE_F32, false) == Mode::Unsupported, "legacy quant conversion intercepted");
}

static void reader_logical_large_and_failure() {
    using namespace sd::browser_weights;
    Reader reader;
    Queue queue;
    auto destination = std::make_shared<Destination>(3111649280ULL);
    uint64_t reads = 0, writes = 0, last_source = 0;
    const uint64_t start = (6ULL << 30) + 64;
    {
        BufferOwner owner{queue, destination};
        require(
            reader.read(
                start, 1555824640ULL, 2, 4, true,
                [&](uint64_t at, void *, size_t bytes) {
                    require(at == start + reads && bytes <= 512 * 1024, "large logical source continuity");
                    reads += bytes;
                    last_source = at;
                    return true;
                },
                [](const void *, void *, size_t elements) { require(elements <= 262144, "large conversion window"); },
                [&](const void *, uint64_t at, size_t bytes) {
                    require(at == writes && bytes <= 1024 * 1024, "large logical destination continuity");
                    queue.set(destination, at, nullptr, bytes);
                    writes += bytes;
                    return true;
                },
                []() { return true; }),
            "logical H3 embedding read failed");
        require(writes == 3111649280ULL && reads == 1555824640ULL && last_source > (1ULL << 32),
                "logical large tensor truncated");
    }
    require(reader.scratch_bytes() == 1536 * 1024 && queue.peak_pending <= 32 * 1024 * 1024 && queue.waits < 100,
            "large tensor grew scratch/window or waited on every slice");
    require(queue.budget.pending_bytes() == 0 && queue.pending.empty(), "large tensor final drain");

    for (int failure_mode = 0; failure_mode < 4; ++failure_mode) {
        Reader failing_reader;
        Queue failing_queue;
        auto storage = std::make_shared<Destination>(4 * 1024 * 1024);
        size_t reads = 0, checks = 0;
        bool reached = false;
        try {
            BufferOwner owner{failing_queue, storage};
            bool ok = failing_reader.read(
                start, 2 * 1024 * 1024, 2, 4, true,
                [&](uint64_t, void *, size_t) {
                    ++reads;
                    if (reads == 2 && failure_mode == 0)
                        return false; // truncated file
                    return true;
                },
                [&](const void *, void *, size_t) {
                    if (reads == 2 && failure_mode == 1)
                        throw std::runtime_error("conversion failed");
                },
                [&](const void *, uint64_t at, size_t bytes) {
                    if (reads == 2 && failure_mode == 2)
                        failing_queue.fail_write = true;
                    failing_queue.set(storage, at, nullptr, bytes);
                    return true;
                },
                [&]() {
                    ++checks;
                    return !(failure_mode == 3 && reads == 2);
                }); // cooperative stop
            reached = !ok;
        } catch (const std::runtime_error &) {
            reached = true;
        }
        require(reached && !storage->alive && failing_queue.pending.empty() && failing_queue.waits == 1,
                "read/convert/write/cancel failure escaped destination drain");
    }
    rejects(
        [&]() {
            reader.read(
                UINT64_MAX - 1, 2, 2, 4, true, [](auto...) { return true; }, [](auto...) {},
                [](auto...) { return true; }, []() { return true; });
        },
        "file offset overflow accepted");
}

int main() {
    try {
        ranges_and_partial_words();
        queue_window_and_ownership();
        reader_numeric_and_raw();
        reader_logical_large_and_failure();
        std::cout << "bounded weight upload contracts passed\n";
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
