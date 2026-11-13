#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

static constexpr uint64_t MiB = 1024 * 1024;
struct device {
    int id;
    uint64_t queued = 0;
};
using ggml_backend_dev_t = device *;
struct backend {
    device * dev;
};
using ggml_backend_t = backend *;
static uint64_t live = 0, waits = 0, initialized = 0, freed = 0;
static bool fail_init = false;
static std::vector<int> trace;
ggml_backend_t ggml_backend_dev_init(device * dev, const char *) {
    trace.push_back(10 + dev->id);
    ++initialized;
    if (fail_init)
        return nullptr;
    ++live;
    return new backend{dev};
}
void ggml_backend_synchronize(backend * value) {
    assert(value);
    ++waits;
    trace.push_back(20 + value->dev->id);
    value->dev->queued = 0;
}
void ggml_backend_free(backend * value) {
    if (value) {
        assert(value->dev->queued == 0);
        --live;
        ++freed;
        trace.push_back(30 + value->dev->id);
        delete value;
    }
}

// Both compile the unchanged actual prepared helper tokens. These are arithmetic
// width tests, not a claim of executing either Wasm ABI or a real GPU backend.
namespace width32 {
using size_t = std::uint32_t;
#include "model-upload-helper.inc"
} // namespace width32
namespace width64 {
using size_t = std::uint64_t;
#include "model-upload-helper.inc"
} // namespace width64

struct state {
    uint64_t waits, pending;
};
static state step(state value, uint64_t charge, uint64_t budget) {
    assert(charge <= budget);
    if (charge > budget - value.pending) {
        ++value.waits;
        value.pending = 0;
    }
    value.pending += charge;
    return value;
}
static state brute(uint64_t size, uint64_t chunk, uint64_t pending, uint64_t budget, uint64_t overhead) {
    state value{0, pending};
    while (size) {
        const uint64_t amount = std::min(size, chunk);
        value = step(value, amount + overhead, budget);
        size -= amount;
    }
    return value;
}

// An independent repeated-state machine skips cycles for SIZE_MAX inputs. It
// does not use the production quotient-based first-batch/full-batch formula.
static state cycle_reference(uint64_t size, uint64_t chunk, uint64_t pending, uint64_t budget, uint64_t overhead) {
    const uint64_t full = size / chunk, tail = size % chunk;
    uint64_t completed = 0;
    state value{0, pending};
    std::map<uint64_t, std::pair<uint64_t, uint64_t>> seen;
    while (completed < full) {
        auto [it, inserted] = seen.emplace(value.pending, std::make_pair(completed, value.waits));
        if (!inserted) {
            const uint64_t cycle = completed - it->second.first;
            const uint64_t cycle_waits = value.waits - it->second.second;
            assert(cycle != 0);
            const uint64_t repeats = (full - completed) / cycle;
            if (repeats != 0) {
                completed += repeats * cycle;
                value.waits += repeats * cycle_waits;
                continue;
            }
        }
        value = step(value, chunk + overhead, budget);
        ++completed;
        assert(seen.size() < 32);
    }
    if (tail != 0)
        value = step(value, tail + overhead, budget);
    return value;
}

static uint64_t projection_cases = 0, policy_cases = 0, tensor_sequences = 0, positive_adoptions = 0;
template <class Helper> static void projection_checks() {
    using Size = decltype(Helper{}.pending);
    const uint64_t budget = Helper::budget, overhead = Helper::overhead;
    for (uint64_t chunk : {uint64_t(1), uint64_t(2), uint64_t(31), uint64_t(1024), uint64_t(65536), 8 * MiB - 448,
                           8 * MiB - 32, 8 * MiB}) {
        for (uint64_t full = 0; full <= 17; ++full) {
            for (uint64_t tail : {uint64_t(0), chunk / 2, chunk - 1}) {
                const uint64_t size = full * chunk + tail;
                for (uint64_t pending : {uint64_t(0), uint64_t(1), overhead, budget / 4 - 1, budget / 4, budget / 2,
                                         budget - chunk, budget - 1, budget}) {
                    const auto actual = Helper::project(Size(size), Size(chunk), Size(pending));
                    const auto expected = brute(size, chunk, pending, budget, overhead);
                    assert(actual.waits == expected.waits && actual.pending == expected.pending);
                    ++projection_cases;
                }
            }
        }
    }
    for (uint64_t size : {uint64_t(std::numeric_limits<Size>::max()), uint64_t(std::numeric_limits<Size>::max()) - 1,
                          uint64_t(0xffffffff)}) {
        for (uint64_t chunk : {8 * MiB - 448, 8 * MiB - 32, 8 * MiB}) {
            for (uint64_t pending : {uint64_t(0), budget - 1, budget}) {
                const auto actual = Helper::project(Size(size), Size(chunk), Size(pending));
                const auto expected = cycle_reference(size, chunk, pending, budget, overhead);
                assert(actual.waits == expected.waits && actual.pending == expected.pending);
                ++projection_cases;
            }
        }
    }
}

template <class Helper> static uint64_t selected(Helper & helper, device & target, uint64_t size, uint64_t alignment) {
    using Size = decltype(Helper{}.pending);
    const uint64_t old_pending = helper.pending;
    const auto old_backend = helper.backend;
    const auto old_device = helper.device;
    const auto before = std::make_pair(initialized, waits);
    const auto before_trace = trace.size();
    const uint64_t chunk = helper.chunk_size_for(&target, Size(size), Size(alignment));
    assert(helper.pending == old_pending && helper.backend == old_backend && helper.device == old_device);
    assert(std::make_pair(initialized, waits) == before && trace.size() == before_trace);
    return chunk;
}

template <class Helper> static void upload(Helper & helper, device & target, uint64_t amount) {
    using Size = decltype(Helper{}.pending);
    helper.reserve(&target, Size(amount));
    target.queued += amount + helper.overhead;
    assert(target.queued <= helper.budget && helper.pending == target.queued);
    trace.push_back(40 + target.id);
}

template <class Helper> static void policy_checks() {
    using Size = decltype(Helper{}.pending);
    device a{1}, b{2};
    const uint64_t budget = Helper::budget;
    for (uint64_t alignment : {uint64_t(8), uint64_t(16), uint64_t(136), uint64_t(328), uint64_t(440), uint64_t(576)}) {
        const uint64_t old_cap = (8 * MiB / alignment) * alignment;
        for (uint64_t multiple = 1; multiple <= 17; ++multiple) {
            for (int64_t delta : {-int64_t(alignment), int64_t(0), int64_t(alignment)}) {
                const uint64_t size = uint64_t(int64_t(multiple * old_cap) + delta);
                for (uint64_t pending : {uint64_t(0), budget / 4, budget / 2, budget - 1, budget}) {
                    Helper helper;
                    helper.device = &a;
                    helper.pending = Size(pending);
                    const uint64_t chunk = selected(helper, a, size, alignment);
                    const uint64_t original = std::min(size, old_cap);
                    assert(chunk <= original && chunk != 0);
                    assert((size - 1) / chunk == (size - 1) / original);
                    const auto before = brute(size, original, pending, budget, helper.overhead);
                    const auto after = brute(size, chunk, pending, budget, helper.overhead);
                    assert(std::make_pair(after.waits, after.pending) <= std::make_pair(before.waits, before.pending));
                    // Device change predictions start at zero without draining now.
                    const uint64_t switched = selected(helper, b, size, alignment);
                    helper.pending = 0;
                    helper.device = nullptr;
                    assert(switched == selected(helper, b, size, alignment));
                    ++policy_cases;
                }
            }
        }
    }
    {
        Helper helper;
        assert(selected(helper, a, 0, 0) == 0);
        assert(selected(helper, a, 0, 8 * MiB + 1) == 0);
        for (uint64_t alignment : {uint64_t(0), 8 * MiB + 1}) {
            bool threw = false;
            try {
                selected(helper, a, 1, alignment);
            } catch (const std::runtime_error &) {
                threw = true;
            }
            assert(threw && helper.pending == 0 && helper.backend == nullptr);
        }
        assert(selected(helper, a, 8 * MiB, 8 * MiB) == 8 * MiB); // preferred cap rounds to zero
        for (uint64_t size :
             {uint64_t(std::numeric_limits<Size>::max()), uint64_t(std::numeric_limits<Size>::max()) - 1}) {
            for (uint64_t alignment : {uint64_t(8), uint64_t(16), uint64_t(328), uint64_t(440)}) {
                const uint64_t chunk = selected(helper, a, size, alignment);
                const uint64_t original = 8 * MiB / alignment * alignment;
                assert((size - 1) / chunk == (size - 1) / original);
                const auto before = cycle_reference(size, original, 0, budget, helper.overhead);
                const auto after = cycle_reference(size, chunk, 0, budget, helper.overhead);
                assert(std::make_pair(after.waits, after.pending) <= std::make_pair(before.waits, before.pending));
            }
        }
    }
    {
        Helper helper;
        for (uint64_t multiple : {uint64_t(1), uint64_t(2), uint64_t(3), uint64_t(4), uint64_t(16)}) {
            const uint64_t size = multiple * 8 * MiB;
            assert(selected(helper, a, size, 16) == 8 * MiB); // no extra read on exact cap multiples
        }
    }
    // Same-count-only would produce four drains. The policy must preserve three
    // for this already-largest-first, complete-quantization-block input sequence.
    const uint64_t start_waits = waits;
    uint64_t count = 0;
    {
        Helper helper;
        for (const auto & [size, alignment] :
             {std::pair<uint64_t, uint64_t>{50331560, 16}, {25170200, 440}, {16772712, 16}}) {
            const uint64_t chunk = selected(helper, a, size, alignment);
            uint64_t done = 0;
            while (done < size) {
                const uint64_t amount = std::min(chunk, size - done);
                upload(helper, a, amount);
                done += amount;
                ++count;
            }
            assert(done == size);
        }
    }
    assert(count == 12 && waits - start_waits == 3 && live == 0 && a.queued == 0);
    ++tensor_sequences;

    // Reject an always-original regression: a representative IQ3_S expert has
    // the same 43 chunks but can fill four full chunks in each charged batch.
    const uint64_t before_positive = waits;
    {
        Helper helper;
        const uint64_t size = 360448000, alignment = 440;
        const uint64_t original = (8 * MiB / alignment) * alignment;
        const uint64_t chunk = selected(helper, a, size, alignment);
        assert(original == 8388600 && chunk == 8388160);
        const auto baseline = brute(size, original, 0, helper.budget, helper.overhead);
        assert(baseline.waits + bool(baseline.pending) == 14);
        uint64_t done = 0, chunks = 0;
        while (done < size) {
            const uint64_t amount = std::min(chunk, size - done);
            upload(helper, a, amount);
            done += amount;
            ++chunks;
        }
        assert(done == size && chunks == 43);
    }
    assert(waits - before_positive == 11 && live == 0 && a.queued == 0);
    ++positive_adoptions;

    // Pure selection precedes a file read. Device acquisition and draining stay
    // in reserve; throwing before the first reserve still retires the old queue.
    const uint64_t before_failure = waits;
    try {
        Helper helper;
        upload(helper, a, 4096);
        const auto old_live = live;
        selected(helper, b, 360448000, 440);
        assert(live == old_live && b.queued == 0 && a.queued != 0);
        throw std::runtime_error("read failed before reserve");
    } catch (const std::runtime_error &) {
    }
    assert(waits == before_failure + 1 && live == 0 && a.queued == 0 && b.queued == 0);
    {
        Helper helper;
        upload(helper, a, 4096);
        selected(helper, b, 360448000, 440);
        const auto before = trace.size();
        upload(helper, b, 4096);
        assert(trace[before] == 21 && trace[before + 1] == 31 && trace[before + 2] == 12 && trace[before + 3] == 42);
        assert(a.queued == 0);
    }
    auto cancelled = [&]() {
        Helper helper;
        const auto chunk = selected(helper, a, 360448000, 440);
        upload(helper, a, chunk);
        return false;
    };
    assert(!cancelled() && live == 0 && a.queued == 0);
    {
        Helper helper;
        upload(helper, a, 4096);
        fail_init = true;
        bool threw = false;
        try {
            upload(helper, b, 4096);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        assert(threw && helper.pending == 0 && helper.backend == nullptr && live == 0 && a.queued == 0);
        fail_init = false;
        upload(helper, b, 1);
    }
    assert(live == 0 && a.queued == 0 && b.queued == 0);
}

int main() {
    projection_checks<width32::llama_webgpu_load_uploads>();
    projection_checks<width64::llama_webgpu_load_uploads>();
    policy_checks<width32::llama_webgpu_load_uploads>();
    policy_checks<width64::llama_webgpu_load_uploads>();
    assert(live == 0);
    std::cout << "{\"passed\":true,\"widths\":[32,64],\"projectionCases\":" << projection_cases
              << ",\"policyCases\":" << policy_cases << ",\"regressionSequences\":" << tensor_sequences
              << ",\"positiveAdoptions\":" << positive_adoptions << ",\"liveBackends\":" << live << ",\"gpu\":false}"
              << std::endl;
}
