#pragma once
#include <cstddef>
#include <array>

namespace allocation_fault {
struct State {
    bool observing = false;
    std::ptrdiff_t fail_after = -1;
    std::size_t allocations = 0;
    std::size_t bytes = 0;
    std::size_t deletions = 0;
    std::array<std::size_t, 2048> sizes{};
};
extern State state;
// External API implementations are observations, not the host allocation cost
// of the production caller. Their own recorder storage is outside that window.
struct ExternalCall {
    bool previous = state.observing;
    ExternalCall() {
        state.observing = false;
    }
    ~ExternalCall() {
        state.observing = previous;
    }
};
} // namespace allocation_fault
