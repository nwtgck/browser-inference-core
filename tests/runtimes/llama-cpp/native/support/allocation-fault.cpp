#include "allocation-fault.hpp"
#include <cstdlib>
#include <new>

allocation_fault::State allocation_fault::state;

void * operator new(size_t bytes) {
    if (allocation_fault::state.observing) {
        auto & state = allocation_fault::state;
        if (state.allocations >= state.sizes.size())
            std::abort();
        state.sizes[state.allocations++] = bytes;
        state.bytes += bytes;
        if (allocation_fault::state.fail_after == 0) {
            allocation_fault::state.fail_after = -1;
            throw std::bad_alloc();
        }
        if (allocation_fault::state.fail_after > 0)
            --allocation_fault::state.fail_after;
    }
    if (auto * value = std::malloc(bytes ? bytes : 1))
        return value;
    throw std::bad_alloc();
}
void * operator new[](size_t bytes) {
    return ::operator new(bytes);
}
void operator delete(void * value) noexcept {
    if (value && allocation_fault::state.observing)
        ++allocation_fault::state.deletions;
    std::free(value);
}
void operator delete(void * value, size_t) noexcept {
    ::operator delete(value);
}
void operator delete[](void * value) noexcept {
    ::operator delete(value);
}
void operator delete[](void * value, size_t) noexcept {
    ::operator delete(value);
}
