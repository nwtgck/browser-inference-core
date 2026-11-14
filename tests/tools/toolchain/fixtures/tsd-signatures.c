#include <stdint.h>
#ifdef TEST_ASYNCIFY
#include <emscripten.h>
#endif

uint64_t raw_i64(uint64_t value) {
    return value ^ UINT64_C(0x123456789abcdef0);
}

uintptr_t converted_pointer(uintptr_t value) {
    return value;
}

uintptr_t optional_pointer(uintptr_t value) {
    return value;
}

uint64_t async_value(uint64_t value) {
    return value + 1;
}

int sync_value(int value) {
    return value + 3;
}

#ifdef TEST_ASYNCIFY
int64_t suspend_i64(int64_t value) {
    emscripten_sleep(0);
    return value + 1;
}
#endif
