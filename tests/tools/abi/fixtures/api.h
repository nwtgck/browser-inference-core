#pragma once

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint16_t bits;
} ggml_bf16_t;

struct ggml_fixture_record {
    uint32_t count;
    bool active;
    union {
        int32_t signed_value;
        uint32_t unsigned_value;
    };
    uint16_t values[3];
    bool flags[2];
    struct {
        uint8_t tag;
        uint32_t payload;
        union {
            struct {
                int8_t code;
            } inner;
            uint32_t spare;
        };
    } nested;
    struct {
        int16_t value;
    } samples[2];
};

enum ggml_fixture_mode { GGML_FIXTURE_NEGATIVE = -3, GGML_FIXTURE_POSITIVE = 7 };
typedef int32_t (*ggml_fixture_callback)(int32_t value);

extern "C" {
static inline int32_t ggml_fixture_header_local(int32_t value) {
    return value;
}
int32_t ggml_fixture_i32(int32_t value);
uint32_t ggml_fixture_u32(uint32_t value);
uint8_t ggml_fixture_u8(uint8_t value);
bool ggml_fixture_bool(bool value);
uint64_t ggml_fixture_u64(uint64_t value);
int64_t ggml_fixture_i64(int64_t value);
size_t ggml_fixture_size(size_t value);
void * ggml_fixture_pointer(void * value);
enum ggml_fixture_mode ggml_fixture_enum(enum ggml_fixture_mode value);
double ggml_fixture_float(double value);
int32_t ggml_fixture_callback_call(ggml_fixture_callback callback, int32_t value);
int32_t ggml_fixture_getter(void);
int32_t ggml_fixture_wait(int32_t value);
struct ggml_fixture_record ggml_fixture_record_default(void);
int32_t ggml_fixture_record_sum(struct ggml_fixture_record value);
}
