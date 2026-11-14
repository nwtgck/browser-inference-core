#include "api.h"

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

enum BicoreRecordLayouts {
    BICORE_ggml_fixture_record = sizeof(ggml_fixture_record),
    BICORE_ggml_bf16_t = sizeof(ggml_bf16_t),
};
enum BicoreMacroConstants : int64_t { BICORE_GGML_FIXTURE_WIDTH = sizeof(void *) };

#ifndef BICORE_ABI_EXTRACT
#include "fingerprint.h"

extern "C" {
uint32_t bicore_abi_version(void) {
    return 3;
}
uint32_t bicore_pointer_bytes(void) {
    return sizeof(void *);
}
const char * bicore_schema_hash(void) {
    return BICORE_ABI_FINGERPRINT;
}
uint32_t bicore_file_backed_rows_version(void) {
    return 0;
}
uint64_t bicore_model_file_backed_shard(uint64_t, uint32_t) {
    return 0;
}

int32_t ggml_fixture_i32(int32_t value) {
    return value;
}
uint32_t ggml_fixture_u32(uint32_t value) {
    return value;
}
uint8_t ggml_fixture_u8(uint8_t value) {
    return value;
}
bool ggml_fixture_bool(bool value) {
    return value;
}
uint64_t ggml_fixture_u64(uint64_t value) {
    return value;
}
int64_t ggml_fixture_i64(int64_t value) {
    return value;
}
size_t ggml_fixture_size(size_t value) {
    return value;
}
void * ggml_fixture_pointer(void * value) {
    return value;
}
enum ggml_fixture_mode ggml_fixture_enum(enum ggml_fixture_mode value) {
    return value;
}
double ggml_fixture_float(double value) {
    return value;
}
int32_t ggml_fixture_callback_call(ggml_fixture_callback callback, int32_t value) {
    return callback(value);
}
int32_t ggml_fixture_getter(void) {
    return 11;
}
int32_t ggml_fixture_wait(int32_t value) {
#ifdef __EMSCRIPTEN__
    emscripten_sleep(1);
#endif
    return value + 1;
}
ggml_fixture_record ggml_fixture_record_default(void) {
    ggml_fixture_record value{};
    value.count = 4;
    value.active = true;
    value.signed_value = -2;
    value.values[2] = 9;
    value.flags[1] = true;
    value.nested.payload = 23;
    value.nested.inner.code = -3;
    value.samples[1].value = -7;
    return value;
}
int32_t ggml_fixture_record_sum(ggml_fixture_record value) {
    return static_cast<int32_t>(value.count) + value.signed_value + value.values[2] + value.nested.payload;
}
}
#endif
