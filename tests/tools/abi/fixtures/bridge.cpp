#include "api.h"

#include <cstdlib>

extern "C" {
void bicore_ggml_fixture_record_default(ggml_fixture_record * output) {
    if (!output) {
        std::abort();
    }
    *output = ggml_fixture_record_default();
}
int32_t bicore_ggml_fixture_record_sum(const ggml_fixture_record * input) {
    if (!input) {
        std::abort();
    }
    return ggml_fixture_record_sum(*input);
}
}
