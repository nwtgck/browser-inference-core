#pragma once

#include <stdexcept>

// Keep fixture checks active under Release/NDEBUG, including boundary doubles.
#define SDC_CHECK(condition)                                                                                           \
    do {                                                                                                               \
        if (!(condition))                                                                                              \
            throw std::runtime_error("contract failed: " #condition);                                                  \
    } while (false)
