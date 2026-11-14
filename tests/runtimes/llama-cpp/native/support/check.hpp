#pragma once

#include <stdexcept>
#include <string>

// Test contracts remain active under Release/NDEBUG. Do not use the target's
// assertion configuration to decide whether a boundary double validates calls.
inline void bicore_check(bool value, const char * expression, const char * file, int line) {
    if (!value) {
        throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + ": " + expression);
    }
}
#define BICORE_CHECK(expression) bicore_check(bool(expression), #expression, __FILE__, __LINE__)
