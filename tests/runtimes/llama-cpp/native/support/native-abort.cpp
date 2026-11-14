#include <cstdarg>
#include <cstdio>
#include <stdexcept>

// Linker interception belongs only to this fixture. Observe abort diagnostics
// and cleanup without claiming that the production caller can recover.
extern "C" [[noreturn]] void __wrap_ggml_abort(const char *, int, const char * format, ...) {
    char message[2048];
    va_list arguments;
    va_start(arguments, format);
    std::vsnprintf(message, sizeof(message), format, arguments);
    va_end(arguments);
    throw std::runtime_error(message);
}
