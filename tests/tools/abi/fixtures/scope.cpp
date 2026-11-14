#include <stddef.h>
#include <stdint.h>

namespace mtmd {
struct mtmd_cpp_deleter {
    void operator()(void *) {}
};
} // namespace mtmd

namespace mtmd_helper {
struct mtmd_helper_cpp_owner {};
} // namespace mtmd_helper

#ifdef BICORE_AMBIGUOUS_ALIAS
namespace hidden {
using size_t = unsigned char;
} // namespace hidden
#endif

extern "C" {
struct mtmd_params {
    int32_t value;
};
size_t mtmd_identity(int32_t value);
}
