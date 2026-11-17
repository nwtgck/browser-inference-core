#include "speech.h"
#ifdef SPCB_VARIANT_TEST
#include "wasm-probes.h"
#endif
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <sys/types.h>

#ifndef _WIN32
static_assert(sizeof(off_t) == 8, "The speech model reader requires 64-bit file offsets");
#endif

namespace {
// Cancellation is the one API operation allowed while a request is running.
[[maybe_unused]] constexpr void (*sync_cancel)(speech_request *) = &speech_request_cancel;
} // namespace

#define BICORE_CONSTANT(name) BICORE_##name = (name)
enum class BicoreMacroConstants : int64_t {
    BICORE_CONSTANT(EINVAL),
    BICORE_CONSTANT(EIO),
    BICORE_CONSTANT(EOVERFLOW),
    BICORE_CONSTANT(EROFS),
    BICORE_CONSTANT(ENOENT),
    BICORE_CONSTANT(SEEK_SET),
    BICORE_CONSTANT(SEEK_CUR),
    BICORE_CONSTANT(SEEK_END),
    BICORE_CONSTANT(SPEECH_API_VERSION_MAJOR),
    BICORE_CONSTANT(SPEECH_API_VERSION_MINOR),
};
#undef BICORE_CONSTANT

// Describe the owned interop boundary, without claiming model inference quality
// or performance from ABI metadata alone. modelIoMask bit 0 denotes GGUF files.
enum class BicoreCapabilities : uint64_t {
    ggufFileOffsetBits = 64,
    callerOwnedRandomAccess = 1,
    upstreamApi = 1,
    modelIoMask = 1,
};

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
uint32_t bicore_model_io_capabilities(void) {
    return static_cast<uint32_t>(BicoreCapabilities::modelIoMask);
}
}
#endif
