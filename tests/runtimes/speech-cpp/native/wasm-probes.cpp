#include "wasm-probes.h"
#include "log.h"

extern "C" void spc_test_emit_log(void) {
    log_message(LogLevel::Info, "speech callback probe");
}
