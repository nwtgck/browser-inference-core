#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Present only in test variants; routes a real upstream log through its C API.
void spc_test_emit_log(void);

#ifdef __cplusplus
}
#endif
