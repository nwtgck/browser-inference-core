#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Present only in test variants; routes a real upstream log through its C API.
void spc_test_emit_log(void);

// Real GGUF parser with bounded header reads and virtual payload above 4 GiB.
int spc_test_gguf_large_file(void);

#ifdef __cplusplus
}
#endif
