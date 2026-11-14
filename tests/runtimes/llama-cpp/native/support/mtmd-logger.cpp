// Metadata-only graph/role fixtures omit mtmd.cpp, which owns this state in
// the runtime. Link this definition only into those test executables.
#include "mtmd-impl.h"

mtmd_logger_state g_logger_state{mtmd_log_callback_default, nullptr};
