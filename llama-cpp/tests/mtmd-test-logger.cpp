// Standalone model/role fixtures omit clip.cpp, which owns this state in the
// real runtime. Link this definition only into those test executables.
#include "clip-impl.h"

clip_logger_state g_logger_state = {clip_log_callback_default, nullptr};
