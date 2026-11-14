if(NOT TARGET llama)
    message(FATAL_ERROR "File-backed rows requires the upstream llama target")
endif()
# Emscripten project() probes wasm32 before the target's MEMORY64 option.
if((EMSCRIPTEN AND NOT LCB_MEMORY64) OR (NOT EMSCRIPTEN AND NOT CMAKE_SIZEOF_VOID_P EQUAL 8))
    message(FATAL_ERROR "File-backed rows requires a 64-bit build")
endif()
# src, every model TU and models.h already belong to one prepared tree. Keep
# upstream unity exclusions and per-source properties on their original paths.
if(NOT EXISTS "${LCB_LLAMA_SOURCE}/src/models/file-backed-rows.h")
    message(FATAL_ERROR "Missing prepared file-backed rows companion")
endif()
