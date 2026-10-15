get_filename_component(LCB_FILE_ROWS_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
if(NOT TARGET llama)
    message(FATAL_ERROR "File-backed rows requires the upstream llama target")
endif()
if((EMSCRIPTEN AND NOT LCB_MEMORY64) OR NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "File-backed rows requires a 64-bit build")
endif()
set(LCB_FILE_ROWS_OVERLAY "${CMAKE_CURRENT_BINARY_DIR}/file-backed-rows-overlay")
execute_process(COMMAND "${Python3_EXECUTABLE}" "${LCB_FILE_ROWS_ROOT}/scripts/prepare_file_backed_rows.py"
    --source "${LCB_LLAMA_SOURCE}" --output "${LCB_FILE_ROWS_OVERLAY}"
    COMMAND_ERROR_IS_FATAL ANY)
get_target_property(LCB_FILE_ROWS_SOURCES llama SOURCES)
set(LCB_FILE_ROWS_REPLACED "")
set(LCB_FILE_ROWS_MODEL_COUNT 0)
set(LCB_FILE_ROWS_API_COUNT 0)
set(LCB_FILE_ROWS_QWEN_COUNT 0)
foreach(LCB_FILE_ROWS_SOURCE IN LISTS LCB_FILE_ROWS_SOURCES)
    get_filename_component(LCB_FILE_ROWS_ABSOLUTE "${LCB_FILE_ROWS_SOURCE}" ABSOLUTE
        BASE_DIR "${LCB_LLAMA_SOURCE}/src")
    file(RELATIVE_PATH LCB_FILE_ROWS_RELATIVE "${LCB_LLAMA_SOURCE}/src" "${LCB_FILE_ROWS_ABSOLUTE}")
    if(LCB_FILE_ROWS_RELATIVE STREQUAL "llama.cpp" OR LCB_FILE_ROWS_RELATIVE STREQUAL "llama-model.cpp" OR LCB_FILE_ROWS_RELATIVE MATCHES "^models/[^/]+[.]cpp$")
        set(LCB_FILE_ROWS_COPY "${LCB_FILE_ROWS_OVERLAY}/src/${LCB_FILE_ROWS_RELATIVE}")
        list(APPEND LCB_FILE_ROWS_REPLACED "${LCB_FILE_ROWS_COPY}")
        if(LCB_FILE_ROWS_RELATIVE STREQUAL "llama-model.cpp")
            math(EXPR LCB_FILE_ROWS_MODEL_COUNT "${LCB_FILE_ROWS_MODEL_COUNT} + 1")
            set_source_files_properties("${LCB_FILE_ROWS_COPY}" TARGET_DIRECTORY llama
                PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
        elseif(LCB_FILE_ROWS_RELATIVE STREQUAL "llama.cpp")
            math(EXPR LCB_FILE_ROWS_API_COUNT "${LCB_FILE_ROWS_API_COUNT} + 1")
        elseif(LCB_FILE_ROWS_RELATIVE STREQUAL "models/qwen4exp.cpp")
            math(EXPR LCB_FILE_ROWS_QWEN_COUNT "${LCB_FILE_ROWS_QWEN_COUNT} + 1")
        endif()
    else()
        # In particular retain the independent WebGPU loader overlay, if any.
        list(APPEND LCB_FILE_ROWS_REPLACED "${LCB_FILE_ROWS_SOURCE}")
    endif()
endforeach()
if(NOT LCB_FILE_ROWS_MODEL_COUNT EQUAL 1 OR NOT LCB_FILE_ROWS_QWEN_COUNT EQUAL 1 OR NOT LCB_FILE_ROWS_API_COUNT EQUAL 1)
    message(FATAL_ERROR "Upstream model source layout changed; review the file-backed rows overlay")
endif()
set_property(TARGET llama PROPERTY SOURCES "${LCB_FILE_ROWS_REPLACED}")
# The relocated llama-model.cpp and every model TU resolve the same models.h.
# No generic model/base class layout is changed.
target_include_directories(llama BEFORE PRIVATE "${LCB_FILE_ROWS_OVERLAY}/src")
file(GLOB_RECURSE LCB_FILE_ROWS_DEPENDENCIES CONFIGURE_DEPENDS "${LCB_LLAMA_SOURCE}/src/*")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    ${LCB_FILE_ROWS_DEPENDENCIES}
    "${LCB_FILE_ROWS_ROOT}/scripts/prepare_file_backed_rows.py"
    "${LCB_FILE_ROWS_ROOT}/bridge/file-backed-rows.h"
    "${LCB_FILE_ROWS_ROOT}/tests/file-backed-rows/inputs.json"
    "${LCB_FILE_ROWS_ROOT}/upstream-patches-only-as-a-last-resort-with-explicit-user-approval/qwen4exp-file-backed-rows.patch")
