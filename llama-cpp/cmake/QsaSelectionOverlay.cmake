get_filename_component(LCB_QSA_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
if(NOT TARGET llama)
    message(FATAL_ERROR "QSA selection requires the upstream llama target")
endif()
# Compose after the optional file-backed rows overlay, including on wasm32 and
# native builds where that extension is absent. Preserve its model header ABI.
get_target_property(LCB_QSA_SOURCES llama SOURCES)
set(LCB_QSA_MATCHES 0)
set(LCB_QSA_REPLACED "")
foreach(LCB_QSA_SOURCE IN LISTS LCB_QSA_SOURCES)
    get_filename_component(LCB_QSA_NAME "${LCB_QSA_SOURCE}" NAME)
    if(LCB_QSA_NAME STREQUAL "qwen4exp.cpp")
        math(EXPR LCB_QSA_MATCHES "${LCB_QSA_MATCHES} + 1")
        get_filename_component(LCB_QSA_INPUT "${LCB_QSA_SOURCE}" ABSOLUTE BASE_DIR "${LCB_LLAMA_SOURCE}/src")
        get_filename_component(LCB_QSA_INCLUDE "${LCB_QSA_INPUT}" DIRECTORY)
        set(LCB_QSA_COPY "${CMAKE_CURRENT_BINARY_DIR}/qsa-selection-overlay/qwen4exp.cpp")
        list(APPEND LCB_QSA_REPLACED "${LCB_QSA_COPY}")
    else()
        list(APPEND LCB_QSA_REPLACED "${LCB_QSA_SOURCE}")
    endif()
endforeach()
if(NOT LCB_QSA_MATCHES EQUAL 1)
    message(FATAL_ERROR "Upstream QSA source layout changed; review selection overlay")
endif()
execute_process(COMMAND "${Python3_EXECUTABLE}" "${LCB_QSA_ROOT}/scripts/prepare_qsa_selection.py"
    --source "${LCB_LLAMA_SOURCE}" --input "${LCB_QSA_INPUT}"
    --output "${CMAKE_CURRENT_BINARY_DIR}/qsa-selection-overlay"
    COMMAND_ERROR_IS_FATAL ANY)
set_property(TARGET llama PROPERTY SOURCES "${LCB_QSA_REPLACED}")
# Never let a unity translation unit select another model's header definition.
set_source_files_properties("${LCB_QSA_COPY}" TARGET_DIRECTORY llama PROPERTIES
    SKIP_UNITY_BUILD_INCLUSION ON INCLUDE_DIRECTORIES "${LCB_QSA_INCLUDE}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${LCB_QSA_INPUT}"
    "${LCB_LLAMA_SOURCE}/src/llama-memory-hybrid-idx.cpp"
    "${LCB_LLAMA_SOURCE}/src/llama-memory-hybrid-idx.h"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml.c"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-cpu/ops.cpp"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-backend.cpp"
    "${LCB_QSA_ROOT}/scripts/prepare_qsa_selection.py"
    "${LCB_QSA_ROOT}/tests/qsa-selection/inputs.json"
    "${LCB_QSA_ROOT}/upstream-patches-only-as-a-last-resort-with-explicit-user-approval/qwen4exp-all-pools-selection.patch")
