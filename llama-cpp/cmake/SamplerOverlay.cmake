get_filename_component(LCB_SAMPLER_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
# Common native sampler path: existing CPU and WebGPU profiles use the same TU.
set(LCB_SAMPLER_OVERLAY "${CMAKE_CURRENT_BINARY_DIR}/sampler-overlay")
execute_process(COMMAND "${Python3_EXECUTABLE}" "${LCB_SAMPLER_ROOT}/scripts/prepare_sampler.py"
    --source "${LCB_LLAMA_SOURCE}" --output "${LCB_SAMPLER_OVERLAY}"
    COMMAND_ERROR_IS_FATAL ANY)
get_target_property(LCB_SAMPLER_SOURCES llama SOURCES)
set(LCB_SAMPLER_MATCHES 0)
foreach(LCB_SAMPLER_SOURCE IN LISTS LCB_SAMPLER_SOURCES)
    if(LCB_SAMPLER_SOURCE STREQUAL "llama-sampler.cpp")
        math(EXPR LCB_SAMPLER_MATCHES "${LCB_SAMPLER_MATCHES} + 1")
    endif()
endforeach()
if(NOT LCB_SAMPLER_MATCHES EQUAL 1)
    message(FATAL_ERROR "Upstream sampler source layout changed; review sampler overlay")
endif()
list(REMOVE_ITEM LCB_SAMPLER_SOURCES "llama-sampler.cpp")
list(APPEND LCB_SAMPLER_SOURCES "${LCB_SAMPLER_OVERLAY}/llama-sampler.cpp")
set_property(TARGET llama PROPERTY SOURCES "${LCB_SAMPLER_SOURCES}")
set_source_files_properties("${LCB_SAMPLER_OVERLAY}/llama-sampler.cpp"
    TARGET_DIRECTORY llama PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
target_include_directories(llama PRIVATE "${LCB_LLAMA_SOURCE}/src")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${LCB_LLAMA_SOURCE}/src/llama-sampler.cpp"
    "${LCB_LLAMA_SOURCE}/src/llama-context.cpp"
    "${LCB_LLAMA_SOURCE}/src/llama-context.h"
    "${LCB_SAMPLER_ROOT}/scripts/prepare_sampler.py"
    "${LCB_SAMPLER_ROOT}/upstream-patches-only-as-a-last-resort-with-explicit-user-approval/llama-sampler-single-sync.patch")
