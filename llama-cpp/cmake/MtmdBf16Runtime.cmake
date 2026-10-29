get_filename_component(LCB_MTMD_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(LCB_MTMD_RUNTIME_OVERLAY "${CMAKE_CURRENT_BINARY_DIR}/mtmd-runtime-overlay")
execute_process(COMMAND "${Python3_EXECUTABLE}" "${LCB_MTMD_ROOT}/scripts/prepare_mtmd.py"
    --source "${LCB_LLAMA_SOURCE}" --output "${LCB_MTMD_RUNTIME_OVERLAY}" --component runtime
    COMMAND_ERROR_IS_FATAL ANY)
get_target_property(LCB_MTMD_RUNTIME_SOURCES mtmd SOURCES)
if(NOT "mtmd.cpp" IN_LIST LCB_MTMD_RUNTIME_SOURCES)
    message(FATAL_ERROR "Upstream mtmd source layout changed; review the storage runtime API")
endif()
list(REMOVE_ITEM LCB_MTMD_RUNTIME_SOURCES "mtmd.cpp")
list(APPEND LCB_MTMD_RUNTIME_SOURCES "${LCB_MTMD_RUNTIME_OVERLAY}/mtmd.cpp")
set_property(TARGET mtmd PROPERTY SOURCES "${LCB_MTMD_RUNTIME_SOURCES}")
target_include_directories(mtmd PRIVATE "${LCB_LLAMA_SOURCE}/tools/mtmd" "${LCB_MTMD_ROOT}/bridge")
if(LCB_WEBGPU AND LCB_WEBGPU_BF16_PROJECTOR)
    target_compile_definitions(mtmd PRIVATE LCB_MTMD_WEBGPU_BF16_RUNTIME=1)
else()
    target_compile_definitions(mtmd PRIVATE LCB_MTMD_WEBGPU_BF16_RUNTIME=0)
    execute_process(COMMAND "${Python3_EXECUTABLE}" "${LCB_MTMD_ROOT}/scripts/prepare_mtmd.py"
        --source "${LCB_LLAMA_SOURCE}" --output "${LCB_MTMD_RUNTIME_OVERLAY}" --component clip-runtime
        COMMAND_ERROR_IS_FATAL ANY)
    get_target_property(LCB_MTMD_CPU_SOURCES mtmd SOURCES)
    if(NOT "clip.cpp" IN_LIST LCB_MTMD_CPU_SOURCES)
        message(FATAL_ERROR "Upstream mtmd source layout changed; review the CPU storage plumbing")
    endif()
    list(REMOVE_ITEM LCB_MTMD_CPU_SOURCES "clip.cpp")
    list(APPEND LCB_MTMD_CPU_SOURCES "${LCB_MTMD_RUNTIME_OVERLAY}/clip.cpp")
    set_property(TARGET mtmd PROPERTY SOURCES "${LCB_MTMD_CPU_SOURCES}")
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${LCB_LLAMA_SOURCE}/tools/mtmd/mtmd.cpp"
    "${LCB_LLAMA_SOURCE}/tools/mtmd/clip.cpp"
    "${LCB_MTMD_ROOT}/upstream-patches-only-as-a-last-resort-with-explicit-user-approval/mtmd-vision-bf16-clip-runtime.patch"
    "${LCB_MTMD_ROOT}/upstream-patches-only-as-a-last-resort-with-explicit-user-approval/mtmd-vision-bf16-runtime.patch"
    "${LCB_MTMD_ROOT}/scripts/prepare_mtmd.py")
