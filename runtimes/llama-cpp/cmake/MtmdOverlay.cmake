get_filename_component(LCB_MTMD_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
get_filename_component(LCB_REPOSITORY_ROOT "${LCB_MTMD_ROOT}/../.." ABSOLUTE)
set(LCB_MTMD_MODE cpu)
set(LCB_MTMD_CAPABILITY 0)
if(LCB_WEBGPU AND LCB_WEBGPU_BF16_PROJECTOR)
    set(LCB_MTMD_MODE webgpu)
    set(LCB_MTMD_CAPABILITY 1)
    execute_process(COMMAND "${Python3_EXECUTABLE}" -m build_tools.runtimes.llama_cpp.sources.retained_bf16
        --source "${LCB_LLAMA_SOURCE}" WORKING_DIRECTORY "${LCB_REPOSITORY_ROOT}" COMMAND_ERROR_IS_FATAL ANY)
    file(READ "${LCB_MTMD_ROOT}/config/retained-bf16-source.json" LCB_RETAINED_CONTRACT)
    string(JSON LCB_RETAINED_FILE_COUNT LENGTH "${LCB_RETAINED_CONTRACT}" files)
    math(EXPR LCB_RETAINED_FILE_LAST "${LCB_RETAINED_FILE_COUNT} - 1")
    foreach(LCB_RETAINED_FILE_INDEX RANGE ${LCB_RETAINED_FILE_LAST})
        string(JSON LCB_RETAINED_FILE MEMBER "${LCB_RETAINED_CONTRACT}" files ${LCB_RETAINED_FILE_INDEX})
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
            "${LCB_LLAMA_SOURCE}/${LCB_RETAINED_FILE}")
    endforeach()
endif()
set(LCB_MTMD_OVERLAY "${CMAKE_CURRENT_BINARY_DIR}/mtmd-source-overlay")
execute_process(COMMAND "${Python3_EXECUTABLE}" -m build_tools.runtimes.llama_cpp.sources.mtmd
    --source "${LCB_LLAMA_SOURCE}" --output "${LCB_MTMD_OVERLAY}" --mode "${LCB_MTMD_MODE}"
    WORKING_DIRECTORY "${LCB_REPOSITORY_ROOT}" COMMAND_ERROR_IS_FATAL ANY)
file(READ "${LCB_MTMD_OVERLAY}/mtmd-source-manifest.json" LCB_MTMD_SOURCE_MANIFEST)
string(JSON LCB_MTMD_MANIFEST_MODE GET "${LCB_MTMD_SOURCE_MANIFEST}" mode)
if(NOT LCB_MTMD_MANIFEST_MODE STREQUAL LCB_MTMD_MODE)
    message(FATAL_ERROR "Prepared mtmd mode disagrees with the target")
endif()
# Reconfigure for every reviewed preparation input in both CPU and WebGPU mode.
string(JSON LCB_MTMD_CONTRACT_COUNT LENGTH "${LCB_MTMD_SOURCE_MANIFEST}" contractInputs)
math(EXPR LCB_MTMD_CONTRACT_LAST "${LCB_MTMD_CONTRACT_COUNT} - 1")
foreach(LCB_MTMD_CONTRACT_INDEX RANGE ${LCB_MTMD_CONTRACT_LAST})
    string(JSON LCB_MTMD_CONTRACT_FILE MEMBER "${LCB_MTMD_SOURCE_MANIFEST}" contractInputs ${LCB_MTMD_CONTRACT_INDEX})
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${LCB_LLAMA_SOURCE}/${LCB_MTMD_CONTRACT_FILE}")
endforeach()
get_target_property(LCB_MTMD_SOURCE_DIR mtmd SOURCE_DIR)
get_target_property(LCB_MTMD_SOURCES mtmd SOURCES)
set(LCB_MTMD_REPLACED "")
set(LCB_MTMD_COMPILED_RELATIVE "")
foreach(LCB_MTMD_SOURCE IN LISTS LCB_MTMD_SOURCES)
    if(IS_ABSOLUTE "${LCB_MTMD_SOURCE}")
        set(LCB_MTMD_ABSOLUTE "${LCB_MTMD_SOURCE}")
    else()
        set(LCB_MTMD_ABSOLUTE "${LCB_MTMD_SOURCE_DIR}/${LCB_MTMD_SOURCE}")
    endif()
    file(RELATIVE_PATH LCB_MTMD_RELATIVE "${LCB_LLAMA_SOURCE}/tools/mtmd" "${LCB_MTMD_ABSOLUTE}")
    if(LCB_MTMD_RELATIVE MATCHES "^\\.\\." OR NOT EXISTS "${LCB_MTMD_ABSOLUTE}")
        message(FATAL_ERROR "mtmd source is outside the reviewed subtree: ${LCB_MTMD_SOURCE}")
    endif()
    string(JSON LCB_MTMD_SOURCE_HASH ERROR_VARIABLE LCB_MTMD_SOURCE_ERROR
        GET "${LCB_MTMD_SOURCE_MANIFEST}" preparedFiles "${LCB_MTMD_RELATIVE}" sha256)
    if(LCB_MTMD_SOURCE_ERROR OR NOT EXISTS "${LCB_MTMD_OVERLAY}/${LCB_MTMD_RELATIVE}")
        message(FATAL_ERROR "mtmd source missing from the prepared manifest: ${LCB_MTMD_RELATIVE}")
    endif()
    get_source_file_property(LCB_MTMD_SKIP_UNITY "${LCB_MTMD_ABSOLUTE}"
        TARGET_DIRECTORY mtmd SKIP_UNITY_BUILD_INCLUSION)
    if(LCB_MTMD_SKIP_UNITY)
        set_source_files_properties("${LCB_MTMD_OVERLAY}/${LCB_MTMD_RELATIVE}"
            TARGET_DIRECTORY mtmd PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
    endif()
    list(APPEND LCB_MTMD_REPLACED "${LCB_MTMD_OVERLAY}/${LCB_MTMD_RELATIVE}")
    list(APPEND LCB_MTMD_COMPILED_RELATIVE "${LCB_MTMD_RELATIVE}")
endforeach()
set_property(TARGET mtmd PROPERTY SOURCES "${LCB_MTMD_REPLACED}")
# Every mtmd TU uses the same private model layout, including unity batches.
# Model builder files are byte-identical copies; only reviewed core files change.
target_include_directories(mtmd BEFORE PRIVATE "${LCB_MTMD_OVERLAY}" "${LCB_MTMD_ROOT}/native/vision")
target_compile_definitions(mtmd PRIVATE LCB_MTMD_WEBGPU_BF16_RUNTIME=${LCB_MTMD_CAPABILITY})
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/mtmd-prepared-sources.txt"
    CONTENT "$<TARGET_PROPERTY:mtmd,SOURCES>")
file(GLOB_RECURSE LCB_MTMD_INPUT_FILES CONFIGURE_DEPENDS "${LCB_LLAMA_SOURCE}/tools/mtmd/*")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    ${LCB_MTMD_INPUT_FILES}
    "${LCB_REPOSITORY_ROOT}/build_tools/runtimes/llama_cpp/sources/mtmd.py"
    "${LCB_REPOSITORY_ROOT}/build_tools/runtimes/llama_cpp/sources/retained_bf16.py"
    "${LCB_MTMD_ROOT}/config/retained-bf16-source.json"
    "${LCB_MTMD_ROOT}/upstream-patches-only-as-a-last-resort-with-explicit-user-approval/mtmd-vision-bf16-runtime.patch"
    "${LCB_MTMD_ROOT}/upstream-patches-only-as-a-last-resort-with-explicit-user-approval/mtmd-webgpu-bf16.patch"
    "${LCB_MTMD_ROOT}/upstream-patches-only-as-a-last-resort-with-explicit-user-approval/mtmd-non-pthread-threadpool.patch")
