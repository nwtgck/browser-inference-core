get_filename_component(LCB_MTMD_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
# Patch only a build-tree translation unit: the pinned submodule stays clean,
# including in concurrent profile builds and artifact provenance checks.
set(LCB_MTMD_OVERLAY "${CMAKE_CURRENT_BINARY_DIR}/mtmd-overlay")
execute_process(COMMAND "${Python3_EXECUTABLE}" "${LCB_MTMD_ROOT}/scripts/verify_retained_bf16_source.py"
    --source "${LCB_LLAMA_SOURCE}" COMMAND_ERROR_IS_FATAL ANY)
# A subsequent ordinary build must re-run the verifier when ANY reviewed
# upstream input changes, not only after clip.cpp or the contract is edited.
file(READ "${LCB_MTMD_ROOT}/config/retained-bf16-source.json" LCB_RETAINED_CONTRACT)
string(JSON LCB_RETAINED_FILE_COUNT LENGTH "${LCB_RETAINED_CONTRACT}" files)
math(EXPR LCB_RETAINED_FILE_LAST "${LCB_RETAINED_FILE_COUNT} - 1")
foreach(LCB_RETAINED_FILE_INDEX RANGE ${LCB_RETAINED_FILE_LAST})
    string(JSON LCB_RETAINED_FILE MEMBER "${LCB_RETAINED_CONTRACT}" files ${LCB_RETAINED_FILE_INDEX})
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${LCB_LLAMA_SOURCE}/${LCB_RETAINED_FILE}")
endforeach()
execute_process(COMMAND "${Python3_EXECUTABLE}" "${LCB_MTMD_ROOT}/scripts/prepare_mtmd.py"
    --source "${LCB_LLAMA_SOURCE}" --output "${LCB_MTMD_OVERLAY}"
    COMMAND_ERROR_IS_FATAL ANY)
# Replace the existing target source after upstream creates mtmd, preserving
# its compile options/dependencies. Compiling both copies would duplicate symbols.
get_target_property(LCB_MTMD_SOURCES mtmd SOURCES)
if(NOT "clip.cpp" IN_LIST LCB_MTMD_SOURCES)
    message(FATAL_ERROR "Upstream mtmd source layout changed; review the BF16 overlay")
endif()
list(REMOVE_ITEM LCB_MTMD_SOURCES "clip.cpp")
list(APPEND LCB_MTMD_SOURCES "${LCB_MTMD_OVERLAY}/clip.cpp")
set_property(TARGET mtmd PROPERTY SOURCES "${LCB_MTMD_SOURCES}")
# The moved translation unit still includes its upstream sibling headers;
# the bridge directory supplies only the local conversion helper.
target_include_directories(mtmd PRIVATE
    "${LCB_LLAMA_SOURCE}/tools/mtmd" "${LCB_MTMD_ROOT}/bridge")
# Regenerate the copy when either input changes, not just when CMake files do.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${LCB_LLAMA_SOURCE}/tools/mtmd/clip.cpp"
    "${LCB_MTMD_ROOT}/upstream-patches-only-as-a-last-resort-with-explicit-user-approval/mtmd-webgpu-bf16.patch"
    "${LCB_MTMD_ROOT}/scripts/prepare_mtmd.py"
    "${LCB_MTMD_ROOT}/scripts/verify_retained_bf16_source.py"
    "${LCB_MTMD_ROOT}/config/retained-bf16-source.json")
