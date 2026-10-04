get_filename_component(LCB_MTMD_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
# Patch only a build-tree translation unit: the pinned submodule stays clean,
# including in concurrent profile builds and artifact provenance checks.
set(LCB_MTMD_OVERLAY "${CMAKE_CURRENT_BINARY_DIR}/mtmd-overlay")
execute_process(COMMAND "${Python3_EXECUTABLE}" "${LCB_MTMD_ROOT}/scripts/prepare_mtmd.py"
    --source-id "${LCB_SOURCE_ID}" --source "${LCB_LLAMA_SOURCE}" --output "${LCB_MTMD_OVERLAY}"
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
    "${LCB_MTMD_ROOT}/scripts/prepare_mtmd.py")

# A per-source patch plan is a configure input, not an implicit shared policy.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_LIST_DIR}/../config/sources.json"
    "${CMAKE_CURRENT_LIST_DIR}/../sources/${LCB_SOURCE_ID}/patches.json")
