get_filename_component(LCB_AUDIO_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
# A separate translation-unit copy composes with the optional vision overlay.
set(LCB_AUDIO_OVERLAY "${CMAKE_CURRENT_BINARY_DIR}/mtmd-audio-overlay")
execute_process(COMMAND "${Python3_EXECUTABLE}" "${LCB_AUDIO_ROOT}/scripts/prepare_mtmd.py"
    --source-id "${LCB_SOURCE_ID}" --source "${LCB_LLAMA_SOURCE}" --output "${LCB_AUDIO_OVERLAY}" --component audio
    COMMAND_ERROR_IS_FATAL ANY)
get_target_property(LCB_AUDIO_SOURCES mtmd SOURCES)
if(NOT "mtmd-audio.cpp" IN_LIST LCB_AUDIO_SOURCES)
    message(FATAL_ERROR "Upstream mtmd source layout changed; review the audio threading overlay")
endif()
list(REMOVE_ITEM LCB_AUDIO_SOURCES "mtmd-audio.cpp")
list(APPEND LCB_AUDIO_SOURCES "${LCB_AUDIO_OVERLAY}/mtmd-audio.cpp")
set_property(TARGET mtmd PROPERTY SOURCES "${LCB_AUDIO_SOURCES}")
target_include_directories(mtmd PRIVATE "${LCB_LLAMA_SOURCE}/tools/mtmd")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${LCB_LLAMA_SOURCE}/tools/mtmd/mtmd-audio.cpp"
    "${LCB_AUDIO_ROOT}/upstream-patches-only-as-a-last-resort-with-explicit-user-approval/mtmd-audio-single-thread.patch"
    "${LCB_AUDIO_ROOT}/scripts/prepare_mtmd.py")

# A per-source patch plan is a configure input, not an implicit shared policy.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_LIST_DIR}/../config/sources.json"
    "${CMAKE_CURRENT_LIST_DIR}/../sources/${LCB_SOURCE_ID}/patches.json")
