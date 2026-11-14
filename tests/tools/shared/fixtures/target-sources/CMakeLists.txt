cmake_minimum_required(VERSION 3.24)
project(target_source_facts NONE)
include("${TARGET_SOURCES_HELPER}")

add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated/schema.json"
    COMMAND "${CMAKE_COMMAND}" -E touch "${CMAKE_CURRENT_BINARY_DIR}/generated/schema.json")
set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/generated/authored.rule" PROPERTIES GENERATED TRUE HEADER_FILE_ONLY TRUE)
add_library(core OBJECT "input, with space.txt" legitimate.rule "${CMAKE_CURRENT_BINARY_DIR}/generated/schema.json" "${CMAKE_CURRENT_BINARY_DIR}/generated/authored.rule")
set_target_properties(core PROPERTIES LINKER_LANGUAGE CXX)
target_sources(core PRIVATE "$<$<CONFIG:Release>:conditional.txt>")

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/target-sources")
bicore_record_target_sources(TARGET core OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/target-sources/core.txt")

# Model a source added after declaration observation, like CMake's generated PCH
# header. No compiler is enabled or executed in this fixture.
target_sources(core PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated/implicit-header.h")
