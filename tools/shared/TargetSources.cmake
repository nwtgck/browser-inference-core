include_guard(GLOBAL)

# Call after project-owned source declarations are complete. The configure-time
# property distinguishes declared inputs from generation-time additions such as
# real PCH headers and internal rule bookkeeping. Keep both observations so the
# recorder never drops a declared input merely because it has a .rule suffix.
function(bicore_record_target_sources)
    cmake_parse_arguments(SOURCE_FACTS "" "TARGET;OUTPUT" "" ${ARGN})
    if(NOT DEFINED SOURCE_FACTS_TARGET OR NOT DEFINED SOURCE_FACTS_OUTPUT)
        message(FATAL_ERROR "bicore_record_target_sources requires TARGET and OUTPUT")
    endif()
    get_property(declared_sources TARGET "${SOURCE_FACTS_TARGET}" PROPERTY SOURCES)
    set_property(TARGET "${SOURCE_FACTS_TARGET}" PROPERTY BICORE_DECLARED_INPUT_SOURCES "${declared_sources}")
    file(GENERATE OUTPUT "${SOURCE_FACTS_OUTPUT}"
        CONTENT "$<TARGET_PROPERTY:${SOURCE_FACTS_TARGET},SOURCE_DIR>\n$<JOIN:$<TARGET_PROPERTY:${SOURCE_FACTS_TARGET},SOURCES>,\n>\n")
    file(GENERATE OUTPUT "${SOURCE_FACTS_OUTPUT}.declared"
        CONTENT "$<TARGET_PROPERTY:${SOURCE_FACTS_TARGET},SOURCE_DIR>\n$<JOIN:$<TARGET_GENEX_EVAL:${SOURCE_FACTS_TARGET},$<TARGET_PROPERTY:${SOURCE_FACTS_TARGET},BICORE_DECLARED_INPUT_SOURCES>>,\n>\n")
endfunction()
