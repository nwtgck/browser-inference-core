cmake_minimum_required(VERSION 3.24)
project(llama_cpu_contracts LANGUAGES C CXX)
enable_testing()
set(CMAKE_CXX_STANDARD 20)
include("../support/ggml.cmake")
option(LCB_NATIVE_CHECKPOINT_WRITER "Validate the wasm-selected context's ordinary host writer" OFF)
if(NOT DEFINED LCB_NATIVE_MTMD_CAPABILITY)
    set(LCB_NATIVE_MTMD_CAPABILITY 0)
endif()
if(NOT LCB_NATIVE_MTMD_CAPABILITY MATCHES "^[01]$")
    message(FATAL_ERROR "LCB_NATIVE_MTMD_CAPABILITY must match the prepared mtmd mode")
endif()

function(mtmd_includes target)
    target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_LIST_DIR}/.."
        "${LCB_COMPILE_ROOT}/runtimes/llama-cpp/native/vision"
        "${LCB_PREPARED_SOURCE}/tools/mtmd" "${LCB_PREPARED_SOURCE}/include"
        "${LCB_PREPARED_SOURCE}/ggml/src")
    target_compile_options(${target} PRIVATE -ffunction-sections -fdata-sections)
    target_compile_definitions(${target} PRIVATE LCB_MTMD_WEBGPU_BF16_RUNTIME=${LCB_NATIVE_MTMD_CAPABILITY})
    target_link_libraries(${target} PRIVATE ggml ggml-cpu)
endfunction()
function(mtmd_contract target)
    add_executable(${target} ${ARGN})
    mtmd_includes(${target})
    target_link_options(${target} PRIVATE -Wl,--gc-sections)
endfunction()
if(LCB_NATIVE_CHECKPOINT_WRITER)
    mtmd_contract(checkpoint-writer "../checkpoint-writer.cpp")
    target_include_directories(checkpoint-writer PRIVATE "${LCB_PREPARED_SOURCE}/src")
    add_test(NAME checkpoint-writer COMMAND checkpoint-writer)
endif()

# Compile each complete production TU once, then link only each fixture's
# required units. No source fragments or alternative private class layouts.
foreach(unit mtmd mtmd-model mtmd-subcontext mtmd-image mtmd-audio)
    add_library("native-${unit}" OBJECT "${LCB_PREPARED_SOURCE}/tools/mtmd/${unit}.cpp")
    mtmd_includes("native-${unit}")
endforeach()

mtmd_contract(mtmd-bf16 "../mtmd-bf16.cpp")
add_test(NAME mtmd-bf16 COMMAND mtmd-bf16)
mtmd_contract(mtmd-retained "../mtmd-retained.cpp" "../support/mtmd-logger.cpp")
add_test(NAME mtmd-retained COMMAND mtmd-retained)
add_test(NAME mtmd-retained-roles COMMAND mtmd-retained roles)
mtmd_contract(mtmd-borrowed "../mtmd-borrowed.cpp" "../support/allocation-fault.cpp"
    $<TARGET_OBJECTS:native-mtmd> $<TARGET_OBJECTS:native-mtmd-model> $<TARGET_OBJECTS:native-mtmd-subcontext>)
add_test(NAME mtmd-borrowed COMMAND mtmd-borrowed)

mtmd_contract(mtmd-entry "../mtmd-entry.cpp"
    $<TARGET_OBJECTS:native-mtmd> $<TARGET_OBJECTS:native-mtmd-model> $<TARGET_OBJECTS:native-mtmd-subcontext>
    $<TARGET_OBJECTS:native-mtmd-audio> $<TARGET_OBJECTS:native-mtmd-image>)
target_link_options(mtmd-entry PRIVATE -Wl,--wrap=gguf_init_from_file)
add_test(NAME mtmd-entry COMMAND mtmd-entry "${CMAKE_CURRENT_BINARY_DIR}/entry-gguf")
mtmd_contract(mtmd-loader "../mtmd-loader.cpp"
    $<TARGET_OBJECTS:native-mtmd-model> $<TARGET_OBJECTS:native-mtmd>)
add_test(NAME mtmd-loader COMMAND mtmd-loader "${CMAKE_CURRENT_BINARY_DIR}/metadata-gguf")

# Compile the entire real model/graph units. Linker section collection removes
# unused model dispatchers; no graph-base or embedding body is extracted.
mtmd_contract(mtmd-model-graphs "../mtmd-graphs.cpp" "../support/mtmd-logger.cpp"
    "${LCB_PREPARED_SOURCE}/tools/mtmd/mtmd-graph.cpp"
    $<TARGET_OBJECTS:native-mtmd-model>)
foreach(model llava qwen2vl qwen3vl siglip gemma4v gemma4uv lfm3vl)
    target_sources(mtmd-model-graphs PRIVATE "${LCB_PREPARED_SOURCE}/tools/mtmd/models/${model}.cpp")
endforeach()
foreach(model llava qwen2vl qwen3vl gemma3 lfm2 gemma4v gemma4uv lfm3vl)
    foreach(side 32 48)
        add_test(NAME "retained-model-${model}-${side}" COMMAND mtmd-model-graphs "${model}" "${side}")
    endforeach()
endforeach()
foreach(shape 480x480 768x768 480x768 768x480 480x336 1536x1680)
    string(REPLACE "x" ";" dims "${shape}")
    add_test(NAME "retained-model-gemma4uv-12b-${shape}" COMMAND mtmd-model-graphs gemma4uv-12b ${dims})
endforeach()
