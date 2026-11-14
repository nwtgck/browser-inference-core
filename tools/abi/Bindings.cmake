include_guard(GLOBAL)

# The ordinary core sources are the authority for target flags and public types.
# Analysis runs after CMake has emitted their real compilation database entries.
function(bicore_add_abi)
    cmake_parse_arguments(ABI "" "TARGET;RUNTIME;CHECKOUT_ROOT;COMPILE_ROOT;TOOLING_ROOT;NODE;OUTPUT;LLAMA_SOURCE;SD_SOURCE;GGML_SOURCE" "" ${ARGN})
    foreach(name TARGET RUNTIME CHECKOUT_ROOT COMPILE_ROOT TOOLING_ROOT NODE OUTPUT)
        if(NOT ABI_${name})
            message(FATAL_ERROR "bicore_add_abi requires ${name}")
        endif()
    endforeach()
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON PARENT_SCOPE)
    set_property(TARGET "${ABI_TARGET}" PROPERTY EXPORT_COMPILE_COMMANDS ON)
    set(abi_sources "${ABI_COMPILE_ROOT}/runtimes/${ABI_RUNTIME}/native/abi")
    set(entry "${ABI_TOOLING_ROOT}/runtimes/${ABI_RUNTIME}/tools/bindings.js")
    set(metadata "${abi_sources}/metadata.cpp")
    set(bridge "${abi_sources}/bridge.cpp")
    set(outputs schema.json schema.mjs functions.d.ts functions.mjs exports.json jspi-exports.json abi-fingerprint.json)
    if(ABI_RUNTIME STREQUAL "stable-diffusion-cpp")
        list(APPEND outputs core.d.ts module-options.d.ts)
    else()
        list(APPEND outputs extensions.d.ts)
    endif()
    list(TRANSFORM outputs PREPEND "${ABI_OUTPUT}/")
    file(GLOB_RECURSE tool_inputs CONFIGURE_DEPENDS "${ABI_CHECKOUT_ROOT}/tools/abi/*.ts" "${ABI_CHECKOUT_ROOT}/tools/shared/*.ts" "${ABI_CHECKOUT_ROOT}/00-dto/shared/*.ts")
    file(GLOB native_inputs CONFIGURE_DEPENDS "${abi_sources}/*.cpp" "${abi_sources}/*.h" "${abi_sources}/*.hpp")
    set(header_inputs)
    foreach(root "${ABI_LLAMA_SOURCE}/include" "${ABI_LLAMA_SOURCE}/ggml/include" "${ABI_LLAMA_SOURCE}/tools/mtmd" "${ABI_SD_SOURCE}/include" "${ABI_GGML_SOURCE}/include")
        if(IS_DIRECTORY "${root}")
            file(GLOB headers CONFIGURE_DEPENDS "${root}/*.h" "${root}/*.hpp")
            list(APPEND header_inputs ${headers})
        endif()
    endforeach()
    if(ABI_RUNTIME STREQUAL "llama-cpp")
        file(GLOB owned_headers CONFIGURE_DEPENDS "${ABI_COMPILE_ROOT}/runtimes/llama-cpp/native/vision/*.h" "${ABI_COMPILE_ROOT}/runtimes/llama-cpp/native/model-graph/*.h" "${ABI_COMPILE_ROOT}/runtimes/llama-cpp/native/diagnostics/*.h")
        list(APPEND header_inputs ${owned_headers})
    elseif(ABI_RUNTIME STREQUAL "stable-diffusion-cpp")
        list(APPEND header_inputs "${ABI_COMPILE_ROOT}/tests/runtimes/stable-diffusion-cpp/native/browser-probes.h")
        list(APPEND header_inputs "${ABI_COMPILE_ROOT}/runtimes/stable-diffusion-cpp/bindings/module-options.d.ts")
    endif()
    set(runtime_outputs "${CMAKE_CURRENT_BINARY_DIR}/runtime/abi.json")
    if(ABI_RUNTIME STREQUAL "stable-diffusion-cpp")
        list(APPEND runtime_outputs "${CMAKE_CURRENT_BINARY_DIR}/runtime/core.d.ts" "${CMAKE_CURRENT_BINARY_DIR}/runtime/module-options.d.ts")
    else()
        list(APPEND runtime_outputs "${CMAKE_CURRENT_BINARY_DIR}/runtime/extensions.d.ts")
        list(APPEND runtime_outputs "${CMAKE_CURRENT_BINARY_DIR}/runtime/model-graph-v1.schema.json" "${CMAKE_CURRENT_BINARY_DIR}/runtime/model-graph-v1.functions.d.ts")
        if(GGML_WEBGPU_GPU_PROFILE)
            list(APPEND runtime_outputs "${CMAKE_CURRENT_BINARY_DIR}/runtime/q8-diagnostic-v1.schema.json" "${CMAKE_CURRENT_BINARY_DIR}/runtime/q8-diagnostic-v1.functions.d.ts")
        endif()
    endif()
    add_custom_command(OUTPUT ${outputs} "${ABI_OUTPUT}/fingerprint.h" ${runtime_outputs}
        COMMAND "${ABI_NODE}" "${entry}"
            --compile-root "${ABI_COMPILE_ROOT}"
            --compilation-database "${CMAKE_BINARY_DIR}/compile_commands.json"
            --translation-unit "${metadata}" --bridge-unit "${bridge}"
            --output "${ABI_OUTPUT}" --runtime-directory "${CMAKE_CURRENT_BINARY_DIR}/runtime"
        COMMAND "${CMAKE_COMMAND}"
            "-DBICORE_FINGERPRINT_JSON=${ABI_OUTPUT}/abi-fingerprint.json"
            "-DBICORE_FINGERPRINT_TEMPLATE=${ABI_COMPILE_ROOT}/tools/abi/fingerprint.h.in"
            "-DBICORE_FINGERPRINT_HEADER=${ABI_OUTPUT}/fingerprint.h"
            -P "${ABI_COMPILE_ROOT}/tools/abi/ConfigureFingerprint.cmake"
        DEPENDS "${entry}" "${ABI_CHECKOUT_ROOT}/runtimes/${ABI_RUNTIME}/tools/bindings.ts"
            ${tool_inputs} ${native_inputs} ${header_inputs} "${CMAKE_BINARY_DIR}/compile_commands.json"
            "${ABI_COMPILE_ROOT}/tools/abi/fingerprint.h.in" "${ABI_COMPILE_ROOT}/tools/abi/ConfigureFingerprint.cmake"
        VERBATIM)
    add_custom_target("${ABI_TARGET}-abi" DEPENDS ${outputs} "${ABI_OUTPUT}/fingerprint.h" ${runtime_outputs})
    add_dependencies("${ABI_TARGET}" "${ABI_TARGET}-abi")
    target_include_directories("${ABI_TARGET}" PRIVATE "${ABI_OUTPUT}")
    set_property(SOURCE "${metadata}" APPEND PROPERTY OBJECT_DEPENDS "${ABI_OUTPUT}/fingerprint.h")
    set_property(TARGET "${ABI_TARGET}" APPEND PROPERTY LINK_DEPENDS ${outputs})
endfunction()
