cmake_minimum_required(VERSION 3.24)
project(llama_native_contracts LANGUAGES C CXX)

enable_testing()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

foreach(required LCB_COMPILE_ROOT LCB_PREPARED_SOURCE LCB_DAWN_PACKAGE)
    if(NOT IS_DIRECTORY "${${required}}")
        message(FATAL_ERROR "${required} must select an existing source input")
    endif()
endforeach()
option(LCB_NATIVE_GPU_PROFILE "Compile the current diagnostic provider" OFF)
option(LCB_NATIVE_CPU_CONTRACTS "Run CPU-backed retained BF16 and model graph cases" ON)
include(support/ggml.cmake)

set(webgpu "${LCB_PREPARED_SOURCE}/ggml/src/ggml-webgpu")
function(native_contract target)
    add_executable(${target} ${ARGN} support/allocation-fault.cpp support/native-abort.cpp)
    target_compile_definitions(${target} PRIVATE __wasi__ GGML_SCHED_MAX_COPIES=4 GGML_WEBGPU_BATCH_PARAM_UPLOADS BICORE_REAL_GGML)
    if(LCB_NATIVE_GPU_PROFILE)
        target_compile_definitions(${target} PRIVATE GGML_WEBGPU_GPU_PROFILE GGML_WEBGPU_CPU_PROFILE)
    endif()
    target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/support" "${webgpu}")
    target_include_directories(${target} SYSTEM PRIVATE
        "${LCB_PREPARED_SOURCE}/ggml/include" "${LCB_PREPARED_SOURCE}/ggml/src"
        "${LCB_DAWN_PACKAGE}/webgpu/include" "${LCB_DAWN_PACKAGE}/webgpu_cpp/include")
    target_compile_options(${target} PRIVATE -ffunction-sections -fdata-sections)
    target_link_options(${target} PRIVATE -Wl,--gc-sections -Wl,--wrap=ggml_abort)
    target_link_libraries(${target} PRIVATE ggml-base)
    add_test(NAME ${target} COMMAND ${target} "${CMAKE_CURRENT_BINARY_DIR}/${target}-shaders")
endfunction()

native_contract(webgpu-provider provider.cpp)
native_contract(webgpu-transfer transfer.cpp)
if(LCB_NATIVE_GPU_PROFILE)
    target_sources(webgpu-transfer PRIVATE
        "${LCB_COMPILE_ROOT}/runtimes/llama-cpp/native/diagnostics/webgpu-q8-diagnostic.cpp")
    target_compile_definitions(webgpu-transfer PRIVATE LCB_WEBGPU_Q8_DIAGNOSTIC=1)
endif()
native_contract(webgpu-readback readback.cpp)
target_compile_definitions(webgpu-readback PRIVATE __EMSCRIPTEN__)
if(NOT LCB_NATIVE_GPU_PROFILE)
    native_contract(hc-admission hc-admission.cpp)
endif()

if(LCB_NATIVE_CPU_CONTRACTS)
    # The real private class and complete prepared TU have one declaration universe.
    # Only external ggml backend/scheduler calls are replaced by the owned fixture.
    add_executable(mtmd-ownership mtmd-ownership.cpp support/allocation-fault.cpp
        "${LCB_PREPARED_SOURCE}/tools/mtmd/mtmd-subcontext.cpp")
    target_include_directories(mtmd-ownership PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}"
        "${LCB_PREPARED_SOURCE}/tools/mtmd" "${LCB_PREPARED_SOURCE}/ggml/include"
        "${LCB_PREPARED_SOURCE}/ggml/src" "${LCB_PREPARED_SOURCE}/include"
        "${LCB_COMPILE_ROOT}/runtimes/llama-cpp/native/vision")
    target_compile_definitions(mtmd-ownership PRIVATE LCB_MTMD_WEBGPU_BF16_RUNTIME=0)
    target_compile_options(mtmd-ownership PRIVATE -ffunction-sections -fdata-sections)
    target_link_options(mtmd-ownership PRIVATE -Wl,--gc-sections)
    add_test(NAME mtmd-ownership COMMAND mtmd-ownership)
    add_executable(mtmd-upload mtmd-upload.cpp)
    target_include_directories(mtmd-upload PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}"
        "${LCB_PREPARED_SOURCE}/tools/mtmd" "${LCB_PREPARED_SOURCE}/ggml/include"
        "${LCB_PREPARED_SOURCE}/ggml/src" "${LCB_PREPARED_SOURCE}/include"
        "${LCB_COMPILE_ROOT}/runtimes/llama-cpp/native/vision")
    add_test(NAME mtmd-upload COMMAND mtmd-upload)
    add_subdirectory(cpu)
endif()
