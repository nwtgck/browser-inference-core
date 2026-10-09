cmake_minimum_required(VERSION 3.24)
project(llama-cpp-browser-core LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(LCB_LLAMA_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama.cpp" CACHE PATH "Pinned llama.cpp checkout")
option(LCB_NATIVE_CHECK "Build a native shared library for boundary tests, not a distribution runtime" OFF)
option(LCB_MEMORY64 "Use 64-bit WebAssembly memory" OFF)
option(LCB_WEBGPU "Link the WebGPU backend" OFF)
# Experimental downstream shader optimization; see docs/moe-direct-slot.md.
option(LCB_WEBGPU_MOE_DIRECT_SLOT "Use the reviewed direct selected-slot MoE shader" OFF)
# Compatibility workaround for the pinned backend, not a new BF16 kernel.
# See docs/webgpu-bf16-projector.md for memory/precision costs and removal criteria.
option(LCB_WEBGPU_BF16_PROJECTOR "Expand vision BF16 weights to F32 for the pinned WebGPU backend" ON)
option(LCB_JSPI "Suspend WebGPU calls with JSPI" ${LCB_WEBGPU})
option(LCB_ASYNCIFY "Suspend WebGPU calls with Asyncify" OFF)
set(LCB_MAXIMUM_MEMORY "4294967296" CACHE STRING "Maximum WebAssembly linear memory in bytes")
set(LCB_VARIANT "browser" CACHE STRING "Runtime variant: browser or test")
set_property(CACHE LCB_VARIANT PROPERTY STRINGS browser test)
if(NOT LCB_VARIANT STREQUAL "browser" AND NOT LCB_VARIANT STREQUAL "test")
    message(FATAL_ERROR "Unknown runtime variant: ${LCB_VARIANT}")
endif()
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/config/variants.json" LCB_VARIANTS)
string(JSON LCB_ASSERTIONS GET "${LCB_VARIANTS}" "${LCB_VARIANT}" assertions)
string(JSON LCB_ENVIRONMENT GET "${LCB_VARIANTS}" "${LCB_VARIANT}" environment)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/config/variants.json")

if(LCB_WEBGPU_MOE_DIRECT_SLOT AND NOT LCB_WEBGPU)
    message(FATAL_ERROR "MoE direct-slot requires WebGPU")
endif()

if(LCB_JSPI AND LCB_ASYNCIFY)
    message(FATAL_ERROR "JSPI and Asyncify are mutually exclusive")
endif()
if(LCB_WEBGPU AND NOT LCB_JSPI AND NOT LCB_ASYNCIFY)
    message(FATAL_ERROR "WebGPU requires JSPI or Asyncify")
endif()
if(LCB_ASYNCIFY AND (NOT LCB_WEBGPU OR LCB_MEMORY64))
    message(FATAL_ERROR "The Asyncify profile requires WebGPU and 32-bit memory")
endif()

if(NOT EXISTS "${LCB_LLAMA_SOURCE}/include/llama.h")
    message(FATAL_ERROR "Missing submodule. Run: git submodule update --init --recursive")
endif()
if(NOT EMSCRIPTEN AND NOT LCB_NATIVE_CHECK)
    message(FATAL_ERROR "Use emcmake, or explicitly enable LCB_NATIVE_CHECK for native tests")
endif()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_program(LCB_CLANG NAMES clang REQUIRED)
set(LCB_GENERATED "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${LCB_GENERATED}")
execute_process(COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_bindings.py"
    --source "${LCB_LLAMA_SOURCE}" --output "${LCB_GENERATED}" --clang "${LCB_CLANG}"
    COMMAND_ERROR_IS_FATAL ANY)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_bindings.py"
    "${LCB_LLAMA_SOURCE}/include/llama.h"
    "${LCB_LLAMA_SOURCE}/ggml/include/ggml.h"
    "${LCB_LLAMA_SOURCE}/ggml/include/ggml-alloc.h"
    "${LCB_LLAMA_SOURCE}/ggml/include/ggml-backend.h"
    "${LCB_LLAMA_SOURCE}/ggml/include/ggml-cpu.h"
    "${LCB_LLAMA_SOURCE}/ggml/include/ggml-opt.h"
    "${LCB_LLAMA_SOURCE}/ggml/include/gguf.h"
    "${LCB_LLAMA_SOURCE}/tools/mtmd/mtmd.h"
    "${LCB_LLAMA_SOURCE}/tools/mtmd/mtmd-helper.h")

# Flags affecting the ABI must also apply to the outer bridge target.
if(EMSCRIPTEN)
    add_compile_options(-msimd128)
    if(LCB_ASYNCIFY)
        # Asyncify cannot instrument Wasm exception handling in the pinned toolchain.
        # Use the same JavaScript exception ABI in every library and the bridge.
        add_compile_options(-fexceptions)
        add_link_options(-fexceptions)
    else()
        add_compile_options(-fwasm-exceptions)
        add_link_options(-fwasm-exceptions)
    endif()
    if(LCB_MEMORY64)
        add_compile_options(-sMEMORY64=1)
        add_link_options(-sMEMORY64=1)
    endif()
endif()
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_WASM_MEM64 ${LCB_MEMORY64} CACHE BOOL "" FORCE)
set(LLAMA_BUILD_HTML OFF CACHE BOOL "" FORCE)
foreach(option BUILD_TESTS BUILD_TOOLS BUILD_EXAMPLES BUILD_SERVER BUILD_APP OPENSSL SUBPROCESS LLGUIDANCE)
    set(LLAMA_${option} OFF CACHE BOOL "" FORCE)
endforeach()
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_MTMD ON CACHE BOOL "" FORCE)
set(MTMD_VIDEO OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
set(GGML_WEBGPU ${LCB_WEBGPU} CACHE BOOL "" FORCE)
set(GGML_WEBGPU_JSPI ${LCB_JSPI} CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
add_subdirectory("${LCB_LLAMA_SOURCE}" "llama")

if(LCB_WEBGPU_MOE_DIRECT_SLOT)
    include(cmake/MoeDirectSlotOverlay.cmake)
endif()

if(LCB_WEBGPU)
    include(cmake/WebgpuSourceOverlay.cmake)
endif()

# All browser profiles are single-threaded, including CPU and WebGPU.
# The overlay retains upstream behavior in native/pthread builds.
include(cmake/MtmdAudioOverlay.cmake)

if(LCB_WEBGPU AND LCB_WEBGPU_BF16_PROJECTOR)
    include(cmake/MtmdOverlay.cmake)
endif()

if(EMSCRIPTEN AND LCB_ASYNCIFY)
    # The pinned backend misspells -fexceptions as -exceptions in its link interface.
    # Correct the integration here without changing the pinned upstream checkout.
    get_target_property(LCB_WEBGPU_LINK_OPTIONS ggml-webgpu INTERFACE_LINK_OPTIONS)
    list(REMOVE_ITEM LCB_WEBGPU_LINK_OPTIONS "-exceptions")
    set_property(TARGET ggml-webgpu PROPERTY INTERFACE_LINK_OPTIONS "${LCB_WEBGPU_LINK_OPTIONS}")
endif()

if(LCB_NATIVE_CHECK)
    add_library(core SHARED "${LCB_GENERATED}/bindings.cpp")
    if(UNIX AND NOT APPLE)
        target_link_options(core PRIVATE -Wl,--no-undefined)
    endif()
else()
    add_executable(core "${LCB_GENERATED}/bindings.cpp" bridge/chat-embind.cpp)
    set_target_properties(core PROPERTIES SUFFIX ".mjs" RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/runtime")
    target_link_options(core PRIVATE
        --bind --no-entry -sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORT_NAME=createLlamaCppCore
        --emit-tsd "${CMAKE_CURRENT_BINARY_DIR}/runtime/core.d.ts"
        "-sENVIRONMENT=${LCB_ENVIRONMENT}" -sALLOW_MEMORY_GROWTH=1
        "-sMAXIMUM_MEMORY=${LCB_MAXIMUM_MEMORY}" -sINITIAL_HEAP=33554432
        -sSTACK_SIZE=2097152 -sWASM_BIGINT=1 -sALLOW_TABLE_GROWTH=1
        -sFORCE_FILESYSTEM=1 "-sASSERTIONS=${LCB_ASSERTIONS}" -sEXIT_RUNTIME=0
        "-sEXPORTED_FUNCTIONS=@${LCB_GENERATED}/exports.json"
        "-sEXPORTED_RUNTIME_METHODS=['HEAPU8','FS','WORKERFS','addFunction','removeFunction','UTF8ToString','stringToUTF8','lengthBytesUTF8','ccall','cwrap']"
        "-sINCOMING_MODULE_JS_API=['locateFile','wasmBinary','wasmMemory','instantiateWasm','print','printErr','onAbort','preRun','postRun','noInitialRun','mainScriptUrlOrBlob']"
        -lworkerfs.js)
    if(LCB_JSPI)
        target_link_options(core PRIVATE -sJSPI=1 "-sJSPI_EXPORTS=@${LCB_GENERATED}/jspi-exports.json")
    elseif(LCB_ASYNCIFY)
        # Keep indirect-call instrumentation: backend dispatch can suspend through
        # function pointers. Leave async import discovery to the linked port/runtime.
        target_link_options(core PRIVATE -sASYNCIFY=1 -sASYNCIFY_STACK_SIZE=2097152)
    endif()
endif()
target_include_directories(core PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${LCB_LLAMA_SOURCE}"
    "${LCB_LLAMA_SOURCE}/include" "${LCB_LLAMA_SOURCE}/ggml/include")
target_link_libraries(core PRIVATE llama-common mtmd llama)
