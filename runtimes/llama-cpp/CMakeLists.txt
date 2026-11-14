cmake_minimum_required(VERSION 3.24)
project(llama-cpp-browser-core LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(LCB_LLAMA_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama.cpp" CACHE PATH "Pinned llama.cpp checkout")
option(LCB_NATIVE_CHECK "Build a native shared library for boundary tests, not a distribution runtime" OFF)
option(LCB_MEMORY64 "Use 64-bit WebAssembly memory" OFF)
# Explicit lazy_mode=ON opts Qwen4Exp into bounded file-backed PLE rows.
option(LCB_FILE_BACKED_PLE "Build the reviewed Qwen4Exp file-backed rows extension" ${LCB_MEMORY64})
option(LCB_QSA_ALL_POOLS "Skip QSA ranking when every pool is selected" ON)
option(LCB_WEBGPU "Link the WebGPU backend" OFF)
# Keep distribution defaults until retained BF16 passes target-model validation.
# See docs/webgpu-bf16-projector.md for memory/precision costs and removal criteria.
option(LCB_WEBGPU_BF16_PROJECTOR "Enable reviewed vision BF16 loader handling" ON)
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
set(LCB_CHECKOUT_ROOT "" CACHE PATH "Checkout that owns the running Node tooling")
set(LCB_COMPILE_ROOT "" CACHE PATH "Snapshot of owned CMake and native compiler inputs")
set(LCB_TOOLING_ROOT "" CACHE PATH "Compiled TypeScript tooling owned by the checkout")
set(LCB_NODE "" CACHE FILEPATH "Node executable for project-owned tooling")
set(LCB_PREPARED_SOURCE "" CACHE PATH "Already prepared coherent llama.cpp tree")
foreach(LCB_REQUIRED_INPUT LCB_CHECKOUT_ROOT LCB_COMPILE_ROOT LCB_TOOLING_ROOT LCB_NODE)
    if(NOT ${LCB_REQUIRED_INPUT})
        message(FATAL_ERROR "${LCB_REQUIRED_INPUT} must be supplied explicitly")
    endif()
endforeach()
if(NOT EXISTS "${LCB_TOOLING_ROOT}/runtimes/llama-cpp/tools/cli.js")
    message(FATAL_ERROR "Compile the selected runtime tooling before configuring CMake")
endif()
set(LCB_GENERATED "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${LCB_GENERATED}")
if(EMSCRIPTEN)
    set(LCB_PREPARATION_TARGET wasm)
else()
    set(LCB_PREPARATION_TARGET native)
endif()
if(NOT LCB_PREPARED_SOURCE)
    set(LCB_PREPARED_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/prepared")
    execute_process(COMMAND "${LCB_NODE}" "${LCB_TOOLING_ROOT}/runtimes/llama-cpp/tools/cli.js" prepare
        --compile-root "${LCB_COMPILE_ROOT}" --target "${LCB_PREPARATION_TARGET}" --upstream "${LCB_LLAMA_SOURCE}" --output "${LCB_PREPARED_OUTPUT}"
        --webgpu "${LCB_WEBGPU}" --gpu-profile "${GGML_WEBGPU_GPU_PROFILE}"
        --file-rows "${LCB_FILE_BACKED_PLE}" --qsa "${LCB_QSA_ALL_POOLS}"
        --bf16-projector "${LCB_WEBGPU_BF16_PROJECTOR}" --python "${Python3_EXECUTABLE}"
        WORKING_DIRECTORY "${LCB_CHECKOUT_ROOT}" COMMAND_ERROR_IS_FATAL ANY)
    set(LCB_PREPARED_SOURCE "${LCB_PREPARED_OUTPUT}/llama.cpp")
endif()
get_filename_component(LCB_PREPARED_PARENT "${LCB_PREPARED_SOURCE}" DIRECTORY)
set(LCB_PREPARATION_RECEIPT "${LCB_PREPARED_PARENT}/source-preparation.json")
if(NOT EXISTS "${LCB_PREPARATION_RECEIPT}")
    message(FATAL_ERROR "Prepared source requires its source-preparation.json receipt")
endif()
file(READ "${LCB_PREPARATION_RECEIPT}" LCB_PREPARATION_RECORD)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${LCB_PREPARATION_RECEIPT}")
foreach(LCB_PREPARATION_PAIR "webgpu;LCB_WEBGPU" "gpuProfile;GGML_WEBGPU_GPU_PROFILE"
        "fileRows;LCB_FILE_BACKED_PLE" "qsa;LCB_QSA_ALL_POOLS")
    list(GET LCB_PREPARATION_PAIR 0 LCB_PREPARATION_KEY)
    list(GET LCB_PREPARATION_PAIR 1 LCB_PREPARATION_FLAG)
    string(JSON LCB_PREPARATION_VALUE GET "${LCB_PREPARATION_RECORD}" options "${LCB_PREPARATION_KEY}")
    if((LCB_PREPARATION_VALUE AND NOT ${LCB_PREPARATION_FLAG}) OR
       (NOT LCB_PREPARATION_VALUE AND ${LCB_PREPARATION_FLAG}))
        message(FATAL_ERROR "Prepared source mode disagrees with ${LCB_PREPARATION_FLAG}")
    endif()
endforeach()
string(JSON LCB_PREPARED_TARGET GET "${LCB_PREPARATION_RECORD}" options target)
if(NOT LCB_PREPARED_TARGET STREQUAL LCB_PREPARATION_TARGET)
    message(FATAL_ERROR "Prepared wasm/native source selection disagrees with the target")
endif()
string(JSON LCB_PREPARATION_MTMD GET "${LCB_PREPARATION_RECORD}" options mtmdMode)
if(LCB_WEBGPU AND LCB_WEBGPU_BF16_PROJECTOR)
    set(LCB_EXPECTED_MTMD webgpu)
else()
    set(LCB_EXPECTED_MTMD cpu)
endif()
if(NOT LCB_PREPARATION_MTMD STREQUAL LCB_EXPECTED_MTMD)
    message(FATAL_ERROR "Prepared mtmd source mode disagrees with the target")
endif()
set(LCB_LLAMA_SOURCE "${LCB_PREPARED_SOURCE}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/prepared-source-root.txt" "${LCB_LLAMA_SOURCE}\n")
# Current inputs and transparent patches determine regeneration. Fixed original,
# intermediate and final source digests are not compatibility gates.
file(GLOB_RECURSE LCB_PREPARATION_DEPENDENCIES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/native/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/upstream-patches-only-as-a-last-resort-with-explicit-user-approval/*.patch"
    "${LCB_COMPILE_ROOT}/tools/abi/*.cmake"
    "${LCB_COMPILE_ROOT}/tools/abi/*.in")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${LCB_PREPARATION_DEPENDENCIES})

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

if(LCB_WEBGPU)
    include(cmake/WebgpuSourceOverlay.cmake)
endif()

if(LCB_FILE_BACKED_PLE)
    include(cmake/FileBackedRowsOverlay.cmake)
endif()

if(LCB_QSA_ALL_POOLS)
    include(cmake/QsaSelectionOverlay.cmake)
endif()

# All browser profiles are single-threaded, including CPU and WebGPU.
# The overlay retains upstream behavior in native/pthread builds.
include(cmake/MtmdOverlay.cmake)

if(EMSCRIPTEN AND LCB_ASYNCIFY)
    # The pinned backend misspells -fexceptions as -exceptions in its link interface.
    # Correct the integration here without changing the pinned upstream checkout.
    get_target_property(LCB_WEBGPU_LINK_OPTIONS ggml-webgpu INTERFACE_LINK_OPTIONS)
    list(REMOVE_ITEM LCB_WEBGPU_LINK_OPTIONS "-exceptions")
    set_property(TARGET ggml-webgpu PROPERTY INTERFACE_LINK_OPTIONS "${LCB_WEBGPU_LINK_OPTIONS}")
endif()

# The selected target's generated ABI owns both complete link export lists.
set(LCB_LINK_EXPORTS "${LCB_GENERATED}/exports.json")
set(LCB_LINK_JSPI_EXPORTS "${LCB_GENERATED}/jspi-exports.json")
if(GGML_WEBGPU_GPU_PROFILE)
    if(EMSCRIPTEN AND NOT LCB_JSPI)
        message(FATAL_ERROR "The Q8 diagnostic bridge requires JSPI in WebAssembly builds")
    endif()
    if(NOT LCB_WEBGPU)
        message(FATAL_ERROR "Q8 diagnostics require the WebGPU GPU-profile build")
    endif()
else()
    # A cache switched back to normal must not package stale diagnostic assets.
    file(REMOVE "${CMAKE_CURRENT_BINARY_DIR}/runtime/q8-diagnostic-v1.schema.json"
                "${CMAKE_CURRENT_BINARY_DIR}/runtime/q8-diagnostic-v1.functions.d.ts")
endif()

if(LCB_NATIVE_CHECK)
    add_library(core SHARED native/abi/bridge.cpp native/abi/metadata.cpp)
    if(UNIX AND NOT APPLE)
        target_link_options(core PRIVATE -Wl,--no-undefined)
    endif()
else()
    add_executable(core native/abi/bridge.cpp native/abi/metadata.cpp native/chat/chat-embind.cpp)
    set_target_properties(core PROPERTIES SUFFIX ".mjs" RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/runtime")
    target_link_options(core PRIVATE
        --bind --no-entry -sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORT_NAME=createLlamaCppCore
        --emit-tsd "${CMAKE_CURRENT_BINARY_DIR}/runtime/core.d.ts"
        "-sENVIRONMENT=${LCB_ENVIRONMENT}" -sALLOW_MEMORY_GROWTH=1
        "-sMAXIMUM_MEMORY=${LCB_MAXIMUM_MEMORY}" -sINITIAL_HEAP=33554432
        -sSTACK_SIZE=2097152 -sWASM_BIGINT=1 -sALLOW_TABLE_GROWTH=1
        -sFORCE_FILESYSTEM=1 "-sASSERTIONS=${LCB_ASSERTIONS}" -sEXIT_RUNTIME=0
        "-sEXPORTED_FUNCTIONS=@${LCB_LINK_EXPORTS}"
        "-sEXPORTED_RUNTIME_METHODS=['HEAPU8','FS','WORKERFS','addFunction','removeFunction','UTF8ToString','stringToUTF8','lengthBytesUTF8','ccall','cwrap']"
        "-sINCOMING_MODULE_JS_API=['locateFile','wasmBinary','wasmMemory','instantiateWasm','print','printErr','onAbort','preRun','postRun','noInitialRun','mainScriptUrlOrBlob']"
        -lworkerfs.js)
    if(LCB_WEBGPU AND LCB_VARIANT STREQUAL "browser")
        # A local navigator would hide the earlier browser-version assertion guard.
        if(NOT LCB_ASSERTIONS EQUAL 0)
            message(FATAL_ERROR "Review host navigator scope before enabling browser assertions")
        endif()
        set(LCB_HOST_NAVIGATOR_PRE_JS "${CMAKE_CURRENT_SOURCE_DIR}/host/emscripten/host-navigator-pre.js")
        target_link_options(core PRIVATE --pre-js "${LCB_HOST_NAVIGATOR_PRE_JS}")
        set_property(TARGET core APPEND PROPERTY LINK_DEPENDS "${LCB_HOST_NAVIGATOR_PRE_JS}")
    endif()
    if(LCB_JSPI)
        target_link_options(core PRIVATE -sJSPI=1 "-sJSPI_EXPORTS=@${LCB_LINK_JSPI_EXPORTS}")
    elseif(LCB_ASYNCIFY)
        # Keep indirect-call instrumentation: backend dispatch can suspend through
        # function pointers. Leave async import discovery to the linked port/runtime.
        target_link_options(core PRIVATE -sASYNCIFY=1 -sASYNCIFY_STACK_SIZE=2097152)
    endif()
endif()
target_include_directories(core PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/native/vision" "${CMAKE_CURRENT_SOURCE_DIR}/native/io" "${LCB_LLAMA_SOURCE}"
    "${LCB_LLAMA_SOURCE}/include" "${LCB_LLAMA_SOURCE}/ggml/include")
target_link_libraries(core PRIVATE llama-common mtmd llama)

if(LCB_FILE_BACKED_PLE)
    target_compile_definitions(core PRIVATE LCB_FILE_BACKED_PLE)
endif()

if(LCB_WEBGPU AND GGML_WEBGPU_GPU_PROFILE)
    target_sources(core PRIVATE native/diagnostics/webgpu-q8-diagnostic.cpp)
    target_compile_definitions(core PRIVATE LCB_WEBGPU_Q8_DIAGNOSTIC=1)
    target_include_directories(core PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/native/diagnostics")
endif()

# The added header is consumed only by the changed chat TU and leaf bindings.
target_include_directories(core PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/native/model-graph")
target_sources(core PRIVATE native/model-graph/model-graph.cpp)

if(LCB_NATIVE_CHECK)
    add_executable(model-graph-boundary-test "${LCB_COMPILE_ROOT}/tests/runtimes/llama_cpp/model-graph/boundary.cpp")
    target_include_directories(model-graph-boundary-test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${LCB_LLAMA_SOURCE}"
        "${LCB_LLAMA_SOURCE}/include" "${LCB_LLAMA_SOURCE}/ggml/include" "${LCB_LLAMA_SOURCE}/common")
    target_link_libraries(model-graph-boundary-test PRIVATE core)
endif()

# ABI generation uses the actual core target compile command after generation.
include("${LCB_COMPILE_ROOT}/tools/abi/Bindings.cmake")
bicore_add_abi(TARGET core RUNTIME llama-cpp CHECKOUT_ROOT "${LCB_CHECKOUT_ROOT}" COMPILE_ROOT "${LCB_COMPILE_ROOT}"
    TOOLING_ROOT "${LCB_TOOLING_ROOT}" NODE "${LCB_NODE}" OUTPUT "${LCB_GENERATED}"
    LLAMA_SOURCE "${LCB_LLAMA_SOURCE}")

# CMake owns the actual target source sets, including all mtmd model builders.
# Consumers record this generated list rather than parsing CMake source text.
include("${LCB_COMPILE_ROOT}/tools/shared/TargetSources.cmake")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/target-sources")
foreach(LCB_INPUT_TARGET core llama llama-common mtmd ggml-base ggml-cpu ggml-webgpu)
    if(TARGET ${LCB_INPUT_TARGET})
        bicore_record_target_sources(TARGET ${LCB_INPUT_TARGET}
            OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/target-sources/${LCB_INPUT_TARGET}.txt")
    endif()
endforeach()
