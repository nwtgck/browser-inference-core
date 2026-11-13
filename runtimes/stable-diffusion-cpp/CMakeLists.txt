cmake_minimum_required(VERSION 3.24)
project(stable-diffusion-cpp-browser-core LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(SDCB_SOURCE "" CACHE PATH "Prepared stable-diffusion source (never vendor)")
set(SDCB_GGML_SOURCE "" CACHE PATH "Prepared ggml source (never vendor)")
set(SDCB_PROVENANCE "" CACHE FILEPATH "Checked source preparation receipt")
option(SDCB_NATIVE_CHECK "Native ABI/math tests, not a browser distribution" OFF)
option(SDCB_MEMORY64 "Use 64-bit Wasm addresses; file offsets are always 64-bit" OFF)
set(SDCB_MAXIMUM_MEMORY "4294967296" CACHE STRING "Wasm linear memory ceiling")
option(SDCB_JSPI "Use JSPI rather than Asyncify" ON)
set(SDCB_VARIANT "browser" CACHE STRING "browser or test")
if(NOT SDCB_VARIANT MATCHES "^(browser|test)$")
    message(FATAL_ERROR "Unknown variant")
endif()
if(NOT EXISTS "${SDCB_SOURCE}/include/stable-diffusion.h" OR NOT EXISTS "${SDCB_GGML_SOURCE}/include/ggml.h")
    message(FATAL_ERROR "Prepare pinned upstream sources first")
endif()
if(NOT EMSCRIPTEN AND NOT SDCB_NATIVE_CHECK)
    message(FATAL_ERROR "Use emcmake or SDCB_NATIVE_CHECK")
endif()
# A prepared source tree has no .git directory. Do not let upstream's version
# probes accidentally identify the enclosing browser-inference-core checkout.
if(NOT EXISTS "${SDCB_PROVENANCE}")
    message(FATAL_ERROR "Pass the checked source preparation receipt as SDCB_PROVENANCE")
endif()
file(READ "${SDCB_PROVENANCE}" SDCB_SOURCE_RECORD)
foreach(role stable-diffusion ggml)
    string(JSON identity ERROR_VARIABLE identity_error GET "${SDCB_SOURCE_RECORD}" upstreams "${role}" commit)
    string(LENGTH "${identity}" identity_length)
    if(identity_error OR NOT identity_length EQUAL 40 OR NOT identity MATCHES "^[0-9a-f]+$")
        message(FATAL_ERROR "Invalid upstream identity in source receipt: ${role}")
    endif()
    set("SDCB_COMMIT_${role}" "${identity}")
endforeach()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${SDCB_PROVENANCE}")
get_filename_component(BICORE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(SDCB_TEST_ROOT "${BICORE_ROOT}/tests/runtimes/stable_diffusion_cpp")
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_program(SDCB_CLANG NAMES clang REQUIRED)
set(SDCB_GENERATED "${CMAKE_CURRENT_BINARY_DIR}/generated")
if(SDCB_JSPI)
    set(SDCB_SUSPENSION jspi)
else()
    set(SDCB_SUSPENSION asyncify)
endif()
execute_process(COMMAND "${Python3_EXECUTABLE}" -m build_tools.runtimes.stable_diffusion_cpp.bindings
    --source "${SDCB_SOURCE}" --ggml-source "${SDCB_GGML_SOURCE}" --output "${SDCB_GENERATED}" --clang "${SDCB_CLANG}"
    --suspension "${SDCB_SUSPENSION}"
    WORKING_DIRECTORY "${BICORE_ROOT}" COMMAND_ERROR_IS_FATAL ANY)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${BICORE_ROOT}/build_tools/runtimes/stable_diffusion_cpp/bindings.py"
    "${BICORE_ROOT}/build_tools/bindings.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/bindings/module-options.d.ts"
    "${SDCB_SOURCE}/include/stable-diffusion.h"
    "${SDCB_GGML_SOURCE}/include/ggml.h"
    "${SDCB_GGML_SOURCE}/include/ggml-backend.h"
    "${SDCB_GGML_SOURCE}/include/ggml-cpu.h"
    "${SDCB_GGML_SOURCE}/include/ggml-alloc.h"
    "${SDCB_GGML_SOURCE}/include/gguf.h")
if(SDCB_MEMORY64 AND NOT SDCB_JSPI)
    message(FATAL_ERROR "The memory64 profile requires JSPI")
endif()
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
foreach(flag OPENMP BLAS METAL CUDA VULKAN BACKEND_DL)
    set(GGML_${flag} OFF CACHE BOOL "" FORCE)
endforeach()
set(GGML_WEBGPU ${EMSCRIPTEN} CACHE BOOL "" FORCE)
set(GGML_WEBGPU_JSPI ${SDCB_JSPI} CACHE BOOL "" FORCE)
set(SD_USE_UPSTREAM_GGML ON CACHE BOOL "" FORCE)
set(SD_GGML_SOURCE_DIR "${SDCB_GGML_SOURCE}" CACHE PATH "" FORCE)
foreach(flag BUILD_EXAMPLES WEBP WEBM BUILD_SHARED_LIBS)
    set(SD_${flag} OFF CACHE BOOL "" FORCE)
endforeach()
if(EMSCRIPTEN)
    add_compile_options(-msimd128)
    if(SDCB_MEMORY64)
        add_compile_options(-sMEMORY64=1)
        add_link_options(-sMEMORY64=1)
    endif()
    if(SDCB_JSPI)
        add_compile_options(-fwasm-exceptions)
        add_link_options(-fwasm-exceptions)
    else()
        add_compile_options(-fexceptions)
        add_link_options(-fexceptions)
    endif()
endif()
function(add_prepared_upstreams)
    # Scope these values to the two prepared projects. No fake .git checkout,
    # source rewriting, or global modification of Git discovery is necessary.
    set(GIT_EXE OFF)
    set(GGML_GIT_DIRTY 0)
    set(GGML_BUILD_COMMIT "${SDCB_COMMIT_ggml}-bicore")
    set(SDCPP_BUILD_COMMIT "${SDCB_COMMIT_stable-diffusion}-bicore")
    set(SDCPP_BUILD_VERSION "bicore")
    add_subdirectory("${SDCB_SOURCE}" upstream)
endfunction()
add_prepared_upstreams()
target_compile_definitions(stable-diffusion PRIVATE SD_BROWSER_WEBGPU=1)
if(EMSCRIPTEN)
    # This image-only prepared backend keeps the original queue submission cadence.
    target_compile_definitions(ggml-webgpu PRIVATE GGML_WEBGPU_BATCH_PARAM_UPLOADS=1)
endif()
if(EMSCRIPTEN AND NOT SDCB_JSPI)
    # Fix the selected backend's typo without modifying the vendor checkout.
    get_target_property(opts ggml-webgpu INTERFACE_LINK_OPTIONS)
    list(REMOVE_ITEM opts "-exceptions")
    set_property(TARGET ggml-webgpu PROPERTY INTERFACE_LINK_OPTIONS "${opts}")
endif()
# No model singleton, JSON dispatcher, pipeline policy or browser storage owner.
if(SDCB_NATIVE_CHECK)
    add_library(core SHARED "${SDCB_GENERATED}/bindings.cpp")
    if(UNIX AND NOT APPLE)
        target_link_options(core PRIVATE -Wl,--no-undefined)
    endif()
else()
    add_executable(core "${SDCB_GENERATED}/bindings.cpp")
    set_target_properties(core PROPERTIES SUFFIX ".mjs" RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/runtime")
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/config/variants.json" variants)
    string(JSON assertions GET "${variants}" "${SDCB_VARIANT}" assertions)
    string(JSON environment GET "${variants}" "${SDCB_VARIANT}" environment)
    if(SDCB_VARIANT STREQUAL "test")
        target_sources(core PRIVATE "${SDCB_TEST_ROOT}/wasm-probes.cpp")
        target_sources(core PRIVATE "${SDCB_TEST_ROOT}/qwen-timestep-probe.cpp")
        target_sources(core PRIVATE "${SDCB_TEST_ROOT}/bf16-weights-probe.cpp")
        target_sources(core PRIVATE "${SDCB_TEST_ROOT}/graph-walk-probe.cpp")
        target_sources(core PRIVATE "${SDCB_TEST_ROOT}/conv3d-bias-probe.cpp")
        target_sources(core PRIVATE "${SDCB_TEST_ROOT}/webgpu-performance-probe.cpp")
        target_include_directories(core PRIVATE "${SDCB_SOURCE}/src" "${SDCB_GGML_SOURCE}/include" "${SDCB_GGML_SOURCE}/src")
        target_compile_definitions(core PRIVATE SD_BROWSER_WEBGPU=1 SD_USE_UPSTREAM_GGML GGML_MAX_NAME=160)
        file(READ "${SDCB_GENERATED}/exports.json" SDCB_EXPORTS)
        foreach(probe sdc_test_gguf_offset sdc_test_gguf_value sdc_test_callbacks sdc_test_safetensors_offset sdc_test_safetensors_value sdc_test_model_tensor_count sdc_test_gguf_shape sdc_test_gguf_original_shape_values sdc_test_qwen_timestep sdc_test_bf16_weights sdc_test_graph_walk sdc_test_conv3d_bias sdc_test_webgpu_performance)
            string(JSON length LENGTH "${SDCB_EXPORTS}")
            string(JSON SDCB_EXPORTS SET "${SDCB_EXPORTS}" ${length} "\"_${probe}\"")
        endforeach()
        file(WRITE "${SDCB_GENERATED}/exports.json" "${SDCB_EXPORTS}")
        # The optional WebGPU probe can suspend in either JSPI or Asyncify.
        file(READ "${SDCB_GENERATED}/jspi-exports.json" SDCB_JSPI_EXPORTS)
        string(JSON length LENGTH "${SDCB_JSPI_EXPORTS}")
        string(JSON SDCB_JSPI_EXPORTS SET "${SDCB_JSPI_EXPORTS}" ${length} "\"sdc_test_qwen_timestep\"")
        string(JSON length LENGTH "${SDCB_JSPI_EXPORTS}")
        string(JSON SDCB_JSPI_EXPORTS SET "${SDCB_JSPI_EXPORTS}" ${length} "\"sdc_test_bf16_weights\"")
        string(JSON length LENGTH "${SDCB_JSPI_EXPORTS}")
        string(JSON SDCB_JSPI_EXPORTS SET "${SDCB_JSPI_EXPORTS}" ${length} "\"sdc_test_conv3d_bias\"")
        string(JSON length LENGTH "${SDCB_JSPI_EXPORTS}")
        string(JSON SDCB_JSPI_EXPORTS SET "${SDCB_JSPI_EXPORTS}" ${length} "\"sdc_test_webgpu_performance\"")
        file(WRITE "${SDCB_GENERATED}/jspi-exports.json" "${SDCB_JSPI_EXPORTS}")
    endif()
    set_property(TARGET core APPEND PROPERTY LINK_DEPENDS
        "${SDCB_GENERATED}/exports.json" "${SDCB_GENERATED}/jspi-exports.json")
    target_link_options(core PRIVATE --no-entry -sMODULARIZE=1 -sEXPORT_ES6=1
        -sEXPORT_NAME=createStableDiffusionCore "-sENVIRONMENT=${environment}"
        -sALLOW_MEMORY_GROWTH=1 "-sMAXIMUM_MEMORY=${SDCB_MAXIMUM_MEMORY}" -sINITIAL_HEAP=33554432
        -sSTACK_SIZE=2097152 -sWASM_BIGINT=1 -sEXIT_RUNTIME=0 -sFORCE_FILESYSTEM=1
        -sALLOW_TABLE_GROWTH=1 "-sASSERTIONS=${assertions}"
        "-sEXPORTED_FUNCTIONS=@${SDCB_GENERATED}/exports.json"
        "-sEXPORTED_RUNTIME_METHODS=['FS','HEAPU8','ccall','addFunction','removeFunction','UTF8ToString']"
        "-sINCOMING_MODULE_JS_API=['wasmBinary','instantiateWasm','locateFile','print','printErr','onAbort']")
    if(SDCB_JSPI)
        target_link_options(core PRIVATE -sJSPI=1 "-sJSPI_EXPORTS=@${SDCB_GENERATED}/jspi-exports.json")
    else()
        target_link_options(core PRIVATE -sASYNCIFY=1 -sASYNCIFY_STACK_SIZE=2097152)
    endif()
    configure_file("${SDCB_GENERATED}/core.d.ts" "${CMAKE_CURRENT_BINARY_DIR}/runtime/core.d.ts" COPYONLY)
endif()
target_include_directories(core PRIVATE "${SDCB_SOURCE}/include" "${SDCB_SOURCE}/thirdparty" "${SDCB_GGML_SOURCE}/include")
target_link_libraries(core PRIVATE stable-diffusion)
if(SDCB_NATIVE_CHECK)
    add_executable(browser-boundary-test "${SDCB_TEST_ROOT}/native.cpp" "${SDCB_TEST_ROOT}/wasm-probes.cpp" "${SDCB_TEST_ROOT}/qwen-timestep-probe.cpp" "${SDCB_TEST_ROOT}/bf16-weights-probe.cpp" "${SDCB_TEST_ROOT}/graph-walk-probe.cpp" "${SDCB_TEST_ROOT}/conv3d-bias-probe.cpp" "${SDCB_TEST_ROOT}/webgpu-performance-probe.cpp")
    target_compile_definitions(browser-boundary-test PRIVATE SD_BROWSER_WEBGPU=1 SD_USE_UPSTREAM_GGML GGML_MAX_NAME=160)
    target_include_directories(browser-boundary-test PRIVATE "${SDCB_SOURCE}/src" "${SDCB_GGML_SOURCE}/include" "${SDCB_GGML_SOURCE}/src")
    target_link_libraries(browser-boundary-test PRIVATE core stable-diffusion)
    add_executable(public-graph-test "${SDCB_TEST_ROOT}/public-graph-test.cpp")
    target_include_directories(public-graph-test PRIVATE "${SDCB_GGML_SOURCE}/include")
    target_link_libraries(public-graph-test PRIVATE core stable-diffusion)
    enable_testing()
    add_test(NAME public-graph COMMAND public-graph-test)
    foreach(name image-causal anima-attention anima-context webgpu-rope webgpu-attention-softmax)
        add_executable(${name}-test "${SDCB_TEST_ROOT}/${name}-test.cpp")
        target_compile_definitions(${name}-test PRIVATE SD_BROWSER_WEBGPU=1 SD_USE_UPSTREAM_GGML GGML_MAX_NAME=160)
        target_include_directories(${name}-test PRIVATE "${SDCB_SOURCE}/src" "${SDCB_GGML_SOURCE}/include" "${SDCB_GGML_SOURCE}/src")
        target_link_libraries(${name}-test PRIVATE stable-diffusion)
        add_test(NAME ${name} COMMAND ${name}-test)
    endforeach()
    set(SDCB_WEBGPU_SOURCE "${SDCB_GGML_SOURCE}/src/ggml-webgpu/ggml-webgpu.cpp")
    foreach(fixture policy transfer)
        add_custom_command(OUTPUT "${SDCB_GENERATED}/webgpu-${fixture}-fixture.cpp"
            COMMAND "${Python3_EXECUTABLE}" "${SDCB_TEST_ROOT}/webgpu-${fixture}-fixture.py"
                --source "${SDCB_WEBGPU_SOURCE}" --output "${SDCB_GENERATED}/webgpu-${fixture}-fixture.cpp"
            DEPENDS "${SDCB_TEST_ROOT}/webgpu-${fixture}-fixture.py"
                "${SDCB_TEST_ROOT}/webgpu_wait_probe.py" "${SDCB_WEBGPU_SOURCE}"
                "${SDCB_GGML_SOURCE}/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp"
                "${SDCB_TEST_ROOT}/webgpu-transfer-fixture.cpp.in"
            VERBATIM)
    endforeach()
    add_executable(webgpu-bounded-test "${SDCB_TEST_ROOT}/webgpu-bounded-test.cpp" "${SDCB_GENERATED}/webgpu-policy-fixture.cpp")
    add_executable(webgpu-qwen-work-test "${SDCB_TEST_ROOT}/webgpu-qwen-work-test.cpp" "${SDCB_GENERATED}/webgpu-policy-fixture.cpp")
    add_executable(webgpu-transfer-test "${SDCB_GENERATED}/webgpu-transfer-fixture.cpp")
    foreach(name webgpu-bounded webgpu-transfer webgpu-qwen-work)
        target_compile_definitions(${name}-test PRIVATE SD_BROWSER_WEBGPU=1 SD_USE_UPSTREAM_GGML GGML_MAX_NAME=160)
        target_include_directories(${name}-test PRIVATE "${SDCB_TEST_ROOT}"
            "${SDCB_SOURCE}/src" "${SDCB_GGML_SOURCE}/include" "${SDCB_GGML_SOURCE}/src")
        target_link_libraries(${name}-test PRIVATE stable-diffusion)
        add_test(NAME ${name} COMMAND ${name}-test)
    endforeach()
    if(UNIX AND NOT APPLE)
        target_compile_definitions(anima-context-test PRIVATE SDCB_TEST_WRAP_CACHE_ALLOC=1)
        target_link_options(anima-context-test PRIVATE -Wl,--wrap=ggml_backend_alloc_ctx_tensors)
    endif()
    add_test(NAME browser-boundary COMMAND browser-boundary-test)
    add_executable(gguf-original-shapes-test "${SDCB_TEST_ROOT}/gguf-original-shapes-test.cpp" "${SDCB_TEST_ROOT}/wasm-probes.cpp")
    target_compile_definitions(gguf-original-shapes-test PRIVATE SD_BROWSER_WEBGPU=1 SD_USE_UPSTREAM_GGML GGML_MAX_NAME=160)
    target_include_directories(gguf-original-shapes-test PRIVATE "${SDCB_SOURCE}/src" "${SDCB_GGML_SOURCE}/include" "${SDCB_GGML_SOURCE}/src")
    target_link_libraries(gguf-original-shapes-test PRIVATE stable-diffusion)
    add_test(NAME gguf-original-shapes COMMAND gguf-original-shapes-test)
    # Link the real prepared IO source without the browser macro as well, so
    # native generic-parser fallback cannot bypass original-shape validation.
    add_executable(gguf-original-shapes-fallback-test "${SDCB_TEST_ROOT}/gguf-original-shapes-test.cpp" "${SDCB_TEST_ROOT}/wasm-probes.cpp"
        "${SDCB_SOURCE}/src/model_io/gguf_io.cpp")
    target_compile_definitions(gguf-original-shapes-fallback-test PRIVATE SD_USE_UPSTREAM_GGML GGML_MAX_NAME=160 SDCB_TEST_GGUF_GENERIC_FALLBACK=1)
    target_include_directories(gguf-original-shapes-fallback-test PRIVATE "${SDCB_SOURCE}/src" "${SDCB_GGML_SOURCE}/include" "${SDCB_GGML_SOURCE}/src")
    target_link_libraries(gguf-original-shapes-fallback-test PRIVATE stable-diffusion)
    add_test(NAME gguf-original-shapes-fallback COMMAND gguf-original-shapes-fallback-test)
    add_executable(bounded-weight-uploads-test "${SDCB_TEST_ROOT}/bounded-weight-uploads.cpp")
    target_compile_definitions(bounded-weight-uploads-test PRIVATE GGML_MAX_NAME=160)
    target_include_directories(bounded-weight-uploads-test PRIVATE "${SDCB_SOURCE}/src"
        "${SDCB_GGML_SOURCE}/include" "${SDCB_GGML_SOURCE}/src/ggml-webgpu")
    target_link_libraries(bounded-weight-uploads-test PRIVATE ggml-base)
    add_test(NAME bounded-weight-uploads COMMAND bounded-weight-uploads-test)
    add_executable(browser-weight-reader-integration-test "${SDCB_TEST_ROOT}/browser-weight-reader-integration.cpp")
    target_compile_definitions(browser-weight-reader-integration-test PRIVATE SD_BROWSER_WEBGPU=1 SD_USE_UPSTREAM_GGML GGML_MAX_NAME=160)
    target_include_directories(browser-weight-reader-integration-test PRIVATE "${SDCB_SOURCE}/src"
        "${SDCB_GGML_SOURCE}/include" "${SDCB_GGML_SOURCE}/src")
    target_link_libraries(browser-weight-reader-integration-test PRIVATE stable-diffusion)
    add_test(NAME browser-weight-reader-integration COMMAND browser-weight-reader-integration-test)
    add_executable(bf16-retention-test "${SDCB_TEST_ROOT}/bf16-retention-test.cpp")
    target_compile_definitions(bf16-retention-test PRIVATE SD_BROWSER_WEBGPU=1 SD_USE_UPSTREAM_GGML GGML_MAX_NAME=160)
    target_include_directories(bf16-retention-test PRIVATE "${SDCB_SOURCE}/src"
        "${SDCB_GGML_SOURCE}/include" "${SDCB_GGML_SOURCE}/src")
    target_link_libraries(bf16-retention-test PRIVATE stable-diffusion)
    add_test(NAME bf16-retention COMMAND bf16-retention-test)
    if(UNIX)
        add_test(NAME webgpu-bf16-weight-kernels-contract COMMAND "${Python3_EXECUTABLE}"
            "${SDCB_TEST_ROOT}/webgpu-bf16-weight-kernels/check_kernels.py"
            --source "${SDCB_WEBGPU_SOURCE}" --cxx "${CMAKE_CXX_COMPILER}")
        add_test(NAME webgpu-dispatch-grid-contract COMMAND "${Python3_EXECUTABLE}"
            "${SDCB_TEST_ROOT}/webgpu-dispatch-grid/check_grid.py"
            --source "${SDCB_WEBGPU_SOURCE}" --cxx "${CMAKE_CXX_COMPILER}")
        add_test(NAME webgpu-registry-contract COMMAND "${Python3_EXECUTABLE}"
            "${SDCB_TEST_ROOT}/webgpu-registry/check_registry.py"
            --source "${SDCB_WEBGPU_SOURCE}" --cxx "${CMAKE_CXX_COMPILER}")
        add_test(NAME webgpu-softmax-cache-contract COMMAND "${Python3_EXECUTABLE}"
            "${SDCB_TEST_ROOT}/webgpu-softmax-cache/check_cache.py"
            --source "${SDCB_GGML_SOURCE}/src/ggml-webgpu/wgsl-shaders/soft_max.wgsl"
            --cxx "${CMAKE_CXX_COMPILER}")
        add_test(NAME webgpu-param-upload-contract COMMAND "${Python3_EXECUTABLE}"
            "${SDCB_TEST_ROOT}/webgpu-param-upload/check_uploads.py"
            --source "${SDCB_WEBGPU_SOURCE}" --cxx "${CMAKE_CXX_COMPILER}")
        add_test(NAME webgpu-completion-wait-contract COMMAND "${Python3_EXECUTABLE}"
            "${SDCB_TEST_ROOT}/webgpu_wait_probe.py"
            --source "${SDCB_GGML_SOURCE}/src/ggml-webgpu/ggml-webgpu.cpp" --cxx "${CMAKE_CXX_COMPILER}")
    endif()
    add_executable(h3-pointwise-test "${SDCB_TEST_ROOT}/h3-pointwise-test.cpp")
    target_compile_definitions(h3-pointwise-test PRIVATE SD_BROWSER_WEBGPU=1 SD_USE_UPSTREAM_GGML GGML_MAX_NAME=160)
    target_include_directories(h3-pointwise-test PRIVATE "${SDCB_SOURCE}/src" "${SDCB_GGML_SOURCE}/include" "${SDCB_GGML_SOURCE}/src")
    target_link_libraries(h3-pointwise-test PRIVATE stable-diffusion)
    add_test(NAME h3-pointwise COMMAND h3-pointwise-test)
    if(UNIX AND NOT APPLE)
        add_executable(h3-video-output-test "${SDCB_TEST_ROOT}/h3-video-output-test.cpp")
        target_compile_definitions(h3-video-output-test PRIVATE SD_BROWSER_WEBGPU=1 SD_USE_UPSTREAM_GGML GGML_MAX_NAME=160)
        target_include_directories(h3-video-output-test PRIVATE "${SDCB_SOURCE}/src" "${SDCB_GGML_SOURCE}/include" "${SDCB_GGML_SOURCE}/src")
        target_link_libraries(h3-video-output-test PRIVATE stable-diffusion)
        target_link_options(h3-video-output-test PRIVATE -Wl,--wrap=malloc -Wl,--wrap=calloc -Wl,--wrap=free)
        add_test(NAME h3-video-output COMMAND h3-video-output-test)
    endif()
endif()
