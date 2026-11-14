# Ordinary contracts consume the same prepared source/header tree as core.
# This file is included by the runtime's native configuration.
set(SDCB_EXISTING_TESTS "${SDCB_COMPILE_ROOT}/tests/runtimes/stable_diffusion_cpp")

function(sdc_contract target)
    add_executable(${target} ${ARGN})
    target_compile_definitions(${target} PRIVATE SD_BROWSER_WEBGPU=1 SD_USE_UPSTREAM_GGML GGML_MAX_NAME=160)
    target_include_directories(${target} PRIVATE "${SDCB_SOURCE}/src"
        "${SDCB_GGML_SOURCE}/include" "${SDCB_GGML_SOURCE}/src" "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(${target} PRIVATE stable-diffusion)
    add_test(NAME ${target} COMMAND ${target})
endfunction()

# The native process owns one implementation of the runtime. Linking both the
# static runtime and core.so duplicates internal callback/static object storage.
# Compile the actual small ABI sources into these contracts; core remains its
# own independently linked production target and generated ABI integration.
add_library(native-abi-contract OBJECT
    "${SDCB_COMPILE_ROOT}/runtimes/stable-diffusion-cpp/native/abi/bridge.cpp"
    "${SDCB_COMPILE_ROOT}/runtimes/stable-diffusion-cpp/native/abi/metadata.cpp")
# ABI analysis must select the production core's flags, not this test target's
# flags. Keep exactly one exported compile command per production ABI source.
set_property(TARGET native-abi-contract PROPERTY EXPORT_COMPILE_COMMANDS OFF)
target_compile_definitions(native-abi-contract PRIVATE SD_BROWSER_WEBGPU=1 SD_USE_UPSTREAM_GGML GGML_MAX_NAME=160)
target_include_directories(native-abi-contract PRIVATE "${SDCB_SOURCE}/include" "${SDCB_SOURCE}/thirdparty"
    "${SDCB_GGML_SOURCE}/include" "${SDCB_GENERATED}" "${SDCB_COMPILE_ROOT}/runtimes/stable-diffusion-cpp/native")
target_link_libraries(native-abi-contract PUBLIC stable-diffusion)
add_dependencies(native-abi-contract core-abi)

sdc_contract(browser-boundary-test "${SDCB_EXISTING_TESTS}/native.cpp"
    "${SDCB_EXISTING_TESTS}/wasm-probes.cpp" "${SDCB_EXISTING_TESTS}/qwen-timestep-probe.cpp"
    "${SDCB_EXISTING_TESTS}/bf16-weights-probe.cpp" "${SDCB_EXISTING_TESTS}/graph-walk-probe.cpp"
    "${SDCB_EXISTING_TESTS}/conv3d-bias-probe.cpp" "${SDCB_EXISTING_TESTS}/webgpu-performance-probe.cpp")
target_link_libraries(browser-boundary-test PRIVATE native-abi-contract)
sdc_contract(public-graph-test "${SDCB_EXISTING_TESTS}/public-graph-test.cpp")
target_link_libraries(public-graph-test PRIVATE native-abi-contract)

foreach(name image-causal anima-attention anima-context webgpu-rope webgpu-attention-softmax
        gguf-original-shapes bf16-retention h3-pointwise)
    sdc_contract(${name}-test "${SDCB_EXISTING_TESTS}/${name}-test.cpp")
endforeach()

target_sources(gguf-original-shapes-test PRIVATE "${SDCB_EXISTING_TESTS}/wasm-probes.cpp")
sdc_contract(browser-weight-reader-integration-test "${SDCB_EXISTING_TESTS}/browser-weight-reader-integration.cpp")
# The generic parser branch must obey the same original-shape rules.
sdc_contract(gguf-original-shapes-fallback-test "${SDCB_EXISTING_TESTS}/gguf-original-shapes-test.cpp"
    "${SDCB_EXISTING_TESTS}/wasm-probes.cpp" "${SDCB_SOURCE}/src/model_io/gguf_io.cpp")
set_property(TARGET gguf-original-shapes-fallback-test PROPERTY COMPILE_DEFINITIONS
    SD_USE_UPSTREAM_GGML GGML_MAX_NAME=160 SDCB_TEST_GGUF_GENERIC_FALLBACK=1)

add_executable(bounded-weight-uploads-test "${SDCB_EXISTING_TESTS}/bounded-weight-uploads.cpp")
target_compile_definitions(bounded-weight-uploads-test PRIVATE GGML_MAX_NAME=160)
target_include_directories(bounded-weight-uploads-test PRIVATE "${SDCB_SOURCE}/src"
    "${SDCB_GGML_SOURCE}/include" "${SDCB_GGML_SOURCE}/src/ggml-webgpu")
target_link_libraries(bounded-weight-uploads-test PRIVATE ggml-base)
add_test(NAME bounded-weight-uploads-test COMMAND bounded-weight-uploads-test)

if(UNIX AND NOT APPLE)
    target_compile_definitions(anima-context-test PRIVATE SDCB_TEST_WRAP_CACHE_ALLOC=1)
    target_link_options(anima-context-test PRIVATE -Wl,--wrap=ggml_backend_alloc_ctx_tensors)
    sdc_contract(h3-video-output-test "${SDCB_EXISTING_TESTS}/h3-video-output-test.cpp")
    target_link_options(h3-video-output-test PRIVATE -Wl,--wrap=malloc -Wl,--wrap=calloc -Wl,--wrap=free)
endif()

if(NOT IS_DIRECTORY "${SDCB_DAWN_PACKAGE}/webgpu_cpp/include")
    message(FATAL_ERROR "Native WebGPU contracts require the pinned SDCB_DAWN_PACKAGE headers")
endif()
if(NOT EXISTS "${SDCB_COMPILE_ROOT}/tests/fixtures/dawn-native-api/include/dawn/webgpu_cpp.h")
    message(FATAL_ERROR "Native WebGPU contracts require the owned Dawn API declaration fixture")
endif()
set(SDCB_WEBGPU_SOURCE "${SDCB_GGML_SOURCE}/src/ggml-webgpu")
file(GLOB shader_inputs CONFIGURE_DEPENDS "${SDCB_WEBGPU_SOURCE}/wgsl-shaders/*.wgsl" "${SDCB_WEBGPU_SOURCE}/wgsl-shaders/*.tmpl")
find_package(Python3 REQUIRED COMPONENTS Interpreter)
# This is the upstream's unmodified embedding generator, not a fixture compiler.
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/shaders/ggml-wgsl-shaders.hpp"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/shaders"
    COMMAND "${Python3_EXECUTABLE}" "${SDCB_WEBGPU_SOURCE}/wgsl-shaders/embed_wgsl.py"
        --input_dir "${SDCB_WEBGPU_SOURCE}/wgsl-shaders"
        --output_file "${CMAKE_CURRENT_BINARY_DIR}/shaders/ggml-wgsl-shaders.hpp"
    DEPENDS ${shader_inputs} "${SDCB_WEBGPU_SOURCE}/wgsl-shaders/embed_wgsl.py" VERBATIM)

add_library(sdc-browser-api INTERFACE)
target_compile_definitions(sdc-browser-api INTERFACE SDCB_TEST_BROWSER_API=1)
target_include_directories(sdc-browser-api SYSTEM INTERFACE "${SDCB_DAWN_PACKAGE}/webgpu/include"
    "${SDCB_DAWN_PACKAGE}/webgpu_cpp/include")
add_library(sdc-native-api INTERFACE)
target_include_directories(sdc-native-api SYSTEM INTERFACE "${SDCB_COMPILE_ROOT}/tests/fixtures/dawn-native-api/include")

function(sdc_webgpu_configuration target)
    target_compile_features(${target} PRIVATE cxx_std_20)
    target_compile_definitions(${target} PRIVATE GGML_MAX_NAME=160 SD_USE_UPSTREAM_GGML SD_BROWSER_WEBGPU=1)
    target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${SDCB_WEBGPU_SOURCE}"
        "${CMAKE_CURRENT_BINARY_DIR}/shaders" "${SDCB_GGML_SOURCE}/include" "${SDCB_GGML_SOURCE}/src")
    target_compile_options(${target} PRIVATE -ffunction-sections -fdata-sections)
    target_link_options(${target} PUBLIC -Wl,--gc-sections -Wl,--wrap=ggml_abort)
    target_link_libraries(${target} PUBLIC ggml-base)
endfunction()

function(sdc_webgpu_contract target source)
    add_executable(${target} "${source}" "${CMAKE_CURRENT_BINARY_DIR}/shaders/ggml-wgsl-shaders.hpp")
    sdc_webgpu_configuration(${target})
    target_link_libraries(${target} PRIVATE sdc-browser-api)
endfunction()

# Both policy suites compile exactly the same complete TU/options once.
add_library(webgpu-policy-implementation OBJECT policy.cpp "${CMAKE_CURRENT_BINARY_DIR}/shaders/ggml-wgsl-shaders.hpp")
sdc_webgpu_configuration(webgpu-policy-implementation)
target_link_libraries(webgpu-policy-implementation PRIVATE sdc-browser-api)
target_include_directories(webgpu-policy-implementation PRIVATE "${SDCB_EXISTING_TESTS}")
sdc_contract(webgpu-policy-contract "${SDCB_EXISTING_TESTS}/webgpu-bounded-test.cpp")
sdc_contract(webgpu-qwen-work-contract "${SDCB_EXISTING_TESTS}/webgpu-qwen-work-test.cpp")
target_include_directories(webgpu-policy-contract PRIVATE "${SDCB_EXISTING_TESTS}" "${SDCB_SOURCE}/src")
target_include_directories(webgpu-qwen-work-contract PRIVATE "${SDCB_EXISTING_TESTS}" "${SDCB_SOURCE}/src")
target_link_libraries(webgpu-policy-contract PRIVATE webgpu-policy-implementation)
target_link_libraries(webgpu-qwen-work-contract PRIVATE webgpu-policy-implementation)

sdc_webgpu_contract(webgpu-transfer-contract transfer.cpp)
target_include_directories(webgpu-transfer-contract PRIVATE "${SDCB_SOURCE}/src")
target_link_libraries(webgpu-transfer-contract PRIVATE stable-diffusion)
target_link_options(webgpu-transfer-contract PRIVATE -Wl,--wrap=ggml_backend_alloc_ctx_tensors)
add_test(NAME webgpu-transfer-contract COMMAND webgpu-transfer-contract)

foreach(mode batched per-kernel)
    sdc_webgpu_contract(webgpu-backend-${mode}-contract backend.cpp)
    # Test the real optional SET_ROWS post-graph copy/map boundary. The native
    # SDK target below also checks the production configuration without it.
    target_compile_definitions(webgpu-backend-${mode}-contract PRIVATE GGML_WEBGPU_CHECK_SET_ROWS=1)
    if(mode STREQUAL batched)
        target_compile_definitions(webgpu-backend-${mode}-contract PRIVATE GGML_WEBGPU_BATCH_PARAM_UPLOADS=1)
    endif()
    add_test(NAME webgpu-backend-${mode}-contract COMMAND webgpu-backend-${mode}-contract)
    add_test(NAME webgpu-backend-${mode}-mixed-queue-contract COMMAND webgpu-backend-${mode}-contract mixed-queue)
endforeach()

add_executable(webgpu-backend-native-contract backend.cpp "${CMAKE_CURRENT_BINARY_DIR}/shaders/ggml-wgsl-shaders.hpp")
sdc_webgpu_configuration(webgpu-backend-native-contract)
target_link_libraries(webgpu-backend-native-contract PRIVATE sdc-native-api)
target_compile_definitions(webgpu-backend-native-contract PRIVATE GGML_WEBGPU_BATCH_PARAM_UPLOADS=1)
add_test(NAME webgpu-backend-native-contract COMMAND webgpu-backend-native-contract)
add_test(NAME webgpu-backend-native-mixed-queue-contract COMMAND webgpu-backend-native-contract mixed-queue)

foreach(case normal null-instance null-adapter no-f16 no-subgroups device-failure reordered concurrent
        held-buffers device-lost adapter-wait-error device-wait-error missing-adapter-callback missing-device-callback no-implicit-sync)
    add_test(NAME webgpu-registry-${case}-contract COMMAND webgpu-backend-batched-contract "${case}")
    add_test(NAME webgpu-native-registry-${case}-contract COMMAND webgpu-backend-native-contract "${case}")
endforeach()
