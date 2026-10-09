get_filename_component(LCB_SOURCE_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
if(NOT LCB_WEBGPU OR NOT TARGET ggml-webgpu)
    message(FATAL_ERROR "WebGPU source overlay requires WebGPU")
endif()
if(NOT EMDAWNWEBGPU_DIR)
    message(FATAL_ERROR "WebGPU source requires the reviewed EMDAWNWEBGPU_DIR package")
endif()
set(LCB_SOURCE_OVERLAY "${CMAKE_CURRENT_BINARY_DIR}/webgpu-source-overlay")
execute_process(COMMAND "${Python3_EXECUTABLE}" "${LCB_SOURCE_ROOT}/scripts/prepare_webgpu_source.py"
    --source "${LCB_LLAMA_SOURCE}" --output "${LCB_SOURCE_OVERLAY}"
    --dawn-package "${EMDAWNWEBGPU_DIR}"
    COMMAND_ERROR_IS_FATAL ANY)
get_target_property(LCB_SOURCE_SOURCES ggml-webgpu SOURCES)
set(LCB_SOURCE_MATCHES 0)
foreach(LCB_SOURCE_SOURCE IN LISTS LCB_SOURCE_SOURCES)
    if(LCB_SOURCE_SOURCE STREQUAL "ggml-webgpu.cpp")
        math(EXPR LCB_SOURCE_MATCHES "${LCB_SOURCE_MATCHES} + 1")
    endif()
endforeach()
if(NOT LCB_SOURCE_MATCHES EQUAL 1)
    message(FATAL_ERROR "Upstream WebGPU source layout changed; review source overlay")
endif()
list(REMOVE_ITEM LCB_SOURCE_SOURCES "ggml-webgpu.cpp")
list(APPEND LCB_SOURCE_SOURCES "${LCB_SOURCE_OVERLAY}/ggml-webgpu.cpp")
set_property(TARGET ggml-webgpu PROPERTY SOURCES "${LCB_SOURCE_SOURCES}")
target_compile_definitions(ggml-webgpu PRIVATE GGML_WEBGPU_BATCH_PARAM_UPLOADS)
# The prepared shader-library header is a sibling of the source above; its
# embedded WGSL include still resolves through the existing MoE BEFORE path.
# Keep sibling headers available after relocating the TU. Append, so the MoE
# BEFORE include retains priority over upstream generated shader headers.
target_include_directories(ggml-webgpu PRIVATE "${LCB_LLAMA_SOURCE}/ggml/src/ggml-webgpu")
# Replace only the model loader on the existing llama target. Other core sources
# and independent mtmd/MoE overlays retain their original target ownership.
get_target_property(LCB_LOADER_SOURCES llama SOURCES)
set(LCB_LOADER_MATCHES 0)
foreach(LCB_LOADER_SOURCE IN LISTS LCB_LOADER_SOURCES)
    if(LCB_LOADER_SOURCE STREQUAL "llama-model-loader.cpp")
        math(EXPR LCB_LOADER_MATCHES "${LCB_LOADER_MATCHES} + 1")
    endif()
endforeach()
if(NOT LCB_LOADER_MATCHES EQUAL 1)
    message(FATAL_ERROR "Upstream model loader layout changed; review source overlay")
endif()
list(REMOVE_ITEM LCB_LOADER_SOURCES "llama-model-loader.cpp")
list(APPEND LCB_LOADER_SOURCES "${LCB_SOURCE_OVERLAY}/llama-model-loader.cpp")
set_property(TARGET llama PROPERTY SOURCES "${LCB_LOADER_SOURCES}")
set_source_files_properties("${LCB_SOURCE_OVERLAY}/llama-model-loader.cpp"
    TARGET_DIRECTORY llama PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
target_include_directories(llama PRIVATE "${LCB_LLAMA_SOURCE}/src")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${EMDAWNWEBGPU_DIR}/webgpu/include/webgpu/webgpu.h"
    "${EMDAWNWEBGPU_DIR}/webgpu_cpp/include/webgpu/webgpu_cpp.h"
    "${EMDAWNWEBGPU_DIR}/webgpu_cpp/include/webgpu/webgpu_cpp_chained_struct.h"
    "${EMDAWNWEBGPU_DIR}/webgpu_cpp/include/webgpu/webgpu_enum_class_bitmasks.h"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-webgpu/ggml-webgpu.cpp"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-webgpu/CMakeLists.txt"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-backend.cpp"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-backend-impl.h"
    "${LCB_LLAMA_SOURCE}/ggml/include/ggml.h"
    "${LCB_LLAMA_SOURCE}/ggml/include/ggml-backend.h"
    "${LCB_LLAMA_SOURCE}/src/llama-model-loader.cpp"
    "${LCB_LLAMA_SOURCE}/src/llama-model-loader.h"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-quants.c"
    "${LCB_LLAMA_SOURCE}/src/llama-mmap.cpp"
    "${LCB_LLAMA_SOURCE}/src/llama-mmap.h"
    "${LCB_LLAMA_SOURCE}/src/CMakeLists.txt"
    "${LCB_SOURCE_ROOT}/upstream-patches-only-as-a-last-resort-with-explicit-user-approval/llama-model-loader-webgpu-chunked-upload.patch"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-webgpu/wgsl-shaders/ssm_conv.wgsl"
    "${LCB_SOURCE_ROOT}/upstream-patches-only-as-a-last-resort-with-explicit-user-approval/ggml-webgpu-ssm-conv-single-token.patch"
    "${LCB_SOURCE_ROOT}/scripts/prepare_webgpu_source.py"
    "${LCB_SOURCE_ROOT}/scripts/prepare_moe_direct_slot.py"
    "${LCB_SOURCE_ROOT}/scripts/prepare_webgpu_tensor_copy.py"
    "${LCB_SOURCE_ROOT}/upstream-patches-only-as-a-last-resort-with-explicit-user-approval/ggml-webgpu-batch-param-uploads.patch"
    "${LCB_SOURCE_ROOT}/upstream-patches-only-as-a-last-resort-with-explicit-user-approval/ggml-webgpu-same-device-tensor-copy.patch")
