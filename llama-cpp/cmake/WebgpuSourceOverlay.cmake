get_filename_component(LCB_SOURCE_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
if(NOT LCB_WEBGPU OR NOT TARGET ggml-webgpu)
    message(FATAL_ERROR "WebGPU source overlay requires WebGPU")
endif()
if(NOT EMDAWNWEBGPU_DIR)
    message(FATAL_ERROR "WebGPU source requires the reviewed EMDAWNWEBGPU_DIR package")
endif()
set(LCB_SOURCE_OVERLAY "${CMAKE_CURRENT_BINARY_DIR}/webgpu-source-overlay")
set(LCB_SOURCE_COPY OFF)
set(LCB_SOURCE_BATCH OFF)
if(LCB_WEBGPU_TENSOR_COPY)
    set(LCB_SOURCE_COPY ON)
endif()
if(LCB_WEBGPU_PARAM_UPLOAD_BATCHING)
    set(LCB_SOURCE_BATCH ON)
endif()
execute_process(COMMAND "${Python3_EXECUTABLE}" "${LCB_SOURCE_ROOT}/scripts/prepare_webgpu_source.py"
    --source "${LCB_LLAMA_SOURCE}" --output "${LCB_SOURCE_OVERLAY}"
    --dawn-package "${EMDAWNWEBGPU_DIR}"
    --tensor-copy "${LCB_SOURCE_COPY}" --param-upload-batching "${LCB_SOURCE_BATCH}"
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
if(LCB_WEBGPU_PARAM_UPLOAD_BATCHING)
    target_compile_definitions(ggml-webgpu PRIVATE GGML_WEBGPU_BATCH_PARAM_UPLOADS)
endif()
# Keep sibling headers available after relocating the TU. Append, so the MoE
# BEFORE include retains priority over upstream generated shader headers.
target_include_directories(ggml-webgpu PRIVATE "${LCB_LLAMA_SOURCE}/ggml/src/ggml-webgpu")
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
    "${LCB_SOURCE_ROOT}/scripts/prepare_webgpu_source.py"
    "${LCB_SOURCE_ROOT}/scripts/prepare_moe_direct_slot.py"
    "${LCB_SOURCE_ROOT}/scripts/prepare_webgpu_tensor_copy.py"
    "${LCB_SOURCE_ROOT}/upstream-patches-only-as-a-last-resort-with-explicit-user-approval/ggml-webgpu-batch-param-uploads.patch"
    "${LCB_SOURCE_ROOT}/upstream-patches-only-as-a-last-resort-with-explicit-user-approval/ggml-webgpu-same-device-tensor-copy.patch")
