get_filename_component(LCB_SOURCE_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
if(NOT LCB_WEBGPU OR NOT TARGET ggml-webgpu OR NOT TARGET ggml-base)
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

# The optional snapshot contract also changes the scheduler in ggml-base.
# Preserve every other source and its order, and fail on upstream layout drift.
get_target_property(LCB_BACKEND_SOURCES ggml-base SOURCES)
set(LCB_BACKEND_MATCHES 0)
set(LCB_BACKEND_REPLACED "")
foreach(LCB_BACKEND_SOURCE IN LISTS LCB_BACKEND_SOURCES)
    if(LCB_BACKEND_SOURCE STREQUAL "ggml-backend.cpp")
        math(EXPR LCB_BACKEND_MATCHES "${LCB_BACKEND_MATCHES} + 1")
        list(APPEND LCB_BACKEND_REPLACED "${LCB_SOURCE_OVERLAY}/ggml-backend.cpp")
    else()
        list(APPEND LCB_BACKEND_REPLACED "${LCB_BACKEND_SOURCE}")
    endif()
endforeach()
if(NOT LCB_BACKEND_MATCHES EQUAL 1)
    message(FATAL_ERROR "Upstream ggml backend source layout changed; review source overlay")
endif()
set_property(TARGET ggml-base PROPERTY SOURCES "${LCB_BACKEND_REPLACED}")
set_source_files_properties("${LCB_SOURCE_OVERLAY}/ggml-backend.cpp"
    TARGET_DIRECTORY ggml-base PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
target_include_directories(ggml-base PRIVATE "${LCB_LLAMA_SOURCE}/ggml/src" "${LCB_SOURCE_OVERLAY}")
# The prepared shader-library header is a sibling of the source above; its
# embedded WGSL include resolves through upstream's generated header. v0.7.0
# already implements direct selected-slot MoE dispatch; no direct-slot shader replacement remains.
# Keep sibling headers available after relocating the translation unit.
target_include_directories(ggml-webgpu PRIVATE "${LCB_LLAMA_SOURCE}/ggml/src/ggml-webgpu")
# Replace the model loader and sampler on the existing llama target. Independent
# model and mtmd overlays retain their original target ownership.
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
get_target_property(LCB_SAMPLER_SOURCES llama SOURCES)
set(LCB_SAMPLER_MATCHES 0)
set(LCB_SAMPLER_REPLACED "")
foreach(LCB_SAMPLER_SOURCE IN LISTS LCB_SAMPLER_SOURCES)
    if(LCB_SAMPLER_SOURCE STREQUAL "llama-sampler.cpp")
        math(EXPR LCB_SAMPLER_MATCHES "${LCB_SAMPLER_MATCHES} + 1")
        list(APPEND LCB_SAMPLER_REPLACED "${LCB_SOURCE_OVERLAY}/llama-sampler.cpp")
    else()
        list(APPEND LCB_SAMPLER_REPLACED "${LCB_SAMPLER_SOURCE}")
    endif()
endforeach()
if(NOT LCB_SAMPLER_MATCHES EQUAL 1)
    message(FATAL_ERROR "Upstream sampler source layout changed; review source overlay")
endif()
set_property(TARGET llama PROPERTY SOURCES "${LCB_SAMPLER_REPLACED}")
set_source_files_properties("${LCB_SOURCE_OVERLAY}/llama-sampler.cpp"
    TARGET_DIRECTORY llama PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
target_include_directories(llama PRIVATE "${LCB_LLAMA_SOURCE}/src")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${LCB_SOURCE_ROOT}/scripts/prepare_sampler_single_sync.py"
    "${LCB_SOURCE_ROOT}/upstream-patches-only-as-a-last-resort-with-explicit-user-approval/llama-sampler-single-sync.patch"
    "${LCB_LLAMA_SOURCE}/src/llama-sampler.cpp"
    "${LCB_LLAMA_SOURCE}/src/llama-sampler.h"
    "${LCB_LLAMA_SOURCE}/src/llama-context.cpp"
    "${LCB_LLAMA_SOURCE}/src/llama-context.h"
    "${LCB_LLAMA_SOURCE}/src/llama-vocab.h"
    "${LCB_LLAMA_SOURCE}/src/llama-model.cpp"
    "${LCB_LLAMA_SOURCE}/src/llama-vocab.cpp"
    "${LCB_LLAMA_SOURCE}/include/llama.h"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-backend.cpp")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-webgpu/wgsl-shaders/unary.wgsl"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-webgpu/wgsl-shaders/binary.wgsl"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-webgpu/wgsl-shaders/scale.wgsl"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-webgpu/wgsl-shaders/repeat.wgsl"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-webgpu/wgsl-shaders/cpy.wgsl"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-webgpu/pre_wgsl.hpp"
    "${LCB_SOURCE_ROOT}/scripts/prepare_webgpu_hc.py"
    "${LCB_SOURCE_ROOT}/upstream-patches-only-as-a-last-resort-with-explicit-user-approval/ggml-webgpu-hc-pre-post.patch"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-webgpu/wgsl-shaders/embed_wgsl.py"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml.c"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-alloc.c"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-cpu/ops.cpp"
    "${LCB_LLAMA_SOURCE}/src/models/qwen4exp.cpp"
    "${LCB_LLAMA_SOURCE}/src/llama-context.cpp"
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
    "${LCB_SOURCE_ROOT}/scripts/prepare_webgpu_tensor_copy.py"
    "${LCB_SOURCE_ROOT}/upstream-patches-only-as-a-last-resort-with-explicit-user-approval/ggml-webgpu-batch-param-uploads.patch"
    "${LCB_SOURCE_ROOT}/upstream-patches-only-as-a-last-resort-with-explicit-user-approval/ggml-webgpu-same-device-tensor-copy.patch")

set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${LCB_SOURCE_ROOT}/scripts/prepare_webgpu_ordered_readback.py"
    "${LCB_SOURCE_ROOT}/upstream-patches-only-as-a-last-resort-with-explicit-user-approval/ggml-webgpu-ordered-readback.patch")

set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${LCB_SOURCE_ROOT}/scripts/prepare_webgpu_host_input_snapshot.py"
    "${LCB_SOURCE_ROOT}/upstream-patches-only-as-a-last-resort-with-explicit-user-approval/ggml-webgpu-host-input-snapshot.patch"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-backend.cpp"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-backend-reg.cpp"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-backend-impl.h"
    "${LCB_LLAMA_SOURCE}/ggml/include/ggml-backend.h"
    "${LCB_LLAMA_SOURCE}/ggml/include/ggml.h"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-alloc.c"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml.c"
    "${LCB_LLAMA_SOURCE}/ggml/src/CMakeLists.txt"
    "${LCB_LLAMA_SOURCE}/src/llama-context.cpp")

set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${LCB_SOURCE_ROOT}/scripts/prepare_webgpu_bind_group_cache.py"
    "${LCB_SOURCE_ROOT}/upstream-patches-only-as-a-last-resort-with-explicit-user-approval/ggml-webgpu-bind-group-cache.patch")

set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${LCB_SOURCE_ROOT}/scripts/prepare_webgpu_rms_scale.py"
    "${LCB_SOURCE_ROOT}/scripts/prepare_webgpu_async_tensor_copy.py"
    "${LCB_SOURCE_ROOT}/scripts/prepare_webgpu_iq3s_subgroup.py"
    "${LCB_SOURCE_ROOT}/upstream-patches-only-as-a-last-resort-with-explicit-user-approval/ggml-webgpu-rms-scale.patch"
    "${LCB_SOURCE_ROOT}/upstream-patches-only-as-a-last-resort-with-explicit-user-approval/ggml-webgpu-async-tensor-copy.patch"
    "${LCB_SOURCE_ROOT}/upstream-patches-only-as-a-last-resort-with-explicit-user-approval/ggml-webgpu-iq3s-id-subgroup.patch"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-webgpu/wgsl-shaders/row_norm.wgsl")

set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-common.h")

set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${LCB_SOURCE_ROOT}/scripts/prepare_webgpu_scale_unary.py"
    "${LCB_SOURCE_ROOT}/scripts/prepare_webgpu_q8_subgroup.py"
    "${LCB_SOURCE_ROOT}/upstream-patches-only-as-a-last-resort-with-explicit-user-approval/ggml-webgpu-scale-unary.patch"
    "${LCB_SOURCE_ROOT}/upstream-patches-only-as-a-last-resort-with-explicit-user-approval/ggml-webgpu-q8-subgroup-matvec.patch"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-webgpu/ggml-webgpu.cpp"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-webgpu/pre_wgsl.hpp"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-webgpu/wgsl-shaders/embed_wgsl.py"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-webgpu/wgsl-shaders/common_decls.tmpl"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_vec.wgsl"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_vec_acc.tmpl"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-webgpu/wgsl-shaders/scale.wgsl"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-webgpu/wgsl-shaders/unary.wgsl"
    "${LCB_LLAMA_SOURCE}/ggml/include/ggml.h"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml.c"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-impl.h"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-alloc.c"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-backend.cpp"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-backend-impl.h")

# Short Q4 uses the upstream generated shaders with a checked host selector.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${LCB_SOURCE_ROOT}/scripts/prepare_webgpu_q4_short.py"
    "${LCB_SOURCE_ROOT}/upstream-patches-only-as-a-last-resort-with-explicit-user-approval/ggml-webgpu-q4-short-workgroup.patch")

# Re-run the checked short-Q8 selection when its upstream shader contract changes.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${LCB_SOURCE_ROOT}/scripts/prepare_webgpu_q8_short.py"
    "${LCB_SOURCE_ROOT}/upstream-patches-only-as-a-last-resort-with-explicit-user-approval/ggml-webgpu-q8-short-workgroup.patch"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-webgpu/pre_wgsl.hpp"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-webgpu/wgsl-shaders/common_decls.tmpl"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-webgpu/wgsl-shaders/embed_wgsl.py"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_vec.wgsl"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_vec_acc.tmpl"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_vec_q_acc.tmpl"
)

# Ordered expert-output sum contract and build inputs.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${LCB_SOURCE_ROOT}/scripts/prepare_webgpu_moe_sum.py"
    "${LCB_SOURCE_ROOT}/upstream-patches-only-as-a-last-resort-with-explicit-user-approval/ggml-webgpu-moe-expert-sum.patch"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-impl.h"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml.c"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-alloc.c"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-backend.cpp"
    "${LCB_LLAMA_SOURCE}/src/llama-graph.cpp"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-webgpu/wgsl-shaders/binary.wgsl"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-webgpu/pre_wgsl.hpp"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-webgpu/wgsl-shaders/embed_wgsl.py"
)

# Single-token expert matvec selection uses upstream shader code unchanged.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${LCB_SOURCE_ROOT}/scripts/prepare_webgpu_expert_short.py"
    "${LCB_SOURCE_ROOT}/upstream-patches-only-as-a-last-resort-with-explicit-user-approval/ggml-webgpu-expert-short-workgroup.patch"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-webgpu/pre_wgsl.hpp"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-webgpu/wgsl-shaders/common_decls.tmpl"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-webgpu/wgsl-shaders/embed_wgsl.py"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_id_vec.wgsl"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_vec_acc.tmpl"
)

# Preserve canonical SSM convolution and unary shaders; embed only the epilogue variant.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${LCB_SOURCE_ROOT}/scripts/prepare_webgpu_ssm_conv_silu.py"
    "${LCB_SOURCE_ROOT}/upstream-patches-only-as-a-last-resort-with-explicit-user-approval/ggml-webgpu-ssm-conv-silu.patch"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-webgpu/wgsl-shaders/ssm_conv.wgsl"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-webgpu/wgsl-shaders/unary.wgsl")
