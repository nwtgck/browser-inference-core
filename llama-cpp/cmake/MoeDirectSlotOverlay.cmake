get_filename_component(LCB_MOE_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
if(NOT TARGET ggml-webgpu)
    message(FATAL_ERROR "The MoE direct-slot overlay requires the WebGPU target")
endif()
# Upstream embeds shaders at build time. Prepare a checked private header using
# that same embedder; leave its original generation target and checkout intact.
set(LCB_MOE_OVERLAY "${CMAKE_CURRENT_BINARY_DIR}/moe-direct-slot-overlay")
execute_process(COMMAND "${Python3_EXECUTABLE}" "${LCB_MOE_ROOT}/scripts/prepare_moe_direct_slot.py"
    --source "${LCB_LLAMA_SOURCE}" --output "${LCB_MOE_OVERLAY}"
    COMMAND_ERROR_IS_FATAL ANY)
# ggml-webgpu-shader-lib.hpp includes this header by name. BEFORE selects this copy
# the actual compiled input; tests inspect the real upstream target include order.
target_include_directories(ggml-webgpu BEFORE PRIVATE "${LCB_MOE_OVERLAY}")
file(GLOB LCB_MOE_SHADER_INPUTS CONFIGURE_DEPENDS
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-webgpu/wgsl-shaders/*.wgsl"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-webgpu/wgsl-shaders/*.tmpl")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    ${LCB_MOE_SHADER_INPUTS}
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-webgpu/ggml-webgpu.cpp"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-webgpu/CMakeLists.txt"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-webgpu/wgsl-shaders/embed_wgsl.py"
    "${LCB_MOE_ROOT}/scripts/prepare_moe_direct_slot.py"
    "${LCB_MOE_ROOT}/upstream-patches-only-as-a-last-resort-with-explicit-user-approval/ggml-webgpu-moe-direct-slot.patch")
