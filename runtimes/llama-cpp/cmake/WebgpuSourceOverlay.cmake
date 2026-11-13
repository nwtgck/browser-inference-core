get_filename_component(LCB_REPOSITORY_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
get_filename_component(LCB_SOURCE_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
if(NOT LCB_WEBGPU OR NOT TARGET ggml-webgpu OR NOT TARGET ggml-base)
    message(FATAL_ERROR "WebGPU source overlay requires WebGPU")
endif()
if(NOT EMDAWNWEBGPU_DIR)
    message(FATAL_ERROR "WebGPU source requires the reviewed EMDAWNWEBGPU_DIR package")
endif()
set(LCB_SOURCE_OVERLAY "${CMAKE_CURRENT_BINARY_DIR}/webgpu-source-overlay")
set(LCB_WEBGPU_PROFILE_PREPARE_ARGS)
if(GGML_WEBGPU_GPU_PROFILE)
    list(APPEND LCB_WEBGPU_PROFILE_PREPARE_ARGS --gpu-profile)
endif()
execute_process(COMMAND "${Python3_EXECUTABLE}" -m build_tools.runtimes.llama_cpp.source
    --source "${LCB_LLAMA_SOURCE}" --output "${LCB_SOURCE_OVERLAY}"
    --dawn-package "${EMDAWNWEBGPU_DIR}" ${LCB_WEBGPU_PROFILE_PREPARE_ARGS}
    WORKING_DIRECTORY "${LCB_REPOSITORY_ROOT}" COMMAND_ERROR_IS_FATAL ANY)
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

# The provider is Emdawn-only. Native builds retain the original host writer.
if(EMSCRIPTEN)
    get_target_property(LCB_CHECKPOINT_SOURCES llama SOURCES)
    set(LCB_CHECKPOINT_MATCHES 0)
    set(LCB_CHECKPOINT_REPLACED "")
    foreach(LCB_CHECKPOINT_SOURCE IN LISTS LCB_CHECKPOINT_SOURCES)
        if(LCB_CHECKPOINT_SOURCE STREQUAL "llama-context.cpp")
            math(EXPR LCB_CHECKPOINT_MATCHES "${LCB_CHECKPOINT_MATCHES} + 1")
            list(APPEND LCB_CHECKPOINT_REPLACED "${LCB_SOURCE_OVERLAY}/llama-context.cpp")
        else()
            list(APPEND LCB_CHECKPOINT_REPLACED "${LCB_CHECKPOINT_SOURCE}")
        endif()
    endforeach()
    if(NOT LCB_CHECKPOINT_MATCHES EQUAL 1)
        message(FATAL_ERROR "Upstream context source layout changed; review checkpoint readback overlay")
    endif()
    set_property(TARGET llama PROPERTY SOURCES "${LCB_CHECKPOINT_REPLACED}")
    set_source_files_properties("${LCB_SOURCE_OVERLAY}/llama-context.cpp"
        TARGET_DIRECTORY llama PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
    # Moving this TU must not expose it to a target-wide model include override.
    # Clang's quoted-only search preserves the original source-directory order.
    set_property(SOURCE "${LCB_SOURCE_OVERLAY}/llama-context.cpp" TARGET_DIRECTORY llama
        APPEND PROPERTY COMPILE_OPTIONS "-iquote" "${LCB_LLAMA_SOURCE}/src")
endif()

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
# The prepared shader-library explicitly includes the checked shared-word
# canonical embedding under a distinct name. Do not include upstream's old
# canonical symbols in that translation unit as well. v0.7.0 already implements
# direct selected-slot MoE dispatch; no direct-slot replacement remains.
# The accepted matrix-only group skip uses its own embedded header.
# Keep sibling headers available after relocating the translation unit.
target_include_directories(ggml-webgpu PRIVATE "${LCB_LLAMA_SOURCE}/ggml/src/ggml-webgpu")
set(LCB_WORD_LOAD_CANONICAL
    "${LCB_SOURCE_OVERLAY}/ggml-webgpu-aligned-word-load-shaders.hpp")
if(NOT EXISTS "${LCB_WORD_LOAD_CANONICAL}")
    message(FATAL_ERROR "Checked shared word-loader canonical embedding is missing")
endif()
target_sources(ggml-webgpu PRIVATE "${LCB_WORD_LOAD_CANONICAL}")
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
# Depend on the inputs actually captured by the source recipe, not on its old
# intermediate test-stage implementations. The receipt has repository-relative
# names; build-machine absolute paths never enter published provenance.
file(READ "${LCB_SOURCE_OVERLAY}/source-preparation.json" LCB_PREPARATION_RECORD)
string(JSON LCB_PREPARATION_INPUT_COUNT LENGTH "${LCB_PREPARATION_RECORD}" inputs)
math(EXPR LCB_PREPARATION_INPUT_LAST "${LCB_PREPARATION_INPUT_COUNT} - 1")
set(LCB_PREPARATION_INPUTS)
foreach(index RANGE ${LCB_PREPARATION_INPUT_LAST})
    string(JSON input MEMBER "${LCB_PREPARATION_RECORD}" inputs ${index})
    if(input MATCHES "^runtime/")
        string(REGEX REPLACE "^runtime/" "" input "${input}")
        list(APPEND LCB_PREPARATION_INPUTS "${LCB_SOURCE_ROOT}/${input}")
    else()
        list(APPEND LCB_PREPARATION_INPUTS "${LCB_LLAMA_SOURCE}/${input}")
    endif()
endforeach()
string(JSON LCB_PREPARATION_PATCH_COUNT LENGTH "${LCB_PREPARATION_RECORD}" patches)
math(EXPR LCB_PREPARATION_PATCH_LAST "${LCB_PREPARATION_PATCH_COUNT} - 1")
foreach(index RANGE ${LCB_PREPARATION_PATCH_LAST})
    string(JSON patch GET "${LCB_PREPARATION_RECORD}" patches ${index} file)
    list(APPEND LCB_PREPARATION_INPUTS
        "${LCB_SOURCE_ROOT}/upstream-patches-only-as-a-last-resort-with-explicit-user-approval/${patch}")
endforeach()
# The receipt tracks existing files. CONFIGURE_DEPENDS additionally detects a
# shader being added or removed, including additions not mentioned by a patch.
file(GLOB LCB_SHADER_INPUTS CONFIGURE_DEPENDS
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-webgpu/wgsl-shaders/*.wgsl"
    "${LCB_LLAMA_SOURCE}/ggml/src/ggml-webgpu/wgsl-shaders/*.tmpl")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    ${LCB_PREPARATION_INPUTS} ${LCB_SHADER_INPUTS}
    "${LCB_REPOSITORY_ROOT}/build_tools/runtimes/llama_cpp/source.py"
    "${LCB_REPOSITORY_ROOT}/build_tools/runtimes/llama_cpp/webgpu_plan.py"
    "${LCB_REPOSITORY_ROOT}/build_tools/source/patches.py"
    "${LCB_REPOSITORY_ROOT}/build_tools/source/wgsl.py"
    "${LCB_REPOSITORY_ROOT}/build_tools/source/files.py"
    "${LCB_REPOSITORY_ROOT}/build_tools/project.py"
    "${LCB_REPOSITORY_ROOT}/build_tools/output.py"
    "${LCB_REPOSITORY_ROOT}/build_tools/toolchain/dawn.py"
    "${LCB_SOURCE_ROOT}/config/webgpu-patches.json"
    "${EMDAWNWEBGPU_DIR}/webgpu/include/webgpu/webgpu.h"
    "${EMDAWNWEBGPU_DIR}/webgpu_cpp/include/webgpu/webgpu_cpp.h"
    "${EMDAWNWEBGPU_DIR}/webgpu_cpp/include/webgpu/webgpu_cpp_chained_struct.h"
    "${EMDAWNWEBGPU_DIR}/webgpu_cpp/include/webgpu/webgpu_enum_class_bitmasks.h")

# Prevent duplicate outputs in Ninja's regeneration rule when another overlay
# spells a shared dependency through an absolute path containing '..'.
get_directory_property(LCB_CONFIGURE_INPUTS CMAKE_CONFIGURE_DEPENDS)
set(LCB_NORMALIZED_CONFIGURE_INPUTS)
foreach(LCB_CONFIGURE_INPUT IN LISTS LCB_CONFIGURE_INPUTS)
    get_filename_component(LCB_CONFIGURE_INPUT "${LCB_CONFIGURE_INPUT}" ABSOLUTE)
    list(APPEND LCB_NORMALIZED_CONFIGURE_INPUTS "${LCB_CONFIGURE_INPUT}")
endforeach()
list(REMOVE_DUPLICATES LCB_NORMALIZED_CONFIGURE_INPUTS)
set_property(DIRECTORY PROPERTY CMAKE_CONFIGURE_DEPENDS "${LCB_NORMALIZED_CONFIGURE_INPUTS}")
