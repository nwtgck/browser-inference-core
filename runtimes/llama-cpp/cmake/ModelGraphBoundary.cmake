# Leaf bindings for independent vocabularies, templates and logits. This is not
# an external graph executor and does not replace llama_decode or GGML kernels.
get_filename_component(LCB_MODEL_GRAPH_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(LCB_MODEL_GRAPH_OVERLAY "${CMAKE_CURRENT_BINARY_DIR}/model-graph-chat-overlay")
execute_process(COMMAND "${Python3_EXECUTABLE}" -m build_tools.runtimes.llama_cpp.sources.model_graph_chat
    --source "${LCB_LLAMA_SOURCE}" --output "${LCB_MODEL_GRAPH_OVERLAY}" WORKING_DIRECTORY "${LCB_REPOSITORY_ROOT}" COMMAND_ERROR_IS_FATAL ANY)
get_target_property(LCB_MODEL_GRAPH_COMMON_SOURCES llama-common SOURCES)
set(LCB_MODEL_GRAPH_REPLACEMENTS "")
set(LCB_MODEL_GRAPH_MATCHES 0)
foreach(SOURCE IN LISTS LCB_MODEL_GRAPH_COMMON_SOURCES)
    get_filename_component(NAME "${SOURCE}" NAME)
    if(NAME STREQUAL "chat.cpp")
        math(EXPR LCB_MODEL_GRAPH_MATCHES "${LCB_MODEL_GRAPH_MATCHES} + 1")
        list(APPEND LCB_MODEL_GRAPH_REPLACEMENTS "${LCB_MODEL_GRAPH_OVERLAY}/common/chat.cpp")
    else()
        list(APPEND LCB_MODEL_GRAPH_REPLACEMENTS "${SOURCE}")
    endif()
endforeach()
if(NOT LCB_MODEL_GRAPH_MATCHES EQUAL 1)
    message(FATAL_ERROR "Review the independent-vocabulary chat source composition")
endif()
set_property(TARGET llama-common PROPERTY SOURCES "${LCB_MODEL_GRAPH_REPLACEMENTS}")
set_source_files_properties("${LCB_MODEL_GRAPH_OVERLAY}/common/chat.cpp" TARGET_DIRECTORY llama-common PROPERTIES
    SKIP_UNITY_BUILD_INCLUSION ON INCLUDE_DIRECTORIES "${LCB_LLAMA_SOURCE}/common")
execute_process(COMMAND "${Python3_EXECUTABLE}" -m build_tools.runtimes.llama_cpp.generate_model_graph
    --header "${LCB_MODEL_GRAPH_ROOT}/native/model-graph/model-graph.h" --output "${LCB_GENERATED}"
    --base-exports "${LCB_LINK_EXPORTS}" --clang "${LCB_CLANG}" WORKING_DIRECTORY "${LCB_REPOSITORY_ROOT}" COMMAND_ERROR_IS_FATAL ANY)
set(LCB_LINK_EXPORTS "${LCB_GENERATED}/model-graph-exports.json")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/runtime")
foreach(FILE model-graph-v1.schema.json model-graph-v1.functions.d.ts)
    configure_file("${LCB_GENERATED}/${FILE}" "${CMAKE_CURRENT_BINARY_DIR}/runtime/${FILE}" COPYONLY)
endforeach()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${LCB_REPOSITORY_ROOT}/build_tools/runtimes/llama_cpp/sources/model_graph_chat.py"
    "${LCB_REPOSITORY_ROOT}/build_tools/runtimes/llama_cpp/generate_model_graph.py"
    "${LCB_REPOSITORY_ROOT}/build_tools/source/patches.py"
    "${LCB_REPOSITORY_ROOT}/build_tools/source/files.py"
    "${LCB_REPOSITORY_ROOT}/build_tools/output.py"
    "${LCB_MODEL_GRAPH_ROOT}/upstream-patches-only-as-a-last-resort-with-explicit-user-approval/common-chat-independent-vocabulary.patch"
    "${LCB_MODEL_GRAPH_ROOT}/native/model-graph/model-graph.h"
    "${LCB_LLAMA_SOURCE}/common/chat.cpp" "${LCB_LLAMA_SOURCE}/common/chat.h"
    "${LCB_LLAMA_SOURCE}/src/llama-model-loader.cpp" "${LCB_LLAMA_SOURCE}/src/llama-model-loader.h"
    "${LCB_LLAMA_SOURCE}/src/llama-vocab.cpp" "${LCB_LLAMA_SOURCE}/src/llama-vocab.h"
    "${LCB_LLAMA_SOURCE}/src/llama-arch.h" "${LCB_LLAMA_SOURCE}/src/llama-sampler.cpp")
