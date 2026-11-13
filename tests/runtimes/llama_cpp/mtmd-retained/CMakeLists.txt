cmake_minimum_required(VERSION 3.24)
project(mtmd_retained_tests LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_CPU ON CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_ACCELERATE OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory("${LCB_LLAMA_SOURCE}/ggml" ggml)
enable_testing()
add_executable(mtmd-retained-test ../mtmd-retained.cpp ../mtmd-test-logger.cpp)
target_include_directories(mtmd-retained-test PRIVATE ../../../../runtimes/llama-cpp/native/vision "${LCB_LLAMA_SOURCE}/ggml/src"
    "${LCB_LLAMA_SOURCE}/tools/mtmd" "${LCB_LLAMA_SOURCE}/include")
target_link_libraries(mtmd-retained-test PRIVATE ggml ggml-cpu)
add_test(NAME mtmd-retained COMMAND mtmd-retained-test)
add_test(NAME mtmd-retained-roles COMMAND mtmd-retained-test roles)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
get_filename_component(LCB_REPOSITORY_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)
execute_process(COMMAND "${Python3_EXECUTABLE}" -m build_tools.runtimes.llama_cpp.sources.retained_bf16
    --source "${LCB_LLAMA_SOURCE}" WORKING_DIRECTORY "${LCB_REPOSITORY_ROOT}"
    COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/../prepare_mtmd_graph_fixture.py"
    --source "${LCB_LLAMA_SOURCE}" --output "${CMAKE_CURRENT_BINARY_DIR}/fixture"
    COMMAND_ERROR_IS_FATAL ANY)
add_executable(mtmd-retained-models ../mtmd-retained-models.cpp ../mtmd-test-logger.cpp)
foreach(model llava qwen2vl qwen3vl siglip gemma4v gemma4uv lfm3vl)
    target_sources(mtmd-retained-models PRIVATE "${LCB_LLAMA_SOURCE}/tools/mtmd/models/${model}.cpp")
endforeach()
target_include_directories(mtmd-retained-models PRIVATE ../../../../runtimes/llama-cpp/native/vision "${CMAKE_CURRENT_BINARY_DIR}/fixture"
    "${LCB_LLAMA_SOURCE}/tools/mtmd" "${LCB_LLAMA_SOURCE}/include")
target_link_libraries(mtmd-retained-models PRIVATE ggml ggml-cpu)
foreach(model llava qwen2vl qwen3vl gemma3 lfm2 gemma4v gemma4uv lfm3vl)
    foreach(side 32 48)
        add_test(NAME "retained-model-${model}-${side}" COMMAND mtmd-retained-models "${model}" "${side}")
    endforeach()
endforeach()

# The real 12B projector dimensions, including the 70/1120-patch pixel limits.
# These allocate synthetic CPU-backed weights/graphs; no model is downloaded.
foreach(shape 480x480 768x768 480x768 768x480 480x336 1536x1680)
    string(REPLACE "x" ";" dims "${shape}")
    add_test(NAME "retained-model-gemma4uv-12b-${shape}"
        COMMAND mtmd-retained-models gemma4uv-12b ${dims})
endforeach()
