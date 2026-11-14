cmake_minimum_required(VERSION 3.24)
project(qsa_selection_tests LANGUAGES C CXX)
if(NOT LCB_LLAMA_SOURCE OR NOT LCB_NATIVE_BUILD)
    message(FATAL_ERROR "Set LCB_LLAMA_SOURCE and LCB_NATIVE_BUILD to a native core build")
endif()
# Use the same complete source/header tree as the linked native library.
# A pristine models.h cannot describe the file-backed model layout.
if(NOT EXISTS "${LCB_NATIVE_BUILD}/prepared-source-root.txt")
    message(FATAL_ERROR "LCB_NATIVE_BUILD must identify a core built from prepared sources")
endif()
file(READ "${LCB_NATIVE_BUILD}/prepared-source-root.txt" LCB_LLAMA_SOURCE)
string(STRIP "${LCB_LLAMA_SOURCE}" LCB_LLAMA_SOURCE)
if(NOT EXISTS "${LCB_LLAMA_SOURCE}/src/models/file-backed-rows.h")
    message(FATAL_ERROR "Native tests require the coherent file-backed source/header tree")
endif()
set(CMAKE_CXX_STANDARD 17)
add_executable(qsa-model model.cpp)
target_include_directories(qsa-model PRIVATE "${LCB_LLAMA_SOURCE}/include" "${LCB_LLAMA_SOURCE}/common"
    "${LCB_LLAMA_SOURCE}/ggml/include" "${LCB_LLAMA_SOURCE}/src" "${LCB_LLAMA_SOURCE}/tests")
target_link_libraries(qsa-model PRIVATE -Wl,--start-group
    "${LCB_NATIVE_BUILD}/llama/common/libllama-common.a"
    "${LCB_NATIVE_BUILD}/llama/common/libllama-common-base.a"
    "${LCB_NATIVE_BUILD}/llama/src/libllama.a"
    "${LCB_NATIVE_BUILD}/llama/ggml/src/libggml.a"
    "${LCB_NATIVE_BUILD}/llama/ggml/src/libggml-cpu.a"
    "${LCB_NATIVE_BUILD}/llama/ggml/src/libggml-base.a"
    "${LCB_NATIVE_BUILD}/llama/vendor/cpp-httplib/libcpp-httplib.a"
    "${LCB_NATIVE_BUILD}/llama/vendor/hash/libvendor-hash.a"
    -Wl,--end-group dl m pthread)
