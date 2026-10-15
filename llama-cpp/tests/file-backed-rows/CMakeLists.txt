cmake_minimum_required(VERSION 3.22)
project(file_backed_rows_tests LANGUAGES C CXX)
if(NOT LCB_LLAMA_SOURCE OR NOT LCB_NATIVE_BUILD)
    message(FATAL_ERROR "Set LCB_LLAMA_SOURCE and LCB_NATIVE_BUILD to a configured native core build with LCB_FILE_BACKED_PLE=ON")
endif()
set(CMAKE_CXX_STANDARD 17)
add_library(core_under_test SHARED IMPORTED)
set_target_properties(core_under_test PROPERTIES IMPORTED_LOCATION "${LCB_NATIVE_BUILD}/libcore.so")
add_executable(file-rows-reader reader.cpp)
target_include_directories(file-rows-reader PRIVATE ../../bridge "${LCB_LLAMA_SOURCE}/ggml/include")
target_link_libraries(file-rows-reader PRIVATE core_under_test)
# Keep fault injection test-only; the production reader calls the real pread.
target_link_options(file-rows-reader PRIVATE "-Wl,--wrap=pread")
add_executable(file-rows-model model.cpp)
target_include_directories(file-rows-model PRIVATE "${LCB_LLAMA_SOURCE}/include" "${LCB_LLAMA_SOURCE}/common"
    "${LCB_LLAMA_SOURCE}/ggml/include" "${LCB_LLAMA_SOURCE}/src" "${LCB_LLAMA_SOURCE}/tests")
target_link_libraries(file-rows-model PRIVATE -Wl,--start-group
    "${LCB_NATIVE_BUILD}/llama/common/libllama-common.a"
    "${LCB_NATIVE_BUILD}/llama/common/libllama-common-base.a"
    "${LCB_NATIVE_BUILD}/llama/src/libllama.a"
    "${LCB_NATIVE_BUILD}/llama/ggml/src/libggml.a"
    "${LCB_NATIVE_BUILD}/llama/ggml/src/libggml-cpu.a"
    "${LCB_NATIVE_BUILD}/llama/ggml/src/libggml-base.a"
    "${LCB_NATIVE_BUILD}/llama/vendor/cpp-httplib/libcpp-httplib.a"
    "${LCB_NATIVE_BUILD}/llama/vendor/hash/libvendor-hash.a"
    -Wl,--end-group dl m pthread)
enable_testing()
add_test(NAME reader COMMAND file-rows-reader)
add_test(NAME model COMMAND file-rows-model "${CMAKE_CURRENT_BINARY_DIR}")
