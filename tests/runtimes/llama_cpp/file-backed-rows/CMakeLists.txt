cmake_minimum_required(VERSION 3.22)
project(file_backed_rows_tests LANGUAGES C CXX)
if(NOT LCB_LLAMA_SOURCE OR NOT LCB_NATIVE_BUILD)
    message(FATAL_ERROR "Set LCB_LLAMA_SOURCE and LCB_NATIVE_BUILD to a configured native core build with LCB_FILE_BACKED_PLE=ON")
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
add_library(core_under_test SHARED IMPORTED)
set_target_properties(core_under_test PROPERTIES IMPORTED_LOCATION "${LCB_NATIVE_BUILD}/libcore.so")
add_executable(file-rows-reader reader.cpp)
target_include_directories(file-rows-reader PRIVATE ../../../../runtimes/llama-cpp/native/io "${LCB_LLAMA_SOURCE}/ggml/include")
target_link_libraries(file-rows-reader PRIVATE core_under_test)
# Keep fault injection test-only; the production reader calls the real pread.
target_link_options(file-rows-reader PRIVATE "-Wl,--wrap=pread")
add_executable(file-rows-identity identity.cpp)
target_include_directories(file-rows-identity PRIVATE ../../../../runtimes/llama-cpp/native/io)
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
add_executable(file-rows-loader-identity loader-identity.cpp)
get_target_property(LCB_FILE_ROWS_TEST_INCLUDES file-rows-model INCLUDE_DIRECTORIES)
get_target_property(LCB_FILE_ROWS_TEST_LIBRARIES file-rows-model LINK_LIBRARIES)
target_include_directories(file-rows-loader-identity PRIVATE ../../../../runtimes/llama-cpp/native/io ${LCB_FILE_ROWS_TEST_INCLUDES})
target_link_libraries(file-rows-loader-identity PRIVATE ${LCB_FILE_ROWS_TEST_LIBRARIES})
enable_testing()
add_test(NAME reader COMMAND file-rows-reader)
add_test(NAME identity COMMAND file-rows-identity "${CMAKE_CURRENT_BINARY_DIR}/identity-files")
add_test(NAME loader-identity COMMAND file-rows-loader-identity "${CMAKE_CURRENT_BINARY_DIR}/loader-identity-files")
add_test(NAME model COMMAND file-rows-model "${CMAKE_CURRENT_BINARY_DIR}")
