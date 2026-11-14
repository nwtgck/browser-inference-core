cmake_minimum_required(VERSION 3.20)
project(bicore_abi_fixture LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_executable(fixture
    "${COMPILE_ROOT}/runtimes/llama-cpp/native/abi/metadata.cpp"
    "${COMPILE_ROOT}/runtimes/llama-cpp/native/abi/bridge.cpp")
target_include_directories(fixture PRIVATE "${CMAKE_BINARY_DIR}/generated")
target_compile_features(fixture PRIVATE cxx_std_17)
target_compile_options(fixture PRIVATE "-sMEMORY64=${POINTER64}")
target_link_options(fixture PRIVATE
    "-sMEMORY64=${POINTER64}" "-sEXPORTED_FUNCTIONS=@${CMAKE_BINARY_DIR}/generated/exports.json"
    "-sEXPORTED_RUNTIME_METHODS=HEAPU8,ccall,addFunction,removeFunction" "-sALLOW_TABLE_GROWTH=1"
    "-sMODULARIZE=1" "-sEXPORT_ES6=1" "-sENVIRONMENT=node" "-sWASM_BIGINT=1")
if(SUSPENSION STREQUAL "asyncify")
    target_link_options(fixture PRIVATE "-sASYNCIFY=1")
elseif(SUSPENSION STREQUAL "jspi")
    target_link_options(fixture PRIVATE "-sJSPI=1" "-sJSPI_EXPORTS=@${CMAKE_BINARY_DIR}/generated/jspi-exports.json")
endif()
set_target_properties(fixture PROPERTIES SUFFIX ".mjs")
