# Link the actual independently built core, so this also detects missing public
# entry points in the native check library.
add_executable(speech-api-contract api-contract.cpp wasm-probes.cpp)
target_include_directories(speech-api-contract PRIVATE "${SPCB_SOURCE}/include" "${SPCB_SOURCE}/src/common")
target_link_libraries(speech-api-contract PRIVATE core)
add_test(NAME speech-api-contract COMMAND speech-api-contract)

add_executable(speech-webgpu-matmul-layout webgpu-matmul-layout.cpp)
target_include_directories(speech-webgpu-matmul-layout PRIVATE
    "${SPCB_GGML_SOURCE}/include" "${SPCB_GGML_SOURCE}/src/ggml-webgpu")
target_link_libraries(speech-webgpu-matmul-layout PRIVATE ggml-cpu)
add_test(NAME speech-webgpu-matmul-layout COMMAND speech-webgpu-matmul-layout)

add_executable(speech-webgpu-buffer-planning webgpu-buffer-planning.cpp)
target_include_directories(speech-webgpu-buffer-planning PRIVATE
    "${SPCB_GGML_SOURCE}/include" "${SPCB_GGML_SOURCE}/src")
target_link_libraries(speech-webgpu-buffer-planning PRIVATE ggml-cpu)
add_test(NAME speech-webgpu-buffer-planning COMMAND speech-webgpu-buffer-planning)

add_executable(speech-gguf-large-file gguf-large-file.cpp)
target_compile_definitions(speech-gguf-large-file PRIVATE SPCB_GGUF_TEST_MAIN=1)
target_include_directories(speech-gguf-large-file PRIVATE "${SPCB_GGML_SOURCE}/include")
target_link_libraries(speech-gguf-large-file PRIVATE ggml-base)
add_test(NAME speech-gguf-large-file COMMAND speech-gguf-large-file)
