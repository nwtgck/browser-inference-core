# Link the actual independently built core, so this also detects missing public
# entry points in the native check library.
add_executable(speech-api-contract api-contract.cpp wasm-probes.cpp)
target_include_directories(speech-api-contract PRIVATE "${SPCB_SOURCE}/include" "${SPCB_SOURCE}/src/common")
target_link_libraries(speech-api-contract PRIVATE core)
add_test(NAME speech-api-contract COMMAND speech-api-contract)
