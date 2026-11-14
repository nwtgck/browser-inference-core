#include "support/webgpu-recorder.hpp"
#include "ggml-webgpu.cpp"
#include "cases/readback.hpp"

int main() {
    readback_cases::scalar();
    readback_cases::batch();
    std::cout
        << "complete provider/public ggml readback: byte ranges, waits, bounded batches and destination lifetime passed\n";
}
