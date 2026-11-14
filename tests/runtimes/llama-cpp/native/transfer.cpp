#include "support/webgpu-recorder.hpp"
#include "ggml-backend.cpp"
#include "ggml-webgpu.cpp"
#include "support/provider-context.hpp"
#include "support/transfer-backend.hpp"
#include "cases/scheduler.hpp"
#include "cases/fusion.hpp"
#ifdef GGML_WEBGPU_GPU_PROFILE
#include "cases/q8-diagnostic.hpp"
#endif

int main() {
    transfer_cases::all();
    fusion_cases::all();
#ifdef GGML_WEBGPU_GPU_PROFILE
    q8_diagnostic_cases::all();
#endif
    std::cout << "complete scheduler/provider contracts passed\n";
}
