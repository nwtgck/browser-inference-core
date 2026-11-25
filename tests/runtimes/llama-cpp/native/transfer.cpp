#include "support/webgpu-recorder.hpp"
#include "ggml-backend.cpp"
#include "ggml-webgpu.cpp"
#include "support/provider-context.hpp"
#include "support/transfer-backend.hpp"
#include "cases/scheduler.hpp"
#include "cases/fusion.hpp"
#include "cases/indexed-decode.hpp"
#include "cases/indexed-q2.hpp"
#include "cases/short-conv.hpp"
#include "cases/depth-mix.hpp"
#include "cases/situ.hpp"
#include "cases/loop-controller.hpp"
#include "cases/dense-q4-swiglu.hpp"
#ifdef GGML_WEBGPU_GPU_PROFILE
#include "cases/q8-diagnostic.hpp"
#endif

int main() {
    transfer_cases::all();
    fusion_cases::all();
    indexed_decode_cases::all();
    indexed_q2_cases::all();
    short_conv_cases::all();
    depth_mix_cases::all();
    situ_cases::all();
    loop_controller_cases::all();
    dense_q4_swiglu_cases::all();
#ifdef GGML_WEBGPU_GPU_PROFILE
    q8_diagnostic_cases::all();
#endif
    std::cout << "complete scheduler/provider contracts passed\n";
    return 0;
}
