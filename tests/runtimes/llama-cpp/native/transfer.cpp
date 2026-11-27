#include "support/webgpu-recorder.hpp"
#include "ggml-backend.cpp"
#include "ggml-webgpu.cpp"
#include "support/provider-context.hpp"
#include "support/transfer-backend.hpp"
#include "cases/scheduler.hpp"
#include "cases/fusion.hpp"
#include "cases/raw-view-fusions.hpp"
#include "cases/indexed-decode.hpp"
#include "cases/indexed-live-grid.hpp"
#include "cases/indexed-q2.hpp"
#include "cases/indexed-weight-windows.hpp"
#include "cases/expert-storage.hpp"
#include "cases/short-conv.hpp"
#include "cases/short-conv-dynamic.hpp"
#include "cases/ssm-conv-gate.hpp"
#include "cases/depth-mix.hpp"
#include "cases/situ.hpp"
#include "cases/loop-controller.hpp"
#include "cases/dense-q4-swiglu.hpp"
#include "cases/q4-geglu.hpp"
#include "cases/q8-sigmoid-mul.hpp"
#include "cases/fwht-1024.hpp"
#include "cases/add-cast-f16.hpp"
#include "cases/flash-mask-binding.hpp"
#ifdef GGML_WEBGPU_GPU_PROFILE
#include "cases/q8-diagnostic.hpp"
#endif

int main() {
    transfer_cases::all();
    fusion_cases::all();
    raw_view_fusion_cases::all();
    indexed_decode_cases::all();
    indexed_live_grid_cases::all();
    indexed_q2_cases::all();
    indexed_weight_window_cases::all();
    expert_storage_cases::all();
    short_conv_cases::all();
    short_conv_dynamic_cases::all();
    ssm_conv_gate_cases::all();
    depth_mix_cases::all();
    situ_cases::all();
    loop_controller_cases::all();
    dense_q4_swiglu_cases::all();
    q4_geglu_cases::all();
    q8_sigmoid_mul_cases::all();
    fwht_1024_cases::all();
    add_cast_f16_cases::all();
    flash_mask_binding_cases::all();
#ifdef GGML_WEBGPU_GPU_PROFILE
    q8_diagnostic_cases::all();
#endif
    std::cout << "complete scheduler/provider contracts passed\n";
    return 0;
}
