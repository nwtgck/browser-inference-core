"""Current Q4 factory/dispatcher; historical byte guards stay in separate tests."""
from build_tools.project import runtime_test_root
from tests.support.llama_cpp.current_matvec import build_complete_matvec_recorder


def build_host(upstream, prepared, output):
    return build_complete_matvec_recorder(upstream, prepared, output,
        case_directory=runtime_test_root('llama-cpp') / 'webgpu-q4-short',
        dispatch_functions=('ggml_webgpu_u32_from_f32', 'compute_2d_workgroups',
                            'ggml_webgpu_mul_mat_vec4_aligned', 'ggml_webgpu_mul_mat_id_vec',
                            'ggml_webgpu_mul_mat'),
        case_files=('dispatch-cases.cpp', 'host-cases.cpp'))
