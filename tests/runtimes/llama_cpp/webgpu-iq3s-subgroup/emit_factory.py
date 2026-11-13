"""Current IQ3 factory/dispatcher; frozen prefix checks live in historical_iq3."""
from pathlib import Path
import os

from build_tools.project import runtime_test_root
from build_tools.runtimes.llama_cpp import webgpu_plan
from tests.support.llama_cpp.current_matvec import build_complete_matvec_recorder

HERE = runtime_test_root('llama-cpp') / 'webgpu-iq3s-subgroup'


def prepare(upstream: Path, output: Path, *, gpu_profile=False):
    if os.environ.get('LCB_TEST_IQ3S_BASE'):
        raise ValueError('Current IQ3_S checks reject historical prefixes')
    return webgpu_plan.prepare(upstream, output, gpu_profile=gpu_profile)


def build_host(upstream: Path, prepared: Path, output: Path) -> Path:
    return build_complete_matvec_recorder(upstream, prepared, output,
        case_directory=HERE,
        dispatch_functions=('compute_2d_workgroups', 'ggml_webgpu_mul_mat_vec4_aligned',
                            'ggml_webgpu_mul_mat_id_vec'))
