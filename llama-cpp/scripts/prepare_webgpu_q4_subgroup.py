"""Reuse the checked independent-subgroup main for guarded ordinary Q4 matvecs."""
from pathlib import Path
import subprocess
import prepare_webgpu_q8_subgroup as q8_subgroup
import prepare_webgpu_tensor_copy as tensor_copy

BASE = Path('ggml/src/ggml-webgpu')
PATCH_NAME = 'ggml-webgpu-q4-subgroup-matvec.patch'
PATCH_SHA256 = '075c6d739005df296e3e4d013f3f4c710409099b377aca58be2bd2f29e889b8d'
CONTRACT_INPUTS = dict(q8_subgroup.CONTRACT_INPUTS)
EMBEDDED_HEADER_NAME = q8_subgroup.HEADER_NAME
EMBEDDED_HEADER_SHA256 = q8_subgroup.HEADER_SHA256
INPUTS = {
    False: {'ggml-webgpu.cpp': 'ef3e19338638464151e5416591e435deb62cb0d4974be45e61e4b8ec7bba0335',
            'ggml-webgpu-shader-lib.hpp': '83c67332bdaa64acec85bd6fc54aa49ee03633b6a519774ca6727b5a3856d7d4'},
    True: {'ggml-webgpu.cpp': '9a6c93ab8c9bfb1736224e5943e3eb899e83aa4cbdaeacd9ff6f533158527791',
           'ggml-webgpu-shader-lib.hpp': 'eab1d1816f9842022770e54248df34abb8708648f9531e1f5c1a899982fcab11'},
}
OUTPUTS = {
    False: {'ggml-webgpu.cpp': 'ef3e19338638464151e5416591e435deb62cb0d4974be45e61e4b8ec7bba0335',
            'ggml-webgpu-shader-lib.hpp': 'd05738a29d0629d60be93224b1ba69c16d67d97300864c058730bc64c4c73297'},
    True: {'ggml-webgpu.cpp': '9a6c93ab8c9bfb1736224e5943e3eb899e83aa4cbdaeacd9ff6f533158527791',
           'ggml-webgpu-shader-lib.hpp': 'c9a8bc051f56126606327e8fd137a8a2e7374daa0490c893a3036182b6b98462'},
}


def prepare(work: Path, source: Path, patch_root: Path, env: dict[str, str], *, gpu_profile: bool) -> None:
    for name, expected in CONTRACT_INPUTS.items():
        if tensor_copy.digest(source / name) != expected:
            raise ValueError('Q4 subgroup matvec contract changed: ' + name)
    for name, expected in INPUTS[gpu_profile].items():
        if tensor_copy.digest(work / BASE / name) != expected:
            raise ValueError('Q4 subgroup matvec composition input changed: ' + name)
    if tensor_copy.digest(work / EMBEDDED_HEADER_NAME) != EMBEDDED_HEADER_SHA256:
        raise ValueError('Q4 subgroup matvec shared embedding changed')
    patch = patch_root / PATCH_NAME
    if tensor_copy.digest(patch) != PATCH_SHA256:
        raise ValueError('Q4 subgroup matvec patch identity changed')
    for flags in (['--check'], []):
        subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags, str(patch.resolve())],
                       cwd=work, env=env, check=True, capture_output=True)
    for name, expected in OUTPUTS[gpu_profile].items():
        if tensor_copy.digest(work / BASE / name) != expected:
            raise ValueError('Unexpected Q4 subgroup matvec output: ' + name)
