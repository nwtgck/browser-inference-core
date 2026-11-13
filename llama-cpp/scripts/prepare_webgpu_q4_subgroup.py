"""Reuse the checked independent-subgroup main for guarded ordinary Q4 matvecs."""
from pathlib import Path
import subprocess
import prepare_webgpu_q8_subgroup as q8_subgroup
import prepare_webgpu_tensor_copy as tensor_copy

BASE = Path('ggml/src/ggml-webgpu')
PATCH_NAME = 'ggml-webgpu-q4-subgroup-matvec.patch'
PATCH_SHA256 = '40823441227f42dab4e9ae1be2c61fd0fa46c693ca258a385cd3734768070952'
CONTRACT_INPUTS = dict(q8_subgroup.CONTRACT_INPUTS)
EMBEDDED_HEADER_NAME = q8_subgroup.HEADER_NAME
EMBEDDED_HEADER_SHA256 = q8_subgroup.HEADER_SHA256
INPUTS = {False: {'ggml-webgpu.cpp': '23873d267578c6bce435eff0f7643f2853f705ad0b46b3ed8e93cbaada2bb8a2',
         'ggml-webgpu-shader-lib.hpp': '96e949d9e6064f82e2d63111fac3ce1240a0dfcab42e1266628eb26fe8d997fd'},
 True: {'ggml-webgpu.cpp': 'de513ccb952c3cc1da2e42f45f4ed1acd9433f6bdea3e38a61f951f437b9c79c',
        'ggml-webgpu-shader-lib.hpp': '2e12ecc4f59bf9ca594b21ee3240642f194ba8ea6232054e1cef09dad5b1ca1e'}}
OUTPUTS = {False: {'ggml-webgpu.cpp': '23873d267578c6bce435eff0f7643f2853f705ad0b46b3ed8e93cbaada2bb8a2',
         'ggml-webgpu-shader-lib.hpp': '62ee9b9d14abec20066aad94296440cbaa48df8d4a9d38aedfdb61dbd7616e2a'},
 True: {'ggml-webgpu.cpp': 'de513ccb952c3cc1da2e42f45f4ed1acd9433f6bdea3e38a61f951f437b9c79c',
        'ggml-webgpu-shader-lib.hpp': 'f3b638a2f2572e2a64e76270753941db056a7a0b48fec6603525af898a379bb8'}}


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
