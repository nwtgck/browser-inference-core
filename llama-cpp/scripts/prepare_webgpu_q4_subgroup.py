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
INPUTS = {False: {'ggml-webgpu.cpp': 'c977f258c8ec2b98e07f69388e1bc22a00646f76c559ce5afee533d7ff6a0c76',
         'ggml-webgpu-shader-lib.hpp': '62e7700902bbc719cc57d3f6dc91ce4aa6dcec89b1eaf0a7af803298ec243556'},
 True: {'ggml-webgpu.cpp': '62e142cdac82c751f3c758e264c3750ea420429f58a0fa883414322756aa7268',
        'ggml-webgpu-shader-lib.hpp': '221509716c3930422fd820248fa19b0fbd935a3838e0ec772ca62ce019945364'}}
OUTPUTS = {False: {'ggml-webgpu.cpp': 'c977f258c8ec2b98e07f69388e1bc22a00646f76c559ce5afee533d7ff6a0c76',
         'ggml-webgpu-shader-lib.hpp': '7c10f05f5e67be55f5435026aeaf6ed19b89762600318ceae223f846657e9bdb'},
 True: {'ggml-webgpu.cpp': '62e142cdac82c751f3c758e264c3750ea420429f58a0fa883414322756aa7268',
        'ggml-webgpu-shader-lib.hpp': 'ba830e5dda312af69d63320d1297c652a9c01b55a3ddc95cecbf02204d40d6c6'}}


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
