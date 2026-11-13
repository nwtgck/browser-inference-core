"""Reuse the existing short workgroups for ordinary single-column Q6_K matvecs."""
from pathlib import Path
import subprocess
from tests.support.llama_cpp.webgpu_stages import q4_short
from tests.support.llama_cpp.webgpu_stages import tensor_copy
BASE = Path('ggml/src/ggml-webgpu')
PATCH_NAME = 'ggml-webgpu-q6-short-workgroup.patch'
PATCH_SHA256 = '511182db3d0d0f64d49ec53188e815c34ed9da62bdc2aebbd862853f26f5095d'
# Pin the shared Q4/Q6 decoder, canonical main, preprocessor and host contracts.
CONTRACT_INPUTS = dict(q4_short.CONTRACT_INPUTS)
INPUTS = {False: {'ggml-webgpu.cpp': 'c50aeea2606f5da330e6b5f2d26115b94579acd87bb90ae3497415879b90a100',
         'ggml-webgpu-shader-lib.hpp': '1c03edf405f8c645a4ef834a0bf5da41ec868037461092765bd33a8d1baf76f0'},
 True: {'ggml-webgpu.cpp': 'f7c1281458464c910ebbbbb9f7e679a20580498fdc7ce9cff5000f1f6dde1607',
        'ggml-webgpu-shader-lib.hpp': 'ab81a43f65058fd9ae02963d98c42da8a08fd1b7483d6f82fc89c2d0c34d6bc6'}}
OUTPUTS = {False: {'ggml-webgpu.cpp': 'c50aeea2606f5da330e6b5f2d26115b94579acd87bb90ae3497415879b90a100',
         'ggml-webgpu-shader-lib.hpp': '32183874f4ecc5091539271971ab9e61353d35afc28972f251a4c21aebe10357'},
 True: {'ggml-webgpu.cpp': 'f7c1281458464c910ebbbbb9f7e679a20580498fdc7ce9cff5000f1f6dde1607',
        'ggml-webgpu-shader-lib.hpp': '212ac7375483581d7ef6515d03c0400b41dbe4b823882feadfe2cce5526a0995'}}


def prepare(work: Path, source: Path, patch_root: Path, env: dict[str, str], *, gpu_profile: bool) -> None:
    for relative, expected in CONTRACT_INPUTS.items():
        if tensor_copy.digest(source / relative) != expected:
            raise ValueError('Q6 short matvec contract changed: ' + relative)
    for name, expected in INPUTS[gpu_profile].items():
        if tensor_copy.digest(work / BASE / name) != expected:
            raise ValueError('Q6 short matvec composition input changed: ' + name)
    patch = patch_root / PATCH_NAME
    if tensor_copy.digest(patch) != PATCH_SHA256:
        raise ValueError('Q6 short matvec patch identity changed')
    for flags in (['--check'], []):
        subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags, str(patch.resolve())],
                       cwd=work, env=env, check=True)
    for name, expected in OUTPUTS[gpu_profile].items():
        if tensor_copy.digest(work / BASE / name) != expected:
            raise ValueError('Unexpected Q6 short matvec output identity: ' + name)
