"""Avoid Emdawn's temporary mapped copy for aligned tensor readback."""
from pathlib import Path
import subprocess
import prepare_webgpu_ordered_readback as ordered_readback
import prepare_webgpu_tensor_copy as tensor_copy

BASE = Path('ggml/src/ggml-webgpu')
PATCH_NAME = 'ggml-webgpu-direct-mapped-read.patch'
PATCH_SHA256 = 'a30e41de9211e13e34f7241731e3ed17330d0094762513cc77c08e6e54e84621'
CONTRACT_INPUTS = dict(ordered_readback.CONTRACT_INPUTS)
# The existing source-preparation CLI verifies these pinned API headers.
# ReadMappedRange is used only by Emdawn; native mapping remains unchanged.
REVIEWED_DAWN_HEADERS = dict(tensor_copy.DAWN_HEADERS)
INPUTS = {False: {'ggml-webgpu.cpp': 'b16a18a3d544577fbfbc3279475743a5c4cd8c4074f7325d7e337f69cd616708',
                 'ggml-webgpu-shader-lib.hpp': 'd7fadf561f5e02020c0aee0b12300ca3aad712bd2c62c40cb01392b74bf76091'},
          True: {'ggml-webgpu.cpp': 'a9b0c18384ba901fc38b73652772f608b7ef221edd9d9f908116097255404e5b',
                 'ggml-webgpu-shader-lib.hpp': '9071a656fae93afcfd97e12de8a4e328a1b9b7145bddf053a414996bf4215ed3'}}
OUTPUTS = {False: {'ggml-webgpu.cpp': 'ef3e19338638464151e5416591e435deb62cb0d4974be45e61e4b8ec7bba0335',
                  'ggml-webgpu-shader-lib.hpp': 'd7fadf561f5e02020c0aee0b12300ca3aad712bd2c62c40cb01392b74bf76091'},
           True: {'ggml-webgpu.cpp': '9a6c93ab8c9bfb1736224e5943e3eb899e83aa4cbdaeacd9ff6f533158527791',
                  'ggml-webgpu-shader-lib.hpp': '9071a656fae93afcfd97e12de8a4e328a1b9b7145bddf053a414996bf4215ed3'}}


def prepare(work: Path, source: Path, patch_root: Path, env: dict[str, str], *, gpu_profile: bool) -> None:
    for name, expected in CONTRACT_INPUTS.items():
        if tensor_copy.digest(source / name) != expected:
            raise ValueError('Direct mapped read contract changed: ' + name)
    for name, expected in INPUTS[gpu_profile].items():
        if tensor_copy.digest(work / BASE / name) != expected:
            raise ValueError('Direct mapped read composition input changed: ' + name)
    patch = patch_root / PATCH_NAME
    if tensor_copy.digest(patch) != PATCH_SHA256:
        raise ValueError('Direct mapped read patch identity changed')
    for flags in (['--check'], []):
        subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags, str(patch.resolve())],
                       cwd=work, env=env, check=True, capture_output=True)
    for name, expected in OUTPUTS[gpu_profile].items():
        if tensor_copy.digest(work / BASE / name) != expected:
            raise ValueError('Unexpected direct mapped read output: ' + name)
