"""Reuse the checked independent-subgroup main for guarded ordinary Q6 matvecs."""
from pathlib import Path
import subprocess
from tests.support.llama_cpp.webgpu_stages import q8_subgroup
from tests.support.llama_cpp.webgpu_stages import tensor_copy
BASE = Path('ggml/src/ggml-webgpu')
PATCH_NAME = 'ggml-webgpu-q6-subgroup-matvec.patch'
PATCH_SHA256 = '9e8355c7cd862acb5bfe2c3e45909916141b1c0f4e57ad9aafd0d9de011f772a'
CONTRACT_INPUTS = dict(q8_subgroup.CONTRACT_INPUTS)
EMBEDDED_HEADER_NAME = q8_subgroup.HEADER_NAME
EMBEDDED_HEADER_SHA256 = '3db91d6d9aeef1df96cb9d55e8109001eb585c172b94d80d7556baaeae395a40'
INPUTS = {False: {'ggml-webgpu.cpp': 'd586843f24c47a12a29357c30b35b4d3f7b186f18af17bab222a52c5612a10c4',
         'ggml-webgpu-shader-lib.hpp': '87370c7273e37342a4e6ec580b3395c1a4c4e8c9e7e50d2a0e3b7fd1a4d2cb58'},
 True: {'ggml-webgpu.cpp': 'a62afc88802e9a84d80a8f03ac11ed50133f05841c25542d57672250f28474ad',
        'ggml-webgpu-shader-lib.hpp': '8543ff1b77cd69af4c0e8416ec5e89561b754f65a0b278df023cb20ee06cb1cc'}}
OUTPUTS = {False: {'ggml-webgpu.cpp': 'd586843f24c47a12a29357c30b35b4d3f7b186f18af17bab222a52c5612a10c4',
         'ggml-webgpu-shader-lib.hpp': 'd4690def8f004088ce3e24a9fb9845df61cf45af9a781b8359680526108531e2'},
 True: {'ggml-webgpu.cpp': 'a62afc88802e9a84d80a8f03ac11ed50133f05841c25542d57672250f28474ad',
        'ggml-webgpu-shader-lib.hpp': 'f27ce7e35d154ab701e7e122b2ab2b41409e853c42b3ec4591c3ccadd7cb1bde'}}


def prepare(work: Path, source: Path, patch_root: Path, env: dict[str, str], *, gpu_profile: bool) -> None:
    for name, expected in CONTRACT_INPUTS.items():
        if tensor_copy.digest(source / name) != expected:
            raise ValueError('Q6 subgroup matvec contract changed: ' + name)
    for name, expected in INPUTS[gpu_profile].items():
        if tensor_copy.digest(work / BASE / name) != expected:
            raise ValueError('Q6 subgroup matvec composition input changed: ' + name)
    patch = patch_root / PATCH_NAME
    if tensor_copy.digest(patch) != PATCH_SHA256:
        raise ValueError('Q6 subgroup matvec patch identity changed')
    for flags in (['--check'], []):
        subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags, str(patch.resolve())],
                       cwd=work, env=env, check=True)
    for name, expected in OUTPUTS[gpu_profile].items():
        if tensor_copy.digest(work / BASE / name) != expected:
            raise ValueError('Unexpected Q6 subgroup matvec output: ' + name)
