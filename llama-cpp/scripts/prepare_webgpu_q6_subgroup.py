"""Reuse the checked independent-subgroup main for guarded ordinary Q6 matvecs."""
from pathlib import Path
import subprocess
import prepare_webgpu_q8_subgroup as q8_subgroup
import prepare_webgpu_tensor_copy as tensor_copy

BASE = Path('ggml/src/ggml-webgpu')
PATCH_NAME = 'ggml-webgpu-q6-subgroup-matvec.patch'
PATCH_SHA256 = '9e8355c7cd862acb5bfe2c3e45909916141b1c0f4e57ad9aafd0d9de011f772a'
CONTRACT_INPUTS = dict(q8_subgroup.CONTRACT_INPUTS)
EMBEDDED_HEADER_NAME = q8_subgroup.HEADER_NAME
EMBEDDED_HEADER_SHA256 = '3db91d6d9aeef1df96cb9d55e8109001eb585c172b94d80d7556baaeae395a40'
INPUTS = {False: {'ggml-webgpu.cpp': 'd9ca2bb081c5b9392b3c46ddfc7dfe8dd0fcf404f884c46a081303ce90d98f14',
         'ggml-webgpu-shader-lib.hpp': 'cc2447023811811fd492d99e9a90b9e83c3a974a2e604a644b77e9ea30177a6c'},
 True: {'ggml-webgpu.cpp': '02403383d69aa126976a6fe56289ffa4f0ebf63310c89dec2a2c8d96c953b567',
        'ggml-webgpu-shader-lib.hpp': 'f5a07a04ad663e748e25cfa5dde4a97b298609ce08b07b4ceaacd60618f0490e'}}
OUTPUTS = {False: {'ggml-webgpu.cpp': 'd9ca2bb081c5b9392b3c46ddfc7dfe8dd0fcf404f884c46a081303ce90d98f14',
         'ggml-webgpu-shader-lib.hpp': 'd0aceb8f9fb8f27697b659da7b83d982b572039d224cae249ab42168c549cd0c'},
 True: {'ggml-webgpu.cpp': '02403383d69aa126976a6fe56289ffa4f0ebf63310c89dec2a2c8d96c953b567',
        'ggml-webgpu-shader-lib.hpp': '44cf27c9d6770e2485417168ae3ef8e5b59b86eed64937492ffdad2ff8513973'}}


def prepare(work: Path, source: Path, patch_root: Path, env: dict[str, str], *, gpu_profile: bool) -> None:
    for name, expected in CONTRACT_INPUTS.items():
        if tensor_copy.digest(source / name) != expected:
            raise ValueError('Q6 subgroup matvec contract changed: ' + name)
    for name, expected in INPUTS[gpu_profile].items():
        if tensor_copy.digest(work / BASE / name) != expected:
            raise ValueError('Q6 subgroup matvec composition input changed: ' + name)
    if tensor_copy.digest(work / BASE / EMBEDDED_HEADER_NAME) != EMBEDDED_HEADER_SHA256:
        raise ValueError('Q6 subgroup matvec shared embedding changed')
    patch = patch_root / PATCH_NAME
    if tensor_copy.digest(patch) != PATCH_SHA256:
        raise ValueError('Q6 subgroup matvec patch identity changed')
    for flags in (['--check'], []):
        subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags, str(patch.resolve())],
                       cwd=work, env=env, check=True, capture_output=True)
    for name, expected in OUTPUTS[gpu_profile].items():
        if tensor_copy.digest(work / BASE / name) != expected:
            raise ValueError('Unexpected Q6 subgroup matvec output: ' + name)
