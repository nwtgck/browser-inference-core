"""Reuse the checked independent-subgroup main for guarded ordinary Q6 matvecs."""
from pathlib import Path
import subprocess
import prepare_webgpu_q8_subgroup as q8_subgroup
import prepare_webgpu_tensor_copy as tensor_copy

BASE = Path('ggml/src/ggml-webgpu')
PATCH_NAME = 'ggml-webgpu-q6-subgroup-matvec.patch'
PATCH_SHA256 = 'ac336e53a6d1eab0bd60f85de777a1366c7b06bdfeaf492ebdf3ea17bf24349f'
CONTRACT_INPUTS = dict(q8_subgroup.CONTRACT_INPUTS)
EMBEDDED_HEADER_NAME = q8_subgroup.HEADER_NAME
EMBEDDED_HEADER_SHA256 = '3db91d6d9aeef1df96cb9d55e8109001eb585c172b94d80d7556baaeae395a40'
INPUTS = {False: {'ggml-webgpu.cpp': '708de1eb3c267527d65cdb097a06d2ae71437f2ce4880071de42acbe098ffc55',
         'ggml-webgpu-shader-lib.hpp': '4947500704450e61424926a34ea69346b70900944896f5e7495eeed0b4d69725'},
 True: {'ggml-webgpu.cpp': '0d7f1dcb2ad06124a83fa6c33f2b388418268ba0d26d6d12262ccb299b860445',
        'ggml-webgpu-shader-lib.hpp': '91eb008a1ca86b10143027e5080b4ca0469512d3625d4028984e11e89e3cf716'}}
OUTPUTS = {False: {'ggml-webgpu.cpp': '708de1eb3c267527d65cdb097a06d2ae71437f2ce4880071de42acbe098ffc55',
         'ggml-webgpu-shader-lib.hpp': '8398034ce17fe5622ed62398d5648f4beac9d450ce1bad6deff65c89aacf479b'},
 True: {'ggml-webgpu.cpp': '0d7f1dcb2ad06124a83fa6c33f2b388418268ba0d26d6d12262ccb299b860445',
        'ggml-webgpu-shader-lib.hpp': '9664af512eb0a281c0453b783ee84e95341f0fb45b95d8bbb72755b3ed581645'}}


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
