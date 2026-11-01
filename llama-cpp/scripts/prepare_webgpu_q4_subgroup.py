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
INPUTS = {False: {'ggml-webgpu.cpp': 'ef3e19338638464151e5416591e435deb62cb0d4974be45e61e4b8ec7bba0335',
         'ggml-webgpu-shader-lib.hpp': 'd0bba4fa2d66e35a4bb77d4af9b68e315110fe145187dd2cb3522f11e2c14b59'},
 True: {'ggml-webgpu.cpp': '9a6c93ab8c9bfb1736224e5943e3eb899e83aa4cbdaeacd9ff6f533158527791',
        'ggml-webgpu-shader-lib.hpp': 'f529d2dede71abcad3b4d4aa935c3be35ab066301220d556b838cc0e43c0a749'}}
OUTPUTS = {False: {'ggml-webgpu.cpp': 'ef3e19338638464151e5416591e435deb62cb0d4974be45e61e4b8ec7bba0335',
         'ggml-webgpu-shader-lib.hpp': '03794a383311b0944469f3e0d7fda2270dba87fb50e08f9dbc34c27a91e8d764'},
 True: {'ggml-webgpu.cpp': '9a6c93ab8c9bfb1736224e5943e3eb899e83aa4cbdaeacd9ff6f533158527791',
        'ggml-webgpu-shader-lib.hpp': '3456af3ac67cc080ca03043d8c8add03808bbaeb330c2573cd275f2900290ee4'}}


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
