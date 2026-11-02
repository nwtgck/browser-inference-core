"""Reuse the existing short workgroups for ordinary single-column Q6_K matvecs."""
from pathlib import Path
import subprocess
import prepare_webgpu_q4_short as q4_short
import prepare_webgpu_tensor_copy as tensor_copy

BASE = Path('ggml/src/ggml-webgpu')
PATCH_NAME = 'ggml-webgpu-q6-short-workgroup.patch'
PATCH_SHA256 = '511182db3d0d0f64d49ec53188e815c34ed9da62bdc2aebbd862853f26f5095d'
# Pin the shared Q4/Q6 decoder, canonical main, preprocessor and host contracts.
CONTRACT_INPUTS = dict(q4_short.CONTRACT_INPUTS)
INPUTS = {False: {'ggml-webgpu.cpp': 'b16a18a3d544577fbfbc3279475743a5c4cd8c4074f7325d7e337f69cd616708',
         'ggml-webgpu-shader-lib.hpp': '2fddc2634df24710adb36a8a159d706a7da83468c306df96d6a287b3c6703b46'},
 True: {'ggml-webgpu.cpp': 'a9b0c18384ba901fc38b73652772f608b7ef221edd9d9f908116097255404e5b',
        'ggml-webgpu-shader-lib.hpp': 'e70e89859a4a853670155218adbaf8c7f946b5e07795859ec240b559f4ee5158'}}
OUTPUTS = {False: {'ggml-webgpu.cpp': 'b16a18a3d544577fbfbc3279475743a5c4cd8c4074f7325d7e337f69cd616708',
         'ggml-webgpu-shader-lib.hpp': '6c46dd6288d4a505ccfd5fd8aeb7c1c0880623e2a75919bfba8f08c58a6a72de'},
 True: {'ggml-webgpu.cpp': 'a9b0c18384ba901fc38b73652772f608b7ef221edd9d9f908116097255404e5b',
        'ggml-webgpu-shader-lib.hpp': '595d1f2aa85e5e46cfe23b72d39b613ba3254b9003e0adb7990ad1f1d4efde84'}}


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
                       cwd=work, env=env, check=True, capture_output=True)
    for name, expected in OUTPUTS[gpu_profile].items():
        if tensor_copy.digest(work / BASE / name) != expected:
            raise ValueError('Unexpected Q6 short matvec output identity: ' + name)
