"""Select eight-row paired-half register tiles for guarded Q4 prefill matrices."""
from pathlib import Path
import subprocess
import prepare_webgpu_quant_half_pairs as half_pairs
import prepare_webgpu_tensor_copy as tensor_copy

BASE = Path('ggml/src/ggml-webgpu')
PATCH_NAME = 'ggml-webgpu-quant-wide-m.patch'
CONTRACT_INPUTS = dict(half_pairs.CONTRACT_INPUTS)
EMBEDDED_HEADER_SHA256 = half_pairs.HEADER_SHA256

PATCH_SHA256 = '18c5f6fd4d4f003f6245bd7a7dabda2e6cc4c82fe6f06105c148ff8effbd7a7d'
INPUTS = {False: {'ggml-webgpu.cpp': 'ef3e19338638464151e5416591e435deb62cb0d4974be45e61e4b8ec7bba0335',
         'ggml-webgpu-shader-lib.hpp': 'd7fadf561f5e02020c0aee0b12300ca3aad712bd2c62c40cb01392b74bf76091'},
 True: {'ggml-webgpu.cpp': '9a6c93ab8c9bfb1736224e5943e3eb899e83aa4cbdaeacd9ff6f533158527791',
        'ggml-webgpu-shader-lib.hpp': '9071a656fae93afcfd97e12de8a4e328a1b9b7145bddf053a414996bf4215ed3'}}
OUTPUTS = {False: {'ggml-webgpu.cpp': 'ef3e19338638464151e5416591e435deb62cb0d4974be45e61e4b8ec7bba0335',
         'ggml-webgpu-shader-lib.hpp': '83c67332bdaa64acec85bd6fc54aa49ee03633b6a519774ca6727b5a3856d7d4'},
 True: {'ggml-webgpu.cpp': '9a6c93ab8c9bfb1736224e5943e3eb899e83aa4cbdaeacd9ff6f533158527791',
        'ggml-webgpu-shader-lib.hpp': 'eab1d1816f9842022770e54248df34abb8708648f9531e1f5c1a899982fcab11'}}



def prepare(work: Path, source: Path, patch_root: Path, env: dict[str, str], *, gpu_profile: bool) -> None:
    for name, expected in CONTRACT_INPUTS.items():
        if tensor_copy.digest(source / name) != expected:
            raise ValueError('Quantized wide-M contract changed: ' + name)
    for name, expected in INPUTS[gpu_profile].items():
        if tensor_copy.digest(work / BASE / name) != expected:
            raise ValueError('Quantized wide-M composition input changed: ' + name)
    if tensor_copy.digest(work / half_pairs.HEADER_NAME) != EMBEDDED_HEADER_SHA256:
        raise ValueError('Quantized wide-M paired embedding changed')
    patch = patch_root / PATCH_NAME
    if tensor_copy.digest(patch) != PATCH_SHA256:
        raise ValueError('Quantized wide-M patch identity changed')
    for flags in (['--check'], []):
        subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags, str(patch.resolve())],
                       cwd=work, env=env, check=True, capture_output=True)
    for name, expected in OUTPUTS[gpu_profile].items():
        if tensor_copy.digest(work / BASE / name) != expected:
            raise ValueError('Unexpected quantized wide-M output: ' + name)
