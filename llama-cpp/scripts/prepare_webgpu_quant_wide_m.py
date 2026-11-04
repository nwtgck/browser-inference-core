"""Select eight-row paired-half register tiles for guarded Q4 prefill matrices."""
from pathlib import Path
import subprocess
import prepare_webgpu_quant_half_pairs as half_pairs
import prepare_webgpu_tensor_copy as tensor_copy

BASE = Path('ggml/src/ggml-webgpu')
PATCH_NAME = 'ggml-webgpu-quant-wide-m.patch'
CONTRACT_INPUTS = dict(half_pairs.CONTRACT_INPUTS)
EMBEDDED_HEADER_SHA256 = half_pairs.HEADER_SHA256

PATCH_SHA256 = '385acda3e874eebfeac26fc03259dec49fc3579a59e0ac76f55e7a86d8d67686'
INPUTS = {False: {'ggml-webgpu.cpp': '34e8d902c07a8770ae473e082b01aa592b5cae486d1541cc4b0a0dfac16c1941',
         'ggml-webgpu-shader-lib.hpp': '341249f8670c2266f53b092e50306a7d73abacb1faf6e1787002d7386560885b'},
 True: {'ggml-webgpu.cpp': '08663d3d2dde4c0345fbeb984d4e73950628365ec1e56321f873957382db7fe2',
        'ggml-webgpu-shader-lib.hpp': '7ad71c726623e5b65e6f195090246d84bcf6cf79d9fe368be3601facdd1400a5'}}
OUTPUTS = {False: {'ggml-webgpu.cpp': '34e8d902c07a8770ae473e082b01aa592b5cae486d1541cc4b0a0dfac16c1941',
         'ggml-webgpu-shader-lib.hpp': '62e7700902bbc719cc57d3f6dc91ce4aa6dcec89b1eaf0a7af803298ec243556'},
 True: {'ggml-webgpu.cpp': '08663d3d2dde4c0345fbeb984d4e73950628365ec1e56321f873957382db7fe2',
        'ggml-webgpu-shader-lib.hpp': '221509716c3930422fd820248fa19b0fbd935a3838e0ec772ca62ce019945364'}}



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
