"""Select eight-row paired-half register tiles for guarded Q4 prefill matrices."""
from pathlib import Path
import subprocess
from tests.support.llama_cpp.frozen_webgpu_stages import quant_half_pairs as half_pairs
from tests.support.llama_cpp.frozen_webgpu_stages import tensor_copy
BASE = Path('ggml/src/ggml-webgpu')
PATCH_NAME = 'ggml-webgpu-quant-wide-m.patch'
CONTRACT_INPUTS = dict(half_pairs.CONTRACT_INPUTS)
EMBEDDED_HEADER_SHA256 = half_pairs.HEADER_SHA256

PATCH_SHA256 = '385acda3e874eebfeac26fc03259dec49fc3579a59e0ac76f55e7a86d8d67686'
INPUTS = {False: {'ggml-webgpu.cpp': 'c977f258c8ec2b98e07f69388e1bc22a00646f76c559ce5afee533d7ff6a0c76',
         'ggml-webgpu-shader-lib.hpp': '341249f8670c2266f53b092e50306a7d73abacb1faf6e1787002d7386560885b'},
 True: {'ggml-webgpu.cpp': '62e142cdac82c751f3c758e264c3750ea420429f58a0fa883414322756aa7268',
        'ggml-webgpu-shader-lib.hpp': '7ad71c726623e5b65e6f195090246d84bcf6cf79d9fe368be3601facdd1400a5'}}
OUTPUTS = {False: {'ggml-webgpu.cpp': 'c977f258c8ec2b98e07f69388e1bc22a00646f76c559ce5afee533d7ff6a0c76',
         'ggml-webgpu-shader-lib.hpp': '62e7700902bbc719cc57d3f6dc91ce4aa6dcec89b1eaf0a7af803298ec243556'},
 True: {'ggml-webgpu.cpp': '62e142cdac82c751f3c758e264c3750ea420429f58a0fa883414322756aa7268',
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
                       cwd=work, env=env, check=True)
    for name, expected in OUTPUTS[gpu_profile].items():
        if tensor_copy.digest(work / BASE / name) != expected:
            raise ValueError('Unexpected quantized wide-M output: ' + name)
