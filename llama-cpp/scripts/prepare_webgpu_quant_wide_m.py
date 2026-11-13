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
INPUTS = {False: {'ggml-webgpu.cpp': '23873d267578c6bce435eff0f7643f2853f705ad0b46b3ed8e93cbaada2bb8a2',
         'ggml-webgpu-shader-lib.hpp': '543c78b27a615dbb2cc52453270092fbb0b3a088ff0c5829b1842cc7ec9ae30e'},
 True: {'ggml-webgpu.cpp': 'de513ccb952c3cc1da2e42f45f4ed1acd9433f6bdea3e38a61f951f437b9c79c',
        'ggml-webgpu-shader-lib.hpp': '611c89c046020d22afcf3211a70ce81266bab3c59328168e365d2cd1ea148bf2'}}
OUTPUTS = {False: {'ggml-webgpu.cpp': '23873d267578c6bce435eff0f7643f2853f705ad0b46b3ed8e93cbaada2bb8a2',
         'ggml-webgpu-shader-lib.hpp': '96e949d9e6064f82e2d63111fac3ce1240a0dfcab42e1266628eb26fe8d997fd'},
 True: {'ggml-webgpu.cpp': 'de513ccb952c3cc1da2e42f45f4ed1acd9433f6bdea3e38a61f951f437b9c79c',
        'ggml-webgpu-shader-lib.hpp': '2e12ecc4f59bf9ca594b21ee3240642f194ba8ea6232054e1cef09dad5b1ca1e'}}



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
