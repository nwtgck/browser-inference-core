"""Keep the WebGPU allocation limit representable on 32-bit hosts."""
from pathlib import Path
import subprocess
import prepare_webgpu_tensor_copy as tensor_copy

BASE = Path('ggml/src/ggml-webgpu')
PATCH_NAME = 'ggml-webgpu-buffer-size-limit.patch'
PATCH_SHA256 = '7094c7d6278b7fa85a59ca88b5093f9c66aba13c3a38feb4b13bb777d8d4dd58'
CONTRACT_INPUTS = {'ggml/src/ggml-webgpu/ggml-webgpu.cpp': 'c3e627af0d18c596dd45f051c7eba612a6f8f53f458bceecd557d8704f52b8ee', 'ggml/src/ggml-alloc.c': 'c4001b08581b87b80283953acc67d0d2ff13ef0a425179e2bf621f45333bf0fa', 'ggml/src/ggml-backend.cpp': '2f724930ea53dd8f50f26706ecf10a177d56601e668f109cf1cd95052b1b9ad8'}
INPUTS = {False: {'ggml-webgpu.cpp': 'd586843f24c47a12a29357c30b35b4d3f7b186f18af17bab222a52c5612a10c4'}, True: {'ggml-webgpu.cpp': 'a62afc88802e9a84d80a8f03ac11ed50133f05841c25542d57672250f28474ad'}}
OUTPUTS = {False: {'ggml-webgpu.cpp': '2dc724684bdc246e2583e9ed2a7547ffae48bb15e4c400211ae1e90a2fe2dc0a'}, True: {'ggml-webgpu.cpp': 'e6c279f867b6d00e95e17ab5734cbeac50b052cf2ef4a0828e6d19c0941807d1'}}


def prepare(work: Path, source: Path, patch_root: Path, env: dict[str, str], *, gpu_profile: bool) -> None:
    for name, expected in CONTRACT_INPUTS.items():
        if tensor_copy.digest(source / name) != expected:
            raise ValueError('WebGPU buffer-size contract changed: ' + name)
    for name, expected in INPUTS[gpu_profile].items():
        if tensor_copy.digest(work / BASE / name) != expected:
            raise ValueError('WebGPU buffer-size composition input changed: ' + name)
    patch = patch_root / PATCH_NAME
    if tensor_copy.digest(patch) != PATCH_SHA256:
        raise ValueError('WebGPU buffer-size patch identity changed')
    for flags in (['--check'], []):
        subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags, str(patch.resolve())],
                       cwd=work, env=env, check=True, capture_output=True)
    for name, expected in OUTPUTS[gpu_profile].items():
        if tensor_copy.digest(work / BASE / name) != expected:
            raise ValueError('Unexpected WebGPU buffer-size output: ' + name)
