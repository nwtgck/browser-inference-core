"""Keep the WebGPU allocation limit representable on 32-bit hosts."""
from pathlib import Path
import subprocess
import prepare_webgpu_tensor_copy as tensor_copy

BASE = Path('ggml/src/ggml-webgpu')
PATCH_NAME = 'ggml-webgpu-buffer-size-limit.patch'
PATCH_SHA256 = '7094c7d6278b7fa85a59ca88b5093f9c66aba13c3a38feb4b13bb777d8d4dd58'
CONTRACT_INPUTS = {'ggml/src/ggml-webgpu/ggml-webgpu.cpp': '617700adc9a58401414cd6828fa48b4efb3bb0563458e6ebe717360e70c77201', 'ggml/src/ggml-alloc.c': '9664bda3e9c4223f5c94ebfeb3c0381f0065a5091ddae707c09a800f7e419a39', 'ggml/src/ggml-backend.cpp': '7ea26359e7eda6957227d6e4c2a24a96aed622cbbca300b066580f7f3d95df5e'}
INPUTS = {False: {'ggml-webgpu.cpp': 'd9ca2bb081c5b9392b3c46ddfc7dfe8dd0fcf404f884c46a081303ce90d98f14'}, True: {'ggml-webgpu.cpp': '02403383d69aa126976a6fe56289ffa4f0ebf63310c89dec2a2c8d96c953b567'}}
OUTPUTS = {False: {'ggml-webgpu.cpp': '1d7ea35c56439dcb915505fae697d2b25e58843e3df82b65562cc01fa813f212'}, True: {'ggml-webgpu.cpp': '81d3a0f6659cd9668549425977465f40dc64ef18ed7459a168bc0d7c82e1f1f1'}}


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
