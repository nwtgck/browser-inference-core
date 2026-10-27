"""Compose a checked F32 RMS_NORM/SCALE epilogue without replacing canonical WGSL.

The existing row_norm source is patched only in the isolated composition. Its
unchanged reduction and normalization are embedded under a dedicated symbol.
"""
from pathlib import Path
import shutil
import subprocess
import sys
import prepare_webgpu_tensor_copy as tensor_copy

PATCH_NAME = 'ggml-webgpu-rms-scale.patch'
PATCH_SHA256 = '7004361d908c7b730a67ed7effaae1bac1def8082eebadadaf91e841be8f8e77'
BASE = Path('ggml/src/ggml-webgpu')
HEADER_NAME = 'ggml-webgpu-rms-scale-shaders.hpp'
INPUTS = {
    'ggml-webgpu-scale-unary.hpp': '78efb6f8ff9b31813cad1da78d5d97e3d38f2e44ba63dc7165d01aebdebe2898',
    'ggml-webgpu.cpp': 'df43feda75cf9f1485f177d88d94bd6a436e7d562df42ee941d97c2c8b0c5ed8',
    'ggml-webgpu-shader-lib.hpp': '560e5881d91c2f693f244142ff3b13dfefa6eb020abe5ddbfe952b9d8fff1ab8',
}
OUTPUTS = {
    'ggml-webgpu.cpp': '5dfefacf79403c98d81e86aec28d4b3476ea70393d58753da6f586a66ba5a7a0',
    'ggml-webgpu-shader-lib.hpp': 'ce7d5423fa56e604a4d601a6d0c4ca5c10f598f203ab011cec34107a9e855f20',
    'ggml-webgpu-rms-scale.hpp': 'e6335ce145c2863d1bcaf3fe4f0d99c4324840a89496156df229e7d34015f9d4',
    'wgsl-shaders/row_norm.wgsl': 'dff02ee996b5e02074ea7cb913ca0b389b2d6fc91ec0f01449b6087946063521',
}
HEADER_SHA256 = 'ab8cff9d4a5954b1426a987386e829835cd06874c1e7e1287e8dd3524de78818'
CONTRACT_INPUTS = {
    'ggml/include/ggml.h': '475beaa7e959be8c383279d9ce9d5191f37426e6b338270cfb12b26a798129d3',
    'ggml/src/ggml.c': '42b238412cda218552812a37329fab95a2f94be6d8bff5139083450d432f796d',
    'ggml/src/ggml-impl.h': '43564db0238aebb7ed68501e346c194866b5dac218d1d37b26baff9f458c00d3',
    'ggml/src/ggml-alloc.c': '9664bda3e9c4223f5c94ebfeb3c0381f0065a5091ddae707c09a800f7e419a39',
    'ggml/src/ggml-backend.cpp': '43dea13b13c2a15afd83f53515f3488c17072fc107a6ba4a5e0370f90b4fe7ff',
    'ggml/src/ggml-backend-impl.h': '4e861c8e23d3d7b0d99d8e0836e68564af58f8597d75d7bbb732857d7a632fa7',
    'ggml/src/ggml-webgpu/wgsl-shaders/row_norm.wgsl': '1676d1c2d90f44b83584d175078654812d2c8a36dd3a261a055e265f267f1a41',
    'ggml/src/ggml-webgpu/wgsl-shaders/scale.wgsl': 'cee80d0720353f4ad24c8845db54000e6f9b1dd1d1c3585acce3f99b693b7a45',
    'ggml/src/ggml-webgpu/pre_wgsl.hpp': '0d6f781eb5723311cf856e585566c70ddcc779f0e97fcfe6d0e3f636c35688e5',
    'ggml/src/ggml-webgpu/wgsl-shaders/embed_wgsl.py': '37d4a93bb46b0ff417c31a0c1d3fa4b9db52556556138cb2eaedae9daa8dcbea',
}


def prepare(work: Path, source: Path, patch_root: Path, env: dict[str, str]) -> dict[str, Path]:
    for relative, expected in CONTRACT_INPUTS.items():
        if tensor_copy.digest(source / relative) != expected:
            raise ValueError('RMS/scale contract changed: ' + relative)
    for relative, expected in INPUTS.items():
        if tensor_copy.digest(work / BASE / relative) != expected:
            raise ValueError('RMS/scale composition input changed: ' + relative)
    for relative in ('ggml-webgpu-rms-scale.hpp', HEADER_NAME, 'wgsl-shaders/rms_norm_scale.wgsl'):
        if (source / BASE / relative).exists():
            raise ValueError('Upstream RMS/scale source now exists: ' + relative)
    shader = work / BASE / 'wgsl-shaders/row_norm.wgsl'
    if shader.exists():
        raise ValueError('Unexpected precomposed row_norm source')
    shader.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / BASE / 'wgsl-shaders/row_norm.wgsl', shader)
    patch = patch_root / PATCH_NAME
    if tensor_copy.digest(patch) != PATCH_SHA256:
        raise ValueError('RMS/scale patch identity changed')
    for flags in (['--check'], []):
        subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags, str(patch.resolve())],
                       cwd=work, env=env, check=True, capture_output=True)
    for relative, expected in OUTPUTS.items():
        if tensor_copy.digest(work / BASE / relative) != expected:
            raise ValueError('Unexpected RMS/scale output: ' + relative)
    shaders = work / 'rms-scale-only-wgsl'
    shaders.mkdir()
    # A separate symbol keeps ordinary canonical row_norm shaders upstream-owned.
    shutil.copyfile(shader, shaders / 'rms_norm_scale.wgsl')
    header = work / HEADER_NAME
    subprocess.run([sys.executable, str(source / BASE / 'wgsl-shaders/embed_wgsl.py'),
                    '--input_dir', str(shaders), '--output_file', str(header)], check=True, capture_output=True)
    if tensor_copy.digest(header) != HEADER_SHA256:
        raise ValueError('Unexpected RMS/scale embedding identity')
    return {HEADER_NAME: header, 'ggml-webgpu-rms-scale.hpp': work / BASE / 'ggml-webgpu-rms-scale.hpp'}
