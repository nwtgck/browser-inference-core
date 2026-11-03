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
    'ggml-webgpu.cpp': 'd7363c0026d0dec265679d3cb1dddeafcf3f5049d5846eca124ec2c70615d17a',
    'ggml-webgpu-shader-lib.hpp': 'f286785f19fe280a4d2a8ab3873bf5f39d7af6a1b45be892537e8cb657e103f6',
}
OUTPUTS = {
    'ggml-webgpu.cpp': '9d4295d16c8f988e109d13f6d8bdbea9b8a4fb7aa644dc5ac4cc20b86d9bf78f',
    'ggml-webgpu-shader-lib.hpp': 'e200d33142efa3e639802f3a62e25df24e7c3a3b635a4c397bd4e167d8046a7f',
    'ggml-webgpu-rms-scale.hpp': 'e6335ce145c2863d1bcaf3fe4f0d99c4324840a89496156df229e7d34015f9d4',
    'wgsl-shaders/row_norm.wgsl': 'dff02ee996b5e02074ea7cb913ca0b389b2d6fc91ec0f01449b6087946063521',
}
HEADER_SHA256 = 'ab8cff9d4a5954b1426a987386e829835cd06874c1e7e1287e8dd3524de78818'
CONTRACT_INPUTS = {
    'ggml/include/ggml.h': '1aef7a5f465b1c19de305dd6f9a0c303f88e485133a94b9a0ee1cce6345c9101',
    'ggml/src/ggml.c': '4b141cb8e781090b9a4c453781a31c9436052cced6018b080774650b7b57ed4c',
    'ggml/src/ggml-impl.h': 'c84c8f9e3c246504602f5cd388ba54e36691e46bf150e4d70ae857bb823c985f',
    'ggml/src/ggml-alloc.c': '9664bda3e9c4223f5c94ebfeb3c0381f0065a5091ddae707c09a800f7e419a39',
    'ggml/src/ggml-backend.cpp': '7ea26359e7eda6957227d6e4c2a24a96aed622cbbca300b066580f7f3d95df5e',
    'ggml/src/ggml-backend-impl.h': '7693def12dcc79a10dd2152e7cb659278756dba7622ebe41937f951dc94944e3',
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
