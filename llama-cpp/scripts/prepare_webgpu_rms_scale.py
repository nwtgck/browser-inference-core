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
    'ggml-webgpu.cpp': '0e001ef27537170355fecd57b664a5a5fb2ecbb225f570533a9618c899841d16',
    'ggml-webgpu-shader-lib.hpp': '9848a31c0e1fe135802ab1111d90415a4b67e8a2aaa6ae5b12a1f08e612c99a4',
}
OUTPUTS = {
    'ggml-webgpu.cpp': 'b8fa0732191d3c34a7da4d4839683e4c1f7fc0c62a9307fc3481d42311b93b54',
    'ggml-webgpu-shader-lib.hpp': 'b2f898e4aab45e54f95c3166b1023c963e30c72d23ed915d0a705616b5114b3d',
    'ggml-webgpu-rms-scale.hpp': 'e6335ce145c2863d1bcaf3fe4f0d99c4324840a89496156df229e7d34015f9d4',
    'wgsl-shaders/row_norm.wgsl': 'dff02ee996b5e02074ea7cb913ca0b389b2d6fc91ec0f01449b6087946063521',
}
HEADER_SHA256 = 'ab8cff9d4a5954b1426a987386e829835cd06874c1e7e1287e8dd3524de78818'
CONTRACT_INPUTS = {
    'ggml/include/ggml.h': 'bdb0e75be8eeb1fc14a742015025505f5810d8200cbb40bda7e981101413440b',
    'ggml/src/ggml.c': '1008000571b896a22dd3861731ca29e6ec3615898cdc15d9cb529fb171815dfc',
    'ggml/src/ggml-impl.h': '75465b01eba2b1ecae8634898508e71294fbff0f7f96ee9ad2caf5bae5b6c5fd',
    'ggml/src/ggml-alloc.c': 'c4001b08581b87b80283953acc67d0d2ff13ef0a425179e2bf621f45333bf0fa',
    'ggml/src/ggml-backend.cpp': '2f724930ea53dd8f50f26706ecf10a177d56601e668f109cf1cd95052b1b9ad8',
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
