"""Checked logical32 skipped-slot clearing for the existing gather."""
from pathlib import Path
import shutil
import subprocess
import sys
import prepare_webgpu_tensor_copy as tensor_copy

BASE = Path('ggml/src/ggml-webgpu')
PATCH_NAME = 'ggml-webgpu-moe-gather-logical32.patch'
PATCH_SHA256 = '784ac621b98de1689f30c240c7e8099560dc839445616df75c051e9ad4555a67'
SHADER_NAME = 'mul_mat_id_gather_logical32.wgsl'
HEADER_NAME = 'ggml-webgpu-mul-mat-id-gather-logical32-shaders.hpp'
INPUT_SHA256 = '17bc04661859b9b96e624b58d6ebceb3f5d89151594650893a9680ef0df79491'
OUTPUT_SHA256 = '392fb81ddddb2ea569a7d6220704831ee975f96ee2a2131f4d17f2a629ebef56'
SHADER_SHA256 = '57d91d28f0a2a8528bc55ed1d95b08f0464bde4a02efc0fd797275202edfd221'
HEADER_SHA256 = '2fd73d87ebf9cebc13c50a444f627cbfc9440d0f03adfa7c906b68ee7228783c'
CONTRACT_INPUTS = {
    'ggml/include/ggml.h': 'bdb0e75be8eeb1fc14a742015025505f5810d8200cbb40bda7e981101413440b',
    'ggml/src/ggml-webgpu/ggml-webgpu.cpp': 'c3e627af0d18c596dd45f051c7eba612a6f8f53f458bceecd557d8704f52b8ee',
    'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp': 'a110a08b87a2b2768c991281e450cc0e89b5c34f67856b58138b7ca9e9aa5f17',
    'ggml/src/ggml-webgpu/pre_wgsl.hpp': '0d6f781eb5723311cf856e585566c70ddcc779f0e97fcfe6d0e3f636c35688e5',
    'ggml/src/ggml-webgpu/wgsl-shaders/embed_wgsl.py': '37d4a93bb46b0ff417c31a0c1d3fa4b9db52556556138cb2eaedae9daa8dcbea',
    'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_id_gather.wgsl': '433c7b72fb9382e6e452004bdb602d7be8f2654543de680a28d31d8a263298e7',
}


def prepare(work: Path, source: Path, patch_root: Path, env: dict[str, str]) -> dict[str, Path]:
    for relative, expected in CONTRACT_INPUTS.items():
        if tensor_copy.digest(source / relative) != expected:
            raise ValueError('MoE logical32 gather contract changed: ' + relative)
    library = work / BASE / 'ggml-webgpu-shader-lib.hpp'
    if tensor_copy.digest(library) != INPUT_SHA256:
        raise ValueError('MoE logical32 gather composition input changed')
    if (work / HEADER_NAME).exists():
        raise ValueError('MoE logical32 gather would replace unreviewed embedded source')
    for relative in (HEADER_NAME, 'wgsl-shaders/' + SHADER_NAME):
        if (work / BASE / relative).exists() or (source / BASE / relative).exists():
            raise ValueError('MoE logical32 gather would replace unreviewed source: ' + relative)
    patch = patch_root / PATCH_NAME
    if tensor_copy.digest(patch) != PATCH_SHA256:
        raise ValueError('MoE logical32 gather patch identity changed')
    shader = work / BASE / 'wgsl-shaders' / SHADER_NAME
    shader.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / BASE / 'wgsl-shaders/mul_mat_id_gather.wgsl', shader)
    for flags in (['--check'], []):
        subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags, str(patch.resolve())],
                       cwd=work, env=env, check=True, capture_output=True)
    if tensor_copy.digest(library) != OUTPUT_SHA256 or tensor_copy.digest(shader) != SHADER_SHA256:
        raise ValueError('Unexpected MoE logical32 gather source identity')
    shaders = work / 'moe-gather-logical32-only-wgsl'
    shaders.mkdir()
    shutil.copyfile(shader, shaders / SHADER_NAME)
    header = work / HEADER_NAME
    subprocess.run([sys.executable, str(source / BASE / 'wgsl-shaders/embed_wgsl.py'),
                    '--input_dir', str(shaders), '--output_file', str(header)], check=True, capture_output=True)
    if tensor_copy.digest(header) != HEADER_SHA256:
        raise ValueError('Unexpected MoE logical32 gather embedded identity')
    return {HEADER_NAME: header}
