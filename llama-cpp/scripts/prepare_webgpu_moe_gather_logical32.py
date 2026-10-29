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
INPUT_SHA256 = '8aa99e74bcef275d0da27f014373c61e2279239af83c48019faa3e37d2038c80'
OUTPUT_SHA256 = 'a302e654b7afd4ed1ba39a2031596c1d8cb1ca3ba19533be4f5b40f28e1281e2'
SHADER_SHA256 = '57d91d28f0a2a8528bc55ed1d95b08f0464bde4a02efc0fd797275202edfd221'
HEADER_SHA256 = '2fd73d87ebf9cebc13c50a444f627cbfc9440d0f03adfa7c906b68ee7228783c'
CONTRACT_INPUTS = {
    'ggml/include/ggml.h': '2b4dd92fa992cb4ef484917a6c62fef5097698744160ccc09214fd0c459c92c7',
    'ggml/src/ggml-webgpu/ggml-webgpu.cpp': '67e0c1b6ffeec0d943eb63aced20c00951d565cf2bbb964e90a99265259dca8e',
    'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp': '52d39024d24f723dc1b88dc8bf8b4ed18f1e50846010c35065cebc364a789e2d',
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
