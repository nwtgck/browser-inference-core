"""Checked logical32 skipped-slot clearing for the existing gather."""
from pathlib import Path
from build_tools.source.wgsl import generate as generate_wgsl
import shutil
import subprocess
from tests.support.llama_cpp.frozen_webgpu_stages import tensor_copy
BASE = Path('ggml/src/ggml-webgpu')
PATCH_NAME = 'ggml-webgpu-moe-gather-logical32.patch'
PATCH_SHA256 = '784ac621b98de1689f30c240c7e8099560dc839445616df75c051e9ad4555a67'
SHADER_NAME = 'mul_mat_id_gather_logical32.wgsl'
HEADER_NAME = 'ggml-webgpu-mul-mat-id-gather-logical32-shaders.hpp'
INPUT_SHA256 = '9857f40b073d83198354a21bbd956162f66a8e9875647ddd4a7adc4e04622181'
OUTPUT_SHA256 = '7756a6ef0fb1521dd896fbb1a34b7431dac71ef52db93003c0db70527dff3536'
SHADER_SHA256 = '57d91d28f0a2a8528bc55ed1d95b08f0464bde4a02efc0fd797275202edfd221'
HEADER_SHA256 = '2fd73d87ebf9cebc13c50a444f627cbfc9440d0f03adfa7c906b68ee7228783c'
CONTRACT_INPUTS = {
    'ggml/include/ggml.h': '1aef7a5f465b1c19de305dd6f9a0c303f88e485133a94b9a0ee1cce6345c9101',
    'ggml/src/ggml-webgpu/ggml-webgpu.cpp': '7fa5eca8734b7d31bf467281c698b3356c5750c2f923008c179a4dae426cffe5',
    'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp': '34522d61990406ddc1e46c55da4531e6aa2ad1561bf5bf512ebbb38843dbd49d',
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
                       cwd=work, env=env, check=True)
    if tensor_copy.digest(library) != OUTPUT_SHA256 or tensor_copy.digest(shader) != SHADER_SHA256:
        raise ValueError('Unexpected MoE logical32 gather source identity')
    shaders = work / 'moe-gather-logical32-only-wgsl'
    shaders.mkdir()
    shutil.copyfile(shader, shaders / SHADER_NAME)
    header = work / HEADER_NAME
    generate_wgsl(shaders, None, header, generator=source / BASE / 'wgsl-shaders/embed_wgsl.py')
    if tensor_copy.digest(header) != HEADER_SHA256:
        raise ValueError('Unexpected MoE logical32 gather embedded identity')
    return {HEADER_NAME: header}
