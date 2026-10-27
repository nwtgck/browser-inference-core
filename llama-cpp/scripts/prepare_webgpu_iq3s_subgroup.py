"""Source-bound fixed-subgroup IQ3_S indexed matvecs; upstream bytes stay read-only.

Reuse the original quantization helpers/accumulator through the pinned emitter;
only the dedicated main, selection, cache and decision wiring are downstream.
"""
from pathlib import Path
import shutil
import subprocess
import sys

import prepare_webgpu_tensor_copy as tensor_copy

PATCH_NAME = 'ggml-webgpu-iq3s-id-subgroup.patch'
PATCH_SHA256 = '6ec906724c7bfd0cf2238eee0c3e1ba348becbc3d67b6f155ac2d4604645e083'
BASE = Path('ggml/src/ggml-webgpu')
HEADER_NAME = 'ggml-webgpu-iq3s-id-subgroup-shaders.hpp'
SHADER_NAME = 'iq3s_id_subgroup.wgsl'
INPUTS = {
    'ggml-webgpu.cpp': 'c99d86ba5a7684c12a1d3840acd11badcb2058e06283a1d5d2f1d9275013afe2',
    'ggml-webgpu-shader-lib.hpp': '01722e3e01b4aa8d913a44225f0c565ebd1551569810caf1db475069e647aa6a',
}
OUTPUTS = {
    'ggml-webgpu.cpp': '363ce73b31868f5676303b5cd6c6e02155802bde61f2918946d3879feeb99fbd',
    'ggml-webgpu-shader-lib.hpp': 'b9cd9fbcaeb2d96153bc4624347420e020cf613f67b547c51311eba90bb41e95',
    'wgsl-shaders/iq3s_id_subgroup.wgsl': 'a24f289e8a528e806f7d5a4fca0ef45f0588683a100da314d4a01f5da2489547',
}
HEADER_SHA256 = '668f7421877db1ca9675b7e3f251f2ba62964c346ffa47f4c970965a6c36e34d'
CONTRACT_INPUTS = {
    'ggml/src/ggml-common.h': '0061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c',
    'ggml/src/ggml-webgpu/ggml-webgpu.cpp': '67e0c1b6ffeec0d943eb63aced20c00951d565cf2bbb964e90a99265259dca8e',
    'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp': '52d39024d24f723dc1b88dc8bf8b4ed18f1e50846010c35065cebc364a789e2d',
    'ggml/src/ggml-webgpu/pre_wgsl.hpp': '0d6f781eb5723311cf856e585566c70ddcc779f0e97fcfe6d0e3f636c35688e5',
    'ggml/src/ggml-webgpu/wgsl-shaders/common_decls.tmpl': '1cfe6cae3ab4a4ccac963a1a68ea9ddf373f4b4beefe7c933807f37969bf5555',
    'ggml/src/ggml-webgpu/wgsl-shaders/embed_wgsl.py': '37d4a93bb46b0ff417c31a0c1d3fa4b9db52556556138cb2eaedae9daa8dcbea',
    'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_id_vec.wgsl': 'cbe87da8b6eb80fcb7905855fe1d212dc5a0afac55b77600c507fbbd2fd7c31d',
    'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_vec_acc.tmpl': '9421bdb7ac569e03e8483365653c28b95e659702f8ff849310b2c077ee64290f',
    'ggml/src/ggml-quants.c': '5574a2dccf7c07e75b143733e04a5412d3d8c819e7945f5217a7b83a2b2ff8ab',
}


def prepare(work: Path, source: Path, patch_root: Path, env: dict[str, str]) -> dict[str, Path]:
    for relative, expected in CONTRACT_INPUTS.items():
        if tensor_copy.digest(source / relative) != expected:
            raise ValueError('IQ3_S subgroup contract changed: ' + relative)
    for relative, expected in INPUTS.items():
        if tensor_copy.digest(work / BASE / relative) != expected:
            raise ValueError('IQ3_S subgroup composition input changed: ' + relative)
    for relative in (HEADER_NAME, 'wgsl-shaders/' + SHADER_NAME):
        if (source / BASE / relative).exists():
            raise ValueError('Upstream IQ3_S subgroup implementation now exists: ' + relative)
    patch = patch_root / PATCH_NAME
    if tensor_copy.digest(patch) != PATCH_SHA256:
        raise ValueError('IQ3_S subgroup patch identity changed')
    for flags in (['--check'], []):
        subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags, str(patch.resolve())],
                       cwd=work, env=env, check=True, capture_output=True)
    for relative, expected in OUTPUTS.items():
        if tensor_copy.digest(work / BASE / relative) != expected:
            raise ValueError('Unexpected IQ3_S subgroup output: ' + relative)
    shaders = work / 'iq3s-id-subgroup-only-wgsl'
    shaders.mkdir()
    shutil.copyfile(work / BASE / 'wgsl-shaders' / SHADER_NAME, shaders / SHADER_NAME)
    for name in ('common_decls.tmpl', 'mul_mat_vec_acc.tmpl'):
        shutil.copyfile(source / BASE / 'wgsl-shaders' / name, shaders / name)
    header = work / HEADER_NAME
    subprocess.run([sys.executable, str(source / BASE / 'wgsl-shaders/embed_wgsl.py'),
                    '--input_dir', str(shaders), '--output_file', str(header)], check=True, capture_output=True)
    if tensor_copy.digest(header) != HEADER_SHA256:
        raise ValueError('IQ3_S subgroup embedding identity changed')
    return {HEADER_NAME: header}
