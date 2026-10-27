"""Source-bound independent-subgroup Q8_0 matvecs for fixed subgroup widths.

Only the dedicated main and host selection are downstream. Embed the unchanged
upstream byte loaders and accumulator using the reviewed upstream emitter.
"""
from pathlib import Path
import shutil
import subprocess
import sys

import prepare_webgpu_tensor_copy as tensor_copy

PATCH_NAME = 'ggml-webgpu-q8-subgroup-matvec.patch'
PATCH_SHA256 = 'dba97f95c2f5c6fd2a7cf48b7063ea672bab1225ef147722e6444141f90a1dc7'
BASE = Path('ggml/src/ggml-webgpu')
HEADER_NAME = 'ggml-webgpu-q8-subgroup-shaders.hpp'
SHADER_NAME = 'q8_subgroup_matvec.wgsl'
INPUTS = {
    'ggml-webgpu.cpp': 'afb7f066d73ce1c742296502087a2881dad9454957e19f90d5b822982e208274',
    'ggml-webgpu-shader-lib.hpp': '84912f50de5cd9489c76228bd06de74bc87208bec89bc8b3a3a5a7510aed756e',
}
OUTPUTS = {
    'ggml-webgpu.cpp': INPUTS['ggml-webgpu.cpp'],
    'ggml-webgpu-shader-lib.hpp': 'a301a9c08cf0c3d92849286e1066d81b17ff7b1b7deb058da0b6870d13456304',
    'wgsl-shaders/q8_subgroup_matvec.wgsl': '503e7785159426aefa700f4edd8625a59a88997c4beee995328744e06cd8f25c',
}
HEADER_SHA256 = '7d151de06e9a3cb5366b29e043052b3b949c4c1948e7b3c2e80e99a944b95cab'
CONTRACT_INPUTS = {
    'ggml/src/ggml-webgpu/ggml-webgpu.cpp': '67e0c1b6ffeec0d943eb63aced20c00951d565cf2bbb964e90a99265259dca8e',
    'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp': '52d39024d24f723dc1b88dc8bf8b4ed18f1e50846010c35065cebc364a789e2d',
    'ggml/src/ggml-webgpu/pre_wgsl.hpp': '0d6f781eb5723311cf856e585566c70ddcc779f0e97fcfe6d0e3f636c35688e5',
    'ggml/src/ggml-webgpu/wgsl-shaders/common_decls.tmpl': '1cfe6cae3ab4a4ccac963a1a68ea9ddf373f4b4beefe7c933807f37969bf5555',
    'ggml/src/ggml-webgpu/wgsl-shaders/embed_wgsl.py': '37d4a93bb46b0ff417c31a0c1d3fa4b9db52556556138cb2eaedae9daa8dcbea',
    'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_vec.wgsl': '4fdbbaf3b3d3b7c965786b62a58b6b35ceb6d2f80b29e53f4a027904d3243f18',
    'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_vec_acc.tmpl': '9421bdb7ac569e03e8483365653c28b95e659702f8ff849310b2c077ee64290f',
}


def prepare(work: Path, source: Path, patch_root: Path, env: dict[str, str]) -> dict[str, Path]:
    for relative, expected in CONTRACT_INPUTS.items():
        if tensor_copy.digest(source / relative) != expected:
            raise ValueError('Q8 subgroup matvec contract changed: ' + relative)
    for relative, expected in INPUTS.items():
        if tensor_copy.digest(work / BASE / relative) != expected:
            raise ValueError('Q8 subgroup matvec composition input changed: ' + relative)
    for relative in (HEADER_NAME, 'wgsl-shaders/' + SHADER_NAME):
        if (source / BASE / relative).exists():
            raise ValueError('Upstream Q8 subgroup matvec now exists: ' + relative)
    patch = patch_root / PATCH_NAME
    if tensor_copy.digest(patch) != PATCH_SHA256:
        raise ValueError('Q8 subgroup matvec patch identity changed')
    for flags in (['--check'], []):
        subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags, str(patch.resolve())],
                       cwd=work, env=env, check=True, capture_output=True)
    for relative, expected in OUTPUTS.items():
        if tensor_copy.digest(work / BASE / relative) != expected:
            raise ValueError('Unexpected Q8 subgroup matvec output: ' + relative)
    shaders = work / 'q8-subgroup-only-wgsl'
    shaders.mkdir()
    shutil.copyfile(work / BASE / 'wgsl-shaders' / SHADER_NAME, shaders / SHADER_NAME)
    # Reuse the exact upstream implementation through its own include expansion.
    for name in ('common_decls.tmpl', 'mul_mat_vec_acc.tmpl'):
        shutil.copyfile(source / BASE / 'wgsl-shaders' / name, shaders / name)
    header = work / HEADER_NAME
    subprocess.run([sys.executable, str(source / BASE / 'wgsl-shaders/embed_wgsl.py'),
                    '--input_dir', str(shaders), '--output_file', str(header)], check=True, capture_output=True)
    if tensor_copy.digest(header) != HEADER_SHA256:
        raise ValueError('Q8 subgroup matvec embedding identity changed')
    return {HEADER_NAME: header}
