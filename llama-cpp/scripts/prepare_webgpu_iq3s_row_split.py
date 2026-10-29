"""Split only the dedicated four-row IQ3_S accumulator's row loop."""
from pathlib import Path
import shutil
import subprocess
import sys

import prepare_webgpu_tensor_copy as tensor_copy

BASE = Path('ggml/src/ggml-webgpu')
HEADER_NAME = 'ggml-webgpu-iq3s-id-subgroup-shaders.hpp'
SHADER_NAME = 'iq3s_id_subgroup.wgsl'
PATCH_NAME = 'ggml-webgpu-iq3s-row-split.patch'
PATCH_SHA256 = '63facaa4bd3d1283ad0b69ad0a2dac5d9be0961f16faf4fa28a125bb27cb8b8c'
PRIOR_HEADER_SHA256 = '24d365982e2126df01e03be9ff46b8f48a409005bca7b7b53c2c95c5d2944154'
HEADER_SHA256 = '6427de75e644afb55e31cc2c8779004087087b60e846dfbdba11f3fecc88acbb'
SHADER_SHA256 = 'a24f289e8a528e806f7d5a4fca0ef45f0588683a100da314d4a01f5da2489547'
ACCUMULATOR_PATH = str(BASE / 'wgsl-shaders/mul_mat_vec_acc.tmpl')
ACCUMULATOR_INPUT_SHA256 = '23ebdf701a4ed2aff8b101acee31496b3bbad75c1ff0f32924cf52468411b952'
ACCUMULATOR_OUTPUT_SHA256 = '6f37bd58a3f66b8e2e09b3c6fba91c5609b8d770e8ac984e30830bd6d5c2f5f1'
ROWS_PER_SUBGROUP = 4
INPUTS = {
    False: {'ggml-webgpu.cpp': '1ad7f5a177473d4e35f79f0463decc485ed5b9aaf0a3b5a87338ec01dbd757c8',
            'ggml-webgpu-shader-lib.hpp': '5bc3e50a164d6a7fed6980c21a208575972ebce118f8b37787ea2fc8fc856e3e'},
    True: {'ggml-webgpu.cpp': 'c4b8dee4e691a7e10a52412a3dcb4478d3992a2946c4011f4b23850e80332e1f',
           'ggml-webgpu-shader-lib.hpp': 'fabc949c00c6196cd660f6f611ed47ce4ee2c34f54e09109ba9feac545c37880'},
}
CONTRACT_INPUTS = {
    str(BASE / 'wgsl-shaders/common_decls.tmpl'): '1cfe6cae3ab4a4ccac963a1a68ea9ddf373f4b4beefe7c933807f37969bf5555',
    ACCUMULATOR_PATH: '9421bdb7ac569e03e8483365653c28b95e659702f8ff849310b2c077ee64290f',
    str(BASE / 'wgsl-shaders/embed_wgsl.py'): '37d4a93bb46b0ff417c31a0c1d3fa4b9db52556556138cb2eaedae9daa8dcbea',
    str(BASE / 'pre_wgsl.hpp'): '0d6f781eb5723311cf856e585566c70ddcc779f0e97fcfe6d0e3f636c35688e5',
    str(BASE / 'ggml-webgpu-shader-lib.hpp'): '52d39024d24f723dc1b88dc8bf8b4ed18f1e50846010c35065cebc364a789e2d',
}


def prepare(work: Path, source: Path, patch_root: Path, env: dict[str, str], *, previous_header: Path, gpu_profile: bool) -> Path:
    for relative, expected in CONTRACT_INPUTS.items():
        if tensor_copy.digest(source / relative) != expected:
            raise ValueError('IQ3_S row-split contract changed: ' + relative)
    for relative, expected in INPUTS[gpu_profile].items():
        if tensor_copy.digest(work / BASE / relative) != expected:
            raise ValueError('IQ3_S row-split composition input changed: ' + relative)
    if previous_header.name != HEADER_NAME or tensor_copy.digest(previous_header) != PRIOR_HEADER_SHA256:
        raise ValueError('IQ3_S row-split prior embedding changed')
    patch = patch_root / PATCH_NAME
    if tensor_copy.digest(patch) != PATCH_SHA256:
        raise ValueError('IQ3_S row-split patch identity changed')

    # The existing factory specializes OUTPUTS_PER_WG=4 before expanding the
    # host dispatch tile. Its complete normal/profile header identities above
    # pin that contract; the first split loop deliberately covers rows 0..1.
    prior = work / 'iq3s-field-loads-source' / BASE / 'wgsl-shaders'
    for name, expected in ((SHADER_NAME, SHADER_SHA256),
                           ('mul_mat_vec_acc.tmpl', ACCUMULATOR_INPUT_SHA256),
                           ('common_decls.tmpl', CONTRACT_INPUTS[str(BASE / 'wgsl-shaders/common_decls.tmpl')])):
        if tensor_copy.digest(prior / name) != expected:
            raise ValueError('IQ3_S row-split prior shader input changed: ' + name)
    stage = work / 'iq3s-row-split-source'
    shaders = stage / BASE / 'wgsl-shaders'
    shaders.mkdir(parents=True)
    for name in (SHADER_NAME, 'common_decls.tmpl', 'mul_mat_vec_acc.tmpl'):
        shutil.copyfile(prior / name, shaders / name)
    for flags in (['--check'], []):
        subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags, str(patch.resolve())],
                       cwd=stage, env=env, check=True, capture_output=True)
    if tensor_copy.digest(shaders / 'mul_mat_vec_acc.tmpl') != ACCUMULATOR_OUTPUT_SHA256:
        raise ValueError('Unexpected IQ3_S row-split accumulator output')
    generated = work / 'iq3s-row-split-generated'
    generated.mkdir()
    header = generated / HEADER_NAME
    subprocess.run([sys.executable, str(source / BASE / 'wgsl-shaders/embed_wgsl.py'),
                    '--input_dir', str(shaders), '--output_file', str(header)], check=True, capture_output=True)
    if tensor_copy.digest(header) != HEADER_SHA256:
        raise ValueError('Unexpected IQ3_S row-split embedding output')
    return header
