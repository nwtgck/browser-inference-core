"""Split only the dedicated four-row IQ3_S accumulator's row loop."""
from pathlib import Path
from build_tools.source.wgsl import generate as generate_wgsl
import shutil
import subprocess

from tests.support.llama_cpp.webgpu_stages import tensor_copy
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
    False: {'ggml-webgpu.cpp': 'c42e621e1318a2c70e6a6e651d6abdfac0197415fcdf8f0745c26fc082888232',
            'ggml-webgpu-shader-lib.hpp': '29e31f71751110558ce8718c6f421b3673a6cadc512de0c05817aa166ec1492f'},
    True: {'ggml-webgpu.cpp': '75f2fcdec2e53eade6c9a008c9e1b86e707a0a37d0064e38e3d513ab07c5a92c',
           'ggml-webgpu-shader-lib.hpp': 'f9f39dfcd8b94c44effbf09451af951dc4cc4cd5504ceb1a7b20cb42e348a6b7'},
}
CONTRACT_INPUTS = {
    str(BASE / 'wgsl-shaders/common_decls.tmpl'): '1cfe6cae3ab4a4ccac963a1a68ea9ddf373f4b4beefe7c933807f37969bf5555',
    ACCUMULATOR_PATH: '9421bdb7ac569e03e8483365653c28b95e659702f8ff849310b2c077ee64290f',
    str(BASE / 'wgsl-shaders/embed_wgsl.py'): '37d4a93bb46b0ff417c31a0c1d3fa4b9db52556556138cb2eaedae9daa8dcbea',
    str(BASE / 'pre_wgsl.hpp'): '0d6f781eb5723311cf856e585566c70ddcc779f0e97fcfe6d0e3f636c35688e5',
    str(BASE / 'ggml-webgpu-shader-lib.hpp'): 'a110a08b87a2b2768c991281e450cc0e89b5c34f67856b58138b7ca9e9aa5f17',
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
                       cwd=stage, env=env, check=True)
    if tensor_copy.digest(shaders / 'mul_mat_vec_acc.tmpl') != ACCUMULATOR_OUTPUT_SHA256:
        raise ValueError('Unexpected IQ3_S row-split accumulator output')
    generated = work / 'iq3s-row-split-generated'
    generated.mkdir()
    header = generated / HEADER_NAME
    generate_wgsl(shaders, None, header, generator=source / BASE / 'wgsl-shaders/embed_wgsl.py')
    if tensor_copy.digest(header) != HEADER_SHA256:
        raise ValueError('Unexpected IQ3_S row-split embedding output')
    return header
