"""Checked paired-half shared storage for fixed-geometry Q4 prefill kernels."""
from pathlib import Path
from build_tools.source.wgsl import generate as generate_wgsl
import shutil
import subprocess
from tests.support.llama_cpp.webgpu_stages import tensor_copy
BASE = Path('ggml/src/ggml-webgpu')
PATCH_NAME = 'ggml-webgpu-quant-half-pairs.patch'
HEADER_NAME = 'ggml-webgpu-quant-half-pairs-shaders.hpp'
HEADER_SHA256 = '693ede233b36b81d0360bfc212a5ecb6c31cf10aae06afa6788682fbf317e19a'
TEMPLATE_COPIES = {
    'mul_mat_reg_tile_half_pairs.wgsl': 'mul_mat_reg_tile.wgsl',
    'mul_mat_half_pairs_decls.tmpl': 'mul_mat_decls.tmpl',
    'quant_half_pairs_inner_loops.tmpl': 'quant_inner_loops.tmpl',
}

PATCH_SHA256 = 'dfdaabd90a6df2cfc6b1a384c41b4aa3fb0005c5e339420c32341649e74c914f'
INPUTS = {False: {'ggml-webgpu.cpp': 'c50aeea2606f5da330e6b5f2d26115b94579acd87bb90ae3497415879b90a100',
         'ggml-webgpu-shader-lib.hpp': '32183874f4ecc5091539271971ab9e61353d35afc28972f251a4c21aebe10357'},
 True: {'ggml-webgpu.cpp': 'f7c1281458464c910ebbbbb9f7e679a20580498fdc7ce9cff5000f1f6dde1607',
        'ggml-webgpu-shader-lib.hpp': '212ac7375483581d7ef6515d03c0400b41dbe4b823882feadfe2cce5526a0995'}}
OUTPUTS = {False: {'ggml-webgpu.cpp': 'c50aeea2606f5da330e6b5f2d26115b94579acd87bb90ae3497415879b90a100',
         'ggml-webgpu-shader-lib.hpp': '543c78b27a615dbb2cc52453270092fbb0b3a088ff0c5829b1842cc7ec9ae30e'},
 True: {'ggml-webgpu.cpp': 'f7c1281458464c910ebbbbb9f7e679a20580498fdc7ce9cff5000f1f6dde1607',
        'ggml-webgpu-shader-lib.hpp': '611c89c046020d22afcf3211a70ce81266bab3c59328168e365d2cd1ea148bf2'}}
CONTRACT_INPUTS = {'ggml/src/ggml-webgpu/ggml-webgpu.cpp': 'c3e627af0d18c596dd45f051c7eba612a6f8f53f458bceecd557d8704f52b8ee',
 'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp': 'a110a08b87a2b2768c991281e450cc0e89b5c34f67856b58138b7ca9e9aa5f17',
 'ggml/src/ggml-webgpu/pre_wgsl.hpp': '0d6f781eb5723311cf856e585566c70ddcc779f0e97fcfe6d0e3f636c35688e5',
 'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_reg_tile.wgsl': '2d1cb611caf10b72d20b021772806ca4b3ea4f40c9d7a93941e135dacf859f81',
 'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_decls.tmpl': 'c461aa2108b752504f47338e41b14ee6570abdd831cd83652e052147c89d69ef',
 'ggml/src/ggml-webgpu/wgsl-shaders/quant_inner_loops.tmpl': '7f023ec774a11265985d92fb630fb94cd8e4599499f941983d1b23b37dacf5ad',
 'ggml/src/ggml-webgpu/wgsl-shaders/common_decls.tmpl': '1cfe6cae3ab4a4ccac963a1a68ea9ddf373f4b4beefe7c933807f37969bf5555',
 'ggml/src/ggml-webgpu/wgsl-shaders/embed_wgsl.py': '37d4a93bb46b0ff417c31a0c1d3fa4b9db52556556138cb2eaedae9daa8dcbea'}
DERIVED_SHA256 = {'wgsl-shaders/mul_mat_reg_tile_half_pairs.wgsl': '92a35cdbfcd772162fb944558f1d725bd167b80ed57a81674ae4f86c7ae2ba6e',
 'wgsl-shaders/mul_mat_half_pairs_decls.tmpl': '00c1f6e1c3c72be2a4b4e17658db61236b3eb85996be81a9be7906976151add6',
 'wgsl-shaders/quant_half_pairs_inner_loops.tmpl': '23cfbd62dc1713c34db1a2cf96e3c0bf9a4cbf29c2ca011be779469ce39fb0b9'}


def prepare(work: Path, source: Path, patch_root: Path, env: dict[str, str], *, gpu_profile: bool) -> dict[str, Path]:
    for relative, expected in CONTRACT_INPUTS.items():
        if tensor_copy.digest(source / relative) != expected:
            raise ValueError('Quantized half-pair contract changed: ' + relative)
    for name, expected in INPUTS[gpu_profile].items():
        if tensor_copy.digest(work / BASE / name) != expected:
            raise ValueError('Quantized half-pair composition input changed: ' + name)
    for name in [HEADER_NAME, *DERIVED_SHA256]:
        if (work / BASE / name).exists() or (source / BASE / name).exists():
            raise ValueError('Quantized half-pair would replace unreviewed source: ' + name)
    if (work / HEADER_NAME).exists():
        raise ValueError('Quantized half-pair embedding already exists')
    patch = patch_root / PATCH_NAME
    if tensor_copy.digest(patch) != PATCH_SHA256:
        raise ValueError('Quantized half-pair patch identity changed')
    for target, original in TEMPLATE_COPIES.items():
        destination = work / BASE / 'wgsl-shaders' / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / BASE / 'wgsl-shaders' / original, destination)
    for flags in (['--check'], []):
        subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags, str(patch.resolve())],
                       cwd=work, env=env, check=True)
    for name, expected in {**OUTPUTS[gpu_profile], **DERIVED_SHA256}.items():
        if tensor_copy.digest(work / BASE / name) != expected:
            raise ValueError('Unexpected quantized half-pair output: ' + name)
    shaders = work / 'quant-half-pairs-only-wgsl'
    shaders.mkdir()
    for name in ('common_decls.tmpl',):
        shutil.copyfile(source / BASE / 'wgsl-shaders' / name, shaders / name)
    for name in TEMPLATE_COPIES:
        shutil.copyfile(work / BASE / 'wgsl-shaders' / name, shaders / name)
    header = work / HEADER_NAME
    generate_wgsl(shaders, None, header, generator=source / BASE / 'wgsl-shaders/embed_wgsl.py')
    if tensor_copy.digest(header) != HEADER_SHA256:
        raise ValueError('Unexpected quantized half-pair embedding identity')
    return {HEADER_NAME: header}
