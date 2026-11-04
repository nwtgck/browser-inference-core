"""Checked paired-half shared storage for fixed-geometry Q4 prefill kernels."""
from pathlib import Path
import shutil
import subprocess
import sys
import prepare_webgpu_tensor_copy as tensor_copy

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
INPUTS = {False: {'ggml-webgpu.cpp': '57a0941372d9daf59f80396df268a44748d8032b8531e2067c4fae7eeb3e6d72',
         'ggml-webgpu-shader-lib.hpp': 'ec4986b146e05a354eaca1af8f354e66fe5a6907973feb2f2fde2906e842efe7'},
 True: {'ggml-webgpu.cpp': 'cf738a47c8593290544480f7939d3f64ba88d9557d0e9438be114adc85c3a62a',
        'ggml-webgpu-shader-lib.hpp': 'd0e1e291969325c74a9c90fc5d1d57780bcddc188c9c471e5a5f59f8caed5faf'}}
OUTPUTS = {False: {'ggml-webgpu.cpp': '57a0941372d9daf59f80396df268a44748d8032b8531e2067c4fae7eeb3e6d72',
         'ggml-webgpu-shader-lib.hpp': '341249f8670c2266f53b092e50306a7d73abacb1faf6e1787002d7386560885b'},
 True: {'ggml-webgpu.cpp': 'cf738a47c8593290544480f7939d3f64ba88d9557d0e9438be114adc85c3a62a',
        'ggml-webgpu-shader-lib.hpp': '7ad71c726623e5b65e6f195090246d84bcf6cf79d9fe368be3601facdd1400a5'}}
CONTRACT_INPUTS = {'ggml/src/ggml-webgpu/ggml-webgpu.cpp': '617700adc9a58401414cd6828fa48b4efb3bb0563458e6ebe717360e70c77201',
 'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp': '34522d61990406ddc1e46c55da4531e6aa2ad1561bf5bf512ebbb38843dbd49d',
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
                       cwd=work, env=env, check=True, capture_output=True)
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
    subprocess.run([sys.executable, str(source / BASE / 'wgsl-shaders/embed_wgsl.py'),
                    '--input_dir', str(shaders), '--output_file', str(header)], check=True, capture_output=True)
    if tensor_copy.digest(header) != HEADER_SHA256:
        raise ValueError('Unexpected quantized half-pair embedding identity')
    return {HEADER_NAME: header}
