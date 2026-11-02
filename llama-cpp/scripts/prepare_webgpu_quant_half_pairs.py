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
INPUTS = {False: {'ggml-webgpu.cpp': 'b16a18a3d544577fbfbc3279475743a5c4cd8c4074f7325d7e337f69cd616708',
         'ggml-webgpu-shader-lib.hpp': '6c46dd6288d4a505ccfd5fd8aeb7c1c0880623e2a75919bfba8f08c58a6a72de'},
 True: {'ggml-webgpu.cpp': 'a9b0c18384ba901fc38b73652772f608b7ef221edd9d9f908116097255404e5b',
        'ggml-webgpu-shader-lib.hpp': '595d1f2aa85e5e46cfe23b72d39b613ba3254b9003e0adb7990ad1f1d4efde84'}}
OUTPUTS = {False: {'ggml-webgpu.cpp': 'b16a18a3d544577fbfbc3279475743a5c4cd8c4074f7325d7e337f69cd616708',
         'ggml-webgpu-shader-lib.hpp': 'd7fadf561f5e02020c0aee0b12300ca3aad712bd2c62c40cb01392b74bf76091'},
 True: {'ggml-webgpu.cpp': 'a9b0c18384ba901fc38b73652772f608b7ef221edd9d9f908116097255404e5b',
        'ggml-webgpu-shader-lib.hpp': '9071a656fae93afcfd97e12de8a4e328a1b9b7145bddf053a414996bf4215ed3'}}
CONTRACT_INPUTS = {'ggml/src/ggml-webgpu/ggml-webgpu.cpp': '67e0c1b6ffeec0d943eb63aced20c00951d565cf2bbb964e90a99265259dca8e',
 'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp': '52d39024d24f723dc1b88dc8bf8b4ed18f1e50846010c35065cebc364a789e2d',
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
