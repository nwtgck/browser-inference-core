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
PATCH_SHA256 = '44d631151bf14d0e02643bf52a424f66050304df84a182630cd41705e0081387'
BASE = Path('ggml/src/ggml-webgpu')
HEADER_NAME = 'ggml-webgpu-iq3s-id-subgroup-shaders.hpp'
SHADER_NAME = 'iq3s_id_subgroup.wgsl'
INPUTS = {
    'ggml-webgpu.cpp': '759d6def005a8c0878c656bbbef0144a5cccd1be4f1bdc1487ee142ec66596f3',
    'ggml-webgpu-shader-lib.hpp': 'ce7d5423fa56e604a4d601a6d0c4ca5c10f598f203ab011cec34107a9e855f20',
}
OUTPUTS = {
    'ggml-webgpu.cpp': '3de621e1507ba7b7dde3952742ffb9542f536f54e992d75bc257371481daa1bc',
    'ggml-webgpu-shader-lib.hpp': '9ce1f2986675d50f0b99b3a23877bf8f9dd673760f39e8822ba26655d3a83438',
    'wgsl-shaders/iq3s_id_subgroup.wgsl': '065a437ab2d7119caa750a711b6d8b29d1613b305381a637e3b6c37fb2ff481c',
}
HEADER_SHA256 = '5a48122966743cd4bf74087fe511cc716d53fce11fe7592e4c82579f4e8529c4'
CONTRACT_INPUTS = {
    'ggml/src/ggml-common.h': '0061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c',
    'ggml/src/ggml-webgpu/ggml-webgpu.cpp': '995b30ce233bdc23cf918da8916485b568aeac0371bcab458d0df3b7b1d172cf',
    'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp': 'ab446e25ba92b0733bbe957b32adab70be2c3973cd8f412fbbcf2dd86818a289',
    'ggml/src/ggml-webgpu/pre_wgsl.hpp': '0d6f781eb5723311cf856e585566c70ddcc779f0e97fcfe6d0e3f636c35688e5',
    'ggml/src/ggml-webgpu/wgsl-shaders/common_decls.tmpl': '1cfe6cae3ab4a4ccac963a1a68ea9ddf373f4b4beefe7c933807f37969bf5555',
    'ggml/src/ggml-webgpu/wgsl-shaders/embed_wgsl.py': '37d4a93bb46b0ff417c31a0c1d3fa4b9db52556556138cb2eaedae9daa8dcbea',
    'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_id_vec.wgsl': '17c2b16491fbca9f653f963171debf6bc6574eb250f45cfeacae47f71f1b09a7',
    'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_vec_acc.tmpl': 'c938be30a28ea800dda6297dae2a9aee71ddef83f189c077b22bf8c8317d98dc',
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
