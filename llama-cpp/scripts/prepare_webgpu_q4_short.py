"""Choose existing upstream workgroups for short single-column Q4_K matvecs.

Apply only to the isolated header composition after the other shader overlays.
The upstream WGSL, quantization and output/dispatch tiling remain unchanged.
"""
from __future__ import annotations
from pathlib import Path
import subprocess
import prepare_webgpu_tensor_copy as tensor_copy

PATCH_NAME = 'ggml-webgpu-q4-short-workgroup.patch'
PATCH_SHA256 = 'f6d95fde6352ac9cf532005a1eeca5209164da7f83dc765f53853b5b99ab892f'
HEADER_PATH = 'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp'
INPUT_SHA256 = '4a4e19985f79f2931f6f426baaf3f1618dfff80bdd2c4de8634df6337c2f92ec'
OUTPUT_SHA256 = 'd0045d4cc1f4fc88a8ed6a24edbe64c6148f5d3eb2e70b7fcabb9346e17900a8'
CONTRACT_INPUTS = {
    'ggml/src/ggml-webgpu/ggml-webgpu.cpp': '995b30ce233bdc23cf918da8916485b568aeac0371bcab458d0df3b7b1d172cf',
    'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp': 'ab446e25ba92b0733bbe957b32adab70be2c3973cd8f412fbbcf2dd86818a289',
    'ggml/src/ggml-webgpu/pre_wgsl.hpp': '0d6f781eb5723311cf856e585566c70ddcc779f0e97fcfe6d0e3f636c35688e5',
    'ggml/src/ggml-webgpu/wgsl-shaders/common_decls.tmpl': '1cfe6cae3ab4a4ccac963a1a68ea9ddf373f4b4beefe7c933807f37969bf5555',
    'ggml/src/ggml-webgpu/wgsl-shaders/embed_wgsl.py': '37d4a93bb46b0ff417c31a0c1d3fa4b9db52556556138cb2eaedae9daa8dcbea',
    'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_id_vec.wgsl': '17c2b16491fbca9f653f963171debf6bc6574eb250f45cfeacae47f71f1b09a7',
    'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_vec.wgsl': '4fdbbaf3b3d3b7c965786b62a58b6b35ceb6d2f80b29e53f4a027904d3243f18',
    'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_vec_acc.tmpl': 'c938be30a28ea800dda6297dae2a9aee71ddef83f189c077b22bf8c8317d98dc',
    'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_vec_q_acc.tmpl': 'f0353ca818b8922ec11f120e78fefa9d9a4695796663a29aeddc133cce98cf4e',
}


def prepare(work: Path, source: Path, patch_root: Path, env: dict[str, str]) -> None:
    for relative, expected in CONTRACT_INPUTS.items():
        if tensor_copy.digest(source / relative) != expected:
            raise ValueError('Q4 short matvec contract changed: ' + relative)
    target = work / HEADER_PATH
    if tensor_copy.digest(target) != INPUT_SHA256:
        raise ValueError('Q4 short matvec composition input changed')
    patch = patch_root / PATCH_NAME
    if tensor_copy.digest(patch) != PATCH_SHA256:
        raise ValueError('Q4 short matvec patch identity changed')
    for flags in (['--check'], []):
        subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags, str(patch.resolve())],
                       cwd=work, env=env, check=True, capture_output=True)
    if tensor_copy.digest(target) != OUTPUT_SHA256:
        raise ValueError('Unexpected Q4 short matvec output identity')
