"""Select a source-bound workgroup for short, single-column Q8_0 matvecs.

No WGSL is copied or changed; the normal upstream emitter/preprocessor is used.
"""
from __future__ import annotations

from pathlib import Path
import subprocess

import prepare_webgpu_tensor_copy as tensor_copy

PATCH_NAME = 'ggml-webgpu-q8-short-workgroup.patch'
PATCH_SHA256 = '50cc0c34c23665f39994e2e84ecd3fda77df6bd577bc6f0947ce3e60424e7808'
HEADER_PATH = 'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp'
INPUT_SHA256 = '544c99d99ab4e986989f3220b0327f787ab200c4cbd23e3f6a5ff8901d7ec42c'
OUTPUT_SHA256 = '0e7176e1b2fc7576ce238516fb2dff437d8376a8ee9694ccda60312492fbdbfa'
CONTRACT_INPUTS = {
    'ggml/src/ggml-webgpu/pre_wgsl.hpp': '0d6f781eb5723311cf856e585566c70ddcc779f0e97fcfe6d0e3f636c35688e5',
    'ggml/src/ggml-webgpu/wgsl-shaders/common_decls.tmpl': '1cfe6cae3ab4a4ccac963a1a68ea9ddf373f4b4beefe7c933807f37969bf5555',
    'ggml/src/ggml-webgpu/wgsl-shaders/embed_wgsl.py': '37d4a93bb46b0ff417c31a0c1d3fa4b9db52556556138cb2eaedae9daa8dcbea',
    'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_vec.wgsl': '4fdbbaf3b3d3b7c965786b62a58b6b35ceb6d2f80b29e53f4a027904d3243f18',
    'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_vec_acc.tmpl': 'c938be30a28ea800dda6297dae2a9aee71ddef83f189c077b22bf8c8317d98dc',
    'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_vec_q_acc.tmpl': 'f0353ca818b8922ec11f120e78fefa9d9a4695796663a29aeddc133cce98cf4e',
}



def prepare(work: Path, source: Path, patch_root: Path, env: dict[str, str]) -> None:
    for relative, expected in CONTRACT_INPUTS.items():
        if tensor_copy.digest(source / relative) != expected:
            raise ValueError('Q8 short matvec contract changed: ' + relative)
    target = work / HEADER_PATH
    if tensor_copy.digest(target) != INPUT_SHA256:
        raise ValueError('Q8 short matvec composition input changed')
    patch = patch_root / PATCH_NAME
    if tensor_copy.digest(patch) != PATCH_SHA256:
        raise ValueError('Q8 short matvec patch identity changed')
    for flags in (['--check'], []):
        subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags, str(patch.resolve())],
                       cwd=work, env=env, check=True, capture_output=True)
    if tensor_copy.digest(target) != OUTPUT_SHA256:
        raise ValueError('Unexpected Q8 short matvec output identity')
