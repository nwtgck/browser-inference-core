"""Choose the upstream WG_SIZE for short single-token Q5_1 / IQ4_NL experts.

The checked shader and quantization code stay unchanged. Apply only to the
isolated composition after the existing overlays, never to the vendor checkout.
"""
from __future__ import annotations
from pathlib import Path
import subprocess
from tests.support.llama_cpp.frozen_webgpu_stages import tensor_copy
PATCH_NAME = 'ggml-webgpu-expert-short-workgroup.patch'
PATCH_SHA256 = '6f15368df5aaab2a4c0408e0e71d02f7c017de89971f8373ceec31f1ee9d77b8'
COMPOSITION_INPUTS = {
    'ggml/src/ggml-webgpu/ggml-webgpu.cpp': '917095a77677f014f42c4967ba5011cf3c72b44bcc5bf074dc5c97e57ca59550',
    'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp': 'f03dc8d390d4e68fa10b2aed66740c33b1217e1e0f4348b034ff27baed328bca',
}
COMPOSITION_OUTPUTS = {
    'ggml/src/ggml-webgpu/ggml-webgpu.cpp': '9a1d8759a0aedba59d86bb56adc3ce449f1b42363a3265ccd68c3eab1dfb00cf',
    'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp': 'e0d7172cf03f148a0ef6cad9e2047d4534ef0c01eec516e91ecb36e62cafa25d',
}
CONTRACT_INPUTS = {
    'ggml/src/ggml-webgpu/pre_wgsl.hpp': '0d6f781eb5723311cf856e585566c70ddcc779f0e97fcfe6d0e3f636c35688e5',
    'ggml/src/ggml-webgpu/wgsl-shaders/common_decls.tmpl': '1cfe6cae3ab4a4ccac963a1a68ea9ddf373f4b4beefe7c933807f37969bf5555',
    'ggml/src/ggml-webgpu/wgsl-shaders/embed_wgsl.py': '37d4a93bb46b0ff417c31a0c1d3fa4b9db52556556138cb2eaedae9daa8dcbea',
    'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_id_vec.wgsl': 'cbe87da8b6eb80fcb7905855fe1d212dc5a0afac55b77600c507fbbd2fd7c31d',
    'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_vec_acc.tmpl': '9421bdb7ac569e03e8483365653c28b95e659702f8ff849310b2c077ee64290f',
}


def prepare(work: Path, source: Path, patch_root: Path, env: dict[str, str]) -> None:
    for relative, expected in CONTRACT_INPUTS.items():
        if tensor_copy.digest(source / relative) != expected:
            raise ValueError('Expert short matvec contract changed: ' + relative)
    for relative, expected in COMPOSITION_INPUTS.items():
        if tensor_copy.digest(work / relative) != expected:
            raise ValueError('Expert short matvec composition input changed: ' + relative)
    patch = patch_root / PATCH_NAME
    if tensor_copy.digest(patch) != PATCH_SHA256:
        raise ValueError('Expert short matvec patch identity changed')
    for flags in (['--check'], []):
        subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags, str(patch.resolve())],
                       cwd=work, env=env, check=True)
    for relative, expected in COMPOSITION_OUTPUTS.items():
        if tensor_copy.digest(work / relative) != expected:
            raise ValueError('Unexpected expert short matvec output identity: ' + relative)
