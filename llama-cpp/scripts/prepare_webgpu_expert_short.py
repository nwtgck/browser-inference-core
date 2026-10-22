"""Choose the upstream WG_SIZE for short single-token Q5_1 / IQ4_NL experts.

The checked shader and quantization code stay unchanged. Apply only to the
isolated composition after the existing overlays, never to the vendor checkout.
"""
from __future__ import annotations
from pathlib import Path
import subprocess
import prepare_webgpu_tensor_copy as tensor_copy

PATCH_NAME = 'ggml-webgpu-expert-short-workgroup.patch'
PATCH_SHA256 = '6f15368df5aaab2a4c0408e0e71d02f7c017de89971f8373ceec31f1ee9d77b8'
COMPOSITION_INPUTS = {
    'ggml/src/ggml-webgpu/ggml-webgpu.cpp': 'fc2f93acf5923e823a0d986a1634918248cfa32901d64e4c0aae13f3524737f8',
    'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp': 'f934c06f188621fd5ffbc022b4241dc5c7654d8755a8d6e156e27bda346153f5',
}
COMPOSITION_OUTPUTS = {
    'ggml/src/ggml-webgpu/ggml-webgpu.cpp': 'e17ea9ace857f4086d4f25041dc507a90baaf7a937987b16f75c50d80c36b140',
    'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp': '4a4e19985f79f2931f6f426baaf3f1618dfff80bdd2c4de8634df6337c2f92ec',
}
CONTRACT_INPUTS = {
    'ggml/src/ggml-webgpu/pre_wgsl.hpp': '0d6f781eb5723311cf856e585566c70ddcc779f0e97fcfe6d0e3f636c35688e5',
    'ggml/src/ggml-webgpu/wgsl-shaders/common_decls.tmpl': '1cfe6cae3ab4a4ccac963a1a68ea9ddf373f4b4beefe7c933807f37969bf5555',
    'ggml/src/ggml-webgpu/wgsl-shaders/embed_wgsl.py': '37d4a93bb46b0ff417c31a0c1d3fa4b9db52556556138cb2eaedae9daa8dcbea',
    'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_id_vec.wgsl': '17c2b16491fbca9f653f963171debf6bc6574eb250f45cfeacae47f71f1b09a7',
    'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_vec_acc.tmpl': 'c938be30a28ea800dda6297dae2a9aee71ddef83f189c077b22bf8c8317d98dc',
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
                       cwd=work, env=env, check=True, capture_output=True)
    for relative, expected in COMPOSITION_OUTPUTS.items():
        if tensor_copy.digest(work / relative) != expected:
            raise ValueError('Unexpected expert short matvec output identity: ' + relative)
