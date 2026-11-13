"""Checked F32 small-M register-tile geometry; preserve canonical TK8 arithmetic."""
from pathlib import Path
import subprocess
import prepare_webgpu_tensor_copy as tensor_copy

BASE = Path('ggml/src/ggml-webgpu')
HEADER_NAME = 'ggml-webgpu-shader-lib.hpp'
HEADER_PATH = BASE / HEADER_NAME
PATCH_NAME = 'ggml-webgpu-f32-small-m.patch'
PATCH_SHA256 = '72db0465a23aa40d55fb20061b20ac8d3d473dea964a0b36e1e607092e01cc29'
INPUT_SHA256 = 'df28317ed56afbdf2a1d0e57a020d9602dd90c5aaed2d463037d77454ef9a57c'
OUTPUT_SHA256 = '17bc04661859b9b96e624b58d6ebceb3f5d89151594650893a9680ef0df79491'
CONTRACT_INPUTS = {
    'ggml/include/ggml.h': 'bdb0e75be8eeb1fc14a742015025505f5810d8200cbb40bda7e981101413440b',
    'ggml/src/ggml-webgpu/ggml-webgpu.cpp': 'c3e627af0d18c596dd45f051c7eba612a6f8f53f458bceecd557d8704f52b8ee',
    'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp': 'a110a08b87a2b2768c991281e450cc0e89b5c34f67856b58138b7ca9e9aa5f17',
    'ggml/src/ggml-webgpu/pre_wgsl.hpp': '0d6f781eb5723311cf856e585566c70ddcc779f0e97fcfe6d0e3f636c35688e5',
    'ggml/src/ggml-webgpu/wgsl-shaders/embed_wgsl.py': '37d4a93bb46b0ff417c31a0c1d3fa4b9db52556556138cb2eaedae9daa8dcbea',
    'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_reg_tile.wgsl': '2d1cb611caf10b72d20b021772806ca4b3ea4f40c9d7a93941e135dacf859f81',
    'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_decls.tmpl': 'c461aa2108b752504f47338e41b14ee6570abdd831cd83652e052147c89d69ef',
    'ggml/src/ggml-webgpu/wgsl-shaders/common_decls.tmpl': '1cfe6cae3ab4a4ccac963a1a68ea9ddf373f4b4beefe7c933807f37969bf5555',
    'ggml/src/ggml-webgpu/wgsl-shaders/quant_inner_loops.tmpl': '7f023ec774a11265985d92fb630fb94cd8e4599499f941983d1b23b37dacf5ad',
}


def prepare(work: Path, source: Path, patch_root: Path, env: dict[str, str]) -> None:
    for relative, expected in CONTRACT_INPUTS.items():
        if tensor_copy.digest(source / relative) != expected:
            raise ValueError('F32 small-M contract changed: ' + relative)
    target = work / HEADER_PATH
    if tensor_copy.digest(target) != INPUT_SHA256:
        raise ValueError('F32 small-M composition input changed')
    patch = patch_root / PATCH_NAME
    if tensor_copy.digest(patch) != PATCH_SHA256:
        raise ValueError('F32 small-M patch identity changed')
    for flags in (['--check'], []):
        subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags, str(patch.resolve())],
                       cwd=work, env=env, check=True, capture_output=True)
    if tensor_copy.digest(target) != OUTPUT_SHA256:
        raise ValueError('Unexpected F32 small-M output identity')
