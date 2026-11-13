"""Compose the bounded dispatch-position bind-group cache into WebGPU source."""
from __future__ import annotations

from pathlib import Path
import subprocess

import prepare_webgpu_tensor_copy as tensor_copy

PATCH_NAME = 'ggml-webgpu-bind-group-cache.patch'
PATCH_SHA256 = 'b1a3a7b61d186446232e7a88ac5d417af239b04891c87b5b72e482957fbba11b'
INPUT_SHA256 = '9d07d2705b5b30c71952c02be3bb902331174a4b219bb5215b60f233d5ebaa88'
OUTPUT_SHA256 = '95ab51d247fde6b917adfdbdd469572884952221051b0a79744bb177c902075d'
CONTRACT_INPUTS = {
    'ggml/src/ggml-webgpu/ggml-webgpu.cpp': 'c3e627af0d18c596dd45f051c7eba612a6f8f53f458bceecd557d8704f52b8ee',
    'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp': 'a110a08b87a2b2768c991281e450cc0e89b5c34f67856b58138b7ca9e9aa5f17',
    'ggml/src/ggml-backend.cpp': '2f724930ea53dd8f50f26706ecf10a177d56601e668f109cf1cd95052b1b9ad8',
    'ggml/src/ggml-backend-impl.h': '7693def12dcc79a10dd2152e7cb659278756dba7622ebe41937f951dc94944e3',
    'ggml/include/ggml-backend.h': '34464da4b46314c710cd634626578c6f2a6811431d41eddaf5e8aa77082fea2a',
    'ggml/include/ggml.h': 'bdb0e75be8eeb1fc14a742015025505f5810d8200cbb40bda7e981101413440b',
}


def prepare(work: Path, source: Path, patch_root: Path, env: dict[str, str]) -> None:
    for relative, expected in CONTRACT_INPUTS.items():
        if tensor_copy.digest(source / relative) != expected:
            raise ValueError('Bind-group cache contract changed: ' + relative)
    target = work / tensor_copy.SOURCE_PATH
    if tensor_copy.digest(target) != INPUT_SHA256:
        raise ValueError('Bind-group cache composition input changed')
    patch = patch_root / PATCH_NAME
    if tensor_copy.digest(patch) != PATCH_SHA256:
        raise ValueError('Bind-group cache patch identity changed')
    for flags in (['--check'], []):
        subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags, str(patch.resolve())],
                       cwd=work, env=env, check=True, capture_output=True)
    if tensor_copy.digest(target) != OUTPUT_SHA256:
        raise ValueError('Unexpected bind-group cache output identity')
