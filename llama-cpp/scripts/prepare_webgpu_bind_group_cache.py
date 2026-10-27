"""Compose the bounded dispatch-position bind-group cache into WebGPU source."""
from __future__ import annotations

from pathlib import Path
import subprocess

import prepare_webgpu_tensor_copy as tensor_copy

PATCH_NAME = 'ggml-webgpu-bind-group-cache.patch'
PATCH_SHA256 = 'b1a3a7b61d186446232e7a88ac5d417af239b04891c87b5b72e482957fbba11b'
INPUT_SHA256 = 'fe53146ac803476cc713e3452d3613cfec3d04036d984f9f8c3cdb59f7341c67'
OUTPUT_SHA256 = 'b5ac93bbf834718ef60002d71b2a60d52dc605a3b64d5c90305599cc07f7d19e'
CONTRACT_INPUTS = {
    'ggml/src/ggml-webgpu/ggml-webgpu.cpp': '995b30ce233bdc23cf918da8916485b568aeac0371bcab458d0df3b7b1d172cf',
    'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp': 'ab446e25ba92b0733bbe957b32adab70be2c3973cd8f412fbbcf2dd86818a289',
    'ggml/src/ggml-backend.cpp': '43dea13b13c2a15afd83f53515f3488c17072fc107a6ba4a5e0370f90b4fe7ff',
    'ggml/src/ggml-backend-impl.h': '4e861c8e23d3d7b0d99d8e0836e68564af58f8597d75d7bbb732857d7a632fa7',
    'ggml/include/ggml-backend.h': '0e5e8e1f16d11ebbf501e38bc6705d79eee819b5f0b17476533f2100ed9674e2',
    'ggml/include/ggml.h': '475beaa7e959be8c383279d9ce9d5191f37426e6b338270cfb12b26a798129d3',
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
