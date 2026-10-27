"""Compose the bounded dispatch-position bind-group cache into WebGPU source."""
from __future__ import annotations

from pathlib import Path
import subprocess

import prepare_webgpu_tensor_copy as tensor_copy

PATCH_NAME = 'ggml-webgpu-bind-group-cache.patch'
PATCH_SHA256 = 'b1a3a7b61d186446232e7a88ac5d417af239b04891c87b5b72e482957fbba11b'
INPUT_SHA256 = '58e74d85b3c3f30d6f4908bb4d039404f0d939411a8fab65d5c09bb97a4c72fd'
OUTPUT_SHA256 = '5bc700928737cbd1afe4ac9546f8d2566583ea65b42668857bb83db7c0ecc30b'
CONTRACT_INPUTS = {
    'ggml/src/ggml-webgpu/ggml-webgpu.cpp': '67e0c1b6ffeec0d943eb63aced20c00951d565cf2bbb964e90a99265259dca8e',
    'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp': '52d39024d24f723dc1b88dc8bf8b4ed18f1e50846010c35065cebc364a789e2d',
    'ggml/src/ggml-backend.cpp': '79ade6b83b3d042e43e035fa81d2c672211a552af9af38e410a047e4ba2723a0',
    'ggml/src/ggml-backend-impl.h': '4e861c8e23d3d7b0d99d8e0836e68564af58f8597d75d7bbb732857d7a632fa7',
    'ggml/include/ggml-backend.h': '0e5e8e1f16d11ebbf501e38bc6705d79eee819b5f0b17476533f2100ed9674e2',
    'ggml/include/ggml.h': '2b4dd92fa992cb4ef484917a6c62fef5097698744160ccc09214fd0c459c92c7',
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
