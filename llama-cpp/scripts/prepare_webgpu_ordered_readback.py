"""Compose the reviewed same-queue blocking readback hook into WebGPU source."""
from __future__ import annotations

from pathlib import Path
import subprocess

import prepare_webgpu_tensor_copy as tensor_copy

PATCH_NAME = 'ggml-webgpu-ordered-readback.patch'
PATCH_SHA256 = 'f46b9c9b55ee13a02c1a68d682cfe3f5cff6fbf39680e88891590c3f72431325'
INPUT_SHA256 = '1beaf5cc190e79710e840f39184269c75d63d92236d66850f610cf9e3c5e17db'
OUTPUT_SHA256 = '58e74d85b3c3f30d6f4908bb4d039404f0d939411a8fab65d5c09bb97a4c72fd'
# The generic get/get_async/synchronize contracts and backend interface are
# part of this hook's correctness, not just context for applying the patch.
CONTRACT_INPUTS = {
    'ggml/src/ggml-webgpu/ggml-webgpu.cpp': '67e0c1b6ffeec0d943eb63aced20c00951d565cf2bbb964e90a99265259dca8e',
    'ggml/src/ggml-backend.cpp': '79ade6b83b3d042e43e035fa81d2c672211a552af9af38e410a047e4ba2723a0',
    'ggml/src/ggml-backend-impl.h': '4e861c8e23d3d7b0d99d8e0836e68564af58f8597d75d7bbb732857d7a632fa7',
    'ggml/include/ggml-backend.h': '0e5e8e1f16d11ebbf501e38bc6705d79eee819b5f0b17476533f2100ed9674e2',
    'ggml/include/ggml.h': '2b4dd92fa992cb4ef484917a6c62fef5097698744160ccc09214fd0c459c92c7',
}


def prepare(work: Path, source: Path, patch_root: Path, env: dict[str, str]) -> None:
    for relative, expected in CONTRACT_INPUTS.items():
        if tensor_copy.digest(source / relative) != expected:
            raise ValueError('Ordered readback contract changed: ' + relative)
    target = work / tensor_copy.SOURCE_PATH
    if tensor_copy.digest(target) != INPUT_SHA256:
        raise ValueError('Ordered readback composition input changed')
    patch = patch_root / PATCH_NAME
    if tensor_copy.digest(patch) != PATCH_SHA256:
        raise ValueError('Ordered readback patch identity changed')
    for flags in (['--check'], []):
        subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags, str(patch.resolve())],
                       cwd=work, env=env, check=True, capture_output=True)
    if tensor_copy.digest(target) != OUTPUT_SHA256:
        raise ValueError('Unexpected ordered readback output identity')
