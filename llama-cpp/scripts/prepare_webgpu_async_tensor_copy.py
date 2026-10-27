"""Checked ordered CPU/WebGPU copies through the existing optional copy hook."""
from pathlib import Path
import subprocess
import prepare_webgpu_tensor_copy as tensor_copy

PATCH_NAME = 'ggml-webgpu-async-tensor-copy.patch'
PATCH_SHA256 = '7643f09e513bfaa9aca614644466b5d05ae4419f9edcb3f920c480644d247022'
INPUT_SHA256 = 'f2baa1f5238c61dd52e4eb847729308a597bb6ee00ec7bb79c8540df8de57353'
OUTPUT_SHA256 = 'c99d86ba5a7684c12a1d3840acd11badcb2058e06283a1d5d2f1d9275013afe2'
CONTRACT_INPUTS = {
    'ggml/src/ggml.c': '45e90696c4a766d5747f3443f817cd3be11f0110cf93cd16926698c0f49ed083',
    'ggml/src/ggml-webgpu/ggml-webgpu.cpp': '67e0c1b6ffeec0d943eb63aced20c00951d565cf2bbb964e90a99265259dca8e',
    'ggml/src/ggml-backend.cpp': '79ade6b83b3d042e43e035fa81d2c672211a552af9af38e410a047e4ba2723a0',
    'ggml/src/ggml-backend-impl.h': '4e861c8e23d3d7b0d99d8e0836e68564af58f8597d75d7bbb732857d7a632fa7',
    'ggml/include/ggml-backend.h': '0e5e8e1f16d11ebbf501e38bc6705d79eee819b5f0b17476533f2100ed9674e2',
    'ggml/include/ggml.h': '2b4dd92fa992cb4ef484917a6c62fef5097698744160ccc09214fd0c459c92c7',
}


def prepare(work: Path, source: Path, patch_root: Path, env: dict[str, str]) -> None:
    for relative, expected in CONTRACT_INPUTS.items():
        if tensor_copy.digest(source / relative) != expected:
            raise ValueError('Async tensor-copy contract changed: ' + relative)
    target = work / tensor_copy.SOURCE_PATH
    if tensor_copy.digest(target) != INPUT_SHA256:
        raise ValueError('Async tensor-copy composition input changed')
    patch = patch_root / PATCH_NAME
    if tensor_copy.digest(patch) != PATCH_SHA256:
        raise ValueError('Async tensor-copy patch identity changed')
    for flags in (['--check'], []):
        subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags, str(patch.resolve())],
                       cwd=work, env=env, check=True, capture_output=True)
    if tensor_copy.digest(target) != OUTPUT_SHA256:
        raise ValueError('Unexpected async tensor-copy output identity')
