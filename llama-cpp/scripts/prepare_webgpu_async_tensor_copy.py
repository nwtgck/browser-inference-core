"""Checked ordered CPU/WebGPU copies through the existing optional copy hook."""
from pathlib import Path
import subprocess
import prepare_webgpu_tensor_copy as tensor_copy

PATCH_NAME = 'ggml-webgpu-async-tensor-copy.patch'
PATCH_SHA256 = '7643f09e513bfaa9aca614644466b5d05ae4419f9edcb3f920c480644d247022'
INPUT_SHA256 = '5dfefacf79403c98d81e86aec28d4b3476ea70393d58753da6f586a66ba5a7a0'
OUTPUT_SHA256 = '759d6def005a8c0878c656bbbef0144a5cccd1be4f1bdc1487ee142ec66596f3'
CONTRACT_INPUTS = {
    'ggml/src/ggml.c': '42b238412cda218552812a37329fab95a2f94be6d8bff5139083450d432f796d',
    'ggml/src/ggml-webgpu/ggml-webgpu.cpp': '995b30ce233bdc23cf918da8916485b568aeac0371bcab458d0df3b7b1d172cf',
    'ggml/src/ggml-backend.cpp': '43dea13b13c2a15afd83f53515f3488c17072fc107a6ba4a5e0370f90b4fe7ff',
    'ggml/src/ggml-backend-impl.h': '4e861c8e23d3d7b0d99d8e0836e68564af58f8597d75d7bbb732857d7a632fa7',
    'ggml/include/ggml-backend.h': '0e5e8e1f16d11ebbf501e38bc6705d79eee819b5f0b17476533f2100ed9674e2',
    'ggml/include/ggml.h': '475beaa7e959be8c383279d9ce9d5191f37426e6b338270cfb12b26a798129d3',
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
