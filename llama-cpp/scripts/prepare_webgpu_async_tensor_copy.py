"""Checked ordered CPU/WebGPU copies through the existing optional copy hook."""
from pathlib import Path
import subprocess
import prepare_webgpu_tensor_copy as tensor_copy

PATCH_NAME = 'ggml-webgpu-async-tensor-copy.patch'
PATCH_SHA256 = '7643f09e513bfaa9aca614644466b5d05ae4419f9edcb3f920c480644d247022'
INPUT_SHA256 = '753829f410cd68a2dc5a83f6c5be055e30a32aae8fed1f45e70703589f2b1a40'
OUTPUT_SHA256 = '57007fb929f38738bf6ba6d1aa76d8772822d9c2aa30410ca83f4d7e9c0d1d39'
CONTRACT_INPUTS = {
    'ggml/src/ggml.c': 'c0e81f0ba55ad76020ada29b01385a7a742c800971e7c974c425a08839750b49',
    'ggml/src/ggml-webgpu/ggml-webgpu.cpp': '617700adc9a58401414cd6828fa48b4efb3bb0563458e6ebe717360e70c77201',
    'ggml/src/ggml-backend.cpp': '7ea26359e7eda6957227d6e4c2a24a96aed622cbbca300b066580f7f3d95df5e',
    'ggml/src/ggml-backend-impl.h': '7693def12dcc79a10dd2152e7cb659278756dba7622ebe41937f951dc94944e3',
    'ggml/include/ggml-backend.h': '34464da4b46314c710cd634626578c6f2a6811431d41eddaf5e8aa77082fea2a',
    'ggml/include/ggml.h': '1aef7a5f465b1c19de305dd6f9a0c303f88e485133a94b9a0ee1cce6345c9101',
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
