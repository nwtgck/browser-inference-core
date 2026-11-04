"""Reduce synchronous host dispatch assembly allocations; preserve queue ownership."""
from __future__ import annotations
from pathlib import Path
import subprocess
import prepare_webgpu_tensor_copy as tensor_copy

PATCH_NAME = 'ggml-webgpu-host-encode-assembly.patch'
PATCH_SHA256 = '36e80570dce3c369739d406e73955d314c7fb435944e776019ea1666fa8c2252'
INPUT_SHA256 = 'ca2f94d35fdddd1af686da72808960d790c4d13fbf5e69aea95547196128b0ca'
OUTPUT_SHA256 = 'f4b7d396cede67ae3511dbda62bfe22f51f6c235b3e9bb12942db9f99a06e96f'
CONTRACT_INPUTS = {
    'ggml/src/ggml-webgpu/ggml-webgpu.cpp': '617700adc9a58401414cd6828fa48b4efb3bb0563458e6ebe717360e70c77201',
    'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp': '34522d61990406ddc1e46c55da4531e6aa2ad1561bf5bf512ebbb38843dbd49d',
}


def prepare(work: Path, source: Path, patch_root: Path, env: dict[str, str]) -> None:
    for relative, expected in CONTRACT_INPUTS.items():
        if tensor_copy.digest(source / relative) != expected:
            raise ValueError('Host encode assembly contract changed: ' + relative)
    target = work / tensor_copy.SOURCE_PATH
    if tensor_copy.digest(target) != INPUT_SHA256:
        raise ValueError('Host encode assembly composition input changed')
    patch = patch_root / PATCH_NAME
    if tensor_copy.digest(patch) != PATCH_SHA256:
        raise ValueError('Host encode assembly patch identity changed')
    for flags in (['--check'], []):
        subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags, str(patch.resolve())],
                       cwd=work, env=env, check=True, capture_output=True)
    if tensor_copy.digest(target) != OUTPUT_SHA256:
        raise ValueError('Unexpected host encode assembly output identity')
