"""Reduce synchronous host dispatch assembly allocations; preserve queue ownership."""
from __future__ import annotations
from pathlib import Path
import subprocess
import prepare_webgpu_tensor_copy as tensor_copy

PATCH_NAME = 'ggml-webgpu-host-encode-assembly.patch'
PATCH_SHA256 = '36e80570dce3c369739d406e73955d314c7fb435944e776019ea1666fa8c2252'
INPUT_SHA256 = '36a2bf151a18cc4b8d56547811d7aae6945462d01476ca9be3902b1a3e7078e3'
OUTPUT_SHA256 = '6035c95aa02f9bb940ea7e67d650d078061986b7857ffa94e220406d026b47a5'
CONTRACT_INPUTS = {
    'ggml/src/ggml-webgpu/ggml-webgpu.cpp': 'c3e627af0d18c596dd45f051c7eba612a6f8f53f458bceecd557d8704f52b8ee',
    'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp': 'a110a08b87a2b2768c991281e450cc0e89b5c34f67856b58138b7ca9e9aa5f17',
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
