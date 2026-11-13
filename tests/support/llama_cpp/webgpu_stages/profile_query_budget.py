"""Reserve profile queries for a whole legal arena batch, including overshoot."""
from __future__ import annotations
from pathlib import Path
import subprocess
from tests.support.llama_cpp.webgpu_stages import tensor_copy
PATCH_NAME = 'ggml-webgpu-profile-query-budget.patch'
PATCH_SHA256 = 'a182e420364f363c47c6c10aedf825f0e5a55613437bddc17cfcafa06cd546a6'
INPUT_SHA256 = '6035c95aa02f9bb940ea7e67d650d078061986b7857ffa94e220406d026b47a5'
OUTPUT_SHA256 = '8ca2a0773d6467acc35e5e56b5f74ef0e6a58379336cfe9d1b0fe8aec80314d0'
CONTRACT_INPUTS = {
    'ggml/src/ggml-webgpu/ggml-webgpu.cpp': 'c3e627af0d18c596dd45f051c7eba612a6f8f53f458bceecd557d8704f52b8ee',
    'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp': 'a110a08b87a2b2768c991281e450cc0e89b5c34f67856b58138b7ca9e9aa5f17',
}


def prepare(work: Path, source: Path, patch_root: Path, env: dict[str, str]) -> None:
    for relative, expected in CONTRACT_INPUTS.items():
        if tensor_copy.digest(source / relative) != expected:
            raise ValueError('Profile query budget contract changed: ' + relative)
    target = work / tensor_copy.SOURCE_PATH
    if tensor_copy.digest(target) != INPUT_SHA256:
        raise ValueError('Profile query budget composition input changed')
    patch = patch_root / PATCH_NAME
    if tensor_copy.digest(patch) != PATCH_SHA256:
        raise ValueError('Profile query budget patch identity changed')
    for flags in (['--check'], []):
        subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags, str(patch.resolve())],
                       cwd=work, env=env, check=True)
    if tensor_copy.digest(target) != OUTPUT_SHA256:
        raise ValueError('Unexpected profile query budget output identity')
