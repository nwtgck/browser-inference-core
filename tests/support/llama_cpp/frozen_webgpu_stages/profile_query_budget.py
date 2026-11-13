"""Reserve profile queries for a whole legal arena batch, including overshoot."""
from __future__ import annotations
from pathlib import Path
import subprocess
from tests.support.llama_cpp.frozen_webgpu_stages import tensor_copy
PATCH_NAME = 'ggml-webgpu-profile-query-budget.patch'
PATCH_SHA256 = 'a182e420364f363c47c6c10aedf825f0e5a55613437bddc17cfcafa06cd546a6'
INPUT_SHA256 = '67940a20bba8d06205ad2f6cf1e909450c4fbb6805b5eba59f574c3576d69779'
OUTPUT_SHA256 = '008f4ca05c7d7f866998c6457d293e7d22548dc8ae45afaacb2e97a2bcb1fc61'
CONTRACT_INPUTS = {
    'ggml/src/ggml-webgpu/ggml-webgpu.cpp': '7fa5eca8734b7d31bf467281c698b3356c5750c2f923008c179a4dae426cffe5',
    'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp': '34522d61990406ddc1e46c55da4531e6aa2ad1561bf5bf512ebbb38843dbd49d',
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
