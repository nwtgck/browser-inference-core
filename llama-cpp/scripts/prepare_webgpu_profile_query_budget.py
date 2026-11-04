"""Reserve profile queries for a whole legal arena batch, including overshoot."""
from __future__ import annotations
from pathlib import Path
import subprocess
import prepare_webgpu_tensor_copy as tensor_copy

PATCH_NAME = 'ggml-webgpu-profile-query-budget.patch'
PATCH_SHA256 = 'a182e420364f363c47c6c10aedf825f0e5a55613437bddc17cfcafa06cd546a6'
INPUT_SHA256 = 'f4b7d396cede67ae3511dbda62bfe22f51f6c235b3e9bb12942db9f99a06e96f'
OUTPUT_SHA256 = 'b357212a2ba089ff0a1d3efa0e86ac0818aa2643061e2037082d399f898fa132'
CONTRACT_INPUTS = {
    'ggml/src/ggml-webgpu/ggml-webgpu.cpp': '617700adc9a58401414cd6828fa48b4efb3bb0563458e6ebe717360e70c77201',
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
                       cwd=work, env=env, check=True, capture_output=True)
    if tensor_copy.digest(target) != OUTPUT_SHA256:
        raise ValueError('Unexpected profile query budget output identity')
