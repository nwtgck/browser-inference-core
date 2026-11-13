"""Avoid Emdawn's temporary mapped copy for aligned tensor readback."""
from pathlib import Path
import subprocess
import prepare_webgpu_ordered_readback as ordered_readback
import prepare_webgpu_tensor_copy as tensor_copy

BASE = Path('ggml/src/ggml-webgpu')
PATCH_NAME = 'ggml-webgpu-direct-mapped-read.patch'
PATCH_SHA256 = 'a30e41de9211e13e34f7241731e3ed17330d0094762513cc77c08e6e54e84621'
CONTRACT_INPUTS = dict(ordered_readback.CONTRACT_INPUTS)
# The existing source-preparation CLI verifies these pinned API headers.
# ReadMappedRange is used only by Emdawn; native mapping remains unchanged.
REVIEWED_DAWN_HEADERS = dict(tensor_copy.DAWN_HEADERS)
INPUTS = {False: {'ggml-webgpu.cpp': 'c50aeea2606f5da330e6b5f2d26115b94579acd87bb90ae3497415879b90a100',
                 'ggml-webgpu-shader-lib.hpp': '543c78b27a615dbb2cc52453270092fbb0b3a088ff0c5829b1842cc7ec9ae30e'},
          True: {'ggml-webgpu.cpp': 'f7c1281458464c910ebbbbb9f7e679a20580498fdc7ce9cff5000f1f6dde1607',
                 'ggml-webgpu-shader-lib.hpp': '611c89c046020d22afcf3211a70ce81266bab3c59328168e365d2cd1ea148bf2'}}
OUTPUTS = {False: {'ggml-webgpu.cpp': '23873d267578c6bce435eff0f7643f2853f705ad0b46b3ed8e93cbaada2bb8a2',
                  'ggml-webgpu-shader-lib.hpp': '543c78b27a615dbb2cc52453270092fbb0b3a088ff0c5829b1842cc7ec9ae30e'},
           True: {'ggml-webgpu.cpp': 'de513ccb952c3cc1da2e42f45f4ed1acd9433f6bdea3e38a61f951f437b9c79c',
                  'ggml-webgpu-shader-lib.hpp': '611c89c046020d22afcf3211a70ce81266bab3c59328168e365d2cd1ea148bf2'}}


def prepare(work: Path, source: Path, patch_root: Path, env: dict[str, str], *, gpu_profile: bool) -> None:
    for name, expected in CONTRACT_INPUTS.items():
        if tensor_copy.digest(source / name) != expected:
            raise ValueError('Direct mapped read contract changed: ' + name)
    for name, expected in INPUTS[gpu_profile].items():
        if tensor_copy.digest(work / BASE / name) != expected:
            raise ValueError('Direct mapped read composition input changed: ' + name)
    patch = patch_root / PATCH_NAME
    if tensor_copy.digest(patch) != PATCH_SHA256:
        raise ValueError('Direct mapped read patch identity changed')
    for flags in (['--check'], []):
        subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags, str(patch.resolve())],
                       cwd=work, env=env, check=True, capture_output=True)
    for name, expected in OUTPUTS[gpu_profile].items():
        if tensor_copy.digest(work / BASE / name) != expected:
            raise ValueError('Unexpected direct mapped read output: ' + name)
