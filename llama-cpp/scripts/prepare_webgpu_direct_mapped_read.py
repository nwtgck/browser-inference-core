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
INPUTS = {False: {'ggml-webgpu.cpp': '57a0941372d9daf59f80396df268a44748d8032b8531e2067c4fae7eeb3e6d72',
                 'ggml-webgpu-shader-lib.hpp': '341249f8670c2266f53b092e50306a7d73abacb1faf6e1787002d7386560885b'},
          True: {'ggml-webgpu.cpp': 'cf738a47c8593290544480f7939d3f64ba88d9557d0e9438be114adc85c3a62a',
                 'ggml-webgpu-shader-lib.hpp': '7ad71c726623e5b65e6f195090246d84bcf6cf79d9fe368be3601facdd1400a5'}}
OUTPUTS = {False: {'ggml-webgpu.cpp': '34e8d902c07a8770ae473e082b01aa592b5cae486d1541cc4b0a0dfac16c1941',
                  'ggml-webgpu-shader-lib.hpp': '341249f8670c2266f53b092e50306a7d73abacb1faf6e1787002d7386560885b'},
           True: {'ggml-webgpu.cpp': '08663d3d2dde4c0345fbeb984d4e73950628365ec1e56321f873957382db7fe2',
                  'ggml-webgpu-shader-lib.hpp': '7ad71c726623e5b65e6f195090246d84bcf6cf79d9fe368be3601facdd1400a5'}}


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
