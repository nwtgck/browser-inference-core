"""Avoid Emdawn's temporary mapped copy for aligned tensor readback."""
from pathlib import Path
import subprocess
from tests.support.llama_cpp.frozen_webgpu_stages import ordered_readback
from tests.support.llama_cpp.frozen_webgpu_stages import tensor_copy
BASE = Path('ggml/src/ggml-webgpu')
PATCH_NAME = 'ggml-webgpu-direct-mapped-read.patch'
PATCH_SHA256 = 'a30e41de9211e13e34f7241731e3ed17330d0094762513cc77c08e6e54e84621'
CONTRACT_INPUTS = dict(ordered_readback.CONTRACT_INPUTS)
# The existing source-preparation CLI verifies these pinned API headers.
# ReadMappedRange is used only by Emdawn; native mapping remains unchanged.
REVIEWED_DAWN_HEADERS = dict(tensor_copy.DAWN_HEADERS)
INPUTS = {False: {'ggml-webgpu.cpp': '6a99af824c58601ccc633e84c266f2efc7091f3616d2552cfe0cf91c312c9318',
                 'ggml-webgpu-shader-lib.hpp': '341249f8670c2266f53b092e50306a7d73abacb1faf6e1787002d7386560885b'},
          True: {'ggml-webgpu.cpp': 'e55179d4198ebd17d3ab0ec20d0736211a823649be2959e63934f6e3670ed451',
                 'ggml-webgpu-shader-lib.hpp': '7ad71c726623e5b65e6f195090246d84bcf6cf79d9fe368be3601facdd1400a5'}}
OUTPUTS = {False: {'ggml-webgpu.cpp': 'c977f258c8ec2b98e07f69388e1bc22a00646f76c559ce5afee533d7ff6a0c76',
                  'ggml-webgpu-shader-lib.hpp': '341249f8670c2266f53b092e50306a7d73abacb1faf6e1787002d7386560885b'},
           True: {'ggml-webgpu.cpp': '62e142cdac82c751f3c758e264c3750ea420429f58a0fa883414322756aa7268',
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
                       cwd=work, env=env, check=True)
    for name, expected in OUTPUTS[gpu_profile].items():
        if tensor_copy.digest(work / BASE / name) != expected:
            raise ValueError('Unexpected direct mapped read output: ' + name)
