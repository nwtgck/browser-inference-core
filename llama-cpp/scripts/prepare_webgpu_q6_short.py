"""Reuse the existing short workgroups for ordinary single-column Q6_K matvecs."""
from pathlib import Path
import subprocess
import prepare_webgpu_q4_short as q4_short
import prepare_webgpu_tensor_copy as tensor_copy

BASE = Path('ggml/src/ggml-webgpu')
PATCH_NAME = 'ggml-webgpu-q6-short-workgroup.patch'
PATCH_SHA256 = '511182db3d0d0f64d49ec53188e815c34ed9da62bdc2aebbd862853f26f5095d'
# Pin the shared Q4/Q6 decoder, canonical main, preprocessor and host contracts.
CONTRACT_INPUTS = dict(q4_short.CONTRACT_INPUTS)
INPUTS = {False: {'ggml-webgpu.cpp': '6a99af824c58601ccc633e84c266f2efc7091f3616d2552cfe0cf91c312c9318',
         'ggml-webgpu-shader-lib.hpp': 'f6e586c2bcfc312dcc1cd953e3f0244e4f7cd8696f399e3dfb600cd3c492a5b8'},
 True: {'ggml-webgpu.cpp': 'e55179d4198ebd17d3ab0ec20d0736211a823649be2959e63934f6e3670ed451',
        'ggml-webgpu-shader-lib.hpp': 'a1e963e47e7c424783c955ac75b13f6757c8a574797fb0da0cec2a7601019ef3'}}
OUTPUTS = {False: {'ggml-webgpu.cpp': '6a99af824c58601ccc633e84c266f2efc7091f3616d2552cfe0cf91c312c9318',
         'ggml-webgpu-shader-lib.hpp': 'ec4986b146e05a354eaca1af8f354e66fe5a6907973feb2f2fde2906e842efe7'},
 True: {'ggml-webgpu.cpp': 'e55179d4198ebd17d3ab0ec20d0736211a823649be2959e63934f6e3670ed451',
        'ggml-webgpu-shader-lib.hpp': 'd0e1e291969325c74a9c90fc5d1d57780bcddc188c9c471e5a5f59f8caed5faf'}}


def prepare(work: Path, source: Path, patch_root: Path, env: dict[str, str], *, gpu_profile: bool) -> None:
    for relative, expected in CONTRACT_INPUTS.items():
        if tensor_copy.digest(source / relative) != expected:
            raise ValueError('Q6 short matvec contract changed: ' + relative)
    for name, expected in INPUTS[gpu_profile].items():
        if tensor_copy.digest(work / BASE / name) != expected:
            raise ValueError('Q6 short matvec composition input changed: ' + name)
    patch = patch_root / PATCH_NAME
    if tensor_copy.digest(patch) != PATCH_SHA256:
        raise ValueError('Q6 short matvec patch identity changed')
    for flags in (['--check'], []):
        subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags, str(patch.resolve())],
                       cwd=work, env=env, check=True, capture_output=True)
    for name, expected in OUTPUTS[gpu_profile].items():
        if tensor_copy.digest(work / BASE / name) != expected:
            raise ValueError('Unexpected Q6 short matvec output identity: ' + name)
