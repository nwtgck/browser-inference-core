"""Use the exact routing-independent capacity for existing MoE slot tiles."""
from pathlib import Path
import subprocess

from tests.support.llama_cpp.frozen_webgpu_stages import tensor_copy
BASE = Path('ggml/src/ggml-webgpu')
PATCH_NAME = 'ggml-webgpu-moe-tight-tile-bound.patch'
PATCH_SHA256 = '791ccb9ccb1f16c5667eee182e543b2ee625a02a56b0caa72deb928b6aaaf5c6'
INPUTS = {
    False: {'ggml-webgpu.cpp': '3bebf87dd28e65ff400d3d5be87956e27a6ca298550c83bb4437b0ea6f675f96',
            'ggml-webgpu-shader-lib.hpp': 'f8bb0bd001ff26226393d6118ebc5666978e0247a266427a967050d89e9f7399'},
    True: {'ggml-webgpu.cpp': '1dcb3dd11010c4c9c24f2e81df68226ea57700ef1c979e7205dfda05da275cfd',
           'ggml-webgpu-shader-lib.hpp': 'e5a3811a965acf9b681efc6af3f06183ec9c9a90c0d6abb48a743dd3493d74a3'},
}
OUTPUTS = {
    False: {'ggml-webgpu.cpp': '8e5058d5729d74d4fa6b871b7e5bdbb5abfbeef51234df157064059561b5039a'},
    True: {'ggml-webgpu.cpp': '9f60948e2e9cf75ea538f9907a389a00a7d1ee100559daae3e1465108d9971e0'},
}
CONTRACT_INPUTS = {
    'ggml/include/ggml.h': '1aef7a5f465b1c19de305dd6f9a0c303f88e485133a94b9a0ee1cce6345c9101',
    'ggml/src/ggml-webgpu/ggml-webgpu.cpp': '7fa5eca8734b7d31bf467281c698b3356c5750c2f923008c179a4dae426cffe5',
    'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp': '34522d61990406ddc1e46c55da4531e6aa2ad1561bf5bf512ebbb38843dbd49d',
    'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_id_gather.wgsl': '433c7b72fb9382e6e452004bdb602d7be8f2654543de680a28d31d8a263298e7',
    'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_id.wgsl': '667d3a0f8c0e8a8b033d1b3cf861b69d299dee276e76caddaa0ecccbe885a683',
    'ggml/src/ggml-alloc.c': '9664bda3e9c4223f5c94ebfeb3c0381f0065a5091ddae707c09a800f7e419a39',
}



def prepare(work: Path, source: Path, patch_root: Path, env: dict[str, str], *, gpu_profile: bool) -> None:
    for relative, expected in CONTRACT_INPUTS.items():
        if tensor_copy.digest(source / relative) != expected:
            raise ValueError('MoE tile-bound contract changed: ' + relative)
    for name, expected in INPUTS[gpu_profile].items():
        if tensor_copy.digest(work / BASE / name) != expected:
            raise ValueError('MoE tile-bound composition input changed: ' + name)
    patch = patch_root / PATCH_NAME
    if tensor_copy.digest(patch) != PATCH_SHA256:
        raise ValueError('MoE tile-bound patch identity changed')
    for flags in (['--check'], []):
        subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags, str(patch.resolve())],
                       cwd=work, env=env, check=True)
    for name, expected in {**INPUTS[gpu_profile], **OUTPUTS[gpu_profile]}.items():
        if tensor_copy.digest(work / BASE / name) != expected:
            raise ValueError('Unexpected MoE tile-bound output identity: ' + name)
