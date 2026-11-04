"""Use the exact routing-independent capacity for existing MoE slot tiles."""
from pathlib import Path
import subprocess

import prepare_webgpu_tensor_copy as tensor_copy

BASE = Path('ggml/src/ggml-webgpu')
PATCH_NAME = 'ggml-webgpu-moe-tight-tile-bound.patch'
PATCH_SHA256 = '791ccb9ccb1f16c5667eee182e543b2ee625a02a56b0caa72deb928b6aaaf5c6'
INPUTS = {
    False: {'ggml-webgpu.cpp': '5628f36ae648da50afd418c744edcc076a19849e7ac9414aa450f01bd3e99322',
            'ggml-webgpu-shader-lib.hpp': 'f8bb0bd001ff26226393d6118ebc5666978e0247a266427a967050d89e9f7399'},
    True: {'ggml-webgpu.cpp': 'cc00a64403944092716c319b33a59a2016364eb36141a053e3907bbc97f257c7',
           'ggml-webgpu-shader-lib.hpp': 'e5a3811a965acf9b681efc6af3f06183ec9c9a90c0d6abb48a743dd3493d74a3'},
}
OUTPUTS = {
    False: {'ggml-webgpu.cpp': '84bc328d54db3cb715e3a48d7c200c3ea1aa9bc93714fbfd52fb02d16857ee9e'},
    True: {'ggml-webgpu.cpp': 'bffa81f88c0660562164a90a0fa12594470835d453696cbb782ac2cad41ce781'},
}
CONTRACT_INPUTS = {
    'ggml/include/ggml.h': '1aef7a5f465b1c19de305dd6f9a0c303f88e485133a94b9a0ee1cce6345c9101',
    'ggml/src/ggml-webgpu/ggml-webgpu.cpp': '617700adc9a58401414cd6828fa48b4efb3bb0563458e6ebe717360e70c77201',
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
                       cwd=work, env=env, check=True, capture_output=True)
    for name, expected in {**INPUTS[gpu_profile], **OUTPUTS[gpu_profile]}.items():
        if tensor_copy.digest(work / BASE / name) != expected:
            raise ValueError('Unexpected MoE tile-bound output identity: ' + name)
