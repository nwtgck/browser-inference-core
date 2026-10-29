"""Use the exact routing-independent capacity for existing MoE slot tiles."""
from pathlib import Path
import subprocess

import prepare_webgpu_tensor_copy as tensor_copy

BASE = Path('ggml/src/ggml-webgpu')
PATCH_NAME = 'ggml-webgpu-moe-tight-tile-bound.patch'
PATCH_SHA256 = '791ccb9ccb1f16c5667eee182e543b2ee625a02a56b0caa72deb928b6aaaf5c6'
INPUTS = {
    False: {'ggml-webgpu.cpp': '0fc20b712ab48f0929f194b3c4e62c725c85e3a8374446728c9a12bdf8b5a3d1',
            'ggml-webgpu-shader-lib.hpp': '5bc3e50a164d6a7fed6980c21a208575972ebce118f8b37787ea2fc8fc856e3e'},
    True: {'ggml-webgpu.cpp': '750df4b2507e5d7e58d900750081afd8dcc37c8ae9ca19d479104d2161d71d1e',
           'ggml-webgpu-shader-lib.hpp': 'fabc949c00c6196cd660f6f611ed47ce4ee2c34f54e09109ba9feac545c37880'},
}
OUTPUTS = {
    False: {'ggml-webgpu.cpp': '1ad7f5a177473d4e35f79f0463decc485ed5b9aaf0a3b5a87338ec01dbd757c8'},
    True: {'ggml-webgpu.cpp': 'c4b8dee4e691a7e10a52412a3dcb4478d3992a2946c4011f4b23850e80332e1f'},
}
CONTRACT_INPUTS = {
    'ggml/include/ggml.h': '2b4dd92fa992cb4ef484917a6c62fef5097698744160ccc09214fd0c459c92c7',
    'ggml/src/ggml-webgpu/ggml-webgpu.cpp': '67e0c1b6ffeec0d943eb63aced20c00951d565cf2bbb964e90a99265259dca8e',
    'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp': '52d39024d24f723dc1b88dc8bf8b4ed18f1e50846010c35065cebc364a789e2d',
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
