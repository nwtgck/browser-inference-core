"""Guard floating-point vec4 matmul loads against actual binding layouts."""
from pathlib import Path
import subprocess

import prepare_webgpu_tensor_copy as tensor_copy

BASE = Path('ggml/src/ggml-webgpu')
PATCHES = {
    False: ('ggml-webgpu-matmul-vec4-layout.patch',
            '92d233efa2bf7ae7b1fe757d2e649cdbd5f9ed167383d647176ec7ec945065c8'),
    True: ('ggml-webgpu-matmul-vec4-layout-profile.patch',
           'ca08d318898394a726845c3c275fed1897a0c151898260ebda6a29ef44fe8fb4'),
}
INPUTS = {
    False: {
        'ggml-webgpu.cpp': 'd5228a8a36613fa8ae7341bb96eee04f318aed5ccd73aa85e57164d118a70a2b',
        'ggml-webgpu-shader-lib.hpp': 'a302e654b7afd4ed1ba39a2031596c1d8cb1ca3ba19533be4f5b40f28e1281e2',
    },
    True: {
        'ggml-webgpu.cpp': 'c604ef0c3a962bcca567c91a0911f77ae456aad3d8e7d939524a4b73764a98ec',
        'ggml-webgpu-shader-lib.hpp': '629f60c48495fa0c3b1b0bb123f9938100961adac78cc5df232393c220eae171',
    },
}
OUTPUTS = {
    False: {
        'ggml-webgpu.cpp': '0fc20b712ab48f0929f194b3c4e62c725c85e3a8374446728c9a12bdf8b5a3d1',
        'ggml-webgpu-shader-lib.hpp': '5bc3e50a164d6a7fed6980c21a208575972ebce118f8b37787ea2fc8fc856e3e',
    },
    True: {
        'ggml-webgpu.cpp': '750df4b2507e5d7e58d900750081afd8dcc37c8ae9ca19d479104d2161d71d1e',
        'ggml-webgpu-shader-lib.hpp': 'fabc949c00c6196cd660f6f611ed47ce4ee2c34f54e09109ba9feac545c37880',
    },
}
CONTRACT_INPUTS = {
    'ggml/include/ggml.h': '2b4dd92fa992cb4ef484917a6c62fef5097698744160ccc09214fd0c459c92c7',
    'ggml/src/ggml-webgpu/ggml-webgpu.cpp': '67e0c1b6ffeec0d943eb63aced20c00951d565cf2bbb964e90a99265259dca8e',
    'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp': '52d39024d24f723dc1b88dc8bf8b4ed18f1e50846010c35065cebc364a789e2d',
    'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_vec.wgsl': '4fdbbaf3b3d3b7c965786b62a58b6b35ceb6d2f80b29e53f4a027904d3243f18',
    'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_vec_acc.tmpl': '9421bdb7ac569e03e8483365653c28b95e659702f8ff849310b2c077ee64290f',
    'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_id_vec.wgsl': 'cbe87da8b6eb80fcb7905855fe1d212dc5a0afac55b77600c507fbbd2fd7c31d',
    'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_decls.tmpl': 'c461aa2108b752504f47338e41b14ee6570abdd831cd83652e052147c89d69ef',
    'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_reg_tile.wgsl': '2d1cb611caf10b72d20b021772806ca4b3ea4f40c9d7a93941e135dacf859f81',
    'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_subgroup_matrix.wgsl': 'cab4d5d7d034923d9bc092dcf02f92e0d8c9d28d05d29378515dfef82a4e15f0',
    'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_id.wgsl': '667d3a0f8c0e8a8b033d1b3cf861b69d299dee276e76caddaa0ecccbe885a683',
}


def prepare(work: Path, source: Path, patch_root: Path, env: dict[str, str], *, gpu_profile: bool) -> None:
    for relative, expected in CONTRACT_INPUTS.items():
        if tensor_copy.digest(source / relative) != expected:
            raise ValueError('Matmul vec4 layout contract changed: ' + relative)
    for name, expected in INPUTS[gpu_profile].items():
        if tensor_copy.digest(work / BASE / name) != expected:
            raise ValueError('Matmul vec4 layout composition input changed: ' + name)
    name, expected = PATCHES[gpu_profile]
    patch = patch_root / name
    if tensor_copy.digest(patch) != expected:
        raise ValueError('Matmul vec4 layout patch identity changed: ' + name)
    for flags in (['--check'], []):
        subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags, str(patch.resolve())],
                       cwd=work, env=env, check=True, capture_output=True)
    for name, expected in OUTPUTS[gpu_profile].items():
        if tensor_copy.digest(work / BASE / name) != expected:
            raise ValueError('Unexpected matmul vec4 layout output identity: ' + name)
