"""Guard floating-point vec4 matmul loads against actual binding layouts."""
from pathlib import Path
import subprocess

from tests.support.llama_cpp.frozen_webgpu_stages import tensor_copy
BASE = Path('ggml/src/ggml-webgpu')
PATCHES = {
    False: ('ggml-webgpu-matmul-vec4-layout.patch',
            '70c084ac952cd205002abe297cafc5b82b0b68a2ae5b7e751ce415551a8ee50c'),
    True: ('ggml-webgpu-matmul-vec4-layout-profile.patch',
           '8646bdd0339bfbcfaeb86bea10cbf7f95971f363097d8d48527ae3453c1faa7a'),
}
INPUTS = {
    False: {
        'ggml-webgpu.cpp': '67940a20bba8d06205ad2f6cf1e909450c4fbb6805b5eba59f574c3576d69779',
        'ggml-webgpu-shader-lib.hpp': '7756a6ef0fb1521dd896fbb1a34b7431dac71ef52db93003c0db70527dff3536',
    },
    True: {
        'ggml-webgpu.cpp': '5e70b3abb85758df2a1a01ff1ae3640d274c2ff5fc7ad28e94a22c30617051d8',
        'ggml-webgpu-shader-lib.hpp': '46a0ac117e6df47308580043e6fbf40987dee6476ef2bd1ed588bd8c99840059',
    },
}
OUTPUTS = {
    False: {
        'ggml-webgpu.cpp': '3bebf87dd28e65ff400d3d5be87956e27a6ca298550c83bb4437b0ea6f675f96',
        'ggml-webgpu-shader-lib.hpp': 'f8bb0bd001ff26226393d6118ebc5666978e0247a266427a967050d89e9f7399',
    },
    True: {
        'ggml-webgpu.cpp': '1dcb3dd11010c4c9c24f2e81df68226ea57700ef1c979e7205dfda05da275cfd',
        'ggml-webgpu-shader-lib.hpp': 'e5a3811a965acf9b681efc6af3f06183ec9c9a90c0d6abb48a743dd3493d74a3',
    },
}
CONTRACT_INPUTS = {
    'ggml/include/ggml.h': '1aef7a5f465b1c19de305dd6f9a0c303f88e485133a94b9a0ee1cce6345c9101',
    'ggml/src/ggml-webgpu/ggml-webgpu.cpp': '7fa5eca8734b7d31bf467281c698b3356c5750c2f923008c179a4dae426cffe5',
    'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp': '34522d61990406ddc1e46c55da4531e6aa2ad1561bf5bf512ebbb38843dbd49d',
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
                       cwd=work, env=env, check=True)
    for name, expected in OUTPUTS[gpu_profile].items():
        if tensor_copy.digest(work / BASE / name) != expected:
            raise ValueError('Unexpected matmul vec4 layout output identity: ' + name)
