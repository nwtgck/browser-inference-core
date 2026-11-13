"""Guard floating-point vec4 matmul loads against actual binding layouts."""
from pathlib import Path
import subprocess

from tests.support.llama_cpp.webgpu_stages import tensor_copy
BASE = Path('ggml/src/ggml-webgpu')
PATCHES = {
    False: ('ggml-webgpu-matmul-vec4-layout.patch',
            '70c084ac952cd205002abe297cafc5b82b0b68a2ae5b7e751ce415551a8ee50c'),
    True: ('ggml-webgpu-matmul-vec4-layout-profile.patch',
           '8646bdd0339bfbcfaeb86bea10cbf7f95971f363097d8d48527ae3453c1faa7a'),
}
INPUTS = {
    False: {
        'ggml-webgpu.cpp': '6035c95aa02f9bb940ea7e67d650d078061986b7857ffa94e220406d026b47a5',
        'ggml-webgpu-shader-lib.hpp': '392fb81ddddb2ea569a7d6220704831ee975f96ee2a2131f4d17f2a629ebef56',
    },
    True: {
        'ggml-webgpu.cpp': '5c96cf065bd1a66eacc7f3b4bb020ccc72a15fd051ac329479d558e9e6a872f5',
        'ggml-webgpu-shader-lib.hpp': '07b2d15e128cc5e0c8afdb5569577900f0afe6a84123b7e4c2e0acc36e816967',
    },
}
OUTPUTS = {
    False: {
        'ggml-webgpu.cpp': 'aa95050e1db9dc3524c878e6e848344a7f35c46afe35de894b4d143eff757245',
        'ggml-webgpu-shader-lib.hpp': '29e31f71751110558ce8718c6f421b3673a6cadc512de0c05817aa166ec1492f',
    },
    True: {
        'ggml-webgpu.cpp': '883b99505b8d2e4ce976edd12b6ff52449f750ba7d01b6477c4cc96f4aaab2fa',
        'ggml-webgpu-shader-lib.hpp': 'f9f39dfcd8b94c44effbf09451af951dc4cc4cd5504ceb1a7b20cb42e348a6b7',
    },
}
CONTRACT_INPUTS = {
    'ggml/include/ggml.h': 'bdb0e75be8eeb1fc14a742015025505f5810d8200cbb40bda7e981101413440b',
    'ggml/src/ggml-webgpu/ggml-webgpu.cpp': 'c3e627af0d18c596dd45f051c7eba612a6f8f53f458bceecd557d8704f52b8ee',
    'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp': 'a110a08b87a2b2768c991281e450cc0e89b5c34f67856b58138b7ca9e9aa5f17',
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
