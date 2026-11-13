"""Use the exact routing-independent capacity for existing MoE slot tiles."""
from pathlib import Path
import subprocess

from tests.support.llama_cpp.webgpu_stages import tensor_copy
BASE = Path('ggml/src/ggml-webgpu')
PATCH_NAME = 'ggml-webgpu-moe-tight-tile-bound.patch'
PATCH_SHA256 = '791ccb9ccb1f16c5667eee182e543b2ee625a02a56b0caa72deb928b6aaaf5c6'
INPUTS = {
    False: {'ggml-webgpu.cpp': 'aa95050e1db9dc3524c878e6e848344a7f35c46afe35de894b4d143eff757245',
            'ggml-webgpu-shader-lib.hpp': '29e31f71751110558ce8718c6f421b3673a6cadc512de0c05817aa166ec1492f'},
    True: {'ggml-webgpu.cpp': '883b99505b8d2e4ce976edd12b6ff52449f750ba7d01b6477c4cc96f4aaab2fa',
           'ggml-webgpu-shader-lib.hpp': 'f9f39dfcd8b94c44effbf09451af951dc4cc4cd5504ceb1a7b20cb42e348a6b7'},
}
OUTPUTS = {
    False: {'ggml-webgpu.cpp': 'c42e621e1318a2c70e6a6e651d6abdfac0197415fcdf8f0745c26fc082888232'},
    True: {'ggml-webgpu.cpp': '75f2fcdec2e53eade6c9a008c9e1b86e707a0a37d0064e38e3d513ab07c5a92c'},
}
CONTRACT_INPUTS = {
    'ggml/include/ggml.h': 'bdb0e75be8eeb1fc14a742015025505f5810d8200cbb40bda7e981101413440b',
    'ggml/src/ggml-webgpu/ggml-webgpu.cpp': 'c3e627af0d18c596dd45f051c7eba612a6f8f53f458bceecd557d8704f52b8ee',
    'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp': 'a110a08b87a2b2768c991281e450cc0e89b5c34f67856b58138b7ca9e9aa5f17',
    'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_id_gather.wgsl': '433c7b72fb9382e6e452004bdb602d7be8f2654543de680a28d31d8a263298e7',
    'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_id.wgsl': '667d3a0f8c0e8a8b033d1b3cf861b69d299dee276e76caddaa0ecccbe885a683',
    'ggml/src/ggml-alloc.c': 'c4001b08581b87b80283953acc67d0d2ff13ef0a425179e2bf621f45333bf0fa',
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
