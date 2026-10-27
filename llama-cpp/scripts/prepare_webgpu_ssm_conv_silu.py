"""Checked F32 SSM convolution/SiLU composition with canonical shader fallback."""
from pathlib import Path
import shutil
import subprocess
import sys
import prepare_webgpu_tensor_copy as tensor_copy

PATCH_NAME = 'ggml-webgpu-ssm-conv-silu.patch'
PATCH_SHA256 = '57ca1a2707c6dbb52893af084043599c01a9427d46fbd7fa0b82dfeded89e0f4'
BASE = Path('ggml/src/ggml-webgpu')
HEADER_NAME = 'ggml-webgpu-ssm-conv-silu-shaders.hpp'
INPUTS = {
    'ggml-webgpu.cpp': '0837380d874c570a0a787edc19b21816638b03527075ea8eb9c37c95e3ca5526',
    'ggml-webgpu-shader-lib.hpp': '9ce1f2986675d50f0b99b3a23877bf8f9dd673760f39e8822ba26655d3a83438',
}
OUTPUTS = {
    'ggml-webgpu.cpp': 'a53ea652fa4bbaec5c8ba503dcf852c672e433086909846fb55f0dc3ede81bfd',
    'ggml-webgpu-shader-lib.hpp': '86567b1e8d20063c7b8a6a12a8b2a10c379ab1c851a0eb18455ed76582fb7ba7',
    'ggml-webgpu-ssm-conv-silu.hpp': 'b4fff88ccabe30bd75ba0dfe310fa2cf83cb8b361678e5f94ccc10cce0acd3dd',
    'wgsl-shaders/ssm_conv_silu.wgsl': 'b2bf3909f6991312675a7dfa662ce7c68c4c158a32328c5ca5ac1d80d28ac35b',
}
HEADER_SHA256 = 'ac43689eb6fc0829918df74eaed2b1dd83587800c2257e231c147da0028f15ec'
CONTRACT_INPUTS = {
    'ggml/include/ggml.h': '475beaa7e959be8c383279d9ce9d5191f37426e6b338270cfb12b26a798129d3',
    'ggml/src/ggml.c': '42b238412cda218552812a37329fab95a2f94be6d8bff5139083450d432f796d',
    'ggml/src/ggml-impl.h': '43564db0238aebb7ed68501e346c194866b5dac218d1d37b26baff9f458c00d3',
    'ggml/src/ggml-alloc.c': '9664bda3e9c4223f5c94ebfeb3c0381f0065a5091ddae707c09a800f7e419a39',
    'ggml/src/ggml-backend.cpp': '43dea13b13c2a15afd83f53515f3488c17072fc107a6ba4a5e0370f90b4fe7ff',
    'ggml/src/ggml-backend-impl.h': '4e861c8e23d3d7b0d99d8e0836e68564af58f8597d75d7bbb732857d7a632fa7',
    'ggml/src/ggml-webgpu/wgsl-shaders/ssm_conv.wgsl': '868a7284cb858811b5ac70f466550340dba0df356b3b9ac1e7568053b7b9c809',
    'ggml/src/ggml-webgpu/wgsl-shaders/unary.wgsl': '7d4d66e2b954b5d1067447bb24082cf19f5fba7e021d7101193e3edc73cefbc1',
    'ggml/src/ggml-webgpu/pre_wgsl.hpp': '0d6f781eb5723311cf856e585566c70ddcc779f0e97fcfe6d0e3f636c35688e5',
    'ggml/src/ggml-webgpu/wgsl-shaders/embed_wgsl.py': '37d4a93bb46b0ff417c31a0c1d3fa4b9db52556556138cb2eaedae9daa8dcbea',
}


def prepare(work: Path, source: Path, patch_root: Path, env: dict[str, str]) -> dict[str, Path]:
    for relative, expected in CONTRACT_INPUTS.items():
        if tensor_copy.digest(source / relative) != expected:
            raise ValueError('SSM convolution/SiLU contract changed: ' + relative)
    for relative, expected in INPUTS.items():
        if tensor_copy.digest(work / BASE / relative) != expected:
            raise ValueError('SSM convolution/SiLU composition input changed: ' + relative)
    for relative in ('ggml-webgpu-ssm-conv-silu.hpp', HEADER_NAME, 'wgsl-shaders/ssm_conv_silu.wgsl'):
        if (source / BASE / relative).exists() or (work / BASE / relative).exists():
            raise ValueError('SSM convolution/SiLU would replace an unreviewed source: ' + relative)
    shader = work / BASE / 'wgsl-shaders/ssm_conv_silu.wgsl'
    shader.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / BASE / 'wgsl-shaders/ssm_conv.wgsl', shader)
    patch = patch_root / PATCH_NAME
    if tensor_copy.digest(patch) != PATCH_SHA256:
        raise ValueError('SSM convolution/SiLU patch identity changed')
    for flags in (['--check'], []):
        subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags, str(patch.resolve())],
                       cwd=work, env=env, check=True, capture_output=True)
    for relative, expected in OUTPUTS.items():
        if tensor_copy.digest(work / BASE / relative) != expected:
            raise ValueError('Unexpected SSM convolution/SiLU output: ' + relative)
    shaders = work / 'ssm-conv-silu-only-wgsl'
    shaders.mkdir()
    shutil.copyfile(shader, shaders / 'ssm_conv_silu.wgsl')
    header = work / HEADER_NAME
    subprocess.run([sys.executable, str(source / BASE / 'wgsl-shaders/embed_wgsl.py'),
                    '--input_dir', str(shaders), '--output_file', str(header)], check=True, capture_output=True)
    if tensor_copy.digest(header) != HEADER_SHA256:
        raise ValueError('Unexpected SSM convolution/SiLU embedding identity')
    return {HEADER_NAME: header, 'ggml-webgpu-ssm-conv-silu.hpp': work / BASE / 'ggml-webgpu-ssm-conv-silu.hpp'}
