"""Checked F32 SSM convolution/SiLU composition with canonical shader fallback."""
from pathlib import Path
from build_tools.source.wgsl import generate as generate_wgsl
import shutil
import subprocess
from tests.support.llama_cpp.webgpu_stages import tensor_copy
PATCH_NAME = 'ggml-webgpu-ssm-conv-silu.patch'
PATCH_SHA256 = '57ca1a2707c6dbb52893af084043599c01a9427d46fbd7fa0b82dfeded89e0f4'
BASE = Path('ggml/src/ggml-webgpu')
HEADER_NAME = 'ggml-webgpu-ssm-conv-silu-shaders.hpp'
INPUTS = {
    'ggml-webgpu.cpp': '9ec1aac6c9272cb626d74d84de6870f97fbec53c9dd08036e938295babb8ee92',
    'ggml-webgpu-shader-lib.hpp': '4920e53ecff7873774fe92c20951652aa825643aa10f683ed1170c04032b1d85',
}
OUTPUTS = {
    'ggml-webgpu.cpp': '5feda13cfe57aedfecf3a25a4b2f61757860fa22692de214e3c8f9ec3ef54800',
    'ggml-webgpu-shader-lib.hpp': '046a200b6414355b0125aaabf69c39308c7bc191394d227f1a417bcccdca3a2f',
    'ggml-webgpu-ssm-conv-silu.hpp': 'b4fff88ccabe30bd75ba0dfe310fa2cf83cb8b361678e5f94ccc10cce0acd3dd',
    'wgsl-shaders/ssm_conv_silu.wgsl': 'b2bf3909f6991312675a7dfa662ce7c68c4c158a32328c5ca5ac1d80d28ac35b',
}
HEADER_SHA256 = 'ac43689eb6fc0829918df74eaed2b1dd83587800c2257e231c147da0028f15ec'
CONTRACT_INPUTS = {
    'ggml/include/ggml.h': 'bdb0e75be8eeb1fc14a742015025505f5810d8200cbb40bda7e981101413440b',
    'ggml/src/ggml.c': '1008000571b896a22dd3861731ca29e6ec3615898cdc15d9cb529fb171815dfc',
    'ggml/src/ggml-impl.h': '75465b01eba2b1ecae8634898508e71294fbff0f7f96ee9ad2caf5bae5b6c5fd',
    'ggml/src/ggml-alloc.c': 'c4001b08581b87b80283953acc67d0d2ff13ef0a425179e2bf621f45333bf0fa',
    'ggml/src/ggml-backend.cpp': '2f724930ea53dd8f50f26706ecf10a177d56601e668f109cf1cd95052b1b9ad8',
    'ggml/src/ggml-backend-impl.h': '7693def12dcc79a10dd2152e7cb659278756dba7622ebe41937f951dc94944e3',
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
                       cwd=work, env=env, check=True)
    for relative, expected in OUTPUTS.items():
        if tensor_copy.digest(work / BASE / relative) != expected:
            raise ValueError('Unexpected SSM convolution/SiLU output: ' + relative)
    shaders = work / 'ssm-conv-silu-only-wgsl'
    shaders.mkdir()
    shutil.copyfile(shader, shaders / 'ssm_conv_silu.wgsl')
    header = work / HEADER_NAME
    generate_wgsl(shaders, None, header, generator=source / BASE / 'wgsl-shaders/embed_wgsl.py')
    if tensor_copy.digest(header) != HEADER_SHA256:
        raise ValueError('Unexpected SSM convolution/SiLU embedding identity')
    return {HEADER_NAME: header, 'ggml-webgpu-ssm-conv-silu.hpp': work / BASE / 'ggml-webgpu-ssm-conv-silu.hpp'}
