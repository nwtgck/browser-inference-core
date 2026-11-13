"""Checked F32 SSM convolution/SiLU composition with canonical shader fallback."""
from pathlib import Path
from build_tools.source.wgsl import generate as generate_wgsl
import shutil
import subprocess
from tests.support.llama_cpp.frozen_webgpu_stages import tensor_copy
PATCH_NAME = 'ggml-webgpu-ssm-conv-silu.patch'
PATCH_SHA256 = '57ca1a2707c6dbb52893af084043599c01a9427d46fbd7fa0b82dfeded89e0f4'
BASE = Path('ggml/src/ggml-webgpu')
HEADER_NAME = 'ggml-webgpu-ssm-conv-silu-shaders.hpp'
INPUTS = {
    'ggml-webgpu.cpp': 'be9fffb8fb8eb1c85aef121a07ffad0a1182eaa7bed2829ba6c9cb7242b1b506',
    'ggml-webgpu-shader-lib.hpp': 'ade9f9485964536aa1ca390bbc768f10a9a54178ebb69bb7e002ba712d3a61cc',
}
OUTPUTS = {
    'ggml-webgpu.cpp': '972928fc0755daad4e24b2da548d2e62329d5ef474ab5ab7e5214365e88d48cf',
    'ggml-webgpu-shader-lib.hpp': 'a242d104ea3841d3041641fdd33561fbcb9157784a6198d6696a0bb6cee93700',
    'ggml-webgpu-ssm-conv-silu.hpp': 'b4fff88ccabe30bd75ba0dfe310fa2cf83cb8b361678e5f94ccc10cce0acd3dd',
    'wgsl-shaders/ssm_conv_silu.wgsl': 'b2bf3909f6991312675a7dfa662ce7c68c4c158a32328c5ca5ac1d80d28ac35b',
}
HEADER_SHA256 = 'ac43689eb6fc0829918df74eaed2b1dd83587800c2257e231c147da0028f15ec'
CONTRACT_INPUTS = {
    'ggml/include/ggml.h': '1aef7a5f465b1c19de305dd6f9a0c303f88e485133a94b9a0ee1cce6345c9101',
    'ggml/src/ggml.c': 'c0e81f0ba55ad76020ada29b01385a7a742c800971e7c974c425a08839750b49',
    'ggml/src/ggml-impl.h': 'c84c8f9e3c246504602f5cd388ba54e36691e46bf150e4d70ae857bb823c985f',
    'ggml/src/ggml-alloc.c': '9664bda3e9c4223f5c94ebfeb3c0381f0065a5091ddae707c09a800f7e419a39',
    'ggml/src/ggml-backend.cpp': '7ea26359e7eda6957227d6e4c2a24a96aed622cbbca300b066580f7f3d95df5e',
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
